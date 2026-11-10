"""
Streaming pose inference: frames are read in chunks, the detector finds the animal(s), the pose
model predicts keypoints inside each box; only the predictions (frames x individuals x keypoints
x 3) are kept, never the movie.

DETECTOR_STRIDE > 1 (single animal only) runs the detector on every k-th frame and on the last
frame of each chunk, and gives the frames in between a box interpolated linearly between the two
neighbouring detections, enlarged by BOX_MARGIN on every side to cover the movement, but only when
those two boxes overlap or their centres are within one box size of each other: a jump means one
of them is something else (a bright fixture of the rig, for example), and sweeping a box across
the arena would give poses of nothing. The detector is the expensive part on a CPU (about 30x the
pose model for a 1100 x 954 frame).

Single animal: a pose whose mean keypoint confidence is below GAP_MIN_CONFIDENCE is treated as
"not found" (NaN), whatever its box came from (a detection, an interpolated box, a carried box),
so a detector false positive on a rig object does not produce a pose, and gap filling can take
over.

FILL_DETECTOR_GAPS (single animal): when the detector finds no animal in a frame (it misses
small, curled-up or rearing mice under a cable, for example), the box is taken from the previous
frame's pose (its confident keypoints, at least the size of the last detected box) and the pose
model runs on it; this continues frame by frame while the mean keypoint confidence stays at or
above GAP_MIN_CONFIDENCE, and stops otherwise until the detector finds the animal again. The
carried box can follow a non-mouse object (a cable, a shadow) until then, as long as the pose
model is confident enough on it.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import numpy as np

log = logging.getLogger("pose-estimate")

CHUNK = 32
BOX_MARGIN = 0.15
PROGRESS_SECONDS = 60
GAP_KEYPOINT_CONFIDENCE = 0.3   # keypoints that define the box carried from the previous frame
GAP_MIN_KEYPOINTS = 5
GAP_MIN_CONFIDENCE = 0.25       # mean keypoint confidence below which the carried box is dropped


@dataclass
class PoseResult:
    frame_index: np.ndarray        # (T,) source video frame numbers
    poses: np.ndarray              # (T, I, K, 3) x, y (full-frame pixels), confidence; NaN = not found
    boxes: np.ndarray              # (T, I, 5) x, y, w, h, score (score NaN for interpolated boxes)
    detector_runs: int
    seconds: float
    frames_box_from_pose: int = 0  # frames whose box came from the previous frame's pose (FILL_DETECTOR_GAPS)
    frames_box_interpolated: int = 0
    frames_dropped_low_confidence: int = 0   # a box was given, but the pose's mean confidence was too low


def _best_boxes(pred: dict, max_ind: int) -> tuple[np.ndarray, np.ndarray]:
    b = np.asarray(pred.get("bboxes", np.zeros((0, 4))), dtype=np.float32).reshape(-1, 4)
    s = np.asarray(pred.get("bbox_scores", np.zeros(0)), dtype=np.float32).reshape(-1)
    order = np.argsort(-s)[:max_ind]
    return b[order], s[order]


def _box_from_pose(prev: np.ndarray, last_size: np.ndarray, size: tuple[int, int]) -> np.ndarray | None:
    """A box around the previous frame's confident keypoints, at least the size of the last detector box."""
    ok = np.isfinite(prev[:, 0]) & (np.nan_to_num(prev[:, 2]) >= GAP_KEYPOINT_CONFIDENCE)
    if ok.sum() < GAP_MIN_KEYPOINTS:
        return None
    x0, y0 = prev[ok, 0].min(), prev[ok, 1].min()
    x1, y1 = prev[ok, 0].max(), prev[ok, 1].max()
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    w, h = 1.5 * (x1 - x0), 1.5 * (y1 - y0)
    if np.isfinite(last_size).all():
        w, h = max(w, last_size[0]), max(h, last_size[1])
    side = max(w, h, 32.0)
    bx0, by0 = max(0.0, cx - side / 2), max(0.0, cy - side / 2)
    bx1, by1 = min(size[0], cx + side / 2), min(size[1], cy + side / 2)
    if bx1 - bx0 < 8 or by1 - by0 < 8:
        return None
    return np.array([bx0, by0, bx1 - bx0, by1 - by0], dtype=np.float32)


def boxes_close(b0: np.ndarray, b1: np.ndarray) -> bool:
    """Two (x, y, w, h) boxes overlap, or their centres are within one box size of each other."""
    ax0, ay0, aw, ah = (float(v) for v in b0[:4])
    bx0, by0, bw, bh = (float(v) for v in b1[:4])
    iw = min(ax0 + aw, bx0 + bw) - max(ax0, bx0)
    ih = min(ay0 + ah, by0 + bh) - max(ay0, by0)
    if iw > 0 and ih > 0:
        return True
    d = np.hypot((ax0 + aw / 2) - (bx0 + bw / 2), (ay0 + ah / 2) - (by0 + bh / 2))
    return bool(d <= max(aw, ah, bw, bh))


def _interp_box(b0, b1, w: float) -> np.ndarray:
    return (1 - w) * b0 + w * b1


def _grow(box: np.ndarray, frac: float, size: tuple[int, int]) -> np.ndarray:
    x, y, w, h = box
    m = frac * max(w, h)
    x0, y0 = max(0.0, x - m), max(0.0, y - m)
    x1, y1 = min(size[0], x + w + m), min(size[1], y + h + m)
    return np.array([x0, y0, x1 - x0, y1 - y0], dtype=np.float32)


def run(frames, n_total: int, pose_runner, detector_runner, n_keypoints: int, max_ind: int,
        detector_stride: int, frame_size: tuple[int, int], offset: tuple[int, int] = (0, 0),
        label: str = "", fill_gaps: bool = True) -> PoseResult:
    """frames: iterator of (frame_index, RGB array) (cropped if CROP); offset: the crop's (x, y)."""
    stride = detector_stride if (detector_runner is not None and max_ind == 1) else 1
    chunk_len = max(CHUNK, stride) if stride == 1 else stride * max(1, round(CHUNK / stride))
    idx_all = np.zeros(n_total, dtype=np.int64)
    poses = np.full((n_total, max_ind, n_keypoints, 3), np.nan, dtype=np.float32)
    boxes = np.full((n_total, max_ind, 5), np.nan, dtype=np.float32)
    t0 = last_log = time.time()
    done, det_runs = 0, 0
    last_size = np.array([np.nan, np.nan], dtype=np.float32)
    counts = {"from_pose": 0, "interpolated": 0, "dropped": 0}
    buf_idx, buf = [], []

    def flush():
        nonlocal done, det_runs
        n = len(buf)
        if n == 0:
            return
        ctx: list[dict | None] = [None] * n
        if detector_runner is None:          # bottom-up model: the whole frame
            sel = list(range(n))
            inputs = buf
        else:
            det_pos = sorted(set(list(range(0, n, stride)) + [n - 1]))
            preds = detector_runner.inference([buf[i] for i in det_pos])
            det_runs += len(det_pos)
            found = {}
            for i, p in zip(det_pos, preds):
                b, s = _best_boxes(p, max_ind)
                if len(b):
                    found[i] = (b, s)
            for i in range(n):
                if i in found:
                    ctx[i] = {"bboxes": found[i][0], "bbox_scores": found[i][1]}
                    continue
                if stride == 1:                       # every frame was given to the detector
                    continue
                before = [j for j in found if j < i]
                after = [j for j in found if j > i]
                if not before and not after:
                    continue
                if before and after and boxes_close(found[max(before)][0][0], found[min(after)][0][0]):
                    j0, j1 = max(before), min(after)
                    w = (i - j0) / (j1 - j0)
                    box = _interp_box(found[j0][0][0], found[j1][0][0], w)
                    score = min(found[j0][1][0], found[j1][1][0])
                elif before and after:
                    continue                  # the two detections disagree: leave it to gap filling
                else:
                    j = max(before) if before else min(after)
                    if abs(j - i) > stride:   # no detection close enough
                        continue
                    box, score = found[j][0][0], found[j][1][0]
                box = _grow(box, BOX_MARGIN, frame_size)
                ctx[i] = {"bboxes": box[None, :], "bbox_scores": np.array([score], dtype=np.float32),
                          "_interpolated": True}
            sel = [i for i in range(n) if ctx[i] is not None]
            inputs = [(buf[i], {k: v for k, v in ctx[i].items() if not k.startswith("_")}) for i in sel]
        if sel:
            out = pose_runner.inference(inputs)
            for i, p in zip(sel, out):
                bp = np.asarray(p["bodyparts"], dtype=np.float32)        # (n_ind, K, 3)
                k = min(len(bp), max_ind)
                row = done + i
                poses[row, :k] = bp[:k, :, :3]
                if ctx[i] is not None:
                    bb = np.asarray(ctx[i]["bboxes"], dtype=np.float32)[:k]
                    boxes[row, :k, :4] = bb
                    sc = np.asarray(ctx[i]["bbox_scores"], dtype=np.float32)[:k]
                    boxes[row, :k, 4] = np.nan if ctx[i].get("_interpolated") else sc
                if max_ind == 1 and not np.nan_to_num(poses[row, 0, :, 2]).mean() >= GAP_MIN_CONFIDENCE:
                    poses[row] = np.nan               # a box with nothing the model recognises: not found
                    boxes[row] = np.nan
                    ctx[i] = None
                    counts["dropped"] += 1
                elif ctx[i] is not None and ctx[i].get("_interpolated"):
                    counts["interpolated"] += 1
        if fill_gaps and detector_runner is not None and max_ind == 1:
            for i in range(n):
                if ctx[i] is not None:
                    det = boxes[done + i, 0]
                    if np.isfinite(det[4]):
                        last_size[:] = det[2:4]
                    continue
                prev = poses[done + i - 1, 0] if done + i > 0 else None
                box = _box_from_pose(prev, last_size, frame_size) if prev is not None else None
                if box is None:
                    continue
                p = pose_runner.inference([(buf[i], {"bboxes": box[None, :], "bbox_scores": np.array([0.0], np.float32)})])
                if not p:
                    continue
                bp = np.asarray(p[0]["bodyparts"], dtype=np.float32)[:1]
                if np.nanmean(bp[0, :, 2]) < GAP_MIN_CONFIDENCE:
                    continue                          # lost the animal: wait for the detector
                poses[done + i, :1] = bp[:, :, :3]
                boxes[done + i, 0, :4] = box
                boxes[done + i, 0, 4] = -1.0          # marks a box from the previous frame's pose
                counts["from_pose"] += 1
        for i in range(n):
            idx_all[done + i] = buf_idx[i]
        done += n
        buf.clear()
        buf_idx.clear()

    for fi, frame in frames:
        buf_idx.append(fi)
        buf.append(frame)
        if len(buf) >= chunk_len:
            flush()
            now = time.time()
            if now - last_log >= PROGRESS_SECONDS:
                rate = done / (now - t0)
                eta = (n_total - done) / rate if rate > 0 else float("nan")
                log.info("%s pose: %d / %d frames (%.0f%%), %.2f frames/s, about %.0f min left",
                         label, done, n_total, 100 * done / max(1, n_total), rate, eta / 60)
                last_log = now
    flush()
    if done < n_total:
        log.warning("%s: the video ended after %d of %d expected frames", label, done, n_total)
        idx_all, poses, boxes = idx_all[:done], poses[:done], boxes[:done]
    poses[..., 0] += offset[0]
    poses[..., 1] += offset[1]
    boxes[..., 0] += offset[0]
    boxes[..., 1] += offset[1]
    return PoseResult(idx_all, poses, boxes, det_runs, time.time() - t0, counts["from_pose"], counts["interpolated"],
                      counts["dropped"])
