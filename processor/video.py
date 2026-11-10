"""
Video preparation: probe, validate, trim, crop, frame stride, streaming frame reader, and the
labelled QC clip.

Everything goes through OpenCV, whose wheels bundle an LGPL build of FFmpeg: decoding, probing,
and the QC clip, encoded as VP9 in WebM (libvpx, BSD), with MJPEG/AVI as the fallback when the
build has no VP9 encoder. Nothing GPL is used or shipped: no ffmpeg executable (imageio-ffmpeg's is
a GPL build) and no PyAV (its wheels link a GPL FFmpeg with libx264).
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger("pose-estimate")


class VideoError(RuntimeError):
    pass


@dataclass
class VideoInfo:
    path: Path
    width: int
    height: int
    fps: float                 # container's nominal frame rate
    n_frames: int              # counted by decoding (grab) the whole stream
    n_frames_header: int       # what the container says
    codec: str
    duration_s: float          # n_frames / fps
    pts_s: np.ndarray = field(repr=False, default_factory=lambda: np.zeros(0))   # container PTS per frame
    vfr: bool = False          # container timestamps deviate from frame / fps by more than half a frame
    rotation_deg: float = 0.0  # display rotation in the container metadata (frames are analysed as stored)

    def as_dict(self) -> dict:
        return {"file": self.path.name, "width": self.width, "height": self.height, "fps": round(self.fps, 6),
                "n_frames": self.n_frames, "n_frames_header": self.n_frames_header, "codec": self.codec,
                "duration_s": round(self.duration_s, 3), "variable_frame_rate": self.vfr,
                "container_duration_s": round(float(self.pts_s[-1]) + 1.0 / self.fps, 3) if len(self.pts_s) else None,
                "rotation_deg": self.rotation_deg}


def bottom_up_raw_avi(path: Path) -> bool:
    """An uncompressed AVI stored bottom-up (positive DIB height): OpenCV 4.14's FFmpeg reader on
    linux/amd64 corrupts memory on these (negative line size), killing the process."""
    if path.suffix.lower() != ".avi":
        return False
    try:
        with path.open("rb") as fh:
            head = fh.read(1 << 16)
    except OSError:
        return False
    i = head.find(b"strf")
    if i < 0 or len(head) < i + 28:
        return False
    import struct
    _, _, height, _, bits, compression = struct.unpack("<IiiHHI", head[i + 8:i + 28])
    return compression == 0 and height > 0 and bits in (24, 32)


def probe(path: Path) -> VideoInfo:
    """Open, validate and count frames (decoding headers only; seconds for a 10-min video)."""
    if bottom_up_raw_avi(path):
        raise VideoError(f"{path.name} is an uncompressed bottom-up AVI, which OpenCV cannot read safely; "
                         "re-encode it (e.g. ffmpeg -i in.avi -c:v libx264 -crf 18 out.mp4)")
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise VideoError(f"cannot open {path.name} as a video (unsupported codec or not a video file)")
    try:
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        n_header = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fourcc = int(cap.get(cv2.CAP_PROP_FOURCC) or 0)
        codec = fourcc.to_bytes(4, "little").decode("ascii", "replace").strip("\x00 ") if fourcc else "?"
        try:
            rotation = float(cap.get(cv2.CAP_PROP_ORIENTATION_META) or 0.0)
        except Exception:  # noqa: BLE001 - older OpenCV
            rotation = 0.0
        pts = []
        while cap.grab():
            pts.append(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0)
    finally:
        cap.release()
    n = len(pts)
    if n == 0 or w <= 0 or h <= 0:
        raise VideoError(f"{path.name}: no decodable video frames")
    if not (0.5 <= fps <= 1000):
        raise VideoError(f"{path.name}: implausible frame rate {fps} in the container; re-encode the video with a fixed rate")
    pts_s = np.asarray(pts, dtype=np.float64)
    vfr = bool(n > 2 and np.nanmax(np.abs((pts_s - pts_s[0]) - np.arange(n) / fps)) > 0.5 / fps)
    return VideoInfo(path, w, h, fps, n, n_header, codec, n / fps, pts_s, vfr, rotation)


def check_crop(crop, info: VideoInfo) -> tuple[int, int, int, int] | None:
    if crop is None:
        return None
    x, y, w, h = crop
    if x + w > info.width or y + h > info.height:
        raise VideoError(f"CROP {x},{y},{w},{h} is outside the {info.width}x{info.height} frame")
    return crop


def read_frames(path: Path, indices: np.ndarray, crop=None):
    """Yield (index, RGB frame) for the sorted frame indices, decoding sequentially (grab() skips)."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise VideoError(f"cannot open {path.name}")
    try:
        want = iter(indices.tolist())
        nxt = next(want, None)
        i = 0
        while nxt is not None:
            if i < nxt:
                if not cap.grab():
                    break
                i += 1
                continue
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if crop is not None:
                x, y, w, h = crop
                frame = np.ascontiguousarray(frame[y:y + h, x:x + w])
            yield nxt, frame
            i += 1
            nxt = next(want, None)
    finally:
        cap.release()


class ClipWriter:
    """The QC clip through OpenCV's LGPL FFmpeg: VP9 in WebM, else MJPEG in AVI (`suffix` says which)."""

    CODECS = ((".webm", "VP90", "vp9"), (".avi", "MJPG", "mjpeg"))

    def __init__(self, stem: Path, width: int, height: int, fps: float):
        self.width, self.height = width - width % 2, height - height % 2
        self._cv = None
        for suffix, fourcc, name in self.CODECS:
            path = stem.with_name(stem.name + suffix)
            w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*fourcc), fps, (self.width, self.height))
            if w.isOpened():
                self._cv, self.path, self.codec = w, path, name
                break
            w.release()
            path.unlink(missing_ok=True)
            log.warning("QC clip: OpenCV cannot write %s here; trying the next format", name)
        if self._cv is None:
            raise VideoError("OpenCV can write neither VP9/WebM nor MJPEG/AVI")

    def write(self, rgb: np.ndarray) -> None:
        rgb = np.ascontiguousarray(rgb[: self.height, : self.width])
        self._cv.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

    def close(self) -> None:
        self._cv.release()


def write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2, default=_jsonable) + "\n")


def _jsonable(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(type(o))
