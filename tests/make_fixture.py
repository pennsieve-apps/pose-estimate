"""
Writes the test input of pose-estimate (also the app review's input, app.yml `review`):

  video/blob.avi                    synthetic: 2 s (60 frames, 30 fps, 256 x 256) of a dark, mouse-shaped
                                    blob (body, head, tail) walking a circle on a grey floor; uncompressed AVI
                                    written with numpy only (no codec library needed)
  video/blob_timestamps.csv         its frame times (frame, time), starting at 12.5 s, so the session clock
                                    path through <stem>_timestamps.csv is exercised
  video/sub-F01_ses-OF-F01-S1_clip.mp4
                                    2 s (60 frames) of real top-view open-field video: DANDI 002047
                                    (Plusnin et al. 2026, CC-BY-4.0), session sub-F01 ses-OF-F01-S1, frames
                                    2998-3057 (t = 100.03-102.00 s), copied from tests/data/
  session/sub-F01_ses-OF-F01-S1_behavior.nwb
                                    when pynwb and ndx-pose are installed: a small session NWB with an
                                    ImageSeries naming that clip (frame timestamps on the session clock) and the
                                    authors' own DeepLabCut pose for those frames (ndx-pose PoseEstimation), so
                                    the agreement check runs; without them the real clip runs on frame / fps
  sessions.csv                      session_id, path, subject_id, condition

Layouts: `linked` (ports in their own folders: video/, session/) and `flat` (all at the root).

Usage: python -m tests.make_fixture OUT_DIR [--layout linked|flat|both]
"""
from __future__ import annotations

import argparse
import shutil
import struct
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CLIP = HERE / "data" / "openfield_F01_S1_100s.mp4"
AUTHORS = HERE / "data" / "openfield_F01_S1_100s_authors_pose.npz"
SIZE, N_FRAMES, FPS = 256, 60, 30


def blob_frames(n: int = N_FRAMES, size: int = SIZE, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """(n, size, size, 3) uint8 RGB frames and the true (n, 3, 2) nose / centre / tail-base positions."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    frames = np.empty((n, size, size, 3), np.uint8)
    truth = np.empty((n, 3, 2), np.float32)
    for i in range(n):
        a = 2 * np.pi * i / n
        cx, cy = size / 2 + 60 * np.cos(a), size / 2 + 60 * np.sin(a)
        hd = a + np.pi / 2                                   # heading along the circle
        ux, uy = np.cos(hd), np.sin(hd)
        img = np.full((size, size), 150.0, np.float32) + rng.normal(0, 3, (size, size))
        # body: ellipse 26 x 13 px half-axes
        dx, dy = xx - cx, yy - cy
        along, across = dx * ux + dy * uy, -dx * uy + dy * ux
        img[(along / 26) ** 2 + (across / 13) ** 2 <= 1] = 35
        # head
        hx, hy = cx + 28 * ux, cy + 28 * uy
        img[((xx - hx) ** 2 + (yy - hy) ** 2) <= 9 ** 2] = 30
        # tail: a thin line behind
        for t in np.linspace(26, 70, 60):
            tx, ty = cx - t * ux, cy - t * uy
            img[((xx - tx) ** 2 + (yy - ty) ** 2) <= 2.2 ** 2] = 60
        frames[i] = np.clip(img, 0, 255).astype(np.uint8)[..., None]
        truth[i] = [[cx + 36 * ux, cy + 36 * uy], [cx, cy], [cx - 26 * ux, cy - 26 * uy]]
    return frames, truth


def write_avi(path: Path, frames: np.ndarray, fps: int = FPS) -> None:
    """Uncompressed (BI_RGB, 24-bit BGR, bottom-up) AVI 1.0, readable by FFmpeg / OpenCV."""
    n, h, w, _ = frames.shape
    row = (w * 3 + 3) & ~3
    fsize = row * h

    def chunk(tag: bytes, data: bytes) -> bytes:
        return tag + struct.pack("<I", len(data)) + data + (b"\0" if len(data) % 2 else b"")

    def lst(tag: bytes, data: bytes) -> bytes:
        return b"LIST" + struct.pack("<I", len(data) + 4) + tag + data

    avih = struct.pack("<10I4I", int(1e6 / fps), fsize * fps, 0, 0x10, n, 0, 1, fsize, w, h, 0, 0, 0, 0)
    strh = struct.pack("<4s4sIHHIIIIIIIIhhhh", b"vids", b"DIB ", 0, 0, 0, 0, 1, fps, 0, n, fsize, 0xFFFFFFFF, 0,
                       0, 0, w, h)
    strf = struct.pack("<IiiHHIIiiII", 40, w, h, 1, 24, 0, fsize, 0, 0, 0, 0)
    hdrl = lst(b"hdrl", chunk(b"avih", avih) + lst(b"strl", chunk(b"strh", strh) + chunk(b"strf", strf)))
    movi_parts, index, offset = [], [], 4
    for f in frames:
        bgr = f[::-1, :, ::-1]
        data = np.zeros((h, row), np.uint8)
        data[:, : w * 3] = bgr.reshape(h, w * 3)
        c = chunk(b"00db", data.tobytes())
        index.append(struct.pack("<4sIII", b"00db", 0x10, offset, fsize))
        offset += len(c)
        movi_parts.append(c)
    movi = lst(b"movi", b"".join(movi_parts))
    idx1 = chunk(b"idx1", b"".join(index))
    body = b"AVI " + hdrl + movi + idx1
    path.write_bytes(b"RIFF" + struct.pack("<I", len(body)) + body)


def write_session_nwb(path: Path, clip_name: str) -> bool:
    """A small session NWB for the real clip; False when pynwb / ndx-pose are not installed."""
    try:
        from datetime import datetime, timezone

        from ndx_pose import PoseEstimation, PoseEstimationSeries, Skeleton, Skeletons
        from pynwb import NWBHDF5IO, NWBFile
        from pynwb.file import Subject
        from pynwb.image import ImageSeries
    except ImportError:
        return False
    d = np.load(AUTHORS)
    ts = d["timestamps"].astype(np.float64)
    start = datetime(2024, 11, 6, 13, 28, 30, tzinfo=timezone.utc)
    subject = Subject(subject_id="F01", species="Mus musculus", sex="F", genotype="5xFAD hemizygous")
    nwb = NWBFile(session_description="open field, session S1, mouse F01 (2 s test excerpt of DANDI 002047)",
                  identifier="02c9a16c-3f8a-4c17-8e05-936be690d47f", session_start_time=start, subject=subject)
    nwb.add_acquisition(ImageSeries(name="BehaviorVideo", description="behaviour video (external file)",
                                    external_file=[f"../video/{clip_name}"], starting_frame=[0], format="external",
                                    timestamps=ts, unit="n.a."))
    names = [str(k) for k in d["keypoints"]]
    series = []
    for k in names:
        series.append(PoseEstimationSeries(
            name=k, description=f"DeepLabCut body part {k}", data=d[f"{k}_data"].astype(np.float64), unit="pixels",
            reference_frame="video pixels", confidence=d[f"{k}_confidence"].astype(np.float64),
            timestamps=ts if not series else series[0]))
    skel = Skeleton(name="skeleton", nodes=names, edges=np.zeros((0, 2), dtype="uint8"))
    pm = nwb.create_processing_module("behavior", "authors' DLC pose (excerpt)")
    pm.add(Skeletons(skeletons=[skel]))
    pm.add(PoseEstimation(name="PoseEstimation", pose_estimation_series=series, skeleton=skel,
                          scorer="DLC_resnet152_MiceUniversal152Oct23shuffle1_1000000", source_software="DeepLabCut",
                          source_software_version="2.3"))
    with NWBHDF5IO(str(path), "w") as io:
        io.write(nwb)
    return True


def write(out: Path, layout: str = "linked") -> dict:
    out.mkdir(parents=True, exist_ok=True)
    vdir = out / "video" if layout == "linked" else out
    sdir = out / "session" if layout == "linked" else out
    vdir.mkdir(parents=True, exist_ok=True)
    sdir.mkdir(parents=True, exist_ok=True)
    frames, truth = blob_frames()
    write_avi(vdir / "blob.avi", frames)
    with (vdir / "blob_timestamps.csv").open("w") as fh:
        fh.write("frame,time\n")
        for i in range(N_FRAMES):
            fh.write(f"{i},{12.5 + i / FPS:.6f}\n")
    clip = "sub-F01_ses-OF-F01-S1_clip.mp4"
    shutil.copyfile(CLIP, vdir / clip)
    has_nwb = write_session_nwb(sdir / "sub-F01_ses-OF-F01-S1_behavior.nwb", clip)
    (out / "sessions.csv").write_text(
        "session_id,path,subject_id,condition\n"
        "blob,blob.avi,synthetic,synthetic\n"
        f"sub-F01_ses-OF-F01-S1,{clip},F01,5xFAD\n")
    np.save(out / ".blob_truth.npy", truth)
    return {"session_nwb": has_nwb}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("out")
    ap.add_argument("--layout", choices=("linked", "flat", "both"), default="both")
    a = ap.parse_args()
    out = Path(a.out)
    layouts = ("linked", "flat") if a.layout == "both" else (a.layout,)
    for lay in layouts:
        info = write(out / lay if a.layout == "both" else out, lay)
        print(f"wrote {lay} fixture ({'with' if info['session_nwb'] else 'without'} session NWB)")


if __name__ == "__main__":
    main()
