"""
Reading a session NWB (port `session`) with h5py: the file's identity (copied into our NWB so the
files of one session line up), video frame times (ImageSeries / PoseEstimation timestamps), and
any pose already in the file (ndx-pose PoseEstimation, e.g. the authors' own DeepLabCut pose),
which we compare our pose with.

h5py rather than pynwb: it reads any NWB/ndx-pose version without loading extensions, and only
what is needed.
"""
from __future__ import annotations

import csv
import logging
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

import h5py
import numpy as np

log = logging.getLogger("pose-estimate")


def _s(v) -> str | None:
    if v is None:
        return None
    if isinstance(v, bytes):
        return v.decode("utf-8", "replace")
    if isinstance(v, np.ndarray):
        return _s(v[()] if v.shape == () else (v[0] if v.size else None))
    return str(v)


def _times(group: h5py.Group, n: int | None = None) -> np.ndarray | None:
    """A TimeSeries' timestamps, or starting_time + i / rate."""
    if "timestamps" in group:
        return np.asarray(group["timestamps"][()], dtype=np.float64)
    if "starting_time" in group:
        st = group["starting_time"]
        rate = st.attrs.get("rate")
        if rate and n:
            return float(st[()]) + np.arange(n) / float(rate)
    return None


@dataclass
class ImageSeriesRef:
    path: str                   # HDF5 path in the NWB
    external_files: list[str]
    timestamps: np.ndarray | None


@dataclass
class PoseRef:
    path: str
    scorer: str | None
    source_software: str | None
    source_software_version: str | None
    series: dict = field(default_factory=dict)   # keypoint -> (data (T,2), confidence (T,), timestamps (T,))


@dataclass
class SessionNWB:
    path: Path
    identity: dict
    image_series: list[ImageSeriesRef]
    poses: list[PoseRef]


def read_session_nwb(path: Path, load_pose: bool = True) -> SessionNWB:
    with h5py.File(path, "r") as f:
        ident = {
            "identifier": _s(f.get("identifier")[()]) if "identifier" in f else None,
            "session_start_time": _s(f["session_start_time"][()]) if "session_start_time" in f else None,
            "timestamps_reference_time": _s(f["timestamps_reference_time"][()]) if "timestamps_reference_time" in f else None,
            "session_description": _s(f["session_description"][()]) if "session_description" in f else None,
            "subject": {},
        }
        for k in ("experimenter", "institution", "lab", "experiment_description", "keywords"):
            if f"general/{k}" in f:
                v = f[f"general/{k}"][()]
                ident[k] = [_s(x) for x in v] if isinstance(v, np.ndarray) and v.shape else _s(v)
        if "general/subject" in f:
            for k, v in f["general/subject"].items():
                if isinstance(v, h5py.Dataset):
                    ident["subject"][k] = _s(v[()])
        images, poses = [], []

        def visit(name, obj):
            if not isinstance(obj, h5py.Group):
                return
            nt = _s(obj.attrs.get("neurodata_type"))
            if "external_file" in obj and nt and nt.endswith("Series"):
                ext = [_s(x) for x in obj["external_file"][()]]
                images.append(ImageSeriesRef("/" + name, ext, _times(obj, None)))
            elif nt == "PoseEstimation" and load_pose:
                ref = PoseRef("/" + name, _s(obj["scorer"][()]) if "scorer" in obj else None,
                              _s(obj["source_software"][()]) if "source_software" in obj else None,
                              _s(obj["source_software"].attrs.get("version")) if "source_software" in obj else None)
                for k, s in obj.items():
                    if isinstance(s, h5py.Group) and _s(s.attrs.get("neurodata_type")) == "PoseEstimationSeries":
                        data = np.asarray(s["data"][()], dtype=np.float64)
                        conf = np.asarray(s["confidence"][()], dtype=np.float64) if "confidence" in s else np.ones(len(data))
                        ref.series[k] = (data[:, :2], conf, _times(s, len(data)))
                poses.append(ref)

        f.visititems(visit)
    return SessionNWB(path, ident, images, poses)


def ids_match(nwb_path: Path, session_id: str, derive) -> bool:
    return derive(nwb_path) == session_id


# --- frame times --------------------------------------------------------------------------------


@dataclass
class TimeBase:
    times: np.ndarray           # one per video frame (session clock, s)
    source: str                 # nwb:ImageSeries / nwb:PoseEstimation / csv / fps
    detail: str
    warnings: list[str] = field(default_factory=list)


def from_csv(path: Path, n_frames: int) -> np.ndarray:
    frames, times = [], []
    with path.open(newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        cols = {c.lower().strip(): c for c in reader.fieldnames or []}
        if "time" not in cols:
            raise ValueError(f"{path.name}: needs columns frame, time")
        for r in reader:
            times.append(float(r[cols["time"]]))
            frames.append(int(float(r[cols["frame"]])) if "frame" in cols else len(frames))
    t = np.full(n_frames, np.nan)
    for fr, tm in zip(frames, times):
        if 0 <= fr < n_frames:
            t[fr] = tm
    if np.isnan(t).any():
        raise ValueError(f"{path.name}: {int(np.isnan(t).sum())} of {n_frames} frames have no time")
    return t


def choose_timebase(video_name: str, n_frames: int, fps: float, session: SessionNWB | None,
                    all_nwbs: list[SessionNWB], csv_path: Path | None) -> tuple[TimeBase, SessionNWB | None]:
    """Frame times (conventions.md, "Time"): NWB ImageSeries/PoseEstimation timestamps, then
    <stem>_timestamps.csv, then frame / fps. Returns the NWB the times came from (or the session's)."""
    warns: list[str] = []
    # 1. an ImageSeries in any NWB that names this very video file (DANDI external_file)
    candidates = [(nwb, s) for nwb in all_nwbs for s in nwb.image_series
                  if any(PurePosixPath(e).name == video_name for e in s.external_files)]
    # 2. ImageSeries / PoseEstimation of the session's NWB
    if session is not None:
        candidates += [(session, s) for s in session.image_series if (session, s) not in candidates]
    for nwb, s in candidates:
        ts = s.timestamps
        if ts is None or len(ts) == 0:
            continue
        named = any(PurePosixPath(e).name == video_name for e in s.external_files)
        if len(ts) == n_frames:
            return TimeBase(ts, "nwb:ImageSeries", f"{nwb.path.name}{s.path}"), nwb
        if named and len(ts) > n_frames:
            warns.append(f"{nwb.path.name}{s.path} has {len(ts)} timestamps for {n_frames} video frames; "
                         f"using the first {n_frames} (frames missing at the end of the video assumed)")
            return TimeBase(ts[:n_frames], "nwb:ImageSeries", f"{nwb.path.name}{s.path}", warns), nwb
        if named:
            warns.append(f"{nwb.path.name}{s.path} has {len(ts)} timestamps for {n_frames} video frames; not used")
    if session is not None:
        for p in session.poses:
            for _, (_, _, ts) in p.series.items():
                if ts is not None and len(ts) == n_frames:
                    return TimeBase(ts, "nwb:PoseEstimation", f"{session.path.name}{p.path}", warns), session
                break
    if csv_path is not None:
        try:
            return TimeBase(from_csv(csv_path, n_frames), "csv", csv_path.name, warns), session
        except (ValueError, KeyError) as exc:
            warns.append(f"{csv_path.name} not used: {exc}")
    return TimeBase(np.arange(n_frames) / fps, "fps", f"frame / {fps:g} fps", warns), session
