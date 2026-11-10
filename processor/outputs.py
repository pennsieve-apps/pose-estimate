"""
Writing the pose: NWB with ndx-pose (PoseEstimation, PoseEstimationSeries per keypoint, Skeleton)
and the DeepLabCut .h5 table.
"""
from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

log = logging.getLogger("pose-estimate")

REFERENCE_FRAME = ("(0, 0) is the top-left corner of the original (uncropped) video frame; x grows to the "
                   "right and y downwards, in pixels")
CONFIDENCE_DEFINITION = ("DeepLabCut keypoint confidence (heatmap peak score, 0-1); NaN where the animal "
                         "was not detected")


def _dt(text: str | None) -> datetime | None:
    if not text:
        return None
    try:
        d = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    return d if d.tzinfo else d.replace(tzinfo=timezone.utc)


def regular_rate(times: np.ndarray) -> float | None:
    """1 / median interval when every frame interval is within 0.1 % of the median (no gaps), else None."""
    t = np.asarray(times, dtype=np.float64)
    if len(t) < 2:
        return None
    d = np.diff(t)
    med = float(np.median(d))
    if med <= 0 or np.any(np.abs(d - med) > 1e-3 * med):
        return None
    return 1.0 / med


def _as_list(v) -> list[str]:
    if not v:
        return []
    return [v] if isinstance(v, str) else [str(x) for x in v]


def individual_names(n: int) -> list[str]:
    return ["animal0"] if n == 1 else [f"animal{i}" for i in range(n)]


def write_nwb(path: Path, *, session_id: str, identity: dict | None, video_path: Path, video_size: tuple[int, int],
              times: np.ndarray, poses: np.ndarray, bodyparts: list[str], edges: list[tuple[int, int]],
              scorer: str, software_version: str, model_description: str, labeled_clip: str | None,
              sheet_columns: dict, app_version: str, licence_notice: str | None = None) -> list[str]:
    """Writes the NWB; returns the subject fields DANDI requires that are unknown (not DANDI-ready)."""
    from hdmf.backends.hdf5.h5_utils import H5DataIO
    from ndx_pose import PoseEstimation, PoseEstimationSeries, Skeleton, Skeletons
    from pynwb import NWBHDF5IO, NWBFile
    from pynwb.file import Subject

    ident = identity or {}
    start = _dt(ident.get("session_start_time"))
    if start is None:
        start = datetime.fromtimestamp(video_path.stat().st_mtime, tz=timezone.utc)
        start_note = " session_start_time is the video file's modification time (no session NWB)."
    else:
        start_note = ""
    ref_time = _dt(ident.get("timestamps_reference_time")) or start
    # a new identifier per file (NWB/DANDI require it unique); the source's goes into notes
    identifier = str(uuid.uuid4())
    source_note = ""
    if ident.get("identifier"):
        source_note = (f" Derived from {ident.get('source_file') or 'the session NWB'} (identifier "
                       f"{ident['identifier']}) by pose-estimate {app_version}.")
    subject = None
    subj = dict(ident.get("subject") or {})
    for k in ("subject_id", "species", "sex", "age", "date_of_birth", "genotype", "strain"):
        if not subj.get(k) and sheet_columns.get(k):          # sessions.csv fills what the NWB lacks
            subj[k] = sheet_columns[k]
    if subj:
        allowed = ("subject_id", "species", "sex", "age", "genotype", "strain", "description", "weight")
        kw = {k: subj[k] for k in allowed if subj.get(k)}
        dob = _dt(subj.get("date_of_birth"))
        if dob:
            kw["date_of_birth"] = dob
        subject = Subject(**kw)
    nwb = NWBFile(
        session_description=ident.get("session_description") or f"Pose estimated from video {video_path.name}",
        identifier=identifier, session_start_time=start, timestamps_reference_time=ref_time,
        subject=subject, source_script=f"pose-estimate {app_version} (Pennsieve)", source_script_file_name="processor/main.py",
        experimenter=ident.get("experimenter") or None, institution=ident.get("institution") or None,
        lab=ident.get("lab") or None, experiment_description=ident.get("experiment_description") or None,
        keywords=list(dict.fromkeys(_as_list(ident.get("keywords")) + ["pose estimation", "DeepLabCut", "behavior"])),
        session_id=ident.get("session_id") or session_id,
        notes=f"Pose from {video_path.name} by pose-estimate {app_version}: {model_description}.{start_note}{source_note}"
              + (f" {licence_notice}" if licence_notice else ""))
    camera = nwb.create_device(name="camera", description=f"camera that recorded {video_path.name}")
    behavior = nwb.create_processing_module(name="behavior", description="Pose estimated from video (pose-estimate)")
    n_ind = poses.shape[1]
    names = individual_names(n_ind)
    skeletons, estimations = [], []
    rate = regular_rate(times)
    edge_arr = np.array(edges, dtype="uint8" if len(bodyparts) < 256 else "uint16").reshape(-1, 2)
    for ii, ind in enumerate(names):
        skel = Skeleton(name=f"skeleton_{ind}" if n_ind > 1 else "skeleton", nodes=list(bodyparts), edges=edge_arr,
                        subject=subject if (subject is not None and n_ind == 1) else None)
        skeletons.append(skel)
        series = []
        ts_ref = None
        for k, bp in enumerate(bodyparts):
            xy = poses[:, ii, k, :2].astype(np.float32)
            conf = poses[:, ii, k, 2].astype(np.float32)
            kw = dict(name=bp, description=f"{bp} ({ind}), estimated by {scorer}",
                      data=H5DataIO(xy, compression="gzip", compression_opts=4),
                      unit="pixels", reference_frame=REFERENCE_FRAME,
                      confidence=H5DataIO(conf, compression="gzip", compression_opts=4),
                      confidence_definition=CONFIDENCE_DEFINITION)
            if rate is not None:          # regular frames: starting_time + rate (conventions.md, "Time")
                s = PoseEstimationSeries(starting_time=float(times[0]), rate=rate, **kw)
            elif ts_ref is None:
                s = PoseEstimationSeries(timestamps=H5DataIO(np.asarray(times, dtype=np.float64), compression="gzip"), **kw)
                ts_ref = s
            else:
                s = PoseEstimationSeries(timestamps=ts_ref, **kw)
            series.append(s)
        pe = PoseEstimation(
            name="PoseEstimation" if n_ind == 1 else f"PoseEstimation_{ind}",
            pose_estimation_series=series,
            description=f"{model_description}. Times are the video frame times on the session clock.",
            original_videos=[video_path.name], labeled_videos=[labeled_clip] if labeled_clip else None,
            dimensions=np.array([list(video_size)], dtype="uint16"), devices=[camera],
            scorer=scorer, source_software="DeepLabCut", source_software_version=software_version,
            skeleton=skel)
        estimations.append(pe)
    behavior.add(Skeletons(skeletons=skeletons))
    for pe in estimations:
        behavior.add(pe)
    with NWBHDF5IO(str(path), "w") as io:
        io.write(nwb)
    missing = [k for k in ("species", "sex") if not subj.get(k)]
    if not (subj.get("age") or subj.get("date_of_birth")):
        missing.append("age or date_of_birth")
    if not subj.get("subject_id"):
        missing.insert(0, "subject_id")
    return missing


def write_dlc_h5(path: Path, poses: np.ndarray, frame_index: np.ndarray, bodyparts: list[str], scorer: str) -> None:
    """DeepLabCut's table: columns (scorer, bodyparts, coords) for one animal, (scorer, individuals,
    bodyparts, coords) for several; coords x, y, likelihood; rows indexed by video frame number."""
    import pandas as pd

    n_t, n_ind, n_k, _ = poses.shape
    if n_ind == 1:
        cols = pd.MultiIndex.from_product([[scorer], bodyparts, ["x", "y", "likelihood"]],
                                          names=["scorer", "bodyparts", "coords"])
        data = poses[:, 0].reshape(n_t, n_k * 3)
    else:
        cols = pd.MultiIndex.from_product([[scorer], individual_names(n_ind), bodyparts, ["x", "y", "likelihood"]],
                                          names=["scorer", "individuals", "bodyparts", "coords"])
        data = poses.reshape(n_t, n_ind * n_k * 3)
    df = pd.DataFrame(data.astype(np.float64), columns=cols, index=pd.Index(frame_index, name="frame"))
    df.to_hdf(path, key="df_with_missing", format="table", mode="w")
