"""
pose-estimate: markerless pose from video, the first step of the Markerless Behaviour workflow
(pose-estimate -> pose-kinematics -> behavior-syllables -> neuro-report).

Per video (one session per video):
  1. probe and validate the video, choose frame times (session NWB ImageSeries / PoseEstimation
     timestamps, <stem>_timestamps.csv, or frame / fps), trim (T_START, DURATION), frame stride,
     crop;
  2. pose with DeepLabCut 3: SuperAnimal (MODEL) zero-shot, optionally adapted to the video
     (VIDEO_ADAPT), or the lab's own model (port `model`);
  3. write <session_id>_pose.nwb (ndx-pose), <session_id>_pose.h5 (DeepLabCut), a labelled QC clip,
     <session_id>_video.json, figures, and, when the session NWB already holds a pose (e.g. the
     authors' DeepLabCut pose on DANDI), the pixel agreement between the two.

Output (one port, `pose`): $OUTPUT_DIR/pose-estimate/[LABEL/] with the per-session files and
sessions.csv, keypoint_qc.csv, summary.json, provenance.json.
"""
from __future__ import annotations

import csv
import json
import logging
import os
import resource
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", stream=sys.stdout)
log = logging.getLogger("pose-estimate")
for noisy in ("matplotlib", "PIL", "fontTools", "h5py", "hdmf", "pynwb", "deeplabcut", "timm", "numexpr"):
    logging.getLogger(noisy).setLevel(logging.WARNING)
# DeepLabCut warns about every empty field of the SuperAnimal project configs; not actionable here
logging.getLogger("deeplabcut.core.config.project_config").setLevel(logging.ERROR)

import numpy as np  # noqa: E402

from processor import params as PR  # noqa: E402
from processor.inputs import derive_session_id, find_inputs, sheet_match  # noqa: E402
from processor.resources import cgroup_memory_peak_mb, choose_device, resolve_threads  # noqa: E402

APP = "pose-estimate"
REPO_ROOT = Path(__file__).resolve().parent.parent
CITATIONS = ["10.1038/s41593-018-0209-y", "10.1038/s41467-024-48792-2", "10.1038/s41596-019-0176-0"]


def version() -> str:
    for p in (REPO_ROOT / "VERSION", Path("/processor/VERSION")):
        if p.is_file():
            return p.read_text().strip()
    return "0.0.0"


def variant() -> str:
    flavour = os.environ.get("TORCH_FLAVOUR", "")
    return {"cpu": "cpu", "cuda": "gpu"}.get(flavour, flavour or "local")


def tool_versions() -> dict:
    from importlib.metadata import PackageNotFoundError, version as v
    out = {}
    for pkg in ("deeplabcut", "torch", "torchvision", "pynwb", "ndx-pose", "hdmf", "opencv-python-headless",
                "numpy", "pandas", "tables"):
        try:
            out[pkg] = v(pkg)
        except PackageNotFoundError:
            pass
    return out


def peak_rss_mb() -> float:
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return round(r / (1024 * 1024) if sys.platform == "darwin" else r / 1024, 1)


def _r(x, n=4):
    return None if x is None or (isinstance(x, float) and not np.isfinite(x)) else round(float(x), n)


def write_json(path: Path, obj) -> None:
    from processor.video import write_json as wj
    wj(path, obj)


# --- per session ---------------------------------------------------------------------------------


def frame_selection(times: np.ndarray, p: dict) -> np.ndarray:
    idx = np.arange(len(times))
    m = np.ones(len(times), dtype=bool)
    if p["T_START"] > 0:
        m &= times >= p["T_START"]
    if p["DURATION"] > 0:
        t0 = times[m][0] if m.any() else times[0]
        m &= times < max(p["T_START"], t0) + p["DURATION"]
    return idx[m][:: p["FRAME_STRIDE"]]


def qc_window(times: np.ndarray, pose: np.ndarray, seconds: float, pcutoff: float) -> tuple[int, int]:
    """[start, stop) rows of the most active, well-tracked window of `seconds` (animal 0)."""
    n = len(times)
    if n == 0 or seconds <= 0:
        return 0, 0
    dt = np.median(np.diff(times)) if n > 1 else 1.0
    L = max(1, min(n, int(round(seconds / dt))))
    conf = np.nan_to_num(pose[:, 0, :, 2], nan=0.0)
    xy = np.where((conf >= pcutoff)[..., None], pose[:, 0, :, :2], np.nan)
    with np.errstate(all="ignore"):
        cen = np.nanmedian(xy, axis=1)
    step = np.linalg.norm(np.diff(cen, axis=0), axis=1)
    step = np.concatenate([[0.0], np.nan_to_num(step, nan=0.0)])
    step = np.minimum(step, np.nanpercentile(step, 99) if np.any(step > 0) else 0)
    good = (conf >= pcutoff).mean(1)
    best, best_s = 0, -1.0
    for s in range(0, max(1, n - L + 1), max(1, L // 4)):
        score = step[s:s + L].sum() * good[s:s + L].mean()
        if score > best_s:
            best, best_s = s, score
    return best, best + L


def process_session(*, sid, video, sheet_cols, sessions_nwb, nwb_by_sid, inp, p, model_base, device, out_dir, notice,
                    runners_cache, warnings_all) -> dict:
    from processor import compare as CP
    from processor import figures as FG
    from processor import infer as IN
    from processor import model as MD
    from processor import outputs as OU
    from processor import sessionnwb as SN
    from processor import video as VD

    t_session = time.time()
    warns: list[str] = []
    info = VD.probe(video)
    crop = VD.check_crop(p["CROP"], info)
    session = nwb_by_sid.get(sid)
    tb, src_nwb = SN.choose_timebase(video.name, info.n_frames, info.fps, session, sessions_nwb,
                                     inp.timestamp_csvs.get(Path(video.name).stem))
    if session is None and src_nwb is not None:
        session = src_nwb
    warns += tb.warnings
    times_all = tb.times
    eff_fps = (len(times_all) - 1) / (times_all[-1] - times_all[0]) if len(times_all) > 1 and times_all[-1] > times_all[0] else info.fps
    drift_pct = 100.0 * (eff_fps - info.fps) / info.fps
    if info.vfr and tb.source == "fps":
        warns.append(f"{video.name}: the container's frame times are irregular (variable frame rate) but no frame "
                     "timestamps were given; times are frame / fps")
    sel = frame_selection(times_all, p)
    if len(sel) == 0:
        raise ValueError(f"no frames in the window T_START={p['T_START']} DURATION={p['DURATION']} "
                         f"(video times {times_all[0]:.2f}-{times_all[-1]:.2f} s)")
    times = times_all[sel]
    log.info("%s: %s, %dx%d, %d frames at %.3f fps (%.1f s), times from %s (%s); analysing %d frames %.2f-%.2f s%s",
             sid, video.name, info.width, info.height, info.n_frames, info.fps, info.duration_s, tb.source, tb.detail,
             len(sel), times[0], times[-1], f", crop {crop}" if crop else "")

    # model (adapted per video when asked)
    model = model_base
    workdir = MD.tmpdir(f"pose-{sid}-")
    try:
        if p["VIDEO_ADAPT"] and model.kind == "superanimal":
            import copy
            import cv2
            model = copy.deepcopy(model_base)
            n_ad = min(p["ADAPT_FRAMES"], len(sel))
            ad_idx = np.unique(sel[np.linspace(0, len(sel) - 1, n_ad).round().astype(int)])
            clip = workdir / f"{sid}_adapt.mp4"
            w = (crop[2], crop[3]) if crop else (info.width, info.height)
            vw = cv2.VideoWriter(str(clip), cv2.VideoWriter_fourcc(*"mp4v"), min(30.0, info.fps), w)
            for _, fr in VD.read_frames(video, ad_idx, crop):
                vw.write(cv2.cvtColor(fr, cv2.COLOR_RGB2BGR))
            vw.release()
            ta = time.time()
            log.info("%s: video adaptation on %d frames sampled across the analysed window ...", sid, len(ad_idx))
            model = MD.adapt(model, clip, workdir, device, p["BATCH_SIZE"], p["MAX_INDIVIDUALS"])
            log.info("%s: video adaptation done in %.0f s", sid, time.time() - ta)
            pose_runner, det_runner = model.runners(device, p["BATCH_SIZE"], p["MAX_INDIVIDUALS"])
        else:
            if "runners" not in runners_cache:
                runners_cache["runners"] = model.runners(device, p["BATCH_SIZE"], p["MAX_INDIVIDUALS"])
            pose_runner, det_runner = runners_cache["runners"]
        if not model.top_down:
            det_runner = None
        size = (crop[2], crop[3]) if crop else (info.width, info.height)
        res = IN.run(VD.read_frames(video, sel, crop), len(sel), pose_runner, det_runner, len(model.bodyparts),
                     p["MAX_INDIVIDUALS"], p["DETECTOR_STRIDE"], size, (crop[0], crop[1]) if crop else (0, 0), sid,
                     fill_gaps=p["FILL_DETECTOR_GAPS"])
    finally:
        MD.cleanup(workdir)
    if len(res.frame_index) < len(sel):
        keep = len(res.frame_index)
        sel, times = sel[:keep], times[:keep]
    poses = res.poses
    bodyparts = model.bodyparts
    pc = p["PCUTOFF"]
    conf0 = poses[:, 0, :, 2]
    from processor.infer import GAP_MIN_CONFIDENCE
    # "found": a pose whose mean keypoint confidence reaches GAP_MIN_CONFIDENCE (0.25)
    detected = np.nan_to_num(conf0, nan=0.0).mean(1) >= GAP_MIN_CONFIDENCE
    c0 = np.nan_to_num(conf0, nan=0.0)
    mean_conf = c0.mean(0)
    frac_low = (c0 < pc).mean(0)
    frame_ok = (c0 >= pc).mean(1) >= 0.5
    fps_proc = len(sel) / res.seconds if res.seconds > 0 else None
    log.info("%s: pose on %d frames in %.0f s (%.2f frames/s; detector on %d frames, box from the previous pose in %d); "
             "interpolated box in %d, dropped as low-confidence %d; animal found in %.1f%% of frames, mean confidence %.2f",
             sid, len(sel), res.seconds, fps_proc or 0, res.detector_runs, res.frames_box_from_pose,
             res.frames_box_interpolated, res.frames_dropped_low_confidence, 100 * detected.mean(), float(c0.mean()))

    # outputs
    prefix = out_dir / sid
    clip_name = None
    a, b = qc_window(times, poses, p["QC_CLIP_SECONDS"], pc)
    clip_written = False
    if b > a:
        scale = min(1.0, 720.0 / info.width)
        out_w, out_h = int(info.width * scale) // 2 * 2, int(info.height * scale) // 2 * 2
        fps_clip = min(60.0, info.fps / p["FRAME_STRIDE"])
        writer = VD.ClipWriter(out_dir / f"{sid}_labeled", out_w, out_h, fps_clip)
        clip_name = writer.path.name
        if writer.codec != "vp9":
            warns.append(f"QC clip written as {writer.codec} ({clip_name}): this OpenCV build has no VP9 encoder")
        row = {fi: r for r, fi in enumerate(sel[a:b])}
        import cv2
        for fi, fr in VD.read_frames(video, sel[a:b], None):
            pz = poses[a + row[fi]].copy()
            img = FG.draw_pose(fr, pz, bodyparts, model.edges, pc, radius=max(3, info.width // 220))
            if scale < 1:
                img = cv2.resize(img, (out_w, out_h), interpolation=cv2.INTER_AREA)
            cv2.putText(img, f"{sid}  t={times[a + row[fi]]:.2f}s", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (255, 255, 255), 1, cv2.LINE_AA)
            writer.write(img)
        writer.close()
        clip_written = True
    ident = session.identity if session is not None else None
    missing = OU.write_nwb(Path(f"{prefix}_pose.nwb"), session_id=sid, identity=ident, video_path=video,
                 video_size=(info.width, info.height), times=times, poses=poses, bodyparts=bodyparts,
                 edges=model.edges, scorer=model.scorer, software_version=tool_versions().get("deeplabcut", "?"),
                 model_description=model.description, labeled_clip=clip_name if clip_written else None,
                 sheet_columns=sheet_cols, app_version=version(), licence_notice=notice)
    if missing:
        warns.append(f"{sid}_pose.nwb is valid NWB but not DANDI-ready: subject {', '.join(missing)} unknown (link the "
                     "session NWB, or add these columns to sessions.csv)")
    OU.write_dlc_h5(Path(f"{prefix}_pose.h5"), poses, sel, bodyparts, model.scorer)
    vjson = {"session_id": sid, **info.as_dict(), "time_source": tb.source, "time_source_detail": tb.detail,
             "fps_effective": _r(eff_fps, 6), "t_first_frame": _r(times_all[0], 6), "t_last_frame": _r(times_all[-1], 6),
             "analysed": {"t_start": _r(times[0], 6), "t_stop": _r(times[-1], 6), "n_frames": int(len(sel)),
                          "frame_stride": p["FRAME_STRIDE"], "first_frame": int(sel[0]), "last_frame": int(sel[-1]),
                          "crop": list(crop) if crop else None},
             "qc_clip": {"file": clip_name, "t_start": _r(times[a], 3), "t_stop": _r(times[b - 1], 3)} if clip_written else None}
    if abs(drift_pct) > 0.1:
        vjson["drift_pct"] = _r(drift_pct, 4)
    write_json(out_dir / f"{sid}_video.json", vjson)

    # agreement with a pose already in the session NWB
    agr = None
    if session is not None and session.poses and p["MAX_INDIVIDUALS"] >= 1:
        ref = session.poses[0]
        agr = CP.compare(times, poses[:, 0], bodyparts, ref, pc)
        if agr is None:
            warns.append(f"{session.path.name}: has a PoseEstimation but none of its keypoints match ours")

    # figures
    figs = []
    ex_rows = np.flatnonzero(detected)
    if len(ex_rows):
        pick = ex_rows[np.linspace(0, len(ex_rows) - 1, min(4, len(ex_rows))).round().astype(int)]
        want = {int(sel[r]): r for r in pick}
        frames = []
        refs = [] if agr else None
        for fi, fr in VD.read_frames(video, np.array(sorted(want)), None):
            r = want[fi]
            frames.append((fi, float(times[r]), fr, poses[r]))
            if agr:
                ref = session.poses[0]
                d = {}
                rt = next(s[2] for s in ref.series.values() if s[2] is not None)
                j = int(np.argmin(np.abs(rt - times[r])))
                for ours, theirs in agr["matched_keypoints"].items():
                    d[ours] = tuple(ref.series[theirs][0][j, :2])
                refs.append(d)
        f1 = f"{sid}_example_frames.png"
        FG.example_frames(out_dir / f1, sid, frames, bodyparts, model.edges, pc, refs)
        figs.append({"path": f1, "caption": (
            f"{sid}: the estimated keypoints on {len(frames)} frames spread over the analysed window (top: whole "
            f"frame, dashed box = zoomed area; bottom: zoom on the animal). Dots are coloured by body region; hollow "
            f"rings are keypoints below the confidence cut-off ({pc:g}), which downstream steps treat as missing. "
            + ("Yellow crosses are the pose stored in the session NWB, for comparison. " if agr else "")
            + "Check that the dots sit on the animal's nose, ears, back and tail base.")})
    f2 = f"{sid}_confidence.png"
    FG.confidence_over_time(out_dir / f2, sid, times, conf0, bodyparts, pc, detected)
    figs.append({"path": f2, "caption": (
        f"{sid}: how confident the model is for each keypoint (rows) over the session (dark = confident). Below: the "
        f"fraction of keypoints above the cut-off ({pc:g}) and (dashed) whether the animal was found (a pose with mean keypoint confidence of at least 0.25). Look for "
        "keypoints that are pale throughout (the model rarely sees them: occluded, or not visible from this camera "
        "angle) and for dips in time (the animal rearing, grooming, at the wall, or out of view).")})
    f3 = f"{sid}_keypoint_confidence.png"
    FG.keypoint_confidence(out_dir / f3, sid, bodyparts, mean_conf, frac_low, pc)
    figs.append({"path": f3, "caption": (
        f"{sid}: mean confidence per keypoint (left) and the fraction of frames in which it falls below the "
        f"cut-off ({pc:g}, right). Keypoints that are low in most frames (often the tail tip, the ears under a "
        "head-mounted implant, or paws seen from above) should not be relied on downstream; the trunk keypoints "
        "(neck, mouse_center, tail_base) are usually the most reliable.")})
    if agr:
        f4 = f"{sid}_agreement.png"
        ours_xy = {k: poses[:, 0, bodyparts.index(k), :2] for k in agr["matched_keypoints"]}
        ref = session.poses[0]
        rt = next(s[2] for s in ref.series.values() if s[2] is not None)
        jj = np.clip(np.searchsorted(rt, times), 1, len(rt) - 1)
        jj = np.where(np.abs(rt[jj - 1] - times) <= np.abs(rt[jj] - times), jj - 1, jj)   # nearest frame
        ref_xy = {k: ref.series[v][0][jj, :2] for k, v in agr["matched_keypoints"].items()}
        FG.agreement(out_dir / f4, sid, times, agr, ours_xy, ref_xy)
        figs.append({"path": f4, "caption": (
            f"{sid}: agreement between this app's pose and the pose already in the session NWB "
            f"({agr.get('reference_scorer') or agr['reference']}), on the keypoints both have, in frames where both "
            f"are confident. Left: median (dark) and 90th-percentile (light) distance per keypoint; the dashed line "
            f"is 10% of the animal's body length. Right: the two trajectories of the body centre, which should "
            f"overlap. Keypoints are defined slightly differently by the two models (e.g. where exactly the "
            f"'tail base' is), so a few pixels of systematic offset are expected.")})

    kp = {b: {"mean_confidence": _r(mean_conf[k], 4), "fraction_low_confidence": _r(frac_low[k], 4)}
          for k, b in enumerate(bodyparts)}
    if agr:
        for k, v in agr["per_keypoint"].items():
            kp[k]["median_px_error"] = v.get("median_px")
    summ = {
        **{c: sheet_cols[c] for c in sheet_cols},
        "status": "ok", "video": video.name, "model": model.name, "scorer": model.scorer,
        "model_description": model.description, "video_adapt": bool(p["VIDEO_ADAPT"] and model.kind == "superanimal"),
        "n_keypoints": len(bodyparts), "width": info.width, "height": info.height,
        "fps": _r(info.fps, 6), "fps_effective": _r(eff_fps, 6), "n_frames_video": info.n_frames,
        "n_frames_analysed": int(len(sel)), "t_start": _r(times[0], 4), "t_stop": _r(times[-1], 4),
        "duration_analysed_s": _r(times[-1] - times[0] + (1.0 / info.fps) * p["FRAME_STRIDE"], 3),
        "time_source": tb.source, "time_source_detail": tb.detail,
        "fraction_frames_found": _r(detected.mean(), 4),
        "fraction_frames_box_interpolated": _r(res.frames_box_interpolated / max(1, len(sel)), 4),
        "fraction_frames_dropped_low_confidence": _r(res.frames_dropped_low_confidence / max(1, len(sel)), 4),
        "fraction_frames_box_from_pose": _r(res.frames_box_from_pose / max(1, len(sel)), 4),
        "mean_confidence": _r(c0.mean(), 4),
        "pcutoff": pc,
        "fraction_low_confidence_frames": _r(1.0 - frame_ok.mean(), 4),
        "keypoints": kp,
        "pose_seconds": _r(res.seconds, 1), "frames_per_second_processed": _r(fps_proc, 3),
        "detector_stride": p["DETECTOR_STRIDE"] if p["MAX_INDIVIDUALS"] == 1 else 1,
    }
    if abs(drift_pct) > 0.1:
        summ["drift_pct"] = _r(drift_pct, 4)
    if agr:
        summ["agreement"] = {k: v for k, v in agr.items() if not k.startswith("_")}
    summ["seconds"] = _r(time.time() - t_session, 1)
    for w in warns:
        log.warning("%s: %s", sid, w)
    warnings_all += [f"{sid}: {w}" for w in warns]
    return {"summary": summ, "figures": figs, "n_frames": info.n_frames, "n_analysed": int(len(sel)),
            "inputs": [str(video)] + ([str(session.path)] if session is not None else [])}


# --- main ---------------------------------------------------------------------------------------


def main() -> int:
    started = datetime.now(timezone.utc)
    t0 = time.time()
    try:
        p = PR.read_parameters()
    except PR.ParameterError as exc:
        log.error("bad parameter: %s", exc)
        return 2
    if p["VIDEO_ADAPT"] and variant() == "cpu":
        log.error("VIDEO_ADAPT=true needs the gpu variant: on the cpu variant it takes hours and peaks at about "
                  "7.5 GB of memory, against the variant's 8 GB")
        return 2
    in_dir = Path(os.environ.get("INPUT_DIR", "/data/input"))
    out_root = Path(os.environ.get("OUTPUT_DIR", "/data/output"))
    out_dir = out_root / APP / p["LABEL"] if p["LABEL"] else out_root / APP
    threads, tsrc = resolve_threads(p["THREADS"])
    import torch
    torch.set_num_threads(threads)
    torch.manual_seed(p["SEED"])
    np.random.seed(p["SEED"])
    dev = choose_device(p["USE_GPU"])
    device = "cuda" if dev["accelerator"] == "gpu" else "cpu"
    log.info("pose-estimate %s (%s variant): %s; %d threads (%s)", version(), variant(), dev["reason"], threads, tsrc)

    if not in_dir.is_dir():
        log.error("INPUT_DIR %s does not exist", in_dir)
        return 1
    inp = find_inputs(in_dir)
    if not inp.videos:
        log.error("no video (%s) found under %s", ", ".join(("mp4", "avi", "mov", "webm", "m4v", "mkv")), in_dir)
        return 1
    out_dir.mkdir(parents=True, exist_ok=True)

    from processor import model as MD
    from processor import sessionnwb as SN

    try:
        model = MD.lab_model(inp.model_files, device, p["MAX_INDIVIDUALS"], MD.tmpdir("pose-labmodel-")) if inp.model_files else None
    except MD.ModelError as exc:
        log.error("%s", exc)
        return 1
    if model is None:
        model = MD.superanimal(p["MODEL"], device, p["MAX_INDIVIDUALS"])
    elif p["VIDEO_ADAPT"]:
        log.warning("VIDEO_ADAPT applies to SuperAnimal models only; the lab's model is used as given")
    log.info("model: %s", model.description)
    notice = MD.licence_notice(model.name) if model.kind == "superanimal" else None
    if notice:
        log.info("USAGE AND LICENCE: %s", notice)

    warnings_all: list[str] = []
    sessions_nwb, nwb_by_sid = [], {}
    for f in inp.nwbs:
        try:
            s = SN.read_session_nwb(f)
        except Exception as exc:  # noqa: BLE001
            warnings_all.append(f"{f.name}: not readable as NWB ({exc})")
            log.warning("%s: not readable as NWB (%s)", f.name, exc)
            continue
        sessions_nwb.append(s)
        row = sheet_match(inp.sheet, f, derive_session_id(f, in_dir))
        sid = row.session_id if row else derive_session_id(f, in_dir)
        nwb_by_sid.setdefault(sid, s)

    results, rows, figures, prov_sessions, inputs_used = {}, [], [], {}, []

    def write_tables_and_summary() -> dict:
        # tables
        keys = ["session_id", "path", "subject_id", "condition", "group", "status", "n_frames", "n_frames_analysed"]
        extra = sorted({k for r in rows for k in r} - set(keys))
        with (out_dir / "sessions.csv").open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys + extra)
            w.writeheader()
            w.writerows(rows)
        with (out_dir / "keypoint_qc.csv").open("w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["session_id", "keypoint", "mean_confidence", "fraction_low_confidence", "median_px_error"])
            for sid, s in results.items():
                for k, v in (s.get("keypoints") or {}).items():
                    w.writerow([sid, k, v["mean_confidence"], v["fraction_low_confidence"], v.get("median_px_error", "")])

        ok = [s for s in results.values() if s.get("status") == "ok"]
        gpu = None
        if device == "cuda":
            gpu = {"name": torch.cuda.get_device_name(0), "peak_vram_mb": round(torch.cuda.max_memory_allocated() / 2**20, 1)}
        summary = {
            "app": APP, "version": version(), "variant": variant(),
            "model": model.name, "scorer": model.scorer, "model_kind": model.kind,
            "licence_notice": notice,
            "n_sessions": len(results), "n_ok": len(ok),
            "sessions": results,
            "figures": figures,
            "tables": [{"path": "sessions.csv", "caption": "Sessions processed, with status and frame counts."},
                       {"path": "keypoint_qc.csv", "caption": "Per session and keypoint: mean confidence, fraction of "
                        "frames below the cut-off, and the median pixel error against the session NWB's pose when there "
                        "is one."}],
            "warnings": warnings_all,
        }
        write_json(out_dir / "summary.json", summary)
        params_out = {k: (",".join(map(str, v)) if isinstance(v, tuple) else ("" if v is None else v)) for k, v in p.items()}
        prov = {
            "app": APP, "version": version(), "variant": variant(), "tools": tool_versions(),
            "parameters": params_out,
            "inputs": sorted({os.path.relpath(i, in_dir) if i.startswith(str(in_dir)) else i for i in inputs_used}
                             | ({os.path.relpath(str(f), in_dir) for f in inp.model_files} if model.kind == "lab" else set())),
            "started": started.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "finished": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "wall_seconds": round(time.time() - t0, 1),
            "peak_rss_mb": max(peak_rss_mb(), cgroup_memory_peak_mb() or 0),
            "gpu": gpu,
            "sessions": prov_sessions,
            "citations": CITATIONS,
            "licence_notice": notice,
        }
        write_json(out_dir / "provenance.json", prov)
        return prov

    used_ids: dict[str, int] = {}
    runners_cache: dict = {}
    for video in inp.videos:
        row = sheet_match(inp.sheet, video, derive_session_id(video, in_dir))
        sid = row.session_id if row else derive_session_id(video, in_dir)
        cols = dict(row.columns) if row else {}
        if not cols.get("subject_id") and sid.startswith("sub-") and "_ses-" in sid:
            cols["subject_id"] = sid[len("sub-"):sid.index("_ses-")]      # from the DANDI/BIDS name
        if sid in used_ids:
            used_ids[sid] += 1
            new = f"{sid}-{used_ids[sid]}"
            warnings_all.append(f"{video.name}: session_id {sid} is taken by another video; written as {new}")
            sid = new
        else:
            used_ids[sid] = 1
        ts = time.time()
        rec = {"session_id": sid, "path": str(video.relative_to(in_dir)) if str(video).startswith(str(in_dir)) else str(video),
               "subject_id": cols.get("subject_id", ""), "condition": cols.get("condition", ""),
               "group": cols.get("group", ""), **{k: v for k, v in cols.items() if k not in ("subject_id", "condition", "group")}}
        try:
            r = process_session(sid=sid, video=video, sheet_cols=cols, sessions_nwb=sessions_nwb, nwb_by_sid=nwb_by_sid,
                                inp=inp, p=p, model_base=model, device=device, out_dir=out_dir, notice=notice,
                                runners_cache=runners_cache, warnings_all=warnings_all)
            results[sid] = r["summary"]
            figures += r["figures"]
            inputs_used += r["inputs"]
            rec.update(status="ok", n_frames=r["n_frames"], n_frames_analysed=r["n_analysed"])
            prov_sessions[sid] = {"status": "ok", "seconds": round(time.time() - ts, 1)}
            s = r["summary"]
            log.info("%s: ok in %.0f s; %d frames, animal found in %.1f%%, mean confidence %.2f%s", sid, time.time() - ts,
                     r["n_analysed"], 100 * s["fraction_frames_found"], s["mean_confidence"],
                     f", median {s['agreement']['median_px_error']} px from the session NWB's pose"
                     if s.get("agreement") else "")
        except Exception as exc:  # noqa: BLE001 - one failed session must not stop the others
            log.error("%s: failed: %s", sid, exc)
            log.debug(traceback.format_exc())
            traceback.print_exc(file=sys.stdout)
            rec.update(status="failed", n_frames="", n_frames_analysed="")
            results[sid] = {**cols, "status": "failed", "error": str(exc)[:500], "video": video.name}
            prov_sessions[sid] = {"status": "failed", "seconds": round(time.time() - ts, 1), "error": str(exc)[:300]}
            warnings_all.append(f"{sid}: failed: {str(exc)[:300]}")
        rows.append(rec)
        write_tables_and_summary()   # after every session, so a killed run keeps what it finished

    prov = write_tables_and_summary()
    ok = [s_ for s_ in results.values() if s_.get("status") == "ok"]
    log.info("done: %d of %d sessions ok in %.0f s (peak RSS %.0f MB)", len(ok), len(results), time.time() - t0,
             prov["peak_rss_mb"])
    return 0 if ok else 1
