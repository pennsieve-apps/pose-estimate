"""
The pose model: DeepLabCut 3 SuperAnimal (baked into the image, zero-shot, optionally adapted to
the video) or the lab's own DeepLabCut 3 PyTorch model from the `model` port.

Lab model, either
  - an exported model: a .pt written by deeplabcut.export_model (a dict with `config`, `pose` and,
    for top-down models, `detector` weights), or
  - a DeepLabCut 3 project folder (or just its train folder): pytorch_config.yaml with
    snapshot-*.pt (and snapshot-detector-*.pt for top-down models) beside it; the `best` snapshot
    when there is one, else the one with the highest epoch;
  - either of these as a .zip.
"""
from __future__ import annotations

import logging
import re
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger("pose-estimate")

POSE_ARCH = "hrnet_w32"
DETECTOR = "fasterrcnn_resnet50_fpn_v2"

# Edges for the NWB Skeleton and the QC drawings (SuperAnimal configs define none).
TOPVIEWMOUSE_EDGES = [
    ("nose", "left_eye"), ("nose", "right_eye"), ("left_eye", "left_ear"), ("right_eye", "right_ear"),
    ("left_ear", "left_ear_tip"), ("right_ear", "right_ear_tip"), ("nose", "head_midpoint"),
    ("head_midpoint", "neck"), ("neck", "mid_back"), ("mid_back", "mouse_center"),
    ("mouse_center", "mid_backend"), ("mid_backend", "mid_backend2"), ("mid_backend2", "mid_backend3"),
    ("mid_backend3", "tail_base"), ("tail_base", "tail1"), ("tail1", "tail2"), ("tail2", "tail3"),
    ("tail3", "tail4"), ("tail4", "tail5"), ("tail5", "tail_end"),
    ("neck", "left_shoulder"), ("left_shoulder", "left_midside"), ("left_midside", "left_hip"),
    ("left_hip", "tail_base"), ("neck", "right_shoulder"), ("right_shoulder", "right_midside"),
    ("right_midside", "right_hip"), ("right_hip", "tail_base"),
]
QUADRUPED_EDGES = [
    ("nose", "upper_jaw"), ("upper_jaw", "lower_jaw"), ("nose", "left_eye"), ("nose", "right_eye"),
    ("left_eye", "left_earbase"), ("right_eye", "right_earbase"), ("left_earbase", "left_earend"),
    ("right_earbase", "right_earend"), ("neck_base", "neck_end"), ("neck_end", "back_base"),
    ("back_base", "back_middle"), ("back_middle", "back_end"), ("back_end", "tail_base"),
    ("tail_base", "tail_end"), ("throat_base", "throat_end"),
    ("front_left_thai", "front_left_knee"), ("front_left_knee", "front_left_paw"),
    ("front_right_thai", "front_right_knee"), ("front_right_knee", "front_right_paw"),
    ("back_left_thai", "back_left_knee"), ("back_left_knee", "back_left_paw"),
    ("back_right_thai", "back_right_knee"), ("back_right_knee", "back_right_paw"),
]


class ModelError(RuntimeError):
    pass


@dataclass
class PoseModel:
    kind: str                       # "superanimal" or "lab"
    name: str                       # superanimal_topviewmouse / the lab model's name
    scorer: str                     # DeepLabCut scorer string (h5 column level, NWB scorer)
    config: object                  # DeepLabCut PoseConfig
    pose_snapshot: Path
    detector_snapshot: Path | None
    bodyparts: list[str]
    edges: list[tuple[int, int]]
    top_down: bool
    description: str
    files: list[str] = field(default_factory=list)   # what the model came from (provenance)

    def runners(self, device: str, batch_size: int, max_individuals: int):
        from deeplabcut.pose_estimation_pytorch.apis.utils import get_inference_runners
        cfg = self.config
        cfg["device"] = device
        if cfg.get("detector") is not None:
            cfg["detector"]["device"] = device
        return get_inference_runners(
            model_config=cfg, snapshot_path=str(self.pose_snapshot), max_individuals=max_individuals,
            num_bodyparts=len(self.bodyparts), num_unique_bodyparts=0, batch_size=batch_size,
            detector_batch_size=1 if device == "cpu" else max(1, min(batch_size, 4)), device=device,
            detector_path=str(self.detector_snapshot) if self.detector_snapshot else None)


def _edges(bodyparts: list[str], named: list[tuple[str, str]]) -> list[tuple[int, int]]:
    idx = {b: i for i, b in enumerate(bodyparts)}
    return [(idx[a], idx[b]) for a, b in named if a in idx and b in idx]


def superanimal(name: str, device: str, max_individuals: int) -> PoseModel:
    from deeplabcut.modelzoo.utils import get_super_animal_scorer
    from deeplabcut.pose_estimation_pytorch.config.pose import PoseConfig
    from deeplabcut.pose_estimation_pytorch.modelzoo.utils import get_super_animal_snapshot_path

    pose = get_super_animal_snapshot_path(name, POSE_ARCH, download=False)
    det = get_super_animal_snapshot_path(name, DETECTOR, download=False)
    for p in (pose, det):
        if not p.is_file():
            raise ModelError(f"SuperAnimal weights missing from the image: {p} (they are baked in at build time)")
    cfg = PoseConfig.build_for_superanimal_inference(super_animal=name, model_name=POSE_ARCH, detector_name=DETECTOR,
                                                     max_individuals=max_individuals, device=device)
    bodyparts = list(cfg["metadata"]["bodyparts"])
    scorer = get_super_animal_scorer(name, pose, det)
    edges = _edges(bodyparts, TOPVIEWMOUSE_EDGES if "topview" in name else QUADRUPED_EDGES)
    desc = (f"DeepLabCut {name} (SuperAnimal, Ye et al. 2024), {POSE_ARCH} top-down pose model with a "
            f"{DETECTOR} detector, zero-shot (no training on this video)")
    return PoseModel("superanimal", name, scorer, cfg, pose, det, bodyparts, edges, True, desc,
                     [pose.name, det.name])


# --- lab model ----------------------------------------------------------------------------------

_EPOCH = re.compile(r"-(\d+)\.pt$")


def _pick_snapshot(folder: Path, detector: bool) -> Path | None:
    snaps = [p for p in folder.glob("snapshot-*.pt") if ("detector" in p.name) == detector]
    if not snaps:
        return None
    best = [p for p in snaps if "-best-" in p.name or p.name.endswith("-best.pt")]
    pool = best or snaps

    def epoch(p: Path) -> int:
        m = _EPOCH.search(p.name)
        return int(m.group(1)) if m else -1
    return max(pool, key=epoch)


def _unzip(files: list[Path], workdir: Path) -> list[Path]:
    """A zipped project / train folder / export in the `model` port is unpacked into workdir ($TMPDIR)."""
    import zipfile
    out = []
    for f in files:
        if f.suffix.lower() != ".zip":
            out.append(f)
            continue
        dest = workdir / f"unzipped-{f.stem}"
        with zipfile.ZipFile(f) as z:
            for m in z.infolist():
                name = Path(m.filename)
                if m.is_dir() or name.is_absolute() or ".." in name.parts:
                    continue
                if not (name.name == "pytorch_config.yaml" or name.suffix in (".pt", ".pth")):
                    continue
                target = dest / name
                target.parent.mkdir(parents=True, exist_ok=True)
                with z.open(m) as src, target.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
                out.append(target)
        log.info("model: unpacked %s", f.name)
    return out


def lab_model(files: list[Path], device: str, max_individuals: int, workdir: Path) -> PoseModel | None:
    """The lab's model from the `model` port, or None when the port holds none."""
    import torch
    from deeplabcut.pose_estimation_pytorch.config.pose import PoseConfig

    files = _unzip(files, workdir)
    configs = [f for f in files if f.name == "pytorch_config.yaml"]
    exported = []
    for f in files:
        if f.suffix == ".pt" and not f.name.startswith("snapshot-"):
            try:
                obj = torch.load(f, map_location="cpu", weights_only=True)
            except Exception as exc:  # noqa: BLE001 - not a DeepLabCut export
                log.warning("model: %s is not a DeepLabCut exported model (%s); skipped", f.name, str(exc)[:120])
                continue
            if isinstance(obj, dict) and "config" in obj and "pose" in obj:
                exported.append((f, obj))
    if exported:
        f, obj = exported[0]
        if len(exported) > 1:
            log.warning("model: %d exported models found; using %s", len(exported), f.name)
        workdir.mkdir(parents=True, exist_ok=True)
        pose_path = workdir / "snapshot-pose.pt"
        torch.save({"model": obj["pose"]}, pose_path)
        det_path = None
        if obj.get("detector") is not None:
            det_path = workdir / "snapshot-detector.pt"
            torch.save({"model": obj["detector"]}, det_path)
        cfg = PoseConfig.from_any(obj["config"])
        name, origin = f.stem, [f.name]
    elif configs:
        # prefer a train/ folder that has snapshots
        cands = [c for c in configs if _pick_snapshot(c.parent, False) is not None]
        if not cands:
            raise ModelError("model: pytorch_config.yaml found but no snapshot-*.pt beside it")
        c = sorted(cands, key=lambda p: str(p))[-1]
        pose_path = _pick_snapshot(c.parent, False)
        det_path = _pick_snapshot(c.parent, True)
        cfg = PoseConfig.from_any(c)
        name = c.parent.parent.name if c.parent.name == "train" else c.parent.name
        origin = [str(c), pose_path.name] + ([det_path.name] if det_path else [])
    else:
        return None
    cfg["device"] = device
    method = str(cfg["method"]).lower()
    top_down = "td" in method or "top" in method
    if top_down and det_path is None:
        raise ModelError("model: a top-down model needs its detector (export it with the detector, or include "
                         "snapshot-detector-*.pt)")
    bodyparts = list(cfg["metadata"]["bodyparts"])
    ub = list(cfg["metadata"].get("unique_bodyparts") or [])
    if ub:
        log.warning("model: unique bodyparts %s are not written (only the per-animal bodyparts)", ub)
    scorer = f"DLC_{name}"
    desc = f"the lab's own DeepLabCut 3 model '{name}' ({'top-down' if top_down else 'bottom-up'}, " \
           f"{len(bodyparts)} keypoints)"
    return PoseModel("lab", name, scorer, cfg, pose_path, det_path, bodyparts, [], top_down, desc, origin)


# --- video adaptation ---------------------------------------------------------------------------


def adapt(model: PoseModel, clip: Path, workdir: Path, device: str, batch_size: int, max_individuals: int) -> PoseModel:
    """SuperAnimal video adaptation (Ye et al. 2024) on a clip of frames sampled from the video:
    pseudo-labels from the zero-shot model, then 4 epochs of self-training of the pose model and
    the detector. Returns the model with the adapted snapshots."""
    import deeplabcut

    out = workdir / "adapt"
    out.mkdir(parents=True, exist_ok=True)
    deeplabcut.video_inference_superanimal(
        videos=[str(clip)], superanimal_name=model.name, model_name=POSE_ARCH, detector_name=DETECTOR,
        dest_folder=str(out), video_adapt=True, max_individuals=max_individuals, device=device,
        batch_size=batch_size, detector_batch_size=max(1, min(batch_size, 8)),
        video_adapt_batch_size=8, pose_epochs=4, detector_epochs=4,
        create_labeled_video=False, plot_bboxes=False, plot_trajectories=False)
    ckpt = clip.with_name(f"pseudo_{clip.stem}") / "checkpoints"
    pose = sorted(ckpt.glob(f"snapshot-{POSE_ARCH}-*.pt"))
    det = sorted(ckpt.glob(f"snapshot-{DETECTOR}-*.pt"))
    if not pose or not det:
        raise ModelError(f"video adaptation produced no checkpoints in {ckpt}")
    model.pose_snapshot, model.detector_snapshot = pose[-1], det[-1]
    model.scorer = model.scorer + "_videoadapt"
    model.description += "; adapted to this video (SuperAnimal video adaptation, 4 pose + 4 detector epochs)"
    return model


def cleanup(path: Path) -> None:
    shutil.rmtree(path, ignore_errors=True)


def tmpdir(prefix: str) -> Path:
    return Path(tempfile.mkdtemp(prefix=prefix))      # under $TMPDIR (never a hard-coded /tmp path)
