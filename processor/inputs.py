"""
Finding the inputs: videos, session NWB files, frame-time CSVs, a sessions.csv sheet and a lab's
DeepLabCut model, by name and content, recursively, following symlinks (INPUT_DIR entries are
symlinks). Ports (`video`, `session`, `model`) arrive in their own folders when linked, or side
by side at the root; neither layout is assumed.

session_id (conventions.md, "Sessions"), in order:
  1. a sessions.csv row whose `path` matches the video's file name or a parent folder's name;
  2. a DANDI/BIDS-style sub-<a>_ses-<b> in the file name, else in a parent folder's name (DANDI
     puts videos in sub-X_ses-Y_.../<uuid>_external_file_0.mp4); <b> is cut to its first 8
     characters when it is a UUID;
  3. the file stem, sanitised ([^A-Za-z0-9_-] -> -).
"""
from __future__ import annotations

import csv
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".m4v", ".mkv", ".webm")
NWB_EXTENSIONS = (".nwb",)
MODEL_HINT_DIRS = ("dlc-models-pytorch", "exported-models-pytorch")

SUB_SES = re.compile(r"sub-([A-Za-z0-9-]+?)_ses-([A-Za-z0-9-]+)")
UUID = re.compile(r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$")
SANITISE = re.compile(r"[^A-Za-z0-9_-]")


def walk_files(root: Path) -> list[Path]:
    """Every regular file under root, following symlinked folders (each real folder once)."""
    out: list[Path] = []
    seen: set[str] = set()
    for dirpath, dirnames, filenames in os.walk(root, followlinks=True):
        real = os.path.realpath(dirpath)
        if real in seen:
            dirnames[:] = []
            continue
        seen.add(real)
        dirnames[:] = sorted(d for d in dirnames if not d.startswith("."))
        for name in sorted(filenames):
            if name.startswith("."):
                continue
            p = Path(dirpath) / name
            if p.is_file():
                out.append(p)
    return out


def sanitise(text: str) -> str:
    s = SANITISE.sub("-", text).strip("-")
    return s or "session"


def sub_ses_id(text: str) -> str | None:
    m = SUB_SES.search(text)
    if not m:
        return None
    sub, ses = m.group(1), m.group(2)
    if UUID.match(ses):
        ses = ses[:8]
    return f"sub-{sub}_ses-{ses}"


def derive_session_id(path: Path, root: Path) -> str:
    """Rules 2 and 3 above (the sheet, rule 1, is applied by the caller)."""
    sid = sub_ses_id(path.name)
    if sid:
        return sid
    try:
        parents = path.relative_to(root).parents
    except ValueError:
        parents = path.parents
    for parent in parents:
        if parent.name:
            sid = sub_ses_id(parent.name)
            if sid:
                return sid
    stem = path.name
    for ext in VIDEO_EXTENSIONS + NWB_EXTENSIONS:
        if stem.lower().endswith(ext):
            stem = stem[: -len(ext)]
            break
    return sanitise(stem)


@dataclass
class SheetRow:
    session_id: str
    path: str
    columns: dict = field(default_factory=dict)   # every other column (subject_id, condition, group, ...)


def read_sheet(files: list[Path]) -> tuple[list[SheetRow], Path | None]:
    """sessions.csv rows (session_id, path, other columns), or [] when there is none."""
    sheets = [f for f in files if f.name.lower() == "sessions.csv"]
    if not sheets:
        return [], None
    sheet = sheets[0]
    rows: list[SheetRow] = []
    with sheet.open(newline="", encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            r = {(k or "").strip(): (v or "").strip() for k, v in r.items()}
            sid, path = r.pop("session_id", ""), r.pop("path", "")
            if not sid:
                continue
            for k in ("status",):
                r.pop(k, None)
            rows.append(SheetRow(sanitise(sid), path, r))
    return rows, sheet


def sheet_match(rows: list[SheetRow], path: Path, derived_id: str | None = None) -> SheetRow | None:
    """The sheet row whose `path` (or `video` column) names this file or one of its parent folders
    (longest match); else the row whose session_id equals the id derived from the file's name."""
    best, best_len = None, -1
    parts = [path.name] + [p.name for p in path.parents if p.name]
    for row in rows:
        for target in (row.path, row.columns.get("video", "")):
            if not target:
                continue
            target = target.rstrip("/")
            tparts = Path(target).parts
            # the whole relative path matches the end of this file's path, or a file / folder name exactly
            if len(tparts) > 1:
                hit = tuple(path.parts[-len(tparts):]) == tparts
            else:
                hit = Path(target).name in parts
            if hit and len(target) > best_len:
                best, best_len = row, len(target)
    if best is None and derived_id:
        best = next((r for r in rows if r.session_id == derived_id), None)
    return best


@dataclass
class Inputs:
    videos: list[Path]
    nwbs: list[Path]
    timestamp_csvs: dict[str, Path]       # video stem -> <stem>_timestamps.csv
    sheet: list[SheetRow]
    sheet_path: Path | None
    model_files: list[Path]               # candidate lab-model files (pytorch_config.yaml / *.pt)


PORTS = ("video", "session", "model")


def port_relative(path: Path, root: Path) -> str:
    """Path below its input port folder: the same dataset file fed to two ports has one key."""
    parts = path.relative_to(root).parts
    if len(parts) > 1 and parts[0] in PORTS:
        parts = parts[1:]
    return "/".join(parts)


def unique_files(files: list[Path], root: Path) -> list[Path]:
    """Drop copies of a file that a workflow staged under more than one port (e.g. the whole dataset
    wired to both video and session): without this every video would be processed once per port."""
    out, seen = [], set()
    for f in files:
        try:
            key = (port_relative(f, root), f.stat().st_size)
        except ValueError:
            key = (str(f), -1)
        if key in seen:
            continue
        seen.add(key)
        out.append(f)
    return out


def find_inputs(root: Path) -> Inputs:
    files = unique_files(walk_files(root), root)
    model_root = root / "model"
    videos, nwbs, csvs, model_files = [], [], {}, []
    for f in files:
        low = f.name.lower()
        in_model = model_root.exists() and _is_under(f, model_root)
        in_model = in_model or any(part in MODEL_HINT_DIRS for part in f.parts)
        if low.endswith((".pt", ".pth")) or low == "pytorch_config.yaml" or (in_model and low.endswith(".zip")):
            model_files.append(f)
            continue
        if in_model:
            continue      # a lab's project folder may hold its own videos and labelled data
        if low.endswith(VIDEO_EXTENSIONS):
            if "_labeled" in low:
                continue  # labelled QC clips (ours or DeepLabCut's) are not raw video
            videos.append(f)
        elif low.endswith(NWB_EXTENSIONS):
            nwbs.append(f)
        elif low.endswith("_timestamps.csv"):
            csvs[f.name[: -len("_timestamps.csv")]] = f
    sheet, sheet_path = read_sheet(files)
    return Inputs(videos, nwbs, csvs, sheet, sheet_path, model_files)


def _is_under(path: Path, folder: Path) -> bool:
    try:
        path.relative_to(folder)
        return True
    except ValueError:
        return False
