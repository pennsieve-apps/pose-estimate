"""
SuperAnimal weights baked into the image at build time (no network at run time).

Each file is downloaded from the Mathis lab's Hugging Face repository at a pinned revision and
checked against its SHA-256 (the Git LFS object id Hugging Face reports), then stored where
DeepLabCut looks for SuperAnimal snapshots (deeplabcut/modelzoo/checkpoints/<dataset>_<model>.pt).

    python -m processor.weights download     # build time (and local tests)
    python -m processor.weights verify       # check what is installed

The weights are licensed for academic, non-commercial use only (licenses/SuperAnimal-LICENSE.txt).
"""
from __future__ import annotations

import hashlib
import shutil
import sys
from pathlib import Path

WEIGHTS = {
    "superanimal_topviewmouse": {
        "repo": "mwmathis/DeepLabCutModelZoo-SuperAnimal-TopViewMouse",
        "revision": "d443257e4b1145f7439e0953c2015d60fcdc1185",
        "files": {
            "superanimal_topviewmouse_hrnet_w32.pt":
                "5e7ed2b4e568fa93e2b1404251f9b060f9bd3ff5d68712f57c8ddb766d0cf6ed",
            "superanimal_topviewmouse_fasterrcnn_resnet50_fpn_v2.pt":
                "1175e31a5486731734948da55e4b25adcee84da4e36eafc5768a771809d536e5",
        },
    },
    "superanimal_quadruped": {
        "repo": "mwmathis/DeepLabCutModelZoo-SuperAnimal-Quadruped",
        "revision": "1ad30fb80cd666f1e5c91578d1cf63bccfa84f80",
        "files": {
            "superanimal_quadruped_hrnet_w32.pt":
                "a42e58d56bb32f8b4f4fe17ea9ed9511cebc7b7949ac56b97fa6f3a49587c31e",
            "superanimal_quadruped_fasterrcnn_resnet50_fpn_v2.pt":
                "8c723bc3c357f9bfe87d3a23c30218be5d450d2289fd1e33f115327cf6c53d54",
        },
    },
}


def checkpoint_dir() -> Path:
    import deeplabcut  # noqa: F401 - locate the installed package without importing torch-heavy parts
    from deeplabcut.utils.auxiliaryfunctions import get_deeplabcut_path
    return Path(get_deeplabcut_path()) / "modelzoo" / "checkpoints"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def present(name: str) -> bool:
    try:
        d = checkpoint_dir()
    except Exception:  # noqa: BLE001
        return False
    return all((d / f).is_file() for f in WEIGHTS[name]["files"])


def download() -> None:
    from huggingface_hub import hf_hub_download
    dest = checkpoint_dir()
    dest.mkdir(parents=True, exist_ok=True)
    for name, spec in WEIGHTS.items():
        for fname, digest in spec["files"].items():
            target = dest / fname
            if target.is_file() and sha256(target) == digest:
                print(f"ok (present) {fname}")
                continue
            got = Path(hf_hub_download(repo_id=spec["repo"], filename=fname, revision=spec["revision"]))
            actual = sha256(got)
            if actual != digest:
                raise SystemExit(f"checksum mismatch for {fname}: {actual} != {digest}")
            shutil.copyfile(got, target)
            print(f"ok {fname} {digest}")


def verify() -> int:
    d = checkpoint_dir()
    bad = 0
    for spec in WEIGHTS.values():
        for fname, digest in spec["files"].items():
            p = d / fname
            ok = p.is_file() and sha256(p) == digest
            bad += not ok
            print(f"{'ok' if ok else 'MISSING/BAD'} {fname}")
    return 1 if bad else 0


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "verify"
    if cmd == "download":
        download()
        sys.exit(verify())
    sys.exit(verify())
