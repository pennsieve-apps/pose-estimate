"""
Parameters (environment variables named as in app.yml), validated once at start-up.

A bad value is a usage error: main() exits 2 with the message.
"""
from __future__ import annotations

import os

from processor.resources import DEVICES, normalise_use_gpu

SUPERANIMALS = ("superanimal_topviewmouse", "superanimal_quadruped")

DEFAULTS = {
    "MODEL": "superanimal_topviewmouse",
    "VIDEO_ADAPT": "false",
    "ADAPT_FRAMES": "1000",
    "MAX_INDIVIDUALS": "1",
    "PCUTOFF": "0.6",
    "T_START": "0",
    "DURATION": "0",
    "FRAME_STRIDE": "1",
    "CROP": "",
    "DETECTOR_STRIDE": "1",
    "FILL_DETECTOR_GAPS": "true",        # gpu variant; the cpu variant sets 5 (app.yml)
    "BATCH_SIZE": "16",
    "QC_CLIP_SECONDS": "10",
    "LABEL": "",
    "SEED": "42",
    "THREADS": "0",
    "USE_GPU": "auto",             # gpu variant; the cpu variant sets false
}
PARAMETERS = tuple(DEFAULTS)


class ParameterError(ValueError):
    pass


def _bool(name: str, raw: str) -> bool:
    v = raw.strip().lower()
    if v in ("true", "1", "yes", "on"):
        return True
    if v in ("false", "0", "no", "off"):
        return False
    raise ParameterError(f"{name} must be true or false, got {raw!r}")


def _int(name: str, raw: str, lo: int | None = None) -> int:
    try:
        f = float(raw)
    except ValueError as exc:
        raise ParameterError(f"{name} must be a whole number, got {raw!r}") from exc
    if f != int(f):
        raise ParameterError(f"{name} must be a whole number, got {raw!r}")
    if lo is not None and f < lo:
        raise ParameterError(f"{name} must be >= {lo}, got {raw!r}")
    return int(f)


def _float(name: str, raw: str, lo: float | None = None, hi: float | None = None) -> float:
    try:
        f = float(raw)
    except ValueError as exc:
        raise ParameterError(f"{name} must be a number, got {raw!r}") from exc
    if f != f or (lo is not None and f < lo) or (hi is not None and f > hi):
        raise ParameterError(f"{name} must be in [{lo}, {hi}], got {raw!r}")
    return f


def parse_crop(raw: str) -> tuple[int, int, int, int] | None:
    """CROP 'x,y,width,height' in pixels of the original frame; '' = the whole frame."""
    raw = (raw or "").strip()
    if not raw:
        return None
    parts = [p.strip() for p in raw.replace(" ", ",").split(",") if p.strip()]
    if len(parts) != 4:
        raise ParameterError(f"CROP must be 'x,y,width,height' in pixels, got {raw!r}")
    try:
        x, y, w, h = (int(float(p)) for p in parts)
    except ValueError as exc:
        raise ParameterError(f"CROP must be four whole numbers, got {raw!r}") from exc
    if x < 0 or y < 0 or w < 16 or h < 16:
        raise ParameterError(f"CROP needs x, y >= 0 and width, height >= 16, got {raw!r}")
    return x, y, w, h


def read_parameters(env=os.environ) -> dict:
    raw = {k: (env.get(k) if env.get(k) not in (None, "") else v) for k, v in DEFAULTS.items()}
    p: dict = {}
    p["MODEL"] = raw["MODEL"].strip().lower()
    if p["MODEL"] not in SUPERANIMALS:
        raise ParameterError(f"MODEL must be one of {', '.join(SUPERANIMALS)}, got {raw['MODEL']!r}")
    p["VIDEO_ADAPT"] = _bool("VIDEO_ADAPT", raw["VIDEO_ADAPT"])
    p["ADAPT_FRAMES"] = _int("ADAPT_FRAMES", raw["ADAPT_FRAMES"], 50)
    p["MAX_INDIVIDUALS"] = _int("MAX_INDIVIDUALS", raw["MAX_INDIVIDUALS"], 1)
    if p["MAX_INDIVIDUALS"] > 20:
        raise ParameterError("MAX_INDIVIDUALS must be <= 20")
    p["PCUTOFF"] = _float("PCUTOFF", raw["PCUTOFF"], 0.0, 1.0)
    p["T_START"] = _float("T_START", raw["T_START"], 0.0)
    p["DURATION"] = _float("DURATION", raw["DURATION"], 0.0)
    p["FRAME_STRIDE"] = _int("FRAME_STRIDE", raw["FRAME_STRIDE"], 1)
    p["CROP"] = parse_crop(raw["CROP"])
    p["DETECTOR_STRIDE"] = _int("DETECTOR_STRIDE", raw["DETECTOR_STRIDE"], 1)
    if p["DETECTOR_STRIDE"] > 30:
        raise ParameterError("DETECTOR_STRIDE must be <= 30 (frames between two detector runs)")
    p["FILL_DETECTOR_GAPS"] = _bool("FILL_DETECTOR_GAPS", raw["FILL_DETECTOR_GAPS"])
    p["BATCH_SIZE"] = _int("BATCH_SIZE", raw["BATCH_SIZE"], 1)
    p["QC_CLIP_SECONDS"] = _float("QC_CLIP_SECONDS", raw["QC_CLIP_SECONDS"], 0.0, 120.0)
    label = raw["LABEL"].strip()
    if label and not all(c.isalnum() or c in "_-" for c in label):
        raise ParameterError(f"LABEL may only hold letters, digits, '_' and '-', got {label!r}")
    p["LABEL"] = label
    p["SEED"] = _int("SEED", raw["SEED"], 0)
    p["THREADS"] = _int("THREADS", raw["THREADS"], 0)
    p["USE_GPU"] = normalise_use_gpu(raw["USE_GPU"])
    if p["USE_GPU"] not in DEVICES:
        raise ParameterError(f"USE_GPU must be auto, true or false, got {raw['USE_GPU']!r}")
    return p
