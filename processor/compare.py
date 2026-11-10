"""
Agreement with a pose already in the session NWB (e.g. the authors' own DeepLabCut model on
DANDI 002047): keypoints both models have are matched by name, frames by time, and the
pixel distance between the two estimates is summarised per keypoint.

Only frames where both estimates are confident (>= PCUTOFF) are compared; the fraction of frames
that qualify is reported with each number.
"""
from __future__ import annotations

import re

import numpy as np

# our name -> normalised names it may have elsewhere (normalised: lower case, no _ - space)
ALIASES = {
    "nose": ["nose", "snout"],
    "left_ear": ["leftear", "earleft", "lear"],
    "right_ear": ["rightear", "earright", "rear"],
    "tail_base": ["tailbase", "tailroot", "basetail", "tail0"],
    "mouse_center": ["bodycenter", "bodycentre", "center", "centre", "centroid", "body", "mousecenter", "spine2"],
    "head_midpoint": ["headcenter", "headcentre", "head", "headmidpoint"],
    "neck": ["neck"],
    "left_eye": ["lefteye"],
    "right_eye": ["righteye"],
    "tail_end": ["tailend", "tailtip"],
    "left_hip": ["lefthip"],
    "right_hip": ["righthip"],
    "left_shoulder": ["leftshoulder"],
    "right_shoulder": ["rightshoulder"],
}
HEADLINE = ("nose", "tail_base", "mouse_center")    # the demo's headline keypoints (centroid = mouse_center)


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def match_keypoints(ours: list[str], theirs: list[str]) -> dict[str, str]:
    by_norm = {_norm(t): t for t in theirs}
    out = {}
    for o in ours:
        for cand in [_norm(o)] + ALIASES.get(o, []):
            if cand in by_norm and by_norm[cand] not in out.values():
                out[o] = by_norm[cand]
                break
    return out


def compare(times: np.ndarray, pose: np.ndarray, bodyparts: list[str], ref, pcutoff: float) -> dict | None:
    """pose: (T, K, 3) of one animal; ref: sessionnwb.PoseRef. None when nothing matches."""
    mapping = match_keypoints(bodyparts, list(ref.series))
    if not mapping:
        return None
    any_ts = next((s[2] for s in ref.series.values() if s[2] is not None), None)
    if any_ts is None:
        return None
    ref_t = np.asarray(any_ts)
    # nearest reference sample to each of our frames, within half a frame
    dt = np.median(np.diff(ref_t)) if len(ref_t) > 1 else 1.0
    j = np.clip(np.searchsorted(ref_t, times), 1, len(ref_t) - 1)
    j = np.where(np.abs(ref_t[j - 1] - times) <= np.abs(ref_t[j] - times), j - 1, j)
    ok_t = np.abs(ref_t[j] - times) <= 0.5 * dt
    per, pooled, tracks = {}, [], {}
    for ours, theirs in mapping.items():
        k = bodyparts.index(ours)
        data, conf, _ = ref.series[theirs]
        a = pose[:, k, :2]
        b = data[j, :2]
        ca, cb = pose[:, k, 2], conf[j]
        valid = ok_t & np.isfinite(a).all(1) & np.isfinite(b).all(1)
        both = valid & (ca >= pcutoff) & (cb >= pcutoff)
        err = np.linalg.norm(a - b, axis=1)
        tracks[ours] = np.where(both, err, np.nan)
        if both.sum() == 0:
            per[ours] = {"reference": theirs, "n_frames": 0}
            continue
        e = err[both]
        pooled.append(e)
        per[ours] = {"reference": theirs, "n_frames": int(both.sum()),
                     "fraction_compared": round(float(both.sum() / max(1, valid.sum())), 4),
                     "median_px": round(float(np.median(e)), 2), "p90_px": round(float(np.percentile(e, 90)), 2),
                     "mean_px": round(float(np.mean(e)), 2),
                     "median_px_all_frames": round(float(np.median(err[valid])), 2) if valid.any() else None}
    # body length of the reference (nose to tail base), for scale
    body = None
    if "nose" in mapping and "tail_base" in mapping:
        n_, t_ = ref.series[mapping["nose"]], ref.series[mapping["tail_base"]]
        good = (n_[1] >= pcutoff) & (t_[1] >= pcutoff)
        if good.any():
            body = float(np.median(np.linalg.norm(n_[0][good] - t_[0][good], axis=1)))
    all_e = np.concatenate(pooled) if pooled else np.zeros(0)
    head = [k for k in HEADLINE if per.get(k, {}).get("n_frames")]
    head_e = np.concatenate([tracks[k][np.isfinite(tracks[k])] for k in head]) if head else np.zeros(0)
    res = {
        "reference": ref.path,
        "reference_software": " ".join(x for x in (ref.source_software, ref.source_software_version) if x) or None,
        "reference_scorer": ref.scorer,
        "matched_keypoints": mapping,
        "per_keypoint": per,
        "frames_time_matched": int(ok_t.sum()),
        "median_px_error_all_matched": round(float(np.median(all_e)), 2) if all_e.size else None,
        "median_px_error": round(float(np.median(head_e)), 2) if head_e.size else None,
        "median_px_error_keypoints": head,
        "reference_body_length_px": round(body, 1) if body else None,
    }
    if body and res["median_px_error"] is not None:
        res["median_error_body_lengths"] = round(res["median_px_error"] / body, 4)
    res["_tracks"] = tracks
    return res
