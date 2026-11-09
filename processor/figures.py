"""
QC figures (PNG, 150 dpi, session id in the title) and the labelled QC clip.

Colours: keypoints are coloured by body region (head, body, tail, limbs/other: four categorical
slots in a fixed order), never one colour per keypoint (27 keypoints cannot be told apart by
colour). Our pose is blue, a reference pose orange. Confidence uses one hue, light to dark.
"""
from __future__ import annotations

import logging
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

log = logging.getLogger("pose-estimate")

BLUE, ORANGE, AQUA, YELLOW, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#6250d6"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
REGION_COLOURS = {"head": BLUE, "body": ORANGE, "tail": AQUA, "other": VIOLET}
CONF_CMAP = LinearSegmentedColormap.from_list("conf", ["#f3f6fb", "#9cc0ec", "#2a78d6", "#0d3a73"])
DPI = 150

plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.titlesize": 10, "axes.titleweight": "bold",
    "axes.grid": False, "grid.color": GRID, "legend.frameon": False,
})


def region(bp: str) -> str:
    b = bp.lower()
    if any(k in b for k in ("nose", "ear", "eye", "head", "jaw", "mouth", "antler", "snout")):
        return "head"
    if "tail" in b:
        return "tail"
    if any(k in b for k in ("neck", "back", "center", "centre", "spine", "body", "belly", "throat", "hip",
                            "shoulder", "midside")):
        return "body"
    return "other"


def _rgb(hex_: str) -> tuple[int, int, int]:
    return tuple(int(hex_[i:i + 2], 16) for i in (1, 3, 5))


def draw_pose(frame: np.ndarray, pose: np.ndarray, bodyparts: list[str], edges, pcutoff: float,
              radius: int = 4, thickness: int = 2) -> np.ndarray:
    """Keypoints >= pcutoff as filled dots (region colour), lower ones as hollow rings; skeleton in white."""
    img = frame.copy()
    for ind in pose:                                   # (K, 3)
        for a, b in edges:
            pa, pb = ind[a], ind[b]
            if np.isfinite(pa).all() and np.isfinite(pb).all() and pa[2] >= pcutoff and pb[2] >= pcutoff:
                cv2.line(img, (int(pa[0]), int(pa[1])), (int(pb[0]), int(pb[1])), (255, 255, 255), 1, cv2.LINE_AA)
        for k, (x, y, c) in enumerate(ind):
            if not (np.isfinite(x) and np.isfinite(y)):
                continue
            col = _rgb(REGION_COLOURS[region(bodyparts[k])])
            if c >= pcutoff:
                cv2.circle(img, (int(x), int(y)), radius, col, -1, cv2.LINE_AA)
                cv2.circle(img, (int(x), int(y)), radius, (255, 255, 255), 1, cv2.LINE_AA)
            else:
                cv2.circle(img, (int(x), int(y)), radius, col, 1, cv2.LINE_AA)
    return img


def _crop_box(pose_k3: np.ndarray, shape, pcutoff: float, min_size: int = 160) -> tuple[int, int, int, int]:
    good = pose_k3[np.isfinite(pose_k3[:, 0]) & (pose_k3[:, 2] >= pcutoff * 0.5)]
    if len(good) == 0:
        good = pose_k3[np.isfinite(pose_k3[:, 0])]
    h, w = shape[:2]
    if len(good) == 0:
        return 0, 0, w, h
    cx, cy = np.median(good[:, 0]), np.median(good[:, 1])
    span = max(min_size, 1.5 * max(np.ptp(good[:, 0]), np.ptp(good[:, 1])))
    x0, y0 = int(max(0, cx - span / 2)), int(max(0, cy - span / 2))
    return x0, y0, int(min(w, x0 + span)), int(min(h, y0 + span))


def example_frames(path: Path, session_id: str, frames: list[tuple[int, float, np.ndarray, np.ndarray]],
                   bodyparts, edges, pcutoff: float, reference: list | None = None) -> None:
    """frames: (frame_index, time, RGB frame, pose (I, K, 3)); reference: optional per-frame dict name->(x, y)."""
    n = len(frames)
    fig, axes = plt.subplots(2, n, figsize=(3.2 * n, 6.6), gridspec_kw={"height_ratios": [1, 1]})
    axes = np.atleast_2d(axes).reshape(2, n)
    for col, (fi, t, img, pose) in enumerate(frames):
        drawn = draw_pose(img, pose, bodyparts, edges, pcutoff, radius=max(3, img.shape[1] // 250))
        ax = axes[0, col]
        ax.imshow(drawn)
        x0, y0, x1, y1 = _crop_box(pose[0], img.shape, pcutoff)
        ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec="white", lw=1.2, ls="--"))
        ax.set_title(f"frame {fi}, t = {t:.1f} s", fontsize=8, fontweight="normal")
        ax.axis("off")
        ax2 = axes[1, col]
        ax2.imshow(drawn[y0:y1, x0:x1], extent=(x0, x1, y1, y0))
        if reference and reference[col]:
            for name, (rx, ry) in reference[col].items():
                if np.isfinite(rx) and np.isfinite(ry):
                    ax2.plot(rx, ry, marker="x", color=YELLOW, ms=8, mew=2.2)
        ax2.set_xlim(x0, x1)
        ax2.set_ylim(y1, y0)
        ax2.axis("off")
    present = {region(b) for b in bodyparts}
    handles = [plt.Line2D([], [], marker="o", ls="", color=c, mec="white", ms=7, label=r)
               for r, c in REGION_COLOURS.items() if r in present]
    handles.append(plt.Line2D([], [], marker="o", ls="", mfc="none", mec=INK2, ms=7, label=f"confidence < {pcutoff:g}"))
    if reference:
        handles.append(plt.Line2D([], [], marker="x", ls="", color=YELLOW, mew=2, ms=7, label="pose in the session NWB"))
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), fontsize=8)
    fig.suptitle(f"{session_id}: estimated pose on example frames (bottom: zoom on the animal, dashed box)", fontsize=10,
                 fontweight="bold")
    fig.tight_layout(rect=(0, 0.05, 1, 0.96))
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def confidence_over_time(path: Path, session_id: str, times: np.ndarray, conf: np.ndarray, bodyparts,
                         pcutoff: float, detected: np.ndarray) -> None:
    """conf: (T, K) for animal 0. Heatmap of the mean confidence per keypoint per time bin, and the
    fraction of keypoints >= pcutoff over time."""
    t = times - times[0] if len(times) else times
    dur = float(t[-1]) if len(t) else 0.0
    n_bins = int(min(400, max(10, len(t) // 3)))
    edges = np.linspace(0, max(dur, 1e-6), n_bins + 1)
    which = np.clip(np.searchsorted(edges, t, side="right") - 1, 0, n_bins - 1)
    c = np.nan_to_num(conf, nan=0.0)
    heat = np.zeros((len(bodyparts), n_bins))
    frac = np.zeros(n_bins)
    for b in range(n_bins):
        m = which == b
        if m.any():
            heat[:, b] = c[m].mean(0)
            frac[b] = (c[m] >= pcutoff).mean()
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(10, 2.2 + 0.17 * len(bodyparts)), sharex=True,
                                   gridspec_kw={"height_ratios": [len(bodyparts), 6]})
    im = ax0.imshow(heat, aspect="auto", cmap=CONF_CMAP, vmin=0, vmax=1, interpolation="nearest",
                    extent=(times[0], times[0] + dur, len(bodyparts) - 0.5, -0.5))
    ax0.set_yticks(range(len(bodyparts)))
    ax0.set_yticklabels(bodyparts, fontsize=7)
    cb = fig.colorbar(im, ax=[ax0, ax1], fraction=0.02, pad=0.01)
    cb.set_label("mean confidence")
    ax0.set_title(f"{session_id}: keypoint confidence over time")
    centers = times[0] + 0.5 * (edges[:-1] + edges[1:])
    ax1.plot(centers, frac, color=BLUE, lw=2)
    det_frac = np.array([detected[which == b].mean() if (which == b).any() else np.nan for b in range(n_bins)])
    ax1.plot(centers, det_frac, color=INK2, lw=1, ls="--")
    ax1.text(centers[-1], 1.02, "animal detected", color=INK2, fontsize=7, ha="right", va="bottom")
    ax1.set_ylim(0, 1.15)
    ax1.set_ylabel(f"fraction\n>= {pcutoff:g}")
    ax1.set_xlabel("time (s, session clock)")
    ax1.grid(axis="y")
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def keypoint_confidence(path: Path, session_id: str, bodyparts, mean_conf: np.ndarray, frac_low: np.ndarray,
                        pcutoff: float) -> None:
    order = np.arange(len(bodyparts))
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(9, 1.2 + 0.2 * len(bodyparts)), sharey=True)
    cols = [REGION_COLOURS[region(b)] for b in bodyparts]
    ax0.barh(order, mean_conf, color=cols, height=0.7)
    ax0.axvline(pcutoff, color=INK2, lw=1, ls="--")
    ax0.set_xlim(0, 1)
    ax0.set_yticks(order)
    ax0.set_yticklabels(bodyparts, fontsize=7)
    ax0.invert_yaxis()
    ax0.set_xlabel("mean confidence")
    ax0.grid(axis="x")
    ax1.barh(order, frac_low, color=cols, height=0.7)
    ax1.set_xlim(0, 1)
    ax1.set_xlabel(f"fraction of frames below {pcutoff:g}")
    ax1.grid(axis="x")
    for i, v in enumerate(frac_low):
        ax1.text(min(v + 0.01, 0.92), i, f"{100 * v:.0f}%", va="center", fontsize=6.5, color=INK2)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c, label=r) for r, c in REGION_COLOURS.items()
               if r in {region(b) for b in bodyparts}]
    fig.legend(handles=handles, loc="upper center", ncol=len(handles), fontsize=8, bbox_to_anchor=(0.55, 1.0))
    fig.suptitle(f"{session_id}: confidence per keypoint", fontsize=10, fontweight="bold", y=1.04)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def agreement(path: Path, session_id: str, times: np.ndarray, agr: dict, ours_xy: dict, ref_xy: dict) -> None:
    """Left: median error per matched keypoint; right: our vs the reference trajectory of the centroid."""
    per = {k: v for k, v in agr["per_keypoint"].items() if v.get("n_frames")}
    names = list(per)
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(10, 4.2), gridspec_kw={"width_ratios": [1, 1.1]})
    y = np.arange(len(names))
    med = [per[n]["median_px"] for n in names]
    p90 = [per[n]["p90_px"] for n in names]
    ax0.barh(y, p90, color="#c6d9f2", height=0.6, label="90th percentile")
    ax0.barh(y, med, color=BLUE, height=0.6, label="median")
    ax0.set_yticks(y)
    ax0.set_yticklabels([f"{n} / {per[n]['reference']}" for n in names], fontsize=7.5)
    ax0.invert_yaxis()
    ax0.set_xlabel("distance between the two estimates (px)")
    for i, v in enumerate(med):
        ax0.text(v, i, f" {v:.1f}", va="center", fontsize=7, color=INK)
    if agr.get("reference_body_length_px"):
        ax0.axvline(agr["reference_body_length_px"] * 0.1, color=INK2, ls="--", lw=1)
        ax0.text(agr["reference_body_length_px"] * 0.1, len(names) - 0.4, " 10% of body length", fontsize=7,
                 color=INK2, va="bottom")
    ax0.legend(loc="lower right", fontsize=7)
    ax0.grid(axis="x")
    ax0.set_title("error per keypoint (both confident)")
    key = "mouse_center" if "mouse_center" in ours_xy else (names[0] if names else None)
    if key:
        a, b = ours_xy[key], ref_xy[key]
        ax1.plot(b[:, 0], b[:, 1], color=ORANGE, lw=1, alpha=0.9, label="reference (session NWB)")
        ax1.plot(a[:, 0], a[:, 1], color=BLUE, lw=1, alpha=0.8, label="this app")
        ax1.invert_yaxis()
        ax1.set_aspect("equal", adjustable="datalim")
        ax1.set_xlabel("x (px)")
        ax1.set_ylabel("y (px)")
        ax1.legend(loc="upper right", fontsize=7)
        ax1.set_title(f"trajectory of {key}")
    fig.suptitle(f"{session_id}: agreement with the pose in the session NWB "
                 f"(median {agr.get('median_px_error')} px)", fontsize=10, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
