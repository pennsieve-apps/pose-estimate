"""The fixture through the processor (no Docker): a synthetic blob video and 2 s of real open-field video.

Needs the SuperAnimal TopViewMouse weights in DeepLabCut's modelzoo/checkpoints folder (the image
has them; locally: `python -m processor.weights download`); the inference tests are skipped
without them.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest

from processor import compare as CP
from processor import inputs as I
from processor import main as P
from processor import params as PR
from processor import sessionnwb as SN
from processor import video as VD
from processor import weights as W
from tests import make_fixture as FX

FAST = {"THREADS": "4", "USE_GPU": "false", "DETECTOR_STRIDE": "5", "QC_CLIP_SECONDS": "1", "BATCH_SIZE": "8"}
SID = "sub-F01_ses-OF-F01-S1"
needs_weights = pytest.mark.skipif(not W.present("superanimal_topviewmouse"), reason="SuperAnimal weights not installed")


@pytest.fixture(scope="session")
def fixture_root(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("fixture")
    for layout in ("linked", "flat"):
        FX.write(root / layout, layout)
    return root


def run(input_dir: Path, output_dir: Path, monkeypatch, **params) -> int:
    monkeypatch.setenv("INPUT_DIR", str(input_dir))
    monkeypatch.setenv("OUTPUT_DIR", str(output_dir))
    for k in PR.PARAMETERS:
        monkeypatch.delenv(k, raising=False)
    for k, v in {**FAST, **params}.items():
        monkeypatch.setenv(k, str(v))
    return P.main()


@pytest.fixture(scope="module")
def linked_run(fixture_root, tmp_path_factory) -> Path:
    if not W.present("superanimal_topviewmouse"):
        pytest.skip("SuperAnimal weights not installed")
    out = tmp_path_factory.mktemp("out-linked")
    with pytest.MonkeyPatch.context() as mp:
        assert run(fixture_root / "linked", out, mp) == 0
    return out / "pose-estimate"


# --- no model needed ----------------------------------------------------------------------------


def test_session_ids():
    root = Path("/in")
    p = Path("/in/video/sub-F01/sub-F01_ses-OF-F01-S1_behavior+image+ophys/314146ca-1394-47b1-9aef-a433b4560398_external_file_0.mp4")
    assert I.derive_session_id(p, root) == "sub-F01_ses-OF-F01-S1"
    p = Path("/in/sub-NR-0031_ses-caa5dddc-1234-4abc-8def-0123456789ab/cam.mp4")
    assert I.derive_session_id(p, root) == "sub-NR-0031_ses-caa5dddc"
    assert I.derive_session_id(Path("/in/my video (1).avi"), root) == "my-video--1"   # each character outside [A-Za-z0-9_-] becomes -


def test_parameters():
    p = PR.read_parameters({})
    assert p["MODEL"] == "superanimal_topviewmouse" and p["VIDEO_ADAPT"] is False and p["CROP"] is None
    assert PR.read_parameters({"CROP": "10, 20, 300, 200"})["CROP"] == (10, 20, 300, 200)
    for bad in ({"MODEL": "superanimal_bird"}, {"CROP": "1,2,3"}, {"PCUTOFF": "2"}, {"VIDEO_ADAPT": "maybe"},
                {"FRAME_STRIDE": "0"}, {"DETECTOR_STRIDE": "1.5"}):
        with pytest.raises(PR.ParameterError):
            PR.read_parameters(bad)


def test_bad_parameter_exits_2(fixture_root, tmp_path, monkeypatch):
    assert run(fixture_root / "linked", tmp_path, monkeypatch, MODEL="nope") == 2


def test_no_video_exits_1(tmp_path, monkeypatch):
    (tmp_path / "in").mkdir()
    assert run(tmp_path / "in", tmp_path / "out", monkeypatch) == 1


def test_probe_and_timebase(fixture_root):
    info = VD.probe(fixture_root / "linked" / "video" / "blob.avi")
    assert (info.width, info.height, info.n_frames) == (FX.SIZE, FX.SIZE, FX.N_FRAMES)
    assert abs(info.fps - FX.FPS) < 1e-6 and not info.vfr
    tb, _ = SN.choose_timebase("blob.avi", info.n_frames, info.fps, None, [],
                               fixture_root / "linked" / "video" / "blob_timestamps.csv")
    assert tb.source == "csv" and abs(tb.times[0] - 12.5) < 1e-9
    clip = fixture_root / "linked" / "video" / "sub-F01_ses-OF-F01-S1_clip.mp4"
    info = VD.probe(clip)
    nwb_path = fixture_root / "linked" / "session" / "sub-F01_ses-OF-F01-S1_behavior.nwb"
    if nwb_path.exists():
        s = SN.read_session_nwb(nwb_path)
        tb, _ = SN.choose_timebase(clip.name, info.n_frames, info.fps, None, [s], None)
        assert tb.source == "nwb:ImageSeries" and abs(tb.times[0] - 100.0267648) < 1e-6
    tb, _ = SN.choose_timebase(clip.name, info.n_frames, info.fps, None, [], None)
    assert tb.source == "fps" and tb.times[1] == pytest.approx(1 / 30)


def test_webm_is_read(tmp_path):
    w = VD.ClipWriter(tmp_path / "clip", 320, 240, 30.0)
    for _, fr in VD.read_frames(FX.CLIP, np.arange(15)):
        w.write(fr[:240, :320])
    w.close()
    out = w.path
    assert w.codec == "vp9" and out.suffix == ".webm"
    assert VD.probe(out).n_frames == 15
    assert I.find_inputs(tmp_path).videos == [out]


def test_regular_rate():
    from processor import outputs as OU
    assert OU.regular_rate(12.5 + np.arange(100) / 30) == pytest.approx(30)
    t = np.cumsum(np.r_[0, np.full(99, 1 / 30)])
    t[50:] += 0.01                                               # a gap
    assert OU.regular_rate(t) is None


def test_frame_selection():
    t = np.arange(300) / 30 + 5.0
    sel = P.frame_selection(t, {"T_START": 6.0, "DURATION": 2.0, "FRAME_STRIDE": 2})
    assert t[sel[0]] >= 6.0 and t[sel[-1]] < 8.0 and np.all(np.diff(sel) == 2)
    assert len(P.frame_selection(t, {"T_START": 0, "DURATION": 0, "FRAME_STRIDE": 1})) == 300


def test_keypoint_matching():
    m = CP.match_keypoints(["nose", "left_ear", "mouse_center", "tail_base", "head_midpoint"],
                           ["nose", "leftear", "bodycenter", "tailbase", "headcenter", "miniscopeUCLA"])
    assert m == {"nose": "nose", "left_ear": "leftear", "mouse_center": "bodycenter", "tail_base": "tailbase",
                 "head_midpoint": "headcenter"}


# --- inference ----------------------------------------------------------------------------------


@needs_weights
def test_outputs_follow_the_contract(linked_run):
    for name in ("summary.json", "provenance.json", "sessions.csv", "keypoint_qc.csv"):
        assert (linked_run / name).is_file(), name
    for sid in ("blob", SID):
        for what in ("_pose.nwb", "_pose.h5", "_video.json", "_confidence.png", "_keypoint_confidence.png"):
            assert (linked_run / f"{sid}{what}").is_file(), sid + what
    prov = json.loads((linked_run / "provenance.json").read_text())
    assert set(prov) == {"app", "version", "variant", "tools", "parameters", "inputs", "started", "finished",
                         "wall_seconds", "peak_rss_mb", "gpu", "sessions", "citations", "licence_notice"}
    for notice in (prov["licence_notice"], json.loads((linked_run / "summary.json").read_text())["licence_notice"]):
        assert "non-commercial" in notice and "10.1038/s41467-024-48792-2" in notice and "mackenziemathislab" in notice
    assert prov["sessions"][SID]["status"] == "ok"
    rows = list(csv.DictReader((linked_run / "sessions.csv").open()))
    assert {r["session_id"]: r["condition"] for r in rows} == {"blob": "synthetic", SID: "5xFAD"}
    summ = json.loads((linked_run / "summary.json").read_text())
    for f in summ["figures"]:
        assert (linked_run / f["path"]).is_file() and len(f["caption"]) > 40


@needs_weights
def test_real_clip_pose_and_agreement(linked_run):
    s = json.loads((linked_run / "summary.json").read_text())["sessions"][SID]
    assert s["fraction_frames_found"] > 0.95
    assert s["keypoints"]["nose"]["mean_confidence"] > 0.5
    if s["time_source"] == "nwb:ImageSeries":
        assert s["t_start"] == pytest.approx(100.0267648, abs=1e-4)
        agr = s["agreement"]
        assert agr["matched_keypoints"]["tail_base"] == "tailbase"
        assert agr["median_px_error"] < 15        # a 1100 x 954 frame; the mouse is ~70 px long
        assert (linked_run / f"{SID}_agreement.png").is_file()
    assert (linked_run / f"{SID}_labeled.webm").stat().st_size > 1000


@needs_weights
def test_nwb_content(linked_run):
    from pynwb import NWBHDF5IO
    with NWBHDF5IO(str(linked_run / f"{SID}_pose.nwb"), "r") as io:
        nwb = io.read()
        pe = nwb.processing["behavior"]["PoseEstimation"]
        assert len(pe.pose_estimation_series) == 27
        nose = pe.pose_estimation_series["nose"]
        assert nose.data.shape == (60, 2) and nose.confidence.shape == (60,)
        assert nose.timestamps is not None and len(nose.timestamps) == 60   # DANDI 002047: irregular frames
        assert nwb.identifier != "02c9a16c-3f8a-4c17-8e05-936be690d47f"      # a new identifier per file
        assert "02c9a16c-3f8a-4c17-8e05-936be690d47f" in nwb.notes and nwb.session_id == "OF_F01_S1"
        assert "non-commercial" in nwb.notes and "10.1038/s41593-018-0209-y" in nwb.notes
        assert pe.source_software == "DeepLabCut" and "superanimal_topviewmouse" in pe.scorer
        assert list(pe.skeleton.nodes[:]) == list(pe.pose_estimation_series.keys()) or len(pe.skeleton.nodes) == 27
        if nwb.subject is not None:
            assert nwb.subject.subject_id == "F01"
    with NWBHDF5IO(str(linked_run / "blob_pose.nwb"), "r") as io:
        pe = io.read().processing["behavior"]["PoseEstimation"]
        nose = pe.pose_estimation_series["nose"]
        assert nose.timestamps is None                            # regular frames: starting_time + rate
        assert nose.starting_time == pytest.approx(12.5) and nose.rate == pytest.approx(30.0, rel=1e-4)
        assert nose.data.shape[0] == 60                           # times from blob_timestamps.csv


@needs_weights
def test_dlc_h5(linked_run):
    df = pd.read_hdf(linked_run / f"{SID}_pose.h5", "df_with_missing")
    assert df.columns.names == ["scorer", "bodyparts", "coords"] and df.shape == (60, 27 * 3)
    assert df.index[0] == 0


@needs_weights
def test_flat_layout_crop_and_stride(fixture_root, tmp_path, monkeypatch):
    out = tmp_path / "out"
    assert run(fixture_root / "flat", out, monkeypatch, CROP="400,250,400,350", FRAME_STRIDE="2", LABEL="crop",
               QC_CLIP_SECONDS="0") == 0
    top = out / "pose-estimate" / "crop"
    s = json.loads((top / "summary.json").read_text())["sessions"]
    assert s["blob"]["status"] == "failed"            # the crop is outside the 256 x 256 blob video
    assert s[SID]["n_frames_analysed"] == 30
    df = pd.read_hdf(top / f"{SID}_pose.h5", "df_with_missing")
    x = df.xs("x", level="coords", axis=1).to_numpy()
    assert np.nanmin(x) >= 400 - 1                    # coordinates are in full-frame pixels
    assert not list(top.glob(f"{SID}_labeled.*"))


@needs_weights
def test_lab_model_port(fixture_root, tmp_path, monkeypatch):
    """An exported DeepLabCut 3 model in the `model` port is used instead of SuperAnimal (here built
    from the SuperAnimal weights, which exercises the same loading path as a lab's own export)."""
    import torch
    from deeplabcut.pose_estimation_pytorch.config.pose import PoseConfig
    from deeplabcut.pose_estimation_pytorch.modelzoo.utils import get_super_animal_snapshot_path
    inp = tmp_path / "in"
    (inp / "video").mkdir(parents=True)
    (inp / "model").mkdir()
    (inp / "video" / "clip.mp4").symlink_to(fixture_root / "linked" / "video" / "sub-F01_ses-OF-F01-S1_clip.mp4")
    cfg = PoseConfig.build_for_superanimal_inference(super_animal="superanimal_topviewmouse", model_name="hrnet_w32",
                                                     detector_name="fasterrcnn_resnet50_fpn_v2", max_individuals=1,
                                                     device="cpu")
    pose = torch.load(get_super_animal_snapshot_path("superanimal_topviewmouse", "hrnet_w32", download=False),
                      map_location="cpu", weights_only=True)["model"]
    det = torch.load(get_super_animal_snapshot_path("superanimal_topviewmouse", "fasterrcnn_resnet50_fpn_v2",
                                                    download=False), map_location="cpu", weights_only=True)["model"]
    cfg_d = cfg.to_dict(normalize=True)   # as deeplabcut.export_model writes it
    export = tmp_path / "DLC_labmouse_hrnet_w32.pt"
    torch.save({"config": cfg_d, "pose": pose, "detector": det}, export)
    import zipfile                                     # the port takes a .zip too
    with zipfile.ZipFile(inp / "model" / "labmodel.zip", "w") as z:
        z.write(export, "exported-models-pytorch/DLC_labmouse_hrnet_w32.pt")
    out = tmp_path / "out"
    assert run(inp, out, monkeypatch, DURATION="1", QC_CLIP_SECONDS="0") == 0
    summ = json.loads((out / "pose-estimate" / "summary.json").read_text())
    assert summ["model_kind"] == "lab" and summ["scorer"] == "DLC_DLC_labmouse_hrnet_w32"
    assert summ["licence_notice"] is None                     # no SuperAnimal weights used
    assert summ["sessions"]["clip"]["n_frames_analysed"] == 30


def test_bottom_up_raw_avi_is_refused(tmp_path):
    """OpenCV's linux FFmpeg reader crashes the process on bottom-up uncompressed AVIs: refuse them."""
    import struct
    frames, _ = FX.blob_frames(4)
    p = tmp_path / "bu.avi"
    FX.write_avi(p, frames)
    raw = bytearray(p.read_bytes())
    i = raw.find(b"strf")
    struct.pack_into("<i", raw, i + 16, FX.SIZE)          # biHeight > 0: bottom-up
    p.write_bytes(bytes(raw))
    assert VD.bottom_up_raw_avi(p)
    with pytest.raises(VD.VideoError):
        VD.probe(p)


def test_boxes_close():
    from processor.infer import boxes_close
    a = np.array([100, 100, 60, 40])
    assert boxes_close(a, np.array([130, 110, 60, 40]))          # overlapping
    assert boxes_close(a, np.array([155, 100, 60, 40]))          # beside it: centres 55 px apart, box size 60 px
    assert not boxes_close(a, np.array([600, 500, 60, 40]))      # a rig object across the arena


def test_video_adapt_refused_on_cpu_variant(fixture_root, tmp_path, monkeypatch):
    monkeypatch.setenv("TORCH_FLAVOUR", "cpu")
    assert run(fixture_root / "linked", tmp_path, monkeypatch, VIDEO_ADAPT="true") == 2


def test_imageseries_rate_timebase():
    s = SN.SessionNWB(Path("x.nwb"), {}, [SN.ImageSeriesRef("/acquisition/cam", ["video/cam.mp4"], None, 5.0, 25.0)], [])
    tb, _ = SN.choose_timebase("cam.mp4", 10, 30.0, None, [s], None)
    assert tb.source == "nwb:ImageSeries" and tb.times[0] == 5.0 and tb.times[1] == pytest.approx(0.04 + 5.0)


def test_sheet_match_is_exact():
    rows = [I.SheetRow("a", "cam1.mp4"), I.SheetRow("b", "subj/cam1.mp4")]
    assert I.sheet_match(rows, Path("/in/video/xcam1.mp4")) is None
    assert I.sheet_match(rows, Path("/in/subj/cam1.mp4")).session_id == "b"
    assert I.sheet_match(rows, Path("/in/other/cam1.mp4")).session_id == "a"
