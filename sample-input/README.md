# Sample input

## Small: the test fixture

```
python -m tests.make_fixture data/fixture --layout both
```

writes `data/fixture/linked/` (ports in their own folders) and `data/fixture/flat/` (everything at
the root):

- `video/blob.avi` + `video/blob_timestamps.csv`: 2 s of a synthetic dark blob walking a circle
  (uncompressed AVI written with numpy only). SuperAnimal is not trained on blobs: this checks the
  plumbing (probing, timestamps from a CSV, outputs), not the pose.
- `video/sub-F01_ses-OF-F01-S1_clip.mp4`: 2 s of real top-view open-field video from DANDI 002047
  (below), committed in `tests/data/` (CC-BY-4.0).
- `session/sub-F01_ses-OF-F01-S1_behavior.nwb` (when pynwb and ndx-pose are installed): frame
  timestamps for the clip and the authors' own DeepLabCut pose for those 60 frames, so the
  agreement check runs.
- `sessions.csv`: `session_id, path, subject_id, condition`.

## Real: DANDI 002047 (top-view open field, WT and 5xFAD mice)

Plusnin, Pospelov et al., "Hippocampal CA1 one-photon calcium imaging and pose-tracked behavior in
wild-type and 5xFAD mice across four behavioral paradigms", DANDI:002047 version 0.261106.0903,
doi:[10.48324/dandi.002047/0.261106.0903](https://doi.org/10.48324/dandi.002047/0.261106.0903),
CC-BY-4.0.

Each session has an NWB file and, beside it in a folder of the same name, the behaviour video
that the NWB's `acquisition/BehaviorVideo` (an external-file `ImageSeries`) points to: 1100 x 954
px, 30 fps nominal, about 12 minutes, about 300 MB. Example, session `sub-F01_ses-OF-F01-S1`:

| File | DANDI asset id | Size |
|---|---|---|
| `sub-F01/sub-F01_ses-OF-F01-S1_behavior+image+ophys.nwb` | `52efa19d-15cf-45b6-93ae-4e16a4f4a056` | 45.7 MB |
| `sub-F01/sub-F01_ses-OF-F01-S1_behavior+image+ophys/314146ca-1394-47b1-9aef-a433b4560398_external_file_0.mp4` (behaviour video) | `7e0f3e3c-abdc-4e93-848b-10771dee0a0d` | 280 MB |

```
curl -L -o video.mp4 https://api.dandiarchive.org/api/assets/7e0f3e3c-abdc-4e93-848b-10771dee0a0d/download/
```

**Not** the 16 MB `fba12271-..._external_file_0.mp4` in the same folder (asset
`bad4c8a1-0e8d-408e-93da-d483c81c729f`): that one is the miniscope's motion-corrected calcium movie
(508 x 508, the NWB's `acquisition/MotionCorrectedVideo`), not the behaviour camera.

Give the videos to the `video` port and the NWB files to the `session` port. Keep DANDI's folder
names: the session id comes from the `sub-<a>_ses-<b>` folder, and the NWB's `ImageSeries` names
the video file, so each video is matched with its NWB, gets the NWB's frame timestamps (the frame
intervals vary from about 7.5 to 59 ms, so `frame / fps` would drift), and is compared with the
authors' own DeepLabCut pose stored in the NWB (14 keypoints; nose, ears, head centre, body centre
and tail base match SuperAnimal keypoints).

The workflow's demo copy (6 sessions, 3 WT and 3 5xFAD, with `sessions.csv` and a
`<video stem>_timestamps.csv` next to each video) is staged by
`workflow-systems-neuroscience/scripts/stage-demo-data.py openfield-5xfad`.

## Your own model

To avoid the SuperAnimal weights (non-commercial licence), give your lab's DeepLabCut 3 model to
the `model` port: the `.pt` from `deeplabcut.export_model(config_path)` (in
`<project>/exported-models-pytorch/`), or the train folder
`<project>/dlc-models-pytorch/iteration-N/<name>/train/` with `pytorch_config.yaml` and its
`snapshot-*.pt` files.
