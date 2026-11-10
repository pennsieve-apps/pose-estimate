# pose-estimate

> **Usage and licence.** The default pose models are the **SuperAnimal** weights of the
> [Mathis Lab, EPFL](https://www.mackenziemathislab.org/) (Ye et al. 2024, *Nat Commun* 15:5165,
> [doi:10.1038/s41467-024-48792-2](https://doi.org/10.1038/s41467-024-48792-2); model cards
> [TopViewMouse](https://huggingface.co/mwmathis/DeepLabCutModelZoo-SuperAnimal-TopViewMouse) and
> [Quadruped](https://huggingface.co/mwmathis/DeepLabCutModelZoo-SuperAnimal-Quadruped)), run with
> [DeepLabCut](https://github.com/DeepLabCut/DeepLabCut). They are used under their licence for
> **academic, non-commercial research only** (`licenses/SuperAnimal-LICENSE.txt`, `THIRD_PARTY.md`).
> **Commercial users** must contact the Mathis Lab (or EPFL's technology transfer office, tto@epfl.ch,
> as the model cards say), or give the app their own DeepLabCut model
> through the `model` input (then the SuperAnimal weights are not used).
> **When publishing results, please cite** Ye et al. 2024
> ([doi:10.1038/s41467-024-48792-2](https://doi.org/10.1038/s41467-024-48792-2)) and Mathis et
> al. 2018, *Nat Neurosci* 21:1281-1289
> ([doi:10.1038/s41593-018-0209-y](https://doi.org/10.1038/s41593-018-0209-y)).
> Every run that uses SuperAnimal writes this notice, with these links, to its log, `summary.json`,
> `provenance.json` (`licence_notice`) and the `notes` of each output NWB.

This app runs **[DeepLabCut](https://github.com/DeepLabCut/DeepLabCut)**, developed by Alexander
Mathis, Mackenzie Weygandt Mathis and the DeepLabCut contributors at the A. & M.W. Mathis Labs
(Rowland Institute at Harvard; EPFL, Lausanne) ([Mathis et al. 2018](https://doi.org/10.1038/s41593-018-0209-y);
[Nath et al. 2019](https://doi.org/10.1038/s41596-019-0176-0);
[Lauer et al. 2022](https://doi.org/10.1038/s41592-022-01443-0)), with the **SuperAnimal**
pretrained models (TopViewMouse, Quadruped) of Shaokai Ye, Anastasiia Filippova, Jessy Lauer,
Steffen Schneider, Maxime Vidal, Tian Qiu, Alexander Mathis and Mackenzie Weygandt Mathis
([Ye et al. 2024](https://doi.org/10.1038/s41467-024-48792-2); weights on
[Hugging Face](https://huggingface.co/mwmathis/DeepLabCutModelZoo-SuperAnimal-TopViewMouse)), and
writes NWB with [PyNWB](https://github.com/NeurodataWithoutBorders/pynwb) and the
[ndx-pose](https://github.com/rly/ndx-pose) extension (Ryan Ly, Ben Dichter, Alexander Mathis,
Talmo Pereira and contributors). **Pennsieve packages these methods; it did not develop them.** The
demo data are the open-field videos and pose of Viktor Plusnin, Nikita Pospelov and colleagues
(Institute for Advanced Brain Studies, Lomonosov Moscow State University;
[DANDI:002047](https://doi.org/10.48324/dandi.002047/0.261106.0903), CC-BY-4.0).

Markerless pose estimation from video, the first step of the **Markerless Behaviour** workflow:

- `pose-estimate → pose-kinematics → behavior-syllables → neuro-report`

Give it a video of an animal (a mouse filmed from above in an open field, a home cage, a maze; or a
four-legged animal from the side) and it finds the animal and its body keypoints (nose, ears, eyes,
neck, spine, hips, shoulders, tail base and tail) in every frame, **without labelling any frames**:
the SuperAnimal models were trained by the Mathis lab on thousands of labelled frames from many
labs, and work zero-shot on new videos. On the demo data the zero-shot pose lands a median 4.6-7.3 pixels
(8-11 % of the mouse's body length) from the pose the dataset's authors obtained with their own,
specially trained DeepLabCut model (see "Demo results").

For a computational analyst: DeepLabCut 3.0.2 (PyTorch engine), top-down: a Faster R-CNN
(ResNet-50 FPN v2) detector finds the animal, an HRNet-w32 pose model predicts 27 (TopViewMouse) or
39 (Quadruped) keypoints in a 256 x 256 crop around it; optional SuperAnimal video adaptation
(pseudo-label self-training on the video); or the lab's own DeepLabCut 3 model. Frames stream from
the video in chunks (the movie is never held in memory). The pose is written on the **session
clock** (the frame timestamps of the session NWB, or a timestamps CSV), as ndx-pose
`PoseEstimation` in an NWB file that copies the session's identity, so `pose-kinematics` and
`neuro-align` can line it up with neural data.

## Ports

| Port | Direction | What |
|---|---|---|
| `video` | in, required | Videos (`.mp4`, `.avi`, `.mov`, `.webm`; `.m4v`, `.mkv` also read), one session per video. Optional `<video stem>_timestamps.csv` (`frame, time`) next to a video, and a `sessions.csv` sheet (`session_id, path, subject_id, condition, group, ...`). |
| `session` | in, optional | Session NWB file(s), e.g. from DANDI: frame timestamps (an `ImageSeries` naming the video, else a `PoseEstimation` / `ImageSeries` of the same session with one timestamp per frame), session identity and subject (copied into our NWB), and any pose already in the file (compared with ours). |
| `model` | in, optional | The lab's own DeepLabCut 3 (PyTorch) model, used instead of SuperAnimal: a `.pt` from `deeplabcut.export_model`, or the project's train folder (`pytorch_config.yaml` + `snapshot-*.pt`, and `snapshot-detector-*.pt` for a top-down model), either also zipped. The licence-clean path. |
| `pose` | out | Everything under `pose-estimate/` (below). |

**Session id** (conventions.md): the `sessions.csv` row whose `path` names the file or a parent
folder; else `sub-<a>_ses-<b>` from the file name or a parent folder (DANDI keeps behaviour videos
in `sub-X_ses-Y_.../<uuid>_external_file_0.mp4`; a UUID session label is cut to 8 characters);
else the file stem, sanitised. The session NWB is matched to the video by session id, or by the
`ImageSeries` that names the video file.

**Frame times** (in order): an NWB `ImageSeries` whose `external_file` is this video (or one of the
same session with exactly one timestamp per frame); a `PoseEstimation` of the same session with one
timestamp per frame; `<video stem>_timestamps.csv`; else `frame / fps`. `summary.json` says which
(`time_source`). When an `ImageSeries` naming the video has *more* timestamps than the video has
frames, the first ones are used and a warning says so (frames lost at the end); fewer, and it is
not used. Real cameras rarely run at a constant rate (DANDI 002047's frame intervals range from
7.5 to 59 ms), so timestamps matter: `fps_effective` and, beyond 0.1 %, `drift_pct` are reported.

## Parameters

| Parameter | Default (gpu / cpu variant) | What it does |
|---|---|---|
| `MODEL` | `superanimal_topviewmouse` | `superanimal_topviewmouse` (mice from above, 27 keypoints) or `superanimal_quadruped` (side view, 39 keypoints). Ignored with a lab model. |
| `VIDEO_ADAPT` | `false` | SuperAnimal video adaptation: pseudo-labels `ADAPT_FRAMES` frames with the zero-shot model and fine-tunes the pose model and the detector on them (4 epochs each, DeepLabCut's defaults). Off by default: it roughly doubles the run time on a T4 and writes about 1 GB of frames to `/tmp`; turn it on when the confidence figure shows long uncertain stretches. |
| `ADAPT_FRAMES` | `1000` | Frames sampled evenly over the analysed window for the adaptation. |
| `MAX_INDIVIDUALS` | `1` | Animals per frame. Above 1 the animals are not tracked across frames (ordered by detector score); the detector then runs on every frame and gaps are not filled. |
| `PCUTOFF` | `0.6` | Confidence cut-off (DeepLabCut's p-cutoff) for the figures and summary. All keypoints are written with their confidence. |
| `T_START`, `DURATION` | `0`, `0` | Analysed window, seconds on the session clock (0 = from the first frame / to the end). |
| `FRAME_STRIDE` | `1` | Analyse every n-th frame. |
| `CROP` | empty | `x,y,width,height`: analyse only this rectangle (e.g. the arena). Coordinates are still written in pixels of the whole frame. **Crop to the arena when the rig has bright fixtures** (wall marks, white patches, lights): the detector can take them for the animal. |
| `DETECTOR_STRIDE` | `1` / `5` | Run the detector on every n-th frame (and the last of each chunk of about 32 frames) and interpolate its box, enlarged by 15 %, in between, but only between two detections whose boxes overlap or whose centres are within one box size (a jump means one of them is not the animal). Single animal only. |
| `FILL_DETECTOR_GAPS` | `true` | Single animal: when the detector finds no animal, use a box around the previous frame's confident keypoints, frame by frame, while the pose's mean keypoint confidence is at least 0.25. The carried box can follow a non-mouse object (a cable, a shadow) until the detector finds the animal again, as long as the pose model is confident on it. |
| `BATCH_SIZE` | `16` / `8` | Frames per pose-model batch (the detector takes 1 on CPU, up to 4 on GPU). |
| `QC_CLIP_SECONDS` | `10` | Length of the labelled QC clip, taken from the most active, well-tracked stretch; 0 = none. |
| `LABEL` | empty | Extra folder level `pose-estimate/<LABEL>/`. |
| `SEED`, `THREADS`, `USE_GPU` | `42`, `0`, `auto` / `false` | Seed; CPU threads (0 = cgroup quota); `auto` uses the GPU when torch sees one and logs `USE_GPU=auto: cuda not available, using cpu` otherwise. |

In single-animal mode a pose whose mean keypoint confidence is below 0.25 counts as **not found**
(written as NaN), whatever its box came from, so a detector false positive on a rig object gives
no pose and gap filling can take over. Bad values exit with code 2; so does `VIDEO_ADAPT=true` on
the cpu variant (it peaks at about 7.5 GB, against that variant's 8 GB). A session that fails is logged and skipped; the step exits 1 only
when every session failed or there was no video.

**Why the detector gets help.** On the demo videos the mouse is small (about 70 px long in a
1100 x 954 frame) and wears a miniscope with a bright white cable; when it curls up, grooms or rears
under the cable the SuperAnimal detector often finds nothing (in one minute of
`sub-F01_ses-OF-F01-S1`, 29 % of frames had no detection; in the frames we checked, nothing even at
a 0.01 score threshold). The pose model itself
copes well with those frames once it is given a box, so `FILL_DETECTOR_GAPS` carries the box from
the previous frame's pose (see "Demo results" for how many frames that concerns).

## Outputs

All in `$OUTPUT_DIR/pose-estimate/` (`/<LABEL>/` when set):

| File | What |
|---|---|
| `<session_id>_pose.nwb` | `processing/behavior/PoseEstimation` (ndx-pose 0.4.0): one `PoseEstimationSeries` per keypoint (`data` x, y in pixels of the original frame, `confidence` 0-1, NaN where the animal was not found; frame times on the session clock: explicit `timestamps`, stored once and linked, when the frame intervals vary (as in DANDI 002047), else `starting_time` + `rate` when every interval is within 0.1 % of the median), `Skeletons/skeleton` (nodes, edges), `scorer` (e.g. `superanimal_topviewmouse_hrnet_w32_fasterrcnn_resnet50_fpn_v2`), `source_software` DeepLabCut + version, `original_videos`, `labeled_videos`, `dimensions`, a `camera` device. Copies `session_start_time`, `timestamps_reference_time`, `session_id`, `session_description`, `subject`, `experimenter`, `institution`, `lab`, `keywords` and `experiment_description` from the session NWB; its `identifier` is a new UUID, and `notes` names the source file and its identifier. Without a session NWB, `session_start_time` is the video file's modification time (said in `notes`) and the subject comes from `sessions.csv` columns (`subject_id`, `species`, `sex`, `age`, `date_of_birth`, `genotype`, `strain`); when species, sex or age are unknown the file is valid NWB but not DANDI-ready, and `summary.json` warns. With several animals: `PoseEstimation_animal<i>`. |
| `<session_id>_pose.h5` | DeepLabCut's table (key `df_with_missing`): columns `(scorer, bodyparts, coords)` with coords `x, y, likelihood` (`(scorer, individuals, bodyparts, coords)` for several animals); rows indexed by the **video frame number**. |
| `<session_id>_labeled.webm` | QC clip (VP9 in WebM, written by OpenCV's LGPL FFmpeg; MJPEG in `_labeled.avi` only if a build lacks VP9, with a warning; at most 720 px wide): keypoints coloured by body region, hollow below `PCUTOFF`, skeleton in white, session time stamped. |
| `<session_id>_video.json` | Video probe and time base: `width, height, fps, n_frames` (decoded count), `n_frames_header`, `codec`, `duration_s`, `variable_frame_rate`, `container_duration_s`, `rotation_deg` (container display rotation; frames are analysed as stored), `time_source`, `time_source_detail`, `fps_effective`, `t_first_frame`, `t_last_frame`, `analysed` (window, frame range, stride, crop), `qc_clip`, `drift_pct` (> 0.1 % only). |
| `<session_id>_example_frames.png`, `_confidence.png`, `_keypoint_confidence.png`, `_agreement.png` | Figures with captions in `summary.json` (agreement only when the session NWB has a pose). |
| `sessions.csv` | `session_id, path, subject_id, condition, group, status, n_frames, n_frames_analysed` + other sheet columns. |
| `keypoint_qc.csv` | `session_id, keypoint, mean_confidence, fraction_low_confidence, median_px_error`. |
| `summary.json` | Per session: sheet columns, `video`, `model`, `scorer`, `video_adapt`, `n_keypoints`, `width`, `height`, `fps`, `fps_effective`, `n_frames_video`, `n_frames_analysed`, `t_start`, `t_stop`, `time_source`, `fraction_frames_detected`, `fraction_frames_box_from_pose`, `mean_confidence`, `fraction_low_confidence_frames` (frames where fewer than half of the keypoints reach `PCUTOFF`, or no animal), `keypoints` (per keypoint `mean_confidence`, `fraction_low_confidence`, `median_px_error`), `pose_seconds`, `frames_per_second_processed`, and `agreement` (below). Plus `figures` (path + caption, in the order the report shows them), `tables`, `warnings`, `licence_notice`. |
| `provenance.json` | The conventions' exact keys (`tools`, `parameters`, `inputs`, times, `peak_rss_mb`, `gpu` name and peak VRAM, per-session status, `citations`). |

**`agreement`** (when the session NWB already has an ndx-pose `PoseEstimation`, e.g. the authors'
own DeepLabCut model on DANDI): keypoints are matched by name (`nose`, `left_ear`/`leftear`,
`right_ear`, `tail_base`/`tailbase`, `mouse_center`/`bodycenter` (the centroid),
`head_midpoint`/`headcenter`, ...), frames by time (nearest within half a frame), and only frames
where *both* estimates reach `PCUTOFF` are compared: `per_keypoint` (`median_px`, `p90_px`,
`mean_px`, `n_frames`, `fraction_compared`, `median_px_all_frames`), `median_px_error` (the
headline: pooled over nose, tail base and centroid), `median_px_error_all_matched`,
`reference_body_length_px` (the reference's median nose-to-tail-base distance) and
`median_error_body_lengths`. Different models define keypoints slightly differently (where exactly
"tail base" or "body centre" is), so part of this is a systematic offset, not error.

**For `pose-kinematics` and other readers.** Read the timestamps with pynwb (or h5py), not
`movement.io.load_poses.from_nwb_file`: movement 0.17 re-creates time as `i / fps` from zero
(its issue #1169), which loses the session clock (our first frame of a window at 100 s comes back
as 0 s). movement does read the file (tested: 27 keypoints, positions and confidence), and also
`<session_id>_pose.h5` with `from_dlc_file`. The NWB opens with ndx-pose 0.2.1 too (a version
warning; the data read correctly).

## Runtime

Measured on DANDI 002047, session `sub-F01_ses-OF-F01-S1` (1100 x 954 px, 30 fps):

All CPU numbers were measured while about five other builders' jobs shared the same 15-core,
8 GB-Docker M-series Mac, so they are pessimistic and noisy.

| Run | Where | Frames | Wall time | Speed | Peak memory |
|---|---|---|---|---|---|
| cpu defaults (`DETECTOR_STRIDE=5`), fixture: 15 + 16 frames | Docker linux/amd64 **under emulation** on an M-series Mac, 4 CPUs, `--memory 4g`, read-only, uid 65534 (cpu image) | 31 | 403 s (incl. ~70 s start-up) | 0.11-0.14 frames/s | 2229 MB (cgroup `memory.peak`) |
| same, gpu image without a GPU (`USE_GPU=auto` falls back to CPU) | same | 31 | 485 s | 0.12-0.13 frames/s | 2169 MB |
| cpu defaults, `sub-F01_ses-OF-F01-S1`, 60 s window (100-160 s), no gap filling | native arm64 (same Mac), 6 threads | 1800 | 1052 s | 1.75 frames/s | 2.7 GB RSS |
| cpu defaults, 30 s windows (100-130 s) of the six demo sessions (commit `a6c9422`) | native arm64, 6 threads | 900 each | 365-576 s per session | 1.6-2.5 frames/s | 2.8 GB RSS |
| detector on every frame (gpu defaults on a CPU) | native arm64, 4 threads, unloaded | 24 | | detector 1.25-1.5 s/frame, pose model 0.05 s/frame | (not measured separately) |
| `VIDEO_ADAPT=true`, `ADAPT_FRAMES=50`, 2 s clip | native arm64, 4 threads | 60 | 1542 s (1405 s adaptation) | | 7.5 GB RSS |

So on 4 vCPU the cpu variant does roughly 1-2 frames per second natively (a minute of 30 fps
video in 15-30 minutes); in amd64 emulation about ten times slower. A whole 12-minute session
(21,583 frames) would take 3-6 hours on 4 vCPU and was **not run on CPU**: whole sessions are for
the gpu variant. The cpu variant's 8 GB / 4 vCPU leaves room for the 3 GB measured (adaptation
needs about 8 GB: not on the cpu variant).

The detector (Faster R-CNN on the whole frame) is about 30x the cost of the pose model on a CPU,
which is why the cpu variant runs it on every 5th frame. **GPU numbers are to be measured on the
dev compute node.** Expected on a T4 (our estimate, not a measurement: roughly 50-150 frames/s for HRNet-w32
top-down with a detector on every frame): a 12-minute 30 fps video in a few minutes without adaptation. VRAM: the pose
model at batch 16 (256 x 256 crops) and the detector at batch 4 (full frames, resized to 800 px)
should fit well within a T4's 16 GB; on a bigger card raise `BATCH_SIZE` to 32-64.

## Variants

| Variant | Image | Defaults | When |
|---|---|---|---|
| `gpu` (default) | CUDA 12.6 PyTorch (`TORCH_FLAVOUR=cuda`), 13.1 GB uncompressed | detector every frame, batch 16, `USE_GPU=auto` | Whole sessions. Runs on CPU (logged) when no GPU is visible. |
| `cpu` | CPU-only PyTorch (`TORCH_FLAVOUR=cpu`), 3.8 GB uncompressed | `DETECTOR_STRIDE=5`, batch 8, `USE_GPU=false`, 4 vCPU / 8 GB | Short clips or excerpts (`T_START`, `DURATION`): roughly a minute of video per 15-30 minutes on 4 vCPU. |

## Demo results

DANDI 002047, the six staged open-field sessions (day 1, `S1`; 3 WT, 3 5xFAD; 1100 x 954 px,
frame times from each NWB's `BehaviorVideo` timestamps, effective rate 29.994 fps), **a 30 s window
per session** (`T_START=100`, `DURATION=30`, 900 frames each) with the cpu variant's defaults
(`DETECTOR_STRIDE=5`, gap filling on). **Stack that produced them:** this repository at commit
`a6c9422` (DeepLabCut 3.0.2, torch 2.15.0, OpenCV 5.0.0.93, as `provenance.json` records; the images
pin torch 2.14.1 and OpenCV 4.14.0.94), run natively (arm64) in the local
virtualenv on an M-series Mac, 6 threads; whole sessions are for the gpu variant on the compute
node. Outputs in `workflow-systems-neuroscience/data/work/openfield-5xfad/pose-estimate/`. The
reference is the authors' own DeepLabCut model stored in each NWB
(`DLC_resnet152_MiceUniversal152Oct23shuffle1_1000000`, DeepLabCut 2.3, 14 keypoints), compared on
frames where both models are confident (>= 0.6):

| Session | Genotype | Pose found (mean keypoint confidence >= 0.25) | Box: interpolated / from previous pose | Mean confidence | Low-confidence frames | **Median error, nose / tail base / centroid** | Body length (ref.) |
|---|---|---|---|---|---|---|---|
| sub-F07_ses-OF-F07-S1 | 5xFAD | 94.2 % | 54.7 % / 31.7 % | 0.60 | 26 % | **5.5 px** (8.0 % of body length) | 68.7 px |
| sub-F14_ses-OF-F14-S1 | 5xFAD | 100 % | 76.1 % / 4.2 % | 0.63 | 18 % | **5.3 px** (8.0 %) | 66.4 px |
| sub-F20_ses-OF-F20-S1 | 5xFAD | 100 % | 65.4 % / 20.7 % | 0.65 | 10 % | **4.6 px** (7.6 %) | 61.2 px |
| sub-F35_ses-OF-F35-S1 | WT | 88.4 % | 54.9 % / 22.0 % | 0.54 | 42 % | **6.9 px** (10.6 %) | 65.0 px |
| sub-F40_ses-OF-F40-S1 | WT | 71.6 % | 14.4 % / 55.0 % | 0.42 | 69 % | **7.3 px** (10.9 %) | 67.1 px |
| sub-F48_ses-OF-F48-S1 | WT | 99.1 % | 61.3 % / 23.4 % | 0.67 | 6 % | **5.2 px** (8.3 %) | 62.9 px |

The rest of the frames got a box from the detector itself (every 5th frame and chunk ends). Frames
where a box gave a pose with mean confidence below 0.25 were dropped as "not found": 0-3.4 % per
session.

Per keypoint, median over the six sessions' median errors: tail base 4.6 px, centroid
(`mouse_center` / `bodycenter`) 6.5 px, nose 7.0 px, left ear 7.7 px, right ear 8.3 px, head
centre 9.0 px. Mean confidence over the six sessions: tail base 0.85, centroid 0.82, left ear 0.80,
nose and right ear 0.77, head midpoint 0.60, ear tips about 0.5 (61 % of frames below the cut-off),
far tail (`tail5`) 0.15: **the miniscope on the head makes the head keypoints (head midpoint, ear
tips, eyes) the least reliable after the tail tip**, while the trunk and tail base are solid. The
weakest session, F40, spends much of the window at the arena wall under the miniscope cable, where
the detector often loses the mouse and the pose sometimes flips head and tail at low confidence
(see its example frames); the median error there is still 7.3 px, about a tenth of a body length.

**Gross errors.** Frames where our centroid is more than 50 px from the authors' (counting every
frame with a pose of ours and a confident reference): F35 2 of 706, F48 2 of 892, and 0-1 in the
other sessions. Before the review fix to `DETECTOR_STRIDE` (interpolation only between detections
that agree, low-confidence poses dropped), the same count was 89 (F35) and 57 (F48): boxes had
been interpolated from detector false positives on rig objects (a wall mark, a white patch) across
the empty arena. All six NWB files pass `nwb-inspect` (schema, NWB Inspector with the DANDI
configuration, DANDI metadata): 0 critical, 0 violations, DANDI-ready.

Video adaptation on 50 frames of the 2 s test clip ran end to end (median 6.3 px on its 60 frames),
as a functional check, not an evaluation.

## Limitations

- **Zero-shot is not ground truth.** SuperAnimal was not trained on these videos. Check the example
  frames and the QC clip; where the confidence figure shows a keypoint pale most of the time (the
  far tail segments, the ear tips), don't use it downstream.
- **Head-mounted implants.** On the demo data the miniscope and its cable sit on the head: the
  ear-tip and eye keypoints are less confident than the trunk, and the detector loses the mouse
  when it curls up under the cable (hence `FILL_DETECTOR_GAPS`). Ears agree with the authors' model
  less well than the nose and tail base, partly because the two models place "ear" differently.
- **Head-tail flips at low confidence.** Curled up at the wall under a cable, the model can swap
  head and tail; those frames are low-confidence (hollow in the figures) and should be dropped
  downstream (`pose-kinematics` filters on confidence).
- **Identity.** With `MAX_INDIVIDUALS > 1`, animals are not tracked across frames; use a lab model
  with DeepLabCut's tracking for multi-animal work.
- **Camera views.** TopViewMouse is for overhead views; angled home-cage cameras and side views are
  worse (use `superanimal_quadruped` for side views).
- **Pixels only.** Positions are in pixels of the video frame; `pose-kinematics` converts to cm when
  the arena size is known.
- **No 3D.**
- **Uncompressed bottom-up AVIs** (raw DIB, positive height) are refused with a message to
  re-encode: OpenCV 4.14's FFmpeg reader on linux/amd64 corrupts memory on them and would kill the
  whole step.
- **VIDEO_ADAPT on a CPU** is slow (50 frames: 23 minutes on 4 threads of an M-series Mac, under
  load) and needs about 8 GB of RAM (peak RSS 7.5 GB): the cpu variant refuses it (exit 2); use
  the gpu variant.
- **Rig objects.** Bright fixtures can be detected as the animal; with a single animal such boxes
  give low-confidence poses, which are dropped, but `CROP` to the arena avoids them altogether.

## Platform notes

The image is Lambda-safe as the brief requires (tested with `--read-only --user 65534:65534
--tmpfs /tmp`, `HOME` and `TMPDIR` under `/tmp/w`): the weights live in the read-only image
(DeepLabCut's `modelzoo/checkpoints` in site-packages); matplotlib, numba, torch and Hugging Face
caches are pointed at `$TMPDIR` at start-up (`processor/entry.py`); all scratch (video adaptation,
an unpacked lab model) goes to `$TMPDIR`. Its sizes (14 GB gpu, 8 GB / 8 h cpu) are beyond Lambda's
limits, as pose inference on a whole session should be.

## Running it locally

```
python3.12 -m venv .venv && .venv/bin/pip install -r requirements-dev.txt
make weights PYTHON=.venv/bin/python     # SuperAnimal weights into the local DeepLabCut, checksummed
make test PYTHON=.venv/bin/python        # pytest: synthetic blob + 2 s of real video (~3 min)
make run                                 # cpu image in Docker over the fixture (no network, read-only)
```

## Licences

Our code is Apache-2.0 (`LICENSE`). DeepLabCut is LGPL-3.0-or-later (used unmodified, as a
library; the LGPL-3.0 incorporates the GPL-3.0's terms, whose text is in `licenses/GPL-3.0.txt`);
PyTorch, pynwb, hdmf and ndx-pose are BSD; OpenCV is Apache-2.0 with an LGPL build of FFmpeg, which
decodes the videos and encodes the QC clip (VP9 via libvpx, BSD). **Nothing GPL is in the image**:
no `ffmpeg` executable (imageio-ffmpeg, a declared DeepLabCut dependency that DeepLabCut does not
import, is left out) and no PyAV. The gpu image contains NVIDIA's CUDA
libraries (proprietary, redistributable). **The SuperAnimal weights are academic, non-commercial
only.** Details in `THIRD_PARTY.md`.

## Citations

Please cite these works if you use this app. Verified on Crossref/DataCite and PubMed; full author
lists in `CITATION.cff`.

- **DeepLabCut**: Mathis A, Mamidanna P, Cury KM, Abe T, Murthy VN, Mathis MW, Bethge M.
  DeepLabCut: markerless pose estimation of user-defined body parts with deep learning. *Nat
  Neurosci* 2018;21(9):1281-1289. doi:10.1038/s41593-018-0209-y, PMID 30127430.
- **DeepLabCut (Python package)**: Nath T, Mathis A, Chen AC, Patel A, Bethge M, Mathis MW. Using
  DeepLabCut for 3D markerless pose estimation across species and behaviors. *Nat Protoc*
  2019;14(7):2152-2176. doi:10.1038/s41596-019-0176-0, PMID 31227823.
- **SuperAnimal models**: Ye S, Filippova A, Lauer J, Schneider S, Vidal M, Qiu T, Mathis A, Mathis
  MW. SuperAnimal pretrained pose estimation models for behavioral analysis. *Nat Commun*
  2024;15:5165. doi:10.1038/s41467-024-48792-2, PMID 38906853.
- **DeepLabCut multi-animal / top-down engine**: Lauer J, Zhou M, Ye S, Menegas W, Schneider S,
  Nath T, Rahman MM, Di Santo V, Soberanes D, Feng G, Murthy VN, Lauder G, Dulac C, Mathis MW,
  Mathis A. Multi-animal pose estimation, identification and tracking with DeepLabCut. *Nat
  Methods* 2022;19(4):496-504. doi:10.1038/s41592-022-01443-0, PMID 35414125.
- **HRNet** (pose model architecture): Sun K, Xiao B, Liu D, Wang J. Deep High-Resolution
  Representation Learning for Human Pose Estimation. *2019 IEEE/CVF Conference on Computer Vision
  and Pattern Recognition (CVPR)*, 5686-5696. doi:10.1109/CVPR.2019.00584.
- **Faster R-CNN** (detector): Ren S, He K, Girshick R, Sun J. Faster R-CNN: Towards Real-Time
  Object Detection with Region Proposal Networks. *IEEE Trans Pattern Anal Mach Intell*
  2017;39(6):1137-1149. doi:10.1109/TPAMI.2016.2577031, PMID 27295650.
- **NWB**: Rübel O, Tritt A, Ly R, Dichter BK, Ghosh S, Niu L, Baker P, Soltesz I, Ng L, Svoboda K,
  Frank L, Bouchard KE. The Neurodata Without Borders ecosystem for neurophysiological data science.
  *eLife* 2022;11:e78362. doi:10.7554/eLife.78362, PMID 36193886.
- **ndx-pose** (software): Ly R, Dichter B, Mathis A, Maree L, Brozdowski C, Mayorquin H, Pereira T,
  Berrigan E, Adkisson P, Trapani A. https://github.com/rly/ndx-pose (version 0.4.0).
- **Demo data**: Plusnin V, Pospelov N. Hippocampal CA1 one-photon calcium imaging and pose-tracked
  behavior in wild-type and 5xFAD mice across four behavioral paradigms. DANDI Archive, 2026.
  doi:10.48324/dandi.002047/0.261106.0903 (contributors listed on DANDI: Plusnin V, Rogozhnikova O,
  Sotskov V, Pospelov N, Ivanova A, Savelev N, Toropova K, Ivashkina O, Anokhin K).

## Acknowledgements

DeepLabCut and the SuperAnimal models are the work of the A. & M.W. Mathis Labs and the DeepLabCut
community; the DeepLabCut project thanks the Rowland Institute at Harvard (2017-2020), EPFL (2020-)
and the Chan Zuckerberg Initiative (Essential Open Source Software for Science; Cycle 6 jointly
with the Kavli Foundation) for funding, and the labs that contributed SuperAnimal training data (listed on the model cards). NWB and ndx-pose
are developed by the Neurodata Without Borders team (Lawrence Berkeley National Laboratory and
partners) and CatalystNeuro; the demo data are shared on the DANDI Archive by the authors of
DANDI:002047.
