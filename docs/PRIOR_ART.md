# Prior art: what already exists, how it runs, and what to borrow

_Researched 2026-09-25 from public READMEs, code and write-ups. Nothing was downloaded or cloned._

## The short version

Nobody has published a working, evaluated All-22 → top-down pipeline. The closest things are
(a) the 2021 Kaggle **Helmet Assignment** solutions, which mapped sideline/endzone helmets to
real NGS positions and are the best evidence for what works on NFL film, (b) **bgyss/football-tracking**,
an active (Sept 2026) offline All-22 tracker with a real test suite but no ground truth yet, and
(c) the soccer stack (**roboflow/sports**, **SoccerNet GSR**), which is where the mature tooling lives.

Our pipeline already matches the consensus design: pretrained detector at 1280 → BoT-SORT with
camera-motion compensation → fit-once colour clustering with per-track voting → bottom-centre foot
point → homography → CSV + overlay + replay. The gaps are all **data-dependent**: a detector
fine-tuned on football (the single biggest win in every source), automatic field registration,
and a labeled benchmark to measure against.

## 1. American football specifically

### Kaggle NFL Helmet Assignment (2021)

Task: label every helmet box in paired sideline+endzone clips with the NGS player ID. Every top-5
solution used the same skeleton: **YOLOv5 helmet detector → tracker → 2-D point-set registration
(ICP / homography) against NGS → Hungarian assignment → per-track label voting.**

| Place | What mattered |
|---|---|
| 1st K_mat | CNN regresses each helmet's bird's-eye XY straight from the frame; ICP with 4 params; team classifier as a registration constraint; averages assignment matrices across frames before a final Hungarian |
| 2nd tito | YOLOv5 at 1664; two-stage k-means on jersey colours (97→98% with track consistency); OpenCV yard-line detection to fix rotation/trapezoid; SORT with a team step penalty so tracks never switch across teams |
| 3rd hirarin | yolov5l6 at 1280, **bench helmets deliberately excluded from training**; DeepSORT ReID features → k-means for team; final Hungarian on per-track label frequency (0.847 → 0.910) |
| 4th CPMP | Rotation sweep → `cv2.findHomography` → refit, warm-started from the previous frame's H; flags player height vs. the planar assumption as the unsolved error |
| 5th Kyle Lee | Ground-truth boxes scored 0.958 vs 0.90 with his detector: **the detector was the bottleneck**; homography/ICP on top of Hungarian didn't help |

Also useful: Xmartlabs (Canny+Hough field lines for rotation; filtering sideline people raised
precision 0.983 → 0.994); K-NKSM (`findHomography(LMEDS)` warm-started from the previous 60 frames).

### Impact Detection (2021) and Player Contact Detection (2023)

Both fused sideline + endzone: the 2023 winner rasterised NGS as a bird's-eye image and stacked it
as a third input, swapped views as augmentation, and used a black image when a view was missing;
2nd place concatenated endzone and sideline crops side by side. Frame-level video models (TSM,
irCSN, X-CLIP) on helmet-centred crops. Relevant later for the value model, not for tracking.

### GitHub repos

| Repo | What it does | Run / test / view | Status |
|---|---|---|---|
| [bgyss/football-tracking](https://github.com/bgyss/football-tracking) | Offline All-22 tracking: RF-DETR, BoT-SORT/ByteTrack, optional SAM2 masks, calibration from reviewed landmark JSON with timeline keyframes and withheld validation points, cross-shot identity joining | `uv run python -m football_tracking run --input v.mp4 --detector rfdetr --tracker botsort --calibration landmarks.json`; 24 pytest files; outputs `trajectories.csv`, `annotated.mp4` with trails, `field-view.png`; gates: IDF1 ≥ 0.90, ≤ 1 switch/shot, median error 1.5 yd | Active Sept 2026, no license, no ground truth yet |
| [Aust1n-moon/coverage-predictor](https://github.com/Aust1n-moon/coverage-predictor) | YOLO11m helmets (on-field vs sideline) + ByteTrack + stitching; **auto homography from Hough yard lines + hash-mark bands** anchored by `--yard-line 40 --side left`; k-means teams; MLP coverage classifier trained on Big Data Bowl 2023 | `python run.py clip.mp4 --yard-line 40 --side left`; Gradio app; no tests | 9 commits, Sept 2026, no license |
| [JShaikh8/formation-zero](https://github.com/JShaikh8/formation-zero) | All-22 → tracking tables via roboflow/sports (`ViewTransformer`, SigLIP+UMAP+KMeans teams) → DuckDB → text-to-SQL chat; anchors registration to yard lines from play-by-play | No tests or visuals documented | 47 commits |
| ruidazeng/the-playmakers | YOLOv8 with 8 position classes, 503 images, mAP50 0.759 | — | MIT |
| trevorfiez/Football-Player-Tracking-Dataset | 2014 coaches-film boxes + tracks, 9 games | — | Videos came via a dead tool |

Not found: any public repo that fuses sideline + endzone into one evaluated top-down track set; any
public YOLO weights trained on All-22.

## 2. Soccer, where the tooling is mature

- **[roboflow/sports](https://github.com/roboflow/sports)** (MIT, 5.4k stars, last commit May 2025, no tests): YOLOv8 player/pitch-keypoint/ball weights; `sv.ByteTrack`; `TeamClassifier` = SigLIP-base embeddings → UMAP(3) → KMeans(2), fit once on crops sampled across the video; `ViewTransformer` = `cv2.findHomography` on the 32 pitch keypoints with conf > 0.5, recomputed per frame, no smoothing; radar drawn with cv2 from a cm-unit `SoccerPitchConfiguration`. Run: `python main.py --source_video_path X --device mps --mode RADAR`.
- **[SoccerNet Game State Reconstruction](https://github.com/SoccerNet/sn-gamestate)** on [tracklab](https://github.com/TrackingLaboratory/tracklab) (MIT, Hydra configs, TrackEval built in): YOLOv8 COCO `person` → part-based ReID → StrongSORT → keypoint calibration (bbox bottom-centre projected to pitch) → jersey OCR → tracklet-level majority vote for role/number, KMeans on averaged ReID embeddings for team. Metric **GS-HOTA**: HOTA where localisation similarity is a Gaussian on pitch-plane distance (τ = 5 m). 2024/2025 winners both credit **detection quality first**, then tracklet split/merge.
- **Field registration with code**: PnLCalib and No-Bells-Just-Whistles (GPL-2, HRNet keypoints + line extremities → DLT), TVCalib (MIT, line segmentation + per-image optimisation), sportsfield_release (patented, avoid). All soccer-template, per-frame, no temporal smoothing, none knows yard lines or hash marks.
- **Trackers** ([roboflow/trackers](https://github.com/roboflow/trackers), Apache 2): on SportsMOT HOTA ByteTrack 73.0 / BoT-SORT 74.1 / C-BIoU 73.1 / McByte 76.5; docs recommend BoT-SORT for moving cameras. The one American-football MOT paper (AFMOT, arXiv 2511.09455): BoT-SORT best at 45.7 HOTA; **fine-tuning the detector gave +30.8 MOTA, fine-tuning ReID only +0.6.**
- **Detectors**: no published sports benchmark for YOLO11/26 vs RF-DETR. Roboflow's 2026 post: RF-DETR-Small fine-tuned on ~1.5k NFL images, mAP50 74.8. RF-DETR on Apple MPS is slow (an upsample op falls back to CPU); YOLO11 → CoreML is reported 3–4× faster than PyTorch MPS.

## 3. Data, labeling and evaluation

| Dataset | Size | Labels | License / access |
|---|---|---|---|
| Kaggle NFL Helmet Assignment (2021) | 120 videos (60 plays × sideline+endzone, ~10 s, 59.94 fps) + 9,947 helmet stills | Per-frame helmet boxes with player IDs; **NGS 10 Hz x/y in yards with `est_frame` → video frame** | Kaggle login + accept rules; non-commercial, no redistribution |
| Kaggle Impact Detection (2020) | Same 120 videos and stills | Helmet boxes, impact labels, NGS | Same |
| Kaggle Player Contact Detection (2023) | Sideline+endzone videos | `train_baseline_helmets.csv` with player labels; NGS at 10 Hz | Same |
| AFMOT (arXiv 2511.09455) | 20 hand-held videos, 11.4k frames, 189k boxes, 505 tracks | Player + referee boxes with track IDs | CC BY-NC-SA 4.0, gated on Hugging Face |
| Roboflow: FH Technikum Wien "American Football Player Detection" | 171 images | 9 classes incl. players, ball, referee | CC BY 4.0 |
| Roboflow: bronkscottema "Football Players" | 755–1,950 images | american-football-players, center, db, lb, qb, skill | Listed public domain |
| Roboflow: NFL-competition / HF keremberke | 9,947 images | 5 helmet classes | Kaggle images re-uploaded; treat as Kaggle-licensed |

**Benchmark**: the Kaggle Helmet Assignment set is the only public "NFL video + true positions for
the same plays". Caveat: NGS positions are the shoulder-pad sensor, not the feet, so expect a
~0.3–0.5 yd systematic offset. Using it needs Michael to accept the competition rules on Kaggle.

**Labeling tool**: **X-AnyLabeling** (`pip install "x-anylabeling-cvhub[cpu]"`, native macOS,
GPL-3). Loads a video, pre-labels with YOLO + ByteTrack, keeps track IDs, exports YOLO/COCO/MOT.
Workflow: pre-label full-body boxes → correct boxes and IDs on keyframes → the bottom-centre is
the foot point, so detection ground truth comes for free. CVAT has the best keyframe interpolation
but its official images are amd64-only; Label Studio can't do points in video; LabelImg is archived.

**Metrics**: image plane — mAP50 for the detector, HOTA/IDF1/MOTA/IDSW via TrackEval or
`motmetrics`. Field plane — mean/median/RMSE yards, % within 1/2/3 yd, and a GS-HOTA-style HOTA
with τ ≈ 3 yd. Count ID swaps per play separately: they dominate positional error late in plays.
`nfl eval` already reports position error, recall, precision, ID switches and MOTA.

**Fine-tuning recipe** (Ultralytics guide + what worked on AFMOT/SoccerNet): imgsz 1280, classes
`player`/`referee` (exclude bench and crowd), 300–600 labeled frames from ≥ 20 plays (~7–13k
instances) should capture most of the gain; stage 1 `freeze=10` for ~10–20 epochs, stage 2
AdamW lr0 0.001 for 30–60 epochs, mosaic on, `scale=0.5`, `fliplr=0.5`, no rotation/shear,
batch 4–8 at 1280 on the M5. `yolo11-p2` head if players fall under ~20 px in the deep end zone.
`nfl prelabel` + `nfl train` implement this.

## 4. Where we stand and what to do next

Already in the pipeline after tonight: BoT-SORT with camera-motion compensation (default), tiled
inference (`--tiles`), capture by window ID, `nfl eval`, `nfl prelabel`, `nfl train`. Already
matched by design: bottom-centre foot point, fit-once team classifier with per-track votes,
imgsz 1280, CSV schema that mirrors Big Data Bowl.

| # | Improvement | Source | Effort | Impact |
|---|---|---|---|---|
| 1 | **Fine-tune the detector** on ~300–600 labeled All-22 frames, `player` vs `referee`, bench people excluded | Kyle Lee (GT boxes 0.958 vs 0.90), AFMOT (+30.8 MOTA), both GSR winners | 1 day + labeling | Highest |
| 2 | **Automatic field registration**: white∩green mask → Hough yard lines → hash-mark bands → 10+ correspondences → `findHomography(RANSAC)`, warm-started from the previous frame's H and smoothed; anchored to the line of scrimmage from the play-by-play (`--yard-line 40 --side left`) | coverage-predictor, tito, K-NKSM, CPMP | 1–2 days | High: replaces 4 clicks per clip, survives pans |
| 3 | **Team-aware association**: penalise cross-team matches in the tracker cost, keep per-track votes | tito, K_mat (+0.02) | 2–4 h | Medium-high on pile-ups |
| 4 | **Play-level late assignment**: accumulate votes weighted by confidence, final Hungarian at the whistle | hirarin (0.847 → 0.910) | 4–6 h | Medium-high, offline only |
| 5 | **Labeled benchmark**: 2–3 own plays fully labeled in X-AnyLabeling; optionally the Kaggle helmet set as a yards answer key | bgyss gates, SoccerNet | 1–2 days | High: makes 1–4 measurable |
| 6 | **Sideline + endzone fusion** in field coordinates: Hungarian-merge the two views, fill occlusions from the other camera | Contact-detection 1st/2nd | 1–2 days | Medium (brief's angle comparison first) |
| 7 | **Offline tracklet split/merge** with OSNet ReID after each play (GTA-link) | GSR 2025 winner | 1 day | Medium on ID swaps |
| 8 | **Auto-snap detection** from the displacement spike with ≥ 14 players on screen | coverage-predictor | 2–3 h | Low-medium |
| 9 | **CoreML export** of the detector for 3–4× inference speed; keep MPS for training | Roboflow/Ultralytics reports | 2 h | Medium once tiling or larger models are used |
| 10 | **Learned foot offset**: helmet/box-bottom to ground contact to cut the planar-assumption error | CPMP, tito | 4 h | Low-medium |

Skip: DeepSORT/ReID fine-tuning as a first move (+0.6 in AFMOT), RF-DETR on MPS (CPU fallback),
sportsfield_release (patented), SigLIP team embeddings (Lab k-means is enough for two NFL jerseys
and runs in microseconds).

## Sources

Kaggle write-ups: [1st](https://www.kaggle.com/competitions/nfl-health-and-safety-helmet-assignment/writeups/k-mat-1st-place-solution), [2nd](https://www.kaggle.com/competitions/nfl-health-and-safety-helmet-assignment/writeups/tito-2nd-place-solution), [3rd](https://www.kaggle.com/competitions/nfl-health-and-safety-helmet-assignment/writeups/fantastic-hirarin-3rd-place-solution-yolov5-deepso), [4th](https://www.kaggle.com/competitions/nfl-health-and-safety-helmet-assignment/writeups/nfl-2nd-half-4th-place-solution), [5th](https://www.kaggle.com/competitions/nfl-health-and-safety-helmet-assignment/writeups/kl-5th-place-solution), [Xmartlabs](https://blog.xmartlabs.com/blog/kaggle-competition-2021/), [scorer](https://github.com/RobMulla/helmet-assignment), [Contact 1st](https://github.com/nvnnghia/nfl3_1st).
Soccer: [roboflow/sports](https://github.com/roboflow/sports), [sn-gamestate](https://github.com/SoccerNet/sn-gamestate), [tracklab](https://github.com/TrackingLaboratory/tracklab), [PnLCalib](https://github.com/mguti97/PnLCalib), [TVCalib](https://github.com/MM4SPA/tvcalib), [trackers benchmark](https://trackers.roboflow.com/latest/learn/evaluate/), [GTA-link](https://github.com/sjc042/gta-link).
Data and tools: [AFMOT paper](https://arxiv.org/html/2511.09455), [X-AnyLabeling](https://github.com/CVHub520/X-AnyLabeling), [TrackEval](https://github.com/JonathonLuiten/TrackEval), [Ultralytics fine-tuning guide](https://docs.ultralytics.com/guides/finetuning-guide), [Roboflow RF-DETR NFL post](https://blog.roboflow.com/american-football-player-tracker/).
