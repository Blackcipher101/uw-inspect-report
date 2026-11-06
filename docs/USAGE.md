# Running, stages, models and known limits (details)

[Back to README](../README.md)

## Run (workstation1, one command)

```
cd ~/uwvio/inspect_report
./run_all.sh ~/uwvio/data/bus_outside/bus_outside.db3 ~/uwvio/data/bus_outside/camchain_bus_outside.yaml \
             ~/uwvio/data/bus_outside/s4_rep.tum ~/uwvio/out/bus_report
```
Creates `.venv` (torch 2.8.0+cu128, transformers 4.57.1, opencv 4.10, rosbags; see `requirements.txt`),
downloads models into the HF cache, writes per-stage logs to `<out>/logs/`, per-event timings to
`<out>/timings.jsonl` (flushed as they occur). Env: `STAGES="keyframes detect localize report eval"`,
`THR=0.15`, `MODE=base|ens`, `VLM=...`, `LABELS=<labels.json>`, `SOURCE_NAME=...`.

Frames for hand labelling only: `STAGES="keyframes" ./run_all.sh <same four args>` then copy
`<out>/label_frames/` (about 40 JPEGs, 800 px wide) back to `label_frames/` in this repo for labelling.
Poses used for the reported run: uw_ct_vio `results/bus_outside/uwct/s4_rep.tum` (~0.34 m Sim3 ATE; standalone cuVSLAM `cuvslam_stereo_slam.tum` had 5.2 m ATE on this bag, SVIn odometry drifts to hundreds of metres). Pass `none` to skip 3D localisation (box-only findings).

Eval: `LABELS=labels.json STAGES=eval ./run_all.sh ... <same out dir>`. `labels.json` format is in `eval.py`.

## Stages and files
| stage | file | notes |
|---|---|---|
| 1 | `keyframes.py` | time (2-4 s) + pose-distance (0.3 m) selection; rectified left/right; label frames |
| 2 | `detect.py` | OWLv2; both prompt modes (`base` first prompt, `ens` max over 3 prompts) from one forward pass |
| 3 | `localize.py`, `geometry.py` | SGBM at 0.5 scale, median depth in inner 50% of box, 1 m / same-class merge |
| 4 | `report.py`, `render.py` | VLM per-crop verification/assessment/severity + summary; tables rendered by code; `--verify-context`, `--inputs` |
| 5 | `eval.py`, `faithfulness.py`, `rescore.py` | runtime, image-level P/R, crop verification, hallucination counts |


## Models and licences
- Detector: `google/owlv2-base-patch16-ensemble` (Apache-2.0).
- VLM: `Qwen/Qwen3-VL-4B-Instruct` (Apache-2.0; bf16 about 9 GB fits 16 GB). 7B/8B bf16 does not fit; `--load-4bit` is wired for
  Qwen2.5-VL-7B-Instruct (Apache-2.0) but untested. Not used: Qwen2.5-VL-3B (research licence), Grounding DINO weights (Apache-2.0, optional alternative, not implemented).

## Known limits (by design, to be stated in any pitch)
- Calibration is the AFRL underwater Kalibr chain; refraction is absorbed by the pinhole/radtan model only approximately.
- Box size = box extent at median depth (fronto-parallel assumption). The VIO world frame is the start-camera frame, not gravity aligned.
- SGBM in murky water has many invalid pixels; findings without valid depth are box-only (no size/position).
- Faithfulness check is lexical (class terms + finding ids); quoted numbers/locations in free text are not verified.
- Severity is a VLM suggestion, not a calibrated assessment.
- One AI annotator, 40 frames and 17 usable crops: all precision/recall values have wide uncertainty; several classes have 0-3 positives.
- Single scene (murky lake, glare from a lamp in about a third of the frames); nothing here shows performance in clear water or on other structures. NTNU data was not run.
- Findings are ranked by detector score (no verifier-aware ranking); the example report includes unconfirmed-by-human items such as F27 (debris, a false positive) and fish/hole_damage crops I could not judge.
- Stereo uses the AFRL calibration; SGBM ran on ws1 CPU. 528 of 641 localised detections got a valid depth; sizes of large boxes (bus panels, windows) are crude.
- Severity labels are VLM suggestions; v1 gave low/medium to 18 of the 20 confirmed findings, including marine-growth patches, with no calibration.
- Qwen3-VL-4B miscounted its own summary ("10" vs 20); the draft must not be used without human review.
- Files: `run_bus*/` hold the raw outputs (detections, findings, crops, VLM outputs, metrics, logs); `examples/` holds the example report with thumbnails (0.5 MB).
