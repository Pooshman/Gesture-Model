# Repository Guidelines

## Project Structure & Module Organization
The repository is split between model lifecycle assets and FPGA hand-off artifacts. Use `training/` to iterate on the MLP (`mlp-training.py`) and logging utilities, and `data-modifications/` to curate CSV landmarks ahead of training. Inference pipelines for webcam or still images live in `gesture-pipelines/`, while converted resources (`gesture_mlp_model.h5`, `quantized_weights.bin`, per-layer CSVs) sit at the repository root for the FPGA toolchain. Keep large media under `images/` and place integration helpers for board bring-up in dedicated subdirectories.

## Build, Test, and Development Commands
Activate the virtual environment with `source gesture-env/bin/activate` (macOS/Linux) before running Python. Regenerate the model by executing `python training/mlp-training.py` from the repo root; it trains on `landmarks_filtered.csv`, rewrites `gesture_mlp_model.h5`, and saves `scaler_params.npz` with normalization stats. Convert weights to the packed binary for the FPGA via `python quantizing.py`. For live gesture capture and soft validation, run `python gesture-pipelines/gesture-webcam.py`, ensuring the `gesture_recognizer.task` model file and `scaler_params.npz` are present.

## Coding Style & Naming Conventions
Follow PEP 8: four-space indentation, descriptive snake_case for variables/functions, and UpperCamelCase for class names. Keep module-level constants in ALL_CAPS. Group imports by standard library, third-party, then local modules. Document non-obvious math or hardware assumptions inline with concise comments. Prefer `argparse` or config objects over hard-coded paths when adding new scripts.

## Testing Guidelines
There is no formal unit test suite yet; validate changes by rerunning `mlp-training.py` and confirming accuracy stays within expected bounds (>0.9 on the provided dataset). When altering preprocessing, use `gesture-pipelines/gesture-webcam.py` to perform a smoke test on representative gestures. Preserve CSV headers and column order to avoid mismatched landmarks. Record accuracy/latency deltas in pull request notes for traceability.

## Commit & Pull Request Guidelines
Write commits in the imperative mood (e.g., “Add FPGA UART bridge”). Reference issue IDs where applicable and keep commits scoped to one concern: data prep, training, or hardware export. Pull requests should summarize motivation, list functional validation steps (training accuracy, live capture checks, FPGA simulation notes), and attach screenshots or logs when UI or sensor outputs change. Request at least one review before merging to main hardware integration branches.

## FPGA & Security Considerations
When introducing RTL or SoC assets, document the target board constraints file and expected I/O mapping. Treat the gesture-based unlock sequence as sensitive: never log raw authentication gestures, and keep secure enclave firmware under review-controlled directories with restricted sample data.
