# Gesture Model FPGA Pipeline

## Overview
This repository contains the software side of a gesture-recognition pipeline that targets a custom SoC on the Arty S7-25 FPGA. A webcam on the host Mac collects hand landmarks with MediaPipe, a lightweight MLP classifies gestures, and the quantized model is exported so the FPGA can drive a BiStable robot via an ESP32 WiFi link. The long-term target is a secure, low-latency loop where the FPGA validates a four-gesture passcode through an enclave block, updates a host-facing display, and streams motion commands to the robot.

## System Pipeline
1. **Capture** – `gesture-pipelines/gesture-webcam.py` streams frames, extracts 21-point landmarks, and soft-validates predictions with the quantized weights.
2. **Train** – `training/mlp-training.py` fits the MLP on curated datasets (`landmarks_filtered.csv`) to refresh `gesture_mlp_model.h5`.
3. **Quantize/Export** – `quantizing.py` converts per-layer CSV weights into `quantized_weights.bin` for FPGA consumption.
4. **Deploy** – FPGA logic consumes the binary weights, runs inference, evaluates the gesture passcode in a secure enclave, and relays unlock + control signals to the robot through the ESP32 peripheral.

## Repository Layout
- `data-modifications/` – CSV utilities for combining, pruning, and labeling landmark data.
- `training/` – Model training and evaluation scripts (`mlp-training.py`, `gesture-logger.py`).
- `gesture-pipelines/` – MediaPipe-based webcam and still-image prototypes.
- `images/` – Sample gesture reference images.
- `gesture_mlp_model.h5`, `quantized_weights.bin`, `layer_*_{weights,bias}.csv` – Latest trained assets staged for FPGA tooling.

## Setup
1. Use Python 3.10+ and create a virtual environment (`python -m venv gesture-env`).
2. Activate the environment (`source gesture-env/bin/activate` on macOS/Linux).
3. Install dependencies:
   ```bash
   pip install -r requirements.txt  # or install mediapipe, tensorflow, scikit-learn, opencv-python, matplotlib, pandas, numpy
   ```
4. Plug in a webcam and verify MediaPipe access before running the pipelines.

## Model Training & Evaluation
Run the MLP training loop from the repository root:
```bash
python training/mlp-training.py
```
The script stratifies the dataset, computes class weights, and reports validation accuracy. It also writes `scaler_params.npz`, which captures the normalization statistics used during training—keep this file with the exported weights so inference on the FPGA or host matches your preprocessing. Keep accuracy above 0.90 to maintain reliable unlock sequences; adjust preprocessing or class weights if performance drops. Use `gesture-logger.py` to collect new samples and extend the dataset when onboarding new gestures.

## Quantization & Deployment Artifacts
After training, regenerate the FPGA-ready weights:
```bash
python quantizing.py
```
This script expects the latest `layer_*` CSV exports in the root directory and rewrites `quantized_weights.bin`. Consume this binary inside your HDL/SoC project to initialize BRAM or ROM blocks. Track the checksum or git hash of each binary when flashing the Arty S7-25 to keep the hardware configuration auditable.

## Live Gesture Testing
Use the webcam prototype to verify predictions end-to-end:
```bash
python gesture-pipelines/gesture-webcam.py
```
The script loads `gesture_recognizer.task`, applies the saved scaler parameters, feeds frames through MediaPipe, and evaluates the quantized weights in Python. Confirm latency and class stability here before synthesizing FPGA builds. When experimenting with new display peripherals, mirror FPGA output in the console to speed up debugging.

## FPGA Integration Notes
- Reserve BRAM for the three dense layers and plan a streaming interface for 21 landmark triplets produced by the host.
- The unlock flow requires buffering four gestures; implement a state machine that mirrors the secure enclave logic.
- Use the ESP32 WROOM module as a WiFi co-processor—define a narrow command protocol (`FORWARD`, `LEFT`, etc.) and expose diagnostics on UART for bring-up.
- Plan to surface FPGA status (locked/unlocked, last gesture, radio link health) back to the Mac for operator visibility.

## Roadmap
Short-term objectives include documenting the passcode enclave interface, scripting an automated export flow (train → quantize → package), and measuring end-to-end latency. Mid-term, add HDL testbenches for the MLP core and integrate display peripherals. Long-term, secure the communication path (host ↔ FPGA ↔ ESP32) and finalize robot control behaviors.
