# Webcam Accident Detection and Vehicle Trajectory Analysis

An end-to-end computer-vision baseline that supports:

- accident/non-accident CNN training from the Kaggle CCTV image dataset;
- real-time webcam processing;
- uploaded-video processing through a Streamlit interface;
- YOLO vehicle detection and ByteTrack identities;
- paths for each tracked vehicle;
- estimation of the vehicles involved in an accident;
- green pre-accident and red post-accident paths;
- collision-point marking and JSON event reports;
- pattern-recognition evaluation using accuracy, precision, recall, F1, and confusion matrices;
- an optional CNN-feature + SVM baseline for module comparison.

## Important scientific limitation

The selected Kaggle dataset contains **still images** divided into Accident and Non Accident classes. It can train a frame-level CNN, but it does not provide vehicle IDs, collision timestamps, or trajectory labels. Therefore:

- the CNN decides whether frames look like an accident;
- temporal smoothing makes frame decisions more stable in video;
- YOLO + ByteTrack provides vehicle IDs and paths;
- a motion-based pair score estimates which vehicles were involved.

This is a strong undergraduate baseline, but not guaranteed forensic accident reconstruction. A future upgrade should use annotated accident videos and road-plane camera calibration.

## System architecture

```text
Webcam or uploaded video
          ↓
YOLO vehicle detection
          ↓
ByteTrack persistent vehicle IDs
          ↓
Vehicle trajectory history
          ↓
CNN accident probability per sampled frame
          ↓
Temporal probability smoothing
          ↓
Accident event trigger
          ↓
Pair scoring: proximity + IoU + deceleration + heading change
          ↓
Involved IDs + collision point + before/after paths + JSON report
```

## 1. Open in VS Code

Open the extracted project folder in VS Code. Select **Terminal → New Terminal**.

## 2. Create the environment on Windows

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\setup_windows.ps1
```

For later terminals:

```powershell
.\.venv\Scripts\Activate.ps1
```

For an NVIDIA GPU, verify that the installed PyTorch build sees CUDA:

```powershell
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

If CUDA is false, install the correct PyTorch CUDA build from the official PyTorch installer page before training.

## 3. Download and place the dataset

See `scripts/download_dataset.md`.

Expected classes:

```text
data/raw/train/Accident
data/raw/train/Non Accident
data/raw/val/Accident
data/raw/val/Non Accident
data/raw/test/Accident
data/raw/test/Non Accident
```

Check the images:

```powershell
python scripts/inspect_dataset.py --data-dir data/raw
```

## 4. Train the CNN

```powershell
python train_classifier.py --data-dir data/raw --epochs 20 --batch-size 16
```

Generated files:

```text
artifacts/best_accident_cnn.pt
artifacts/best_accident_cnn.metrics.json
```

The training pipeline uses MobileNetV3-Small transfer learning, augmentation, AdamW, early stopping, validation macro-F1 selection, and a final held-out test report.

## 5. Evaluate with pattern-recognition metrics

```powershell
python evaluate_classifier.py --data-dir data/raw
```

Outputs:

```text
artifacts/evaluation/classification_report.json
artifacts/evaluation/confusion_matrix.png
```

Use these results in your report:

- accuracy;
- per-class precision;
- per-class recall;
- macro and weighted F1-score;
- confusion matrix;
- false-positive and false-negative discussion.

## 6. Optional pattern-recognition SVM baseline

This extracts pretrained CNN feature vectors and trains an RBF-SVM with standardized features and stratified five-fold grid search.

```powershell
python train_svm_baseline.py --data-dir data/raw
```

Compare the SVM macro-F1 against the end-to-end fine-tuned CNN. This directly connects the project to a conventional pattern-recognition workflow:

```text
Image → feature extraction → feature scaling → classifier → evaluation
```

## 7. Process an uploaded or recorded video

```powershell
python process_video.py --input "D:\path\traffic_video.mp4" --show
```

Press `q` to close the preview. The annotated MP4 and JSON report are saved in `outputs/`.

## 8. Run real-time webcam mode

```powershell
python run_webcam.py --camera 0
```

Try camera `1` if camera `0` is not your intended webcam:

```powershell
python run_webcam.py --camera 1
```

Press `q` to stop.

## 9. Run the upload website

```powershell
streamlit run app.py
```

The browser application allows video upload, processing, result preview, and downloading of the annotated video and JSON event report.

## Output colours

```text
Light blue path = tracked vehicle with no selected involvement
Green path      = selected vehicle path before the accident frame
Red path        = selected vehicle path after the accident frame
Yellow X        = estimated collision point
Red box         = vehicle selected as involved
```

## How involved vehicles are estimated

For every pair visible when the CNN event is triggered, the code calculates:

```text
pair score = 0.35 × IoU
           + 0.35 × proximity
           + 0.20 × deceleration
           + 0.10 × heading change
```

The highest-scoring pair is reported. A low pair score is marked as `uncertain` in the JSON report rather than presented as certain.

## Tune false detections

Edit `config.yaml`:

```yaml
classifier:
  accident_on_threshold: 0.70
  accident_off_threshold: 0.45
  smoothing_window: 12
  minimum_positive_predictions: 5
```

Increase the on-threshold and positive count to reduce false alarms. Decrease them to improve sensitivity.

## Suggested upgrades

1. Train on video clips with exact impact timestamps.
2. Replace the frame CNN with CNN-LSTM, R(2+1)D, Video Swin, or VideoMAE.
3. Add optical-flow features.
4. Calibrate the fixed camera using homography for speed in metres per second.
5. Add lane detection and time-to-collision features.
6. Collect hard negative examples such as traffic jams, sudden braking, and visual overlap without collision.
7. Evaluate event precision, event recall, false alarms per hour, time-to-accident, IDF1, and HOTA.

## Troubleshooting

### CNN checkpoint not found

Train first:

```powershell
python train_classifier.py --data-dir data/raw
```

### Webcam does not open

Close Teams, Zoom, browsers, and camera applications. Then try `--camera 1`.

### Very slow inference

- use `yolo11n.pt`, which is already the default;
- reduce the webcam/video resolution;
- confirm CUDA is enabled;
- set `classify_every_n_frames: 3` or `4` in `config.yaml`.

### Output video does not preview in the browser

The file is still downloadable. Install FFmpeg and convert it to H.264 if your browser does not support the OpenCV MP4 codec.

## Project files

```text
accident_trajectory_cv/
├── app.py
├── config.yaml
├── evaluate_classifier.py
├── process_video.py
├── run_webcam.py
├── setup_windows.ps1
├── train_classifier.py
├── train_svm_baseline.py
├── requirements.txt
├── src/
│   ├── config.py
│   ├── geometry.py
│   ├── incident.py
│   ├── model.py
│   ├── pipeline.py
│   └── trajectory.py
├── scripts/
│   ├── download_dataset.md
│   └── inspect_dataset.py
└── tests/
    └── test_geometry.py
```
