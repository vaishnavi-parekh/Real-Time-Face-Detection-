# Real-Time Face Detection (Image & Video Processing Project)

Detects faces live from a webcam or a video file, with a simple desktop GUI.

## Setup
1. Install Python 3.9+ (on Linux also: `sudo apt install python3-tk`)
2. `pip install -r requirements.txt`
3. `python face_detection_app.py`

## Using it
- **Source**: Webcam or Video file
- **Detector**:
  - *Haar Cascade* - classical method, works offline
  - *YuNet (DNN)* - more accurate, shows confidence and 5 facial landmarks.
    The small model file downloads automatically the first time (needs internet).
- **Start / Stop**, **Snapshot** (annotated frame), **Save faces** (cropped faces)
- **Blur faces** anonymizes detected faces. Outputs go to `snapshots/`.

## Pipeline
Capture frame -> downscale to 640 px wide -> detect (Haar: grayscale + histogram
equalization + cascade; YuNet: CNN) -> scale boxes back -> draw boxes, scores,
FPS -> display.

## Troubleshooting
- Camera won't open: close Zoom/Teams/other camera apps; check OS camera permission.
- Try `cv2.VideoCapture(1)` in the code if you have several cameras.
- YuNet download blocked: download `face_detection_yunet_2023mar.onnx` from
  github.com/opencv/opencv_zoo (models/face_detection_yunet) into a `models/` folder.
