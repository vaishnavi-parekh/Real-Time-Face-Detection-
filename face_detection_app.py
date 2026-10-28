import os
import sys
import time
import urllib.request
from datetime import datetime

import cv2
import numpy as np

try:
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox
    from PIL import Image, ImageTk
except ImportError as exc:  # pragma: no cover
    tk = None
    _GUI_IMPORT_ERROR = exc

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
SNAP_DIR = os.path.join(BASE_DIR, "snapshots")
YUNET_FILE = os.path.join(MODEL_DIR, "face_detection_yunet_2023mar.onnx")
YUNET_URLS = [
    "https://github.com/opencv/opencv_zoo/raw/main/models/"
    "face_detection_yunet/face_detection_yunet_2023mar.onnx",
    "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/"
    "face_detection_yunet/face_detection_yunet_2023mar.onnx",
]

DETECT_WIDTH = 640          # frames are downscaled to this width for detection
DISPLAY_MAX = (960, 620)    # max size of the video area in the window


# --------------------------------------------------------------------------
# Detectors  (each returns a list of dicts: box=(x,y,w,h), score, landmarks)
# --------------------------------------------------------------------------
class HaarDetector:
    name = "Haar Cascade"

    def __init__(self):
        path = os.path.join(cv2.data.haarcascades,
                            "haarcascade_frontalface_default.xml")
        self.cascade = cv2.CascadeClassifier(path)
        if self.cascade.empty():
            raise RuntimeError("Could not load Haar cascade file:\n" + path)

    def detect(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        h = frame.shape[0]
        min_side = max(30, int(h * 0.08))
        boxes = self.cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5,
            minSize=(min_side, min_side))
        return [{"box": tuple(int(v) for v in b), "score": None,
                 "landmarks": None} for b in boxes]


class YuNetDetector:
    name = "YuNet (DNN)"

    def __init__(self, score_threshold=0.7):
        ensure_yunet_model()
        self.threshold = score_threshold
        self.net = cv2.FaceDetectorYN.create(
            YUNET_FILE, "", (320, 320), score_threshold, 0.3, 5000)

    def detect(self, frame):
        h, w = frame.shape[:2]
        self.net.setInputSize((w, h))
        _, faces = self.net.detect(frame)
        results = []
        if faces is not None:
            for f in faces:
                x, y, bw, bh = (int(v) for v in f[:4])
                lm = f[4:14].reshape(5, 2).astype(int)
                results.append({"box": (x, y, bw, bh),
                                "score": float(f[14]), "landmarks": lm})
        return results


def ensure_yunet_model():
    """Download the YuNet ONNX model if it is not already on disk."""
    if os.path.isfile(YUNET_FILE) and os.path.getsize(YUNET_FILE) > 50_000:
        return
    os.makedirs(MODEL_DIR, exist_ok=True)
    last_error = None
    for url in YUNET_URLS:
        try:
            tmp = YUNET_FILE + ".part"
            urllib.request.urlretrieve(url, tmp)
            if os.path.getsize(tmp) < 50_000:      # Git-LFS pointer, not model
                raise RuntimeError("downloaded file too small")
            os.replace(tmp, YUNET_FILE)
            return
        except Exception as exc:                    # try next mirror
            last_error = exc
    raise RuntimeError(
        "Could not download the YuNet model (check your internet).\n"
        f"Details: {last_error}\n\nYou can also download\n"
        "face_detection_yunet_2023mar.onnx from the OpenCV Zoo "
        "(opencv/opencv_zoo, models/face_detection_yunet)\n"
        f"and place it in:\n{MODEL_DIR}")


DETECTORS = {HaarDetector.name: HaarDetector, YuNetDetector.name: YuNetDetector}


# --------------------------------------------------------------------------
# Drawing / processing helpers
# --------------------------------------------------------------------------
def process_frame(frame, detector, blur=False, show_landmarks=True):
    """Detect faces on a downscaled copy, draw results on the full frame.
    Returns (annotated_frame, detections_in_full_frame_coords)."""
    h, w = frame.shape[:2]
    scale = DETECT_WIDTH / w if w > DETECT_WIDTH else 1.0
    small = cv2.resize(frame, None, fx=scale, fy=scale) if scale != 1.0 else frame
    dets = detector.detect(small)

    out = frame.copy()
    full = []
    for d in dets:
        x, y, bw, bh = (int(round(v / scale)) for v in d["box"])
        x, y = max(0, x), max(0, y)
        bw, bh = min(bw, w - x), min(bh, h - y)
        if bw <= 0 or bh <= 0:
            continue
        lm = None
        if d["landmarks"] is not None:
            lm = (d["landmarks"] / scale).astype(int)
        full.append({"box": (x, y, bw, bh), "score": d["score"], "landmarks": lm})

        if blur:
            roi = out[y:y + bh, x:x + bw]
            k = max(15, (bw // 3) | 1)
            out[y:y + bh, x:x + bw] = cv2.GaussianBlur(roi, (k, k), 0)

        cv2.rectangle(out, (x, y), (x + bw, y + bh), (0, 220, 0), 2)
        label = "face" if d["score"] is None else f"face {d['score']:.2f}"
        cv2.putText(out, label, (x, max(15, y - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 0), 2)
        if show_landmarks and lm is not None:
            for px, py in lm:
                cv2.circle(out, (int(px), int(py)), 3, (0, 120, 255), -1)
    return out, full


# --------------------------------------------------------------------------
# GUI
# --------------------------------------------------------------------------
class App:
    def __init__(self, root):
        self.root = root
        root.title("Real-Time Face Detection")
        root.protocol("WM_DELETE_WINDOW", self.on_close)

        self.cap = None
        self.running = False
        self.detector = None
        self.last_raw = None
        self.last_annotated = None
        self.last_faces = []
        self.fps = 0.0
        self.video_path = ""
        self.photo = None

        self.source_var = tk.StringVar(value="Webcam")
        self.detector_var = tk.StringVar(value=HaarDetector.name)
        self.blur_var = tk.BooleanVar(value=False)
        self.lm_var = tk.BooleanVar(value=True)
        self.status_var = tk.StringVar(value="Ready. Choose a source and press Start.")

        self.build_ui()

    def build_ui(self):
        top = ttk.Frame(self.root, padding=8)
        top.pack(fill="x")

        ttk.Label(top, text="Source:").pack(side="left")
        src = ttk.Combobox(top, textvariable=self.source_var, width=12,
                           values=["Webcam", "Video file"], state="readonly")
        src.pack(side="left", padx=(4, 12))

        ttk.Label(top, text="Detector:").pack(side="left")
        det = ttk.Combobox(top, textvariable=self.detector_var, width=16,
                           values=list(DETECTORS), state="readonly")
        det.pack(side="left", padx=(4, 12))
        det.bind("<<ComboboxSelected>>", lambda e: self.change_detector())

        self.start_btn = ttk.Button(top, text="Start", command=self.start)
        self.start_btn.pack(side="left", padx=2)
        self.stop_btn = ttk.Button(top, text="Stop", command=self.stop,
                                   state="disabled")
        self.stop_btn.pack(side="left", padx=2)
        ttk.Button(top, text="Snapshot", command=self.snapshot).pack(side="left", padx=2)
        ttk.Button(top, text="Save faces", command=self.save_faces).pack(side="left", padx=2)

        opts = ttk.Frame(self.root, padding=(8, 0))
        opts.pack(fill="x")
        ttk.Checkbutton(opts, text="Blur faces (anonymize)",
                        variable=self.blur_var).pack(side="left", padx=(0, 12))
        ttk.Checkbutton(opts, text="Show landmarks (YuNet)",
                        variable=self.lm_var).pack(side="left")

        self.video_label = tk.Label(self.root, bg="black",
                                    width=DISPLAY_MAX[0] // 8,
                                    height=DISPLAY_MAX[1] // 16)
        self.video_label.pack(padx=8, pady=8)

        ttk.Label(self.root, textvariable=self.status_var,
                  relief="sunken", anchor="w").pack(fill="x", side="bottom")

    # ---- detector / source handling --------------------------------------
    def change_detector(self):
        try:
            self.detector = DETECTORS[self.detector_var.get()]()
        except Exception as exc:
            messagebox.showerror("Detector error", str(exc))
            self.detector_var.set(HaarDetector.name)
            self.detector = HaarDetector()

    def open_source(self):
        if self.source_var.get() == "Video file":
            path = filedialog.askopenfilename(
                title="Choose a video",
                filetypes=[("Video", "*.mp4 *.avi *.mov *.mkv *.wmv"),
                           ("All files", "*.*")])
            if not path:
                return None
            self.video_path = path
            cap = cv2.VideoCapture(path)
        else:
            if sys.platform.startswith("win"):
                cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            else:
                cap = cv2.VideoCapture(0)
        if not cap or not cap.isOpened():
            messagebox.showerror(
                "Source error",
                "Could not open the video source.\n"
                "If using a webcam, close other apps that use the camera "
                "and check camera permissions.")
            return None
        return cap

    # ---- controls ---------------------------------------------------------
    def start(self):
        if self.running:
            return
        self.status_var.set("Loading detector...")
        self.root.update_idletasks()
        try:
            self.detector = DETECTORS[self.detector_var.get()]()
        except Exception as exc:
            messagebox.showerror("Detector error", str(exc))
            self.status_var.set("Detector failed to load.")
            return
        cap = self.open_source()
        if cap is None:
            self.status_var.set("Ready.")
            return
        self.cap = cap
        self.running = True
        self.start_btn.config(state="disabled")
        self.stop_btn.config(state="normal")
        self.loop()

    def stop(self):
        self.running = False
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.start_btn.config(state="normal")
        self.stop_btn.config(state="disabled")
        self.status_var.set("Stopped.")

    def loop(self):
        if not self.running:
            return
        t0 = time.time()
        ok, frame = self.cap.read()
        if not ok:
            self.stop()
            self.status_var.set("Video finished / no more frames.")
            return

        if self.source_var.get() == "Webcam":
            frame = cv2.flip(frame, 1)          # mirror view

        try:
            out, faces = process_frame(frame, self.detector,
                                       blur=self.blur_var.get(),
                                       show_landmarks=self.lm_var.get())
        except Exception as exc:
            self.stop()
            messagebox.showerror("Processing error", str(exc))
            return

        self.last_raw, self.last_annotated, self.last_faces = frame, out, faces
        dt = max(time.time() - t0, 1e-6)
        self.fps = 0.9 * self.fps + 0.1 * (1.0 / dt) if self.fps else 1.0 / dt
        cv2.putText(out, f"FPS: {self.fps:.1f}  Faces: {len(faces)}", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        self.show(out)
        self.status_var.set(
            f"{self.detector.name} | FPS: {self.fps:.1f} | Faces: {len(faces)}")
        self.root.after(1, self.loop)

    def show(self, bgr):
        h, w = bgr.shape[:2]
        s = min(DISPLAY_MAX[0] / w, DISPLAY_MAX[1] / h, 1.0)
        if s < 1.0:
            bgr = cv2.resize(bgr, (int(w * s), int(h * s)))
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        self.photo = ImageTk.PhotoImage(Image.fromarray(rgb))
        self.video_label.config(image=self.photo, width=self.photo.width(),
                                height=self.photo.height())

    # ---- saving -----------------------------------------------------------
    def snapshot(self):
        if self.last_annotated is None:
            messagebox.showinfo("Snapshot", "Nothing to save yet. Press Start first.")
            return
        os.makedirs(SNAP_DIR, exist_ok=True)
        path = os.path.join(SNAP_DIR, f"snapshot_{datetime.now():%Y%m%d_%H%M%S}.png")
        cv2.imwrite(path, self.last_annotated)
        self.status_var.set(f"Saved {path}")

    def save_faces(self):
        if self.last_raw is None or not self.last_faces:
            messagebox.showinfo("Save faces", "No faces in the current frame.")
            return
        os.makedirs(SNAP_DIR, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        for i, f in enumerate(self.last_faces, 1):
            x, y, w, h = f["box"]
            cv2.imwrite(os.path.join(SNAP_DIR, f"face_{stamp}_{i}.png"),
                        self.last_raw[y:y + h, x:x + w])
        self.status_var.set(f"Saved {len(self.last_faces)} face crop(s) to {SNAP_DIR}")

    def on_close(self):
        self.stop()
        self.root.destroy()


def main():
    if tk is None:
        print("GUI libraries missing:", _GUI_IMPORT_ERROR)
        print("Install with:  pip install pillow   (and tkinter for your OS)")
        sys.exit(1)
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
