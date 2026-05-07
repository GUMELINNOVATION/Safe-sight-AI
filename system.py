"""
================================================================================
  ENTERPRISE PPE COMPLIANCE SYSTEM  |  Powered by YOLOv8
  AI-driven real-time safety monitoring for industrial environments.
  Version: 6.5.1 (Fixed Vest Detection + Working HTML Records)
================================================================================

SETUP FOR VEST DETECTION:
-------------------------
Option A - Use Pre-trained YOLO Model (BEST):
  1. Download: https://huggingface.co/Tanishjain9/yolov8n-ppe-detection-6classes
  2. Save as "vest_best.pt" next to this script
  3. Set VEST_MODEL_PATH below

Option B - Color Fallback (CURRENT - improved):
  Uses strict HSV + skin-tone exclusion + narrow center zones
  Works for standard hi-vis vests but can have false positives

Option C - Train Your Own:
  yolo detect train data=your_data.yaml model=yolov8n.pt epochs=50
"""

import cv2
import csv
import time
import sys
import os
import numpy as np
import urllib.request
from datetime import datetime
from pathlib import Path
from ultralytics import YOLO

# ═══════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

# Anchor all relative paths to the script's own directory so the script works
# regardless of the working directory it is launched from.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

HELMET_MODEL_PATH = os.path.join(BASE_DIR, "best_new.pt")
PERSON_MODEL_PATH = "yolov8n.pt"
VEST_MODEL_PATH   = os.path.join(BASE_DIR, "vest_best.pt")    # <-- SET YOUR VEST MODEL HERE (optional)
REPORT_PATH       = os.path.join(BASE_DIR, "worker_safety_report.csv")
CONFIDENCE        = 0.30

# Vest detection thresholds
VEST_IOU_THRESHOLD = 0.25
VEST_CONFIDENCE  = 0.20

# Color fallback thresholds (v6.6 - more permissive for real-world use)
VEST_TORSO_RATIO = 0.05       # was 0.08 — lowered to catch partial torso coverage
VEST_FULL_RATIO  = 0.03       # was 0.05
VEST_MIN_RATIO   = 0.01       # was 0.02
VEST_SHOULDER_RATIO = 0.03    # was 0.05
VEST_MIN_AREA = 200           # was 500 — allow smaller detections

# STRICT HSV ranges for fluorescent hi-vis (high brightness + saturation)
VEST_HSV_RANGES = [
    ((15, 100, 150), (45, 255, 255)),   # Yellow-green fluorescent
    ((5, 120, 160), (20, 255, 255)),    # Orange-red fluorescent
    ((20, 80, 180), (40, 255, 255)),    # Neon yellow
    ((10, 90, 150), (40, 255, 255)),    # Catch-all fluorescent
]

# Skin tone exclusion (prevents arms/hands from triggering)
SKIN_HSV_RANGE = ((0, 20, 50), (20, 170, 255))

KNOWN_FACES_DIR = os.path.join(BASE_DIR, "known_faces")
os.makedirs(KNOWN_FACES_DIR, exist_ok=True)

# Model auto-download URLs
FACE_DETECTOR_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
FACE_RECOGNIZER_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx"
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)
FACE_DETECTOR_PATH = os.path.join(MODELS_DIR, "face_detection_yunet_2023mar.onnx")
FACE_RECOGNIZER_PATH = os.path.join(MODELS_DIR, "face_recognition_sface_2021dec.onnx")

VIOLATION_COOLDOWN = 5.0

# Face recognition thresholds
FACE_DETECTION_CONFIDENCE = 0.50
FACE_SIMILARITY_THRESHOLD = 0.20
FACE_MIN_SIZE = 80
FACE_ENROLL_SAMPLES = 5
FACE_ENROLL_CONFIDENCE = 0.75

# ═══════════════════════════════════════════════════════════════════════════════
#  MODEL DOWNLOAD HELPER
# ═══════════════════════════════════════════════════════════════════════════════

def _download_file(url: str, dest: str):
    if os.path.exists(dest):
        return
    print(f"[DOWNLOAD] Fetching {os.path.basename(dest)}...")
    try:
        urllib.request.urlretrieve(url, dest)
        print(f"[DOWNLOAD] Saved to {dest}")
    except Exception as e:
        print(f"[ERROR] Failed to download {os.path.basename(dest)}: {e}")
        sys.exit(1)

# ═══════════════════════════════════════════════════════════════════════════════
#  FACE ID MANAGER
# ═══════════════════════════════════════════════════════════════════════════════

class FaceIDManager:
    def __init__(self):
        self.known_faces: dict[int, np.ndarray] = {}
        self._load_all()

    def _id_path(self, face_id: int) -> str:
        return os.path.join(KNOWN_FACES_DIR, f"face_{face_id}.npy")

    def _counter_path(self) -> str:
        return os.path.join(KNOWN_FACES_DIR, "next_id.txt")

    def _read_next_id(self) -> int:
        p = self._counter_path()
        if os.path.exists(p):
            with open(p) as f:
                try:
                    return int(f.read().strip())
                except ValueError:
                    pass
        ids = [int(f[5:-4]) for f in os.listdir(KNOWN_FACES_DIR)
               if f.startswith("face_") and f.endswith(".npy")]
        return max(ids) + 1 if ids else 1

    def _write_next_id(self, val: int):
        with open(self._counter_path(), "w") as f:
            f.write(str(val))

    def _load_all(self):
        for fname in os.listdir(KNOWN_FACES_DIR):
            if fname.startswith("face_") and fname.endswith(".npy"):
                try:
                    face_id = int(fname[5:-4])
                    self.known_faces[face_id] = np.load(os.path.join(KNOWN_FACES_DIR, fname))
                except (ValueError, IOError):
                    pass
        if self.known_faces:
            print(f"[Face ID] Loaded {len(self.known_faces)} known face(s) from disk.")

    def _save_face(self, face_id: int, embedding: np.ndarray):
        np.save(self._id_path(face_id), embedding)

    @staticmethod
    def extract_embedding(face_img: np.ndarray, face_box: np.ndarray, recognizer):
        if face_img is None or face_img.size == 0:
            return None
        try:
            aligned = recognizer.alignCrop(face_img, face_box)
            embedding = recognizer.feature(aligned)
            if embedding is None or embedding.size == 0:
                return None
            embedding = embedding.flatten().astype(np.float32)
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
            return embedding
        except Exception as e:
            print(f"[Face Debug] extract_embedding error: {e}")
            return None

    def recognize(self, face_img: np.ndarray, face_box: np.ndarray, recognizer):
        embedding = self.extract_embedding(face_img, face_box, recognizer)
        if embedding is None:
            return None, 0.0
        if not self.known_faces:
            return None, 0.0
        best_score, best_id = -1.0, None
        scores = []
        for fid, known_emb in self.known_faces.items():
            score = float(np.dot(embedding, known_emb))
            scores.append((fid, score))
            if score > best_score:
                best_score, best_id = score, fid
        scores.sort(key=lambda x: x[1], reverse=True)
        top3 = scores[:3]
        print(f"[Face Debug] Top matches: {[(f'ID:{fid}', f'{s:.3f}') for fid, s in top3]}")
        if best_score >= FACE_SIMILARITY_THRESHOLD:
            print(f"[Face Debug] MATCH -> ID {best_id} (score={best_score:.3f})")
            return best_id, best_score
        else:
            print(f"[Face Debug] NO MATCH (best={best_score:.3f} < threshold={FACE_SIMILARITY_THRESHOLD})")
            return None, best_score

    def enroll(self, face_img: np.ndarray, face_box: np.ndarray, recognizer):
        embedding = self.extract_embedding(face_img, face_box, recognizer)
        if embedding is None:
            return -1
        new_id = self._read_next_id()
        self.known_faces[new_id] = embedding
        self._save_face(new_id, embedding)
        self._write_next_id(new_id + 1)
        print(f"[Face ID] Enrolled new face -> ID {new_id}")
        return new_id

    def get_or_enroll(self, face_img: np.ndarray, face_box: np.ndarray, recognizer):
        fid, score = self.recognize(face_img, face_box, recognizer)
        if fid is not None:
            return fid
        return self.enroll(face_img, face_box, recognizer)


# ═══════════════════════════════════════════════════════════════════════════════
#  WORKER REGISTRY
# ═══════════════════════════════════════════════════════════════════════════════

face_manager = FaceIDManager()
face_to_track: dict[int, int] = {}
face_sample_buffer: dict[int, list] = {}
worker_registry: dict[int, dict] = {}


def _new_worker_entry() -> dict:
    return {
        "face_id": None, "name": "Unknown Worker",
        "helmet_status": "Checking...", "helmet_placement": "none",
        "vest_status": "Checking...", "vest_placement": "none",
        "vest_safe": False, "is_safe": False,
        "last_violation_time": 0.0, "identified": False, "face_attempts": 0,
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  HUD / UI CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

HUD_HEIGHT = 120
HUD_ALPHA = 0.82
FONT_MAIN = cv2.FONT_HERSHEY_DUPLEX
FONT_SMALL = cv2.FONT_HERSHEY_SIMPLEX

COLOR_SAFE = (50, 205, 50)
COLOR_CRITICAL = (0, 0, 220)
COLOR_WARNING = (30, 165, 255)
COLOR_STANDBY = (200, 200, 200)
COLOR_ACCENT = (0, 180, 255)
COLOR_WHITE = (255, 255, 255)
COLOR_DARK = (20, 20, 20)
COLOR_PERSON = (255, 100, 50)
COLOR_HELMET = (50, 220, 50)
COLOR_VEST = (0, 220, 180)

WINDOW_NAME = "Enterprise PPE Compliance System v6.5.1  |  Press Q to Exit"

# ═══════════════════════════════════════════════════════════════════════════════
#  LOAD YOLO MODELS
# ═══════════════════════════════════════════════════════════════════════════════

if not os.path.exists(HELMET_MODEL_PATH):
    print(f"\n[FATAL ERROR] Helmet model '{HELMET_MODEL_PATH}' not found.")
    sys.exit(1)

try:
    helmet_model = YOLO(HELMET_MODEL_PATH)
    HELMET_CLASSES = helmet_model.names
    print(f"[OK] Helmet model  | classes: {HELMET_CLASSES}")

    person_model = YOLO(PERSON_MODEL_PATH)
    PERSON_CLASS_ID = 0
    print(f"[OK] Person model  | yolov8n (COCO)")

    vest_model = None
    VEST_CLASSES = {}
    if os.path.exists(VEST_MODEL_PATH):
        print(f"[LOAD] Loading vest detection model ({VEST_MODEL_PATH})...")
        vest_model = YOLO(VEST_MODEL_PATH)
        VEST_CLASSES = vest_model.names
        print(f"[OK] Vest model    | classes: {VEST_CLASSES}")
        print("[INFO] Using DEDICATED YOLO vest detection (shape-based)")
    else:
        print(f"[WARN] Vest model '{VEST_MODEL_PATH}' not found.")
        print("[INFO] Using COLOR-BASED fallback for vest detection.")

except Exception as exc:
    print(f"[FATAL ERROR] {exc}")
    sys.exit(1)

# ═══════════════════════════════════════════════════════════════════════════════
#  LOAD FACE RECOGNITION MODELS
# ═══════════════════════════════════════════════════════════════════════════════

_download_file(FACE_DETECTOR_URL, FACE_DETECTOR_PATH)
_download_file(FACE_RECOGNIZER_URL, FACE_RECOGNIZER_PATH)

try:
    face_detector = cv2.FaceDetectorYN.create(
        model=FACE_DETECTOR_PATH, config="", input_size=(320, 320),
        score_threshold=FACE_DETECTION_CONFIDENCE, nms_threshold=0.3, top_k=5000)
    print("[OK] Face detector  | YuNet (OpenCV DNN)")
    face_recognizer = cv2.FaceRecognizerSF.create(
        model=FACE_RECOGNIZER_PATH, config="", backend_id=0, target_id=0)
    print("[OK] Face recognizer | SFace 128-D embeddings")
except Exception as exc:
    print(f"[FATAL ERROR] Face model loading failed: {exc}")
    sys.exit(1)

# ═══════════════════════════════════════════════════════════════════════════════
#  CAMERA SOURCE SELECTION
# ═══════════════════════════════════════════════════════════════════════════════

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("[FATAL ERROR] No camera found. Exiting.")
    sys.exit(1)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

print(f"[OK] Camera ready | {int(cap.get(3))}x{int(cap.get(4))}")

# Warm-up flush
for _ in range(10):
    try:
        cap.read()
    except Exception:
        break



# ═══════════════════════════════════════════════════════════════════════════════
#  CSV REPORT
# ═══════════════════════════════════════════════════════════════════════════════

if not os.path.exists(REPORT_PATH):
    with open(REPORT_PATH, mode="w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(
            ["Timestamp", "Face_ID", "Track_ID", "Name", "Event", "Helmet_Status", "Vest_Status"])

csv_file = open(REPORT_PATH, mode="a", newline="", encoding="utf-8")
csv_writer = csv.writer(csv_file)
print(f"[OK] Report System  | {REPORT_PATH}")
print("\n[SYSTEM READY]  Press Q or ESC to exit.\n")

# ═══════════════════════════════════════════════════════════════════════════════
#  VEST DETECTION (v6.5.1 - Improved Color Fallback)
# ═══════════════════════════════════════════════════════════════════════════════

VEST_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))


def create_vest_mask(hsv_img: np.ndarray) -> np.ndarray:
    """Create vest mask with skin-tone exclusion."""
    masks = []
    for (lower, upper) in VEST_HSV_RANGES:
        masks.append(cv2.inRange(hsv_img, lower, upper))
    combined = masks[0]
    for mask in masks[1:]:
        combined = cv2.bitwise_or(combined, mask)

    # Exclude skin tones
    skin_mask = cv2.inRange(hsv_img, SKIN_HSV_RANGE[0], SKIN_HSV_RANGE[1])
    combined = cv2.bitwise_and(combined, cv2.bitwise_not(skin_mask))

    # Morphological cleanup
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, VEST_KERNEL)
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, VEST_KERNEL)
    return combined


def detect_vest_yolo(frame, x1, y1, x2, y2, vest_results):
    """Shape-based vest detection using YOLO model."""
    ph, pw = y2 - y1, x2 - x1
    if ph < 50 or pw < 30:
        return "Checking...", False, "none", None

    best_vest_box = None
    best_iou = 0.0
    best_conf = 0.0
    vest_found = False

    for box in vest_results[0].boxes:
        cls_id = int(box.cls[0])
        cls_name = VEST_CLASSES.get(cls_id, "").lower()
        conf = float(box.conf[0])

        if conf < VEST_CONFIDENCE:
            continue
        if "vest" not in cls_name and "jacket" not in cls_name:
            continue

        vx1, vy1, vx2, vy2 = map(int, box.xyxy[0])

        ix1 = max(x1, vx1); iy1 = max(y1, vy1)
        ix2 = min(x2, vx2); iy2 = min(y2, vy2)

        if ix2 <= ix1 or iy2 <= iy1:
            continue

        inter_area = (ix2 - ix1) * (iy2 - iy1)
        vest_area = max(1, (vx2 - vx1) * (vy2 - vy1))
        iou = inter_area / vest_area

        if iou > best_iou:
            best_iou = iou
            best_conf = conf
            best_vest_box = (vx1, vy1, vx2, vy2)
            vest_found = True

    print(f"[Vest Debug] YOLO: found={vest_found}, iou={best_iou:.3f}, conf={best_conf:.3f}")

    if not vest_found:
        return "No Vest detected", False, "none", None

    if best_iou >= VEST_IOU_THRESHOLD:
        # Additional position check: vest should be in torso area (25%-75% of person height)
        vcy = (best_vest_box[1] + best_vest_box[3]) / 2  # vest center Y
        rel_y = (vcy - y1) / ph  # relative position in person bbox

        # Vest should be in torso area (roughly 25%-75% of person height)
        if 0.25 <= rel_y <= 0.75:
            # Also check horizontal centering (vest should overlap significantly with person center)
            vest_center_x = (best_vest_box[0] + best_vest_box[2]) / 2
            person_center_x = (x1 + x2) / 2
            center_diff = abs(vest_center_x - person_center_x) / pw

            if center_diff <= 0.4:  # vest center within 40% of person width from center
                return "Vest worn correctly", True, "worn", best_vest_box

        # Vest detected but not properly positioned
        return "Vest detected, not worn properly", False, "not_worn", best_vest_box
    elif best_iou >= 0.05:
        return "Vest detected, not worn properly", False, "not_worn", best_vest_box
    else:
        return "Vest detected", False, "not_worn", best_vest_box


def detect_vest_color(frame, x1, y1, x2, y2):
    """
    v6.5.1: Improved color-based fallback with:
    - Skin-tone exclusion (prevents arms/hands triggering)
    - Narrower center zones (excludes background objects at edges)
    - Strict brightness/saturation requirements
    """
    ph, pw = y2 - y1, x2 - x1
    if ph < 50 or pw < 30:
        return "Checking...", False, "none", None

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = create_vest_mask(hsv)

    # NARROWER zones - center 40% of body width (was 60%)
    # This avoids background objects at the sides of the person bbox
    sz1 = y1 + int(ph * 0.15)
    sz2 = y1 + int(ph * 0.45)
    sx1 = x1 + int(pw * 0.30)
    sx2 = x2 - int(pw * 0.30)

    tz1 = y1 + int(ph * 0.30)
    tz2 = y1 + int(ph * 0.70)
    tx1 = x1 + int(pw * 0.30)
    tx2 = x2 - int(pw * 0.30)

    # Ensure valid zones
    if sx2 <= sx1 or tx2 <= tx1:
        return "No Vest detected", False, "none", None

    shoulder_crop = mask[sz1:sz2, sx1:sx2]
    torso_crop = mask[tz1:tz2, tx1:tx2]
    full_crop = mask[y1:y2, x1:x2]

    shoulder_ratio = (cv2.countNonZero(shoulder_crop) / max(1, shoulder_crop.size)
                      if shoulder_crop.size > 0 else 0.0)
    torso_ratio = (cv2.countNonZero(torso_crop) / max(1, torso_crop.size)
                   if torso_crop.size > 0 else 0.0)
    full_ratio = (cv2.countNonZero(full_crop) / max(1, full_crop.size)
                  if full_crop.size > 0 else 0.0)

    # Find vest bounding box and largest contour area
    vest_bbox = None
    max_contour_area = 0
    contours, _ = cv2.findContours(mask[y1:y2, x1:x2], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        all_pts = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 200:
                all_pts.extend(cnt.reshape(-1, 2).tolist())
                if area > max_contour_area:
                    max_contour_area = area
        if all_pts:
            all_pts = np.array(all_pts)
            vx1 = x1 + int(np.min(all_pts[:, 0]))
            vy1 = y1 + int(np.min(all_pts[:, 1]))
            vx2 = x1 + int(np.max(all_pts[:, 0]))
            vy2 = y1 + int(np.max(all_pts[:, 1]))
            vest_bbox = (vx1, vy1, vx2, vy2)

    print(f"[Vest Debug] Color: shoulder={shoulder_ratio:.3f}, torso={torso_ratio:.3f}, "
          f"full={full_ratio:.3f}, area={max_contour_area}")

    if max_contour_area < VEST_MIN_AREA:
        return "No Vest detected", False, "none", None

    # Primary check: torso coverage alone is sufficient.
    # Shoulder zone (15%–45%) overlaps with the helmet area, so shoulder_ratio
    # is almost always ~0 when a helmet is worn — do NOT require it.
    if torso_ratio >= VEST_TORSO_RATIO:
        if vest_bbox is not None:
            vcy = (vest_bbox[1] + vest_bbox[3]) / 2
            rel_y = (vcy - y1) / ph

            # Vest center must be in the lower 20%–95% of person bbox
            if 0.20 <= rel_y <= 0.95:
                vest_center_x = (vest_bbox[0] + vest_bbox[2]) / 2
                person_center_x = (x1 + x2) / 2
                center_diff = abs(vest_center_x - person_center_x) / pw

                if center_diff <= 0.60:
                    return "Vest worn correctly", True, "worn", vest_bbox

        # Strong torso coverage even without a tight bbox → still worn
        if torso_ratio >= VEST_TORSO_RATIO * 2.0:
            return "Vest worn correctly", True, "worn", vest_bbox

        return "Vest detected, not worn properly", False, "not_worn", vest_bbox
    elif full_ratio >= VEST_FULL_RATIO:
        return "Vest detected, not worn properly", False, "not_worn", vest_bbox
    elif full_ratio >= VEST_MIN_RATIO:
        return "Vest detected", False, "not_worn", vest_bbox
    else:
        return "No Vest detected", False, "none", None


# ═══════════════════════════════════════════════════════════════════════════════
#  HELMET PLACEMENT
# ═══════════════════════════════════════════════════════════════════════════════

def helmet_placement(hx1, hy1, hx2, hy2, px1, py1, px2, py2, head_zone_frac=0.30):
    ix1, iy1 = max(hx1, px1), max(hy1, py1)
    ix2, iy2 = min(hx2, px2), min(hy2, py2)
    if ix2 <= ix1 or iy2 <= iy1:
        return "none"
    inter = (ix2 - ix1) * (iy2 - iy1)
    h_area = max(1, (hx2 - hx1) * (hy2 - hy1))
    if inter / h_area < 0.05:
        return "none"
    hcy = (hy1 + hy2) / 2
    ph = max(1, py2 - py1)
    rel_y = (hcy - py1) / ph
    return "worn" if rel_y <= head_zone_frac else "not_worn"


# ═══════════════════════════════════════════════════════════════════════════════
#  DRAWING HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def draw_stat_card(frame, x, y, label, value, color):
    cv2.putText(frame, label, (x, y), FONT_SMALL, 0.48, COLOR_STANDBY, 1, cv2.LINE_AA)
    cv2.putText(frame, str(value), (x, y + 22), FONT_MAIN, 0.85, color, 2, cv2.LINE_AA)


def draw_label(frame, text, x1, y1, bg_color, text_color=None):
    if text_color is None:
        text_color = COLOR_DARK
    (lw, lh), _ = cv2.getTextSize(text, FONT_SMALL, 0.50, 1)
    cv2.rectangle(frame, (x1, y1 - lh - 8), (x1 + lw + 6, y1), bg_color, -1)
    cv2.putText(frame, text, (x1 + 3, y1 - 4), FONT_SMALL, 0.50, text_color, 1, cv2.LINE_AA)


def _status_color(status):
    sl = status.lower()
    if "correctly" in sl:
        return COLOR_SAFE
    if "not worn" in sl or "not properly" in sl or "not covering" in sl:
        return COLOR_WARNING
    if "detected" in sl:
        return COLOR_ACCENT
    if "no " in sl or "checking" in sl:
        return COLOR_CRITICAL
    return COLOR_STANDBY


def _helmet_icon(placement):
    if placement == "worn":
        return "OK"
    if placement == "not_worn":
        return "!!"
    return "--"


def _vest_icon(placement):
    if placement == "worn":
        return "OK"
    if placement == "not_worn":
        return "!!"
    return "--"


def draw_status_panel(frame, x1, y2, x2, name,
                       helmet_status, helmet_placement_str,
                       vest_status, vest_placement,
                       is_safe, frame_h):
    panel_x = x1
    base_y = y2 + 4
    row_h = 18
    font_sz = 0.40
    th_small = 1
    max_y = frame_h - 5

    h_icon = _helmet_icon(helmet_placement_str)
    v_icon = _vest_icon(vest_placement)
    h_color = _status_color(helmet_status)
    v_color = _status_color(vest_status)

    if base_y < max_y:
        cv2.putText(frame, name, (panel_x, base_y + row_h * 0),
                    FONT_SMALL, font_sz + 0.05, COLOR_WHITE, 1, cv2.LINE_AA)

    row1_y = base_y + row_h * 1
    if row1_y < max_y:
        h_text = f"H: {h_icon} {helmet_status}"
        cv2.putText(frame, h_text, (panel_x, row1_y),
                    FONT_SMALL, font_sz, h_color, th_small, cv2.LINE_AA)

    row2_y = base_y + row_h * 2
    if row2_y < max_y:
        v_text = f"V: {v_icon} {vest_status}"
        cv2.putText(frame, v_text, (panel_x, row2_y),
                    FONT_SMALL, font_sz, v_color, th_small, cv2.LINE_AA)

    row3_y = base_y + row_h * 3
    if row3_y < max_y:
        if is_safe:
            safety_text, safety_color = "Safety: SAFE", COLOR_SAFE
        else:
            safety_text, safety_color = "Safety: VIOLATION", COLOR_CRITICAL
        cv2.putText(frame, safety_text, (panel_x, row3_y),
                    FONT_SMALL, font_sz + 0.02, safety_color, 1, cv2.LINE_AA)


def draw_helmet_box(frame, hx1, hy1, hx2, hy2, placement):
    if placement == "worn":
        color, label_txt = COLOR_SAFE, "Helmet: Worn"
    elif placement == "not_worn":
        color, label_txt = COLOR_WARNING, "Helmet: Not Worn"
    else:
        return
    cv2.rectangle(frame, (hx1, hy1), (hx2, hy2), color, 2, cv2.LINE_AA)
    draw_label(frame, label_txt, hx1, hy1, color, COLOR_DARK)


def draw_vest_box(frame, vx1, vy1, vx2, vy2, placement):
    if placement == "worn":
        color, label_txt = COLOR_SAFE, "Vest: Worn"
    elif placement == "not_worn":
        color, label_txt = COLOR_WARNING, "Vest: Not Worn"
    else:
        return
    cv2.rectangle(frame, (vx1, vy1), (vx2, vy2), color, 2, cv2.LINE_AA)
    draw_label(frame, label_txt, vx1, vy1, color, COLOR_DARK)


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN LOOP SETUP
# ═══════════════════════════════════════════════════════════════════════════════

violation_flash = False
session_violations = 0
fps_timer = time.time()
fps_value = 0.0
frame_count = 0

cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN LOOP
# ═══════════════════════════════════════════════════════════════════════════════

try:
    while True:
        try:
            ret, frame = cap.read()
        except Exception:
            # Bad frame — skip it silently
            time.sleep(0.05)
            continue

        if not ret or frame is None or frame.size == 0:
            time.sleep(0.05)
            continue

        h, w = frame.shape[:2]
        frame_count += 1

        if frame_count % 15 == 0:
            elapsed = time.time() - fps_timer
            fps_value = 15 / elapsed if elapsed > 0 else 0
            fps_timer = time.time()

        person_results = person_model.track(
            frame, conf=CONFIDENCE, verbose=False, classes=[PERSON_CLASS_ID], persist=True)
        helmet_results = helmet_model(frame, conf=CONFIDENCE, verbose=False)

        vest_results = None
        if vest_model is not None:
            vest_results = vest_model(frame, conf=VEST_CONFIDENCE, verbose=False)

        hardhat_boxes = []
        for box in helmet_results[0].boxes:
            cls_name = HELMET_CLASSES.get(int(box.cls[0]), "")
            if cls_name == "Hardhat":
                hx1, hy1, hx2, hy2 = map(int, box.xyxy[0])
                hardhat_boxes.append([hx1, hy1, hx2, hy2, None])

        person_count = 0
        person_data = []

        if person_results[0].boxes.id is not None:
            boxes = person_results[0].boxes.xyxy.cpu().numpy().astype(int)
            ids = person_results[0].boxes.id.cpu().numpy().astype(int)

            for box, track_id in zip(boxes, ids):
                x1, y1, x2, y2 = box
                person_count += 1

                if track_id not in worker_registry:
                    worker_registry[track_id] = _new_worker_entry()
                    face_sample_buffer[track_id] = []

                entry = worker_registry[track_id]

                if not entry["identified"]:
                    head_h = max(int((y2 - y1) * 0.40), FACE_MIN_SIZE)
                    head_crop = frame[y1 : min(y1 + head_h, h), x1:x2]

                    if head_crop.size > 0 and head_crop.shape[0] >= FACE_MIN_SIZE and head_crop.shape[1] >= FACE_MIN_SIZE:
                        face_detector.setInputSize((head_crop.shape[1], head_crop.shape[0]))
                        _, faces = face_detector.detect(head_crop)

                        entry["face_attempts"] += 1

                        if faces is not None and len(faces) > 0:
                            best_face = max(faces, key=lambda f: f[14])
                            conf = float(best_face[14])
                            fx, fy, fw_f, fh_f = int(best_face[0]), int(best_face[1]), int(best_face[2]), int(best_face[3])
                            fx = max(0, fx); fy = max(0, fy)
                            fw_f = min(fw_f, head_crop.shape[1] - fx)
                            fh_f = min(fh_f, head_crop.shape[0] - fy)

                            if fw_f >= FACE_MIN_SIZE and fh_f >= FACE_MIN_SIZE and conf >= FACE_ENROLL_CONFIDENCE:
                                face_box = best_face.copy()
                                face_box[0] = fx; face_box[1] = fy
                                face_box[2] = fw_f; face_box[3] = fh_f

                                print(f"[Face Debug] Track {track_id}: face detected (conf={conf:.2f}, size={fw_f}x{fh_f})")

                                fid, score = face_manager.recognize(head_crop, face_box, face_recognizer)

                                if fid is not None:
                                    if fid in face_to_track:
                                        old_tid = face_to_track[fid]
                                        if old_tid != track_id and old_tid in worker_registry:
                                            old = worker_registry[old_tid]
                                            entry["face_id"] = old["face_id"]
                                            entry["name"] = old["name"]
                                            entry["last_violation_time"] = old["last_violation_time"]
                                            del worker_registry[old_tid]
                                            print(f"[Face ID] Re-entry: face {fid} linked track {old_tid} -> {track_id}")

                                    entry["face_id"] = fid
                                    entry["name"] = f"Worker_{fid:03d}"
                                    entry["identified"] = True
                                    face_to_track[fid] = track_id
                                    face_sample_buffer[track_id] = []
                                    print(f"[Face ID] Recognized existing worker -> {entry['name']} (score={score:.3f})")
                                else:
                                    face_sample_buffer[track_id].append((head_crop, face_box, conf))
                                    print(f"[Face Debug] Track {track_id}: collected sample {len(face_sample_buffer[track_id])}/{FACE_ENROLL_SAMPLES}")

                                    if len(face_sample_buffer[track_id]) >= FACE_ENROLL_SAMPLES:
                                        best_sample = max(face_sample_buffer[track_id], key=lambda s: s[2])
                                        best_face_img, best_face_box, best_conf = best_sample
                                        fid2, _ = face_manager.recognize(best_face_img, best_face_box, face_recognizer)
                                        if fid2 is not None:
                                            entry["face_id"] = fid2
                                            entry["name"] = f"Worker_{fid2:03d}"
                                            entry["identified"] = True
                                            face_to_track[fid2] = track_id
                                            face_sample_buffer[track_id] = []
                                            print(f"[Face ID] Late match found -> {entry['name']}")
                                        else:
                                            new_fid = face_manager.enroll(best_face_img, best_face_box, face_recognizer)
                                            if new_fid > 0:
                                                entry["face_id"] = new_fid
                                                entry["name"] = f"Worker_{new_fid:03d}"
                                                entry["identified"] = True
                                                face_to_track[new_fid] = track_id
                                                face_sample_buffer[track_id] = []
                                                print(f"[Face ID] New worker enrolled -> {entry['name']}")

                person_data.append((track_id, x1, y1, x2, y2))

        active_violations = 0
        helmet_count = 0
        vest_count = 0

        for track_id, x1, y1, x2, y2 in person_data:
            if track_id not in worker_registry:
                continue

            entry = worker_registry[track_id]

            best_placement = "none"
            best_helmet_box = None

            for hbox in hardhat_boxes:
                hx1, hy1, hx2, hy2 = hbox[:4]
                result = helmet_placement(hx1, hy1, hx2, hy2, x1, y1, x2, y2)
                if result == "worn":
                    best_placement = "worn"
                    best_helmet_box = (hx1, hy1, hx2, hy2)
                    hbox[4] = track_id
                    break
                elif result == "not_worn" and best_placement == "none":
                    best_placement = "not_worn"
                    best_helmet_box = (hx1, hy1, hx2, hy2)
                    hbox[4] = track_id

            entry["helmet_placement"] = best_placement

            if best_placement == "worn":
                entry["helmet_status"] = "Helmet worn correctly"
                helmet_count += 1
            elif best_placement == "not_worn":
                entry["helmet_status"] = "Helmet detected, not worn"
            else:
                entry["helmet_status"] = "No Helmet detected"

            # VEST DETECTION
            if vest_model is not None and vest_results is not None:
                vest_status, vest_safe, vest_place, vest_bbox = detect_vest_yolo(
                    frame, x1, y1, x2, y2, vest_results)
            else:
                vest_status, vest_safe, vest_place, vest_bbox = detect_vest_color(
                    frame, x1, y1, x2, y2)

            entry["vest_status"] = vest_status
            entry["vest_safe"] = vest_safe
            entry["vest_placement"] = vest_place
            if vest_safe:
                vest_count += 1

            helmet_ok = (best_placement == "worn")
            is_safe = helmet_ok and vest_safe
            entry["is_safe"] = is_safe

            if not is_safe:
                active_violations += 1

            if not is_safe:
                now = time.time()
                if now - entry["last_violation_time"] >= VIOLATION_COOLDOWN:
                    entry["last_violation_time"] = now
                    csv_writer.writerow([
                        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        entry["face_id"] if entry["face_id"] is not None else "Unknown",
                        track_id, entry["name"], "VIOLATION",
                        entry["helmet_status"], entry["vest_status"],
                    ])
                    csv_file.flush()

            box_color = COLOR_SAFE if is_safe else COLOR_CRITICAL
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2, cv2.LINE_AA)
            draw_label(frame, entry["name"], x1, y1, box_color, COLOR_WHITE)

            if best_helmet_box is not None:
                draw_helmet_box(frame, *best_helmet_box, best_placement)

            if vest_bbox is not None and vest_place != "none":
                draw_vest_box(frame, *vest_bbox, vest_place)

            # ── Always-visible violation badges inside the person box ──────────
            badge_y = y1 + 28
            badge_x = x1 + 6
            if best_placement != "worn":
                badge_label = "NO HELMET" if best_placement == "none" else "HELMET NOT WORN"
                (bw, bh), _ = cv2.getTextSize(badge_label, FONT_SMALL, 0.46, 1)
                cv2.rectangle(frame, (badge_x - 2, badge_y - bh - 4),
                              (badge_x + bw + 4, badge_y + 2), COLOR_CRITICAL, -1)
                cv2.putText(frame, badge_label, (badge_x, badge_y),
                            FONT_SMALL, 0.46, COLOR_WHITE, 1, cv2.LINE_AA)
                badge_y += bh + 10

            if not vest_safe:
                v_label = "NO VEST" if vest_place == "none" else "VEST NOT WORN"
                (bw, bh), _ = cv2.getTextSize(v_label, FONT_SMALL, 0.46, 1)
                cv2.rectangle(frame, (badge_x - 2, badge_y - bh - 4),
                              (badge_x + bw + 4, badge_y + 2), (0, 100, 200), -1)
                cv2.putText(frame, v_label, (badge_x, badge_y),
                            FONT_SMALL, 0.46, COLOR_WHITE, 1, cv2.LINE_AA)

            draw_status_panel(
                frame, x1, y2, x2, entry["name"],
                entry["helmet_status"], best_placement,
                vest_status, vest_place,
                is_safe, h,
            )

        if person_count == 0:
            risk_level, status_text, hud_color = "STANDBY", "MONITORING ACTIVE", COLOR_STANDBY
        elif active_violations == 0:
            risk_level, status_text, hud_color = "SAFE", "STATUS: SAFE  |  FULL PPE COMPLIANCE", COLOR_SAFE
        else:
            risk_level, status_text, hud_color = "CRITICAL", f"ALERT: {active_violations} WORKER(S) UNSAFE", COLOR_CRITICAL
            session_violations += 1
            violation_flash = not violation_flash
            if violation_flash:
                cv2.rectangle(frame, (0, 0), (w - 1, h - 1), COLOR_CRITICAL, 12)

        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, HUD_HEIGHT), COLOR_DARK, -1)
        frame = cv2.addWeighted(overlay, HUD_ALPHA, frame, 1 - HUD_ALPHA, 0)
        cv2.rectangle(frame, (0, 0), (6, HUD_HEIGHT), hud_color, -1)

        draw_stat_card(frame, w - 490, 28, "WORKERS", person_count, COLOR_PERSON)
        draw_stat_card(frame, w - 375, 28, "HELMETS", helmet_count, COLOR_HELMET)
        draw_stat_card(frame, w - 260, 28, "VESTS", vest_count, COLOR_VEST)
        draw_stat_card(frame, w - 145, 28, "UNSAFE", active_violations,
                       COLOR_CRITICAL if active_violations > 0 else COLOR_SAFE)

        cv2.putText(frame, "PPE COMPLIANCE SYSTEM v6.5.1",
                    (16, 30), FONT_MAIN, 0.65, COLOR_ACCENT, 1, cv2.LINE_AA)
        cv2.putText(frame, status_text,
                    (16, 88), FONT_MAIN, 0.62, hud_color, 2, cv2.LINE_AA)
        cv2.putText(frame, f"FPS: {fps_value:.1f}",
                    (w - 90, h - 12), FONT_SMALL, 0.42, COLOR_ACCENT, 1, cv2.LINE_AA)

        cv2.imshow(WINDOW_NAME, frame)
        if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q"), 27):
            break

except KeyboardInterrupt:
    print("\n[INFO] Interrupted.")
except Exception as exc:
    import traceback
    traceback.print_exc()
    print(f"\n[ERROR] {exc}")
finally:
    cap.release()
    csv_file.close()
    cv2.destroyAllWindows()
    print(f"\n[SESSION] Violations: {session_violations}  |  Report: {REPORT_PATH}\n")