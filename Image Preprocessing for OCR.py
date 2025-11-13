# ==============================================================
# ANPR (YOLOv8 + EasyOCR) – Raspberry Pi Optimized Version
# ==============================================================

import cv2
import json
import re
import warnings
from ultralytics import YOLO
import easyocr
from difflib import SequenceMatcher
import logging

# Suppress all unnecessary warnings and YOLO logs
warnings.filterwarnings("ignore")
logging.getLogger("ultralytics").setLevel(logging.ERROR)

# --------------------------------------------------------------
# Load Configurations
# --------------------------------------------------------------
with open("config.json", "r") as f:
    config = json.load(f)

with open("registered_plates.json", "r") as f:
    registered_data = json.load(f)

registered_plates = [plate.upper().replace(" ", "") for plate in registered_data["registered"]]

# --------------------------------------------------------------
# Initialize YOLO + EasyOCR
# --------------------------------------------------------------
model = YOLO("license_plate_detector.pt")  # Prefer lightweight model (yolov8n)
reader = easyocr.Reader(['en'], gpu=False)  # Force CPU mode on Raspberry Pi

ip_stream = config.get("ip_stream")
frame_interval = config.get("frame_interval", 100)
confidence_threshold = config.get("confidence_threshold", 0.5)
fuzzy_threshold = config.get("fuzzy_match_threshold", 0.7)

# --------------------------------------------------------------
# Helper: Clean OCR text
# --------------------------------------------------------------
def clean_plate_text(text):
    text = re.sub(r'[HU]?IND(?:IA)?', '', text, flags=re.IGNORECASE)
    text = text.upper()
    text = re.sub(r'[^A-Z0-9]', '', text)
    text = text.replace("I6", "G").replace("16", "G").replace("1G", "G")
    text = text.replace("O", "0")
    text = text.replace("B", "8") if re.match(r"[A-Z]{2}\d", text) else text.replace("8", "B")
    return text.strip()

# --------------------------------------------------------------
# Helper: Fuzzy match with tolerance
# --------------------------------------------------------------
def is_plate_registered(cleaned, registered_plates, threshold):
    for plate in registered_plates:
        ratio = SequenceMatcher(None, cleaned, plate).ratio()
        if ratio >= threshold:
            return True, plate
    return False, None

# --------------------------------------------------------------
# Preprocessing Function for OCR (improves accuracy on Pi)
# --------------------------------------------------------------
def preprocess_plate(img_bgr):
    # Convert to RGB (EasyOCR expects RGB)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Resize to ~400px on the long side for better OCR
    h, w = img_rgb.shape[:2]
    scale = max(1.0, 400.0 / max(w, 1))
    if scale != 1.0:
        new_w = int(w * scale)
        new_h = int(h * scale)
        img_rgb = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Convert to grayscale
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

    # Enhance contrast (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # Denoise and threshold
    gray = cv2.bilateralFilter(gray, 9, 75, 75)
    th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, 15, 8)

    # Return as RGB for EasyOCR input
    img_rgb_post = cv2.cvtColor(th, cv2.COLOR_GRAY2RGB)
    return img_rgb_post

# --------------------------------------------------------------
# Process Stream (Headless)
# --------------------------------------------------------------
cap = cv2.VideoCapture(ip_stream)
frame_count = 0

if not cap.isOpened():
    print(f"Unable to open stream: {ip_stream}")
    exit(1)

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % frame_interval != 0:
            continue

        # Run YOLO detection
        results = model(frame, imgsz=416, verbose=False)
        boxes = results[0].boxes

        for box in boxes:
            if float(box.conf) < confidence_threshold:
                continue

            # Extract bounding box safely
            xyxy = box.xyxy[0].cpu().numpy() if hasattr(box.xyxy[0], "cpu") else box.xyxy[0]
            x1, y1, x2, y2 = map(int, xyxy)

            # Pad the crop slightly to ensure full plate capture
            pad = 6
            h, w = frame.shape[:2]
            x1 = max(0, x1 - pad)
            y1 = max(0, y1 - pad)
            x2 = min(w - 1, x2 + pad)
            y2 = min(h - 1, y2 + pad)

            plate_crop = frame[y1:y2, x1:x2]
            if plate_crop.size == 0:
                continue

            # --- Apply preprocessing before OCR ---
            processed_plate = preprocess_plate(plate_crop)
            ocr_result = reader.readtext(processed_plate, detail=0)

            if ocr_result:
                raw = " ".join(ocr_result)
                cleaned = clean_plate_text(raw)
                matched, reg_plate = is_plate_registered(cleaned, registered_plates, threshold=fuzzy_threshold)

                if matched:
                    print(f"✅ Registered plate '{reg_plate}' detected → Open boom barrier")
                else:
                    print(f"⛔ Unregistered plate '{cleaned}' detected → Access denied")

except KeyboardInterrupt:
    pass
finally:
    cap.release()
