# ==============================================================
# ANPR (YOLOv8 + EasyOCR) – Clean Console Output
# ==============================================================

import cv2
import json
import re
import warnings
from ultralytics import YOLO
import easyocr
from difflib import SequenceMatcher

# Suppress all unnecessary warnings and YOLO logs
warnings.filterwarnings("ignore")
import logging
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
model = YOLO("license_plate_detector.pt")
reader = easyocr.Reader(['en'])

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

        results = model.predict(frame, verbose=False)  # disable YOLO logs
        boxes = results[0].boxes

        for box in boxes:
            if box.conf < confidence_threshold:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            plate_crop = frame[y1:y2, x1:x2]
            ocr_result = reader.readtext(plate_crop, detail=0)

            if ocr_result:
                raw = " ".join(ocr_result)
                cleaned = clean_plate_text(raw)
                matched, reg_plate = is_plate_registered(cleaned, registered_plates, threshold=fuzzy_threshold)

                if matched:
                    print(f"Registered plate '{reg_plate}' detected → Open boom barrier")
                else:
                    print(f"Unregistered plate '{cleaned}' detected → Access denied")

except KeyboardInterrupt:
    pass
finally:
    cap.release()
