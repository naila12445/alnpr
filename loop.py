# ==============================================================
# ANPR on Live IP Webcam Stream (YOLOv8 + EasyOCR)
# Checks registered vehicles from registered_plates.json
# ==============================================================

import cv2
from ultralytics import YOLO
import easyocr
import re
import time
import json
import os

# -----------------------------
# 1. Load Configuration
# -----------------------------
CONFIG_PATH = "config.json"
REG_PATH = "registered_plates.json"

# Load config.json
if not os.path.exists(CONFIG_PATH):
    raise FileNotFoundError("Missing config.json file!")

with open(CONFIG_PATH, "r") as f:
    config = json.load(f)

FRAME_INTERVAL = config.get("frame_interval", 100)
CONF_THRESHOLD = config.get("confidence_threshold", 0.5)
IP_STREAM = config.get("ip_stream", 0)  # default webcam if not provided

# Load registered_plates.json
if not os.path.exists(REG_PATH):
    raise FileNotFoundError("Missing registered_plates.json file!")

with open(REG_PATH, "r") as f:
    registered_data = json.load(f)
REGISTERED_PLATES = set([p.strip().upper() for p in registered_data.get("registered", [])])

# -----------------------------
# 2. Load YOLO model & OCR
# -----------------------------
model = YOLO("license_plate_detector.pt")
reader = easyocr.Reader(['en'])

# -----------------------------
# 3. Load Live Stream
# -----------------------------
cap = cv2.VideoCapture(IP_STREAM)
if not cap.isOpened():
    raise IOError("Could not open IP Webcam stream. Check IP and Wi-Fi connection!")

# -----------------------------
# 4. Function to clean OCR text
# -----------------------------
def clean_plate_text(text):
    text = re.sub(r'\bIND(?:IA)?\b', '', text, flags=re.IGNORECASE)
    text = text.upper()
    text = re.sub(r'[^A-Z0-9]', '', text)
    text = text.replace("I6", "G").replace("16", "G").replace("1G", "G")
    if re.match(r"[A-Z]{2}\d", text):
        text = text.replace("B", "8")
    else:
        text = text.replace("8", "B")
    return text.strip()

# -----------------------------
# 5. Main Processing Loop
# -----------------------------
frame_count = 0
start_time = time.time()

print("Starting ANPR Live Stream Processing... (Press Ctrl+C to stop)\n")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Stream ended or cannot fetch frame.")
            break

        frame_count += 1
        if frame_count % FRAME_INTERVAL != 0:
            continue

        results = model(frame, verbose=False)

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])

                if conf < CONF_THRESHOLD:
                    continue

                plate_crop = frame[y1:y2, x1:x2]
                if plate_crop.size == 0:
                    continue

                # Perform OCR
                plate_crop = cv2.resize(plate_crop, None, fx=2, fy=2)
                ocr_result = reader.readtext(plate_crop, detail=0)

                if ocr_result:
                    raw_text = " ".join(ocr_result)
                    cleaned = clean_plate_text(raw_text)

                    print(f"Detected: {cleaned} (Raw: {raw_text}, Conf: {conf:.2f})")

                    # Check if registered
                    if cleaned in REGISTERED_PLATES:
                        print(f"✅ Registered Vehicle — {cleaned} | OPEN BARRIER")
                    else:
                        print(f"❌ Unregistered Vehicle — {cleaned} | ACCESS DENIED")

        # Optional FPS info
        if frame_count % 50 == 0:
            elapsed = time.time() - start_time
            fps = frame_count / elapsed if elapsed > 0 else 0
            print(f"Processed {frame_count} frames | FPS: {fps:.2f}")

except KeyboardInterrupt:
    print("\nStopped manually by user.")

# -----------------------------
# 6. Cleanup
# -----------------------------
cap.release()
print("Live ANPR processing ended.")
