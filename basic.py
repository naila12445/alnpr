# ==============================================================
# ANPR (YOLOv8 + EasyOCR)
# Only Text Extraction — From Live IP Webcam Stream
# ==============================================================

import cv2
from ultralytics import YOLO
import easyocr
import re
import time

# -----------------------------
# 1. Load YOLO model & OCR
# -----------------------------
model = YOLO("license_plate_detector.pt")   # Trained YOLOv8 model
reader = easyocr.Reader(['en'])             # OCR engine

# -----------------------------
# 2. IP Webcam Stream
# -----------------------------
ip_stream = "http://192.168.29.31:8080/video"  # Replace with your IP cam URL
cap = cv2.VideoCapture(ip_stream)

if not cap.isOpened():
    raise IOError("Could not open IP Webcam stream. Check IP and Wi-Fi connection.")

# -----------------------------
# 3. Smart OCR Cleaning Function
# -----------------------------
def clean_plate_text(text):
    """Cleans and standardizes OCR text into Indian number plate format."""
    text = re.sub(r'\bIND(?:IA)?\b', '', text, flags=re.IGNORECASE)
    text = re.sub(r'[^A-Za-z0-9]', '', text).upper()

    # Character correction maps
    letter_to_digit = {'O': '0', 'D': '0', 'Q': '0', 'I': '1', 'L': '1', 'Z': '2', 'S': '5', 'G': '6', 'B': '8'}
    digit_to_letter = {'0': 'O', '6': 'G', '8': 'B', '5': 'S', '2': 'Z', '1': 'I'}

    corrected = []
    for i, ch in enumerate(text):
        if i < 2 or (4 <= i < 6):  # Letters region
            corrected.append(digit_to_letter.get(ch, ch))
        else:
            corrected.append(letter_to_digit.get(ch, ch))
    text = ''.join(corrected)

    # Handle common OCR noise
    if text.startswith(("IN", "ID")):
        text = text[2:]

    match = re.match(r'([A-Z]{2})(\d{1,2})([A-Z]{1,2})(\d{3,4})([A-Z]{0,2})', text)
    if match:
        text = ''.join([p for p in match.groups() if p])
    return text[:10]

# -----------------------------
# 4. Main Processing Loop
# -----------------------------
frame_count = 0
start_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Stream ended or cannot fetch frame.")
        break

    frame_count += 1
    results = model(frame, verbose=False)

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            if conf < 0.5:
                continue

            # Crop detected plate
            plate_crop = frame[y1:y2, x1:x2]
            if plate_crop.size == 0:
                continue

            # OCR
            plate_crop = cv2.resize(plate_crop, None, fx=2, fy=2)
            ocr_result = reader.readtext(plate_crop, detail=0)

            if ocr_result:
                raw_text = " ".join(ocr_result)
                cleaned = clean_plate_text(raw_text)
                print(f"Raw: {raw_text}  ->  Cleaned: {cleaned}")

                # Draw on frame
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                cv2.putText(frame, cleaned, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    # FPS counter
    elapsed_time = time.time() - start_time
    fps = frame_count / elapsed_time if elapsed_time > 0 else 0
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

    # Display
    cv2.imshow("ANPR - Text Extraction Only", frame)

    # Stop manually
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Stream stopped by user.")
        break

# -----------------------------
# 5. Cleanup
# -----------------------------
cap.release()
cv2.destroyAllWindows()
print("Live text extraction ended.")
