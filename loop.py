# ==============================================================
# ANPR on Live IP Webcam Stream (YOLOv8 + EasyOCR + Smart Correction)
# Stops automatically when a registered vehicle is detected
# ==============================================================

import cv2
from ultralytics import YOLO
import easyocr
import re
import time

# -----------------------------
# 1. Load YOLO model & OCR
# -----------------------------
model = YOLO("license_plate_detector.pt")  # Your trained YOLO model
reader = easyocr.Reader(['en'])  # English OCR reader

# -----------------------------
# 2. Load Live Stream from IP Webcam
# -----------------------------
ip_stream = "http://192.168.29.31:8080/video"  # Replace with your IP webcam URL
cap = cv2.VideoCapture(ip_stream)

if not cap.isOpened():
    raise IOError("❌ Could not open IP Webcam stream. Check IP and Wi-Fi connection!")

# -----------------------------
# 3. Registered vehicles list
# -----------------------------
registered_numbers = {"KA18EQ0001", "GJ34B9790", "MH12AB1234"}  # Example set

# -----------------------------
# 4. Smart OCR Cleaning Function
# -----------------------------
def clean_plate_text(text):
    # Remove "IND" etc.
    text = re.sub(r'\bIND(?:IA)?\b', '', text, flags=re.IGNORECASE)

    # Keep only alphanumerics and uppercase
    text = re.sub(r'[^A-Za-z0-9]', '', text).upper()

    # Mapping for likely OCR mixups
    letter_to_digit = {'O': '0', 'D': '0', 'Q': '0', 'I': '1', 'L': '1', 'Z': '2', 'S': '5', 'G': '6', 'B': '8'}
    digit_to_letter = {'0': 'O', '6': 'G', '8': 'B', '5': 'S', '2': 'Z', '1': 'I'}
    similar_letters = {'X': 'K', 'K': 'X'}  # Common confusion pair

    corrected = []
    for i, ch in enumerate(text):
        # Rough Indian plate pattern: LL NN LL NNNN
        if i < 2 or (4 <= i < 6):  # Letter zones
            if ch in digit_to_letter:
                corrected.append(digit_to_letter[ch])
            elif ch in similar_letters:
                corrected.append(similar_letters[ch])
            else:
                corrected.append(ch)
        else:  # Numeric zones
            if ch in letter_to_digit:
                corrected.append(letter_to_digit[ch])
            else:
                corrected.append(ch)

    return ''.join(corrected)

# -----------------------------
# 5. Main Processing Loop
# -----------------------------
frame_count = 0
start_time = time.time()
authorized_detected = False  # Flag to stop loop when registered vehicle found

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Stream ended or cannot fetch frame.")
        break

    frame_count += 1
    results = model(frame, verbose=False)

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            if conf < 0.5:
                continue

            plate_crop = frame[y1:y2, x1:x2]
            if plate_crop.size == 0:
                continue

            # Perform OCR every 5th frame
            if frame_count % 5 == 0:
                plate_crop = cv2.resize(plate_crop, None, fx=2, fy=2)
                ocr_result = reader.readtext(plate_crop, detail=0)

                if ocr_result:
                    raw_text = " ".join(ocr_result)
                    cleaned = clean_plate_text(raw_text)

                    print(f"Raw: {raw_text}  →  Cleaned: {cleaned}")

                    # Check registration
                    if cleaned in registered_numbers:
                        print(f"✅ {cleaned} recognized — 🚗 Barrier Open ✅")
                        color = (0, 255, 0)
                        status_text = f"{cleaned} (Authorized)"
                        authorized_detected = True  # Stop loop
                    else:
                        print(f"❌ {cleaned} not recognized — ⛔ Access Denied")
                        color = (0, 0, 255)
                        status_text = f"{cleaned} (Denied)"

                    # Draw on frame
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, status_text, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    # Display FPS
    elapsed_time = time.time() - start_time
    fps = frame_count / elapsed_time if elapsed_time > 0 else 0
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    cv2.imshow("🔍 ANPR - Live Stream", frame)

    # Stop loop if authorized vehicle detected
    if authorized_detected:
        print("🛑 Authorized vehicle detected — stopping ANPR.")
        time.sleep(2)  # Optional delay for barrier open simulation
        break

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("🛑 Stream stopped by user.")
        break

# -----------------------------
# 6. Cleanup
# -----------------------------
cap.release()
cv2.destroyAllWindows()
print("✅ Live ANPR processing ended.")
