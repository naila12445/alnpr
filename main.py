# ==============================================================
# 📹 IP Webcam Live Stream Test (Phone → Laptop)
# ==============================================================
import cv2
import time

# --------------------------------------------------------------
# 🔹 STEP 1: Enter your phone IP Webcam stream URL
# Open IP Webcam app on your phone → Start Server → check the URL shown
# Example: http://192.168.29.31:8080
# --------------------------------------------------------------
ip = "192.168.29.31"  # 👈 change this to match your phone IP
url = f"http://{ip}:8080/video"  # main video feed

# --------------------------------------------------------------
# 🔹 STEP 2: Try to connect
# --------------------------------------------------------------
print("⏳ Connecting to IP Webcam stream...")
time.sleep(2)

# Try using FFMPEG backend first
cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
time.sleep(1)

if not cap.isOpened():
    print("⚠️ Couldn't open stream with FFMPEG. Trying alternate endpoints...")
    # Try other common endpoints
    alt_urls = [
        f"http://{ip}:8080/mjpegfeed",
        f"http://{ip}:8080/mjpeg",
        f"http://{ip}:8080/shot.jpg"
    ]
    for test_url in alt_urls:
        cap = cv2.VideoCapture(test_url)
        if cap.isOpened():
            url = test_url
            print(f"✅ Connected successfully with {test_url}")
            break

if not cap.isOpened():
    raise IOError("❌ Could not open IP Webcam stream. Check IP, Wi-Fi connection, and IP Webcam app!")

print(f"✅ Connected to: {url}")
print("Press 'q' to exit.")

# --------------------------------------------------------------
# 🔹 STEP 3: Show live video
# --------------------------------------------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Frame not received. Retrying...")
        time.sleep(0.5)
        continue

    cv2.imshow("📱 IP Webcam Live Stream", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("👋 Exiting stream.")
        break

cap.release()
cv2.destroyAllWindows()
