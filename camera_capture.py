from picamera2 import Picamera2
import time
from datetime import datetime
import cv2
import numpy as np
import boto3
import os

BUCKET_NAME = "rpi-camera-photo-plants"
REGION = "eu-north-1"
S3_PREFIX = ""   # boş bırak → root’a atar (frontend direkt görür)

s3 = boto3.client("s3", region_name=REGION)

def upload_to_s3(file_path):
    try:
        file_name = os.path.basename(file_path)

        now = datetime.now()

        year = now.strftime("%Y")
        month = now.strftime("%m")
        day = now.strftime("%d")

        s3_key = f"{year}/{month}/{day}/{file_name}" # koyulacak path: 2024/06/15/photo.png gibi

        ext = os.path.splitext(file_name)[1].lower()
        if ext == ".png":
            content_type = "image/png"
        elif ext in [".jpg", ".jpeg"]:
            content_type = "image/jpeg"
        else:
            content_type = "application/octet-stream"

        s3.upload_file(
            file_path,
            BUCKET_NAME,
            s3_key,
            ExtraArgs={"ContentType": content_type}
        )

        print(f"Uploaded → {s3_key}")

    except Exception as e:
        print(f"S3 upload error: {e}")


data = np.load("Calibration_Results/camera_calibration.npz")
camera_matrix = data["mtx"]
dist_coeffs = data["dist"]

camera = Picamera2()

config = camera.create_still_configuration(
    main={"size": (3280, 2464)},
    controls={"AeEnable": 1, "AwbEnable": 1}
)
camera.configure(config)

time.sleep(2)
camera.start()

frame = camera.capture_array()

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")



raw_filename = f"Captured/raw_{timestamp}.png"
undist_filename = f"Captured/undist_{timestamp}.png"
# opencv use bgr format, picamera use rgb format, so we need to convert to bgr
# to work in opencv
frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

#local ve remote kaydet
cv2.imwrite(raw_filename, frame)
upload_to_s3(raw_filename)

# =========================
# UNDISTORT
# =========================
"""
h, w = frame.shape[:2]

new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
    camera_matrix, dist_coeffs, (w, h), 1, (w, h)
)

undistorted = cv2.undistort(
    frame, camera_matrix, dist_coeffs, None, new_camera_matrix
)

# UNDISTORTED kaydet
cv2.imwrite(undist_filename, undistorted)
"""

camera.stop()
camera.close()

print("Fotoğraf çekildi:")
print(f"  RAW: {raw_filename}")
#print(f"  UNDISTORTED: {undist_filename}")

print(f"Fotoğraf çekildi ve '{raw_filename}' olarak kaydedildi.")
#print(f"Fotoğraf çekildi ve '{undist_filename}' olarak kaydedildi.")
"""
except subprocess.CalledProcessError as e:
    print(f"Hata: Fotoğraf çekilemedi. Hata kodu: {e.returncode}")
except FileNotFoundError:
    print("Hata: rpicam-still komutu bulunamadı. Lütfen rpicam-apps paketinin yüklü olduğundan emin olun.")
    """

"""
Yazılımsal Olarak Değiştirilebilir Ayarlar (IMX219 için):
Çözünürlük (Resolution): Konfigürasyon sırasında belirlenir, runtime'da değiştirilemez
Pozlama Süresi (ExposureTime): Mikrosaniye cinsinden (ör. 10000 = 10ms)
Kazanç (Gain): AnalogGain (1.0-8.0) ve DigitalGain
Beyaz Dengesi (White Balance): ColourGains (kırmızı/mavi kazanç)
Otomatik Pozlama/AWB: AeEnable, AwbEnable (0/1)
Renk Ayarları: Saturation, Contrast, Sharpness, Brightness
Kare Hızı (FrameRate): Hedef FPS
Dijital Kırpma (ScalerCrop): Zoom için bölge seçimi """

# !!! libcamera kernel e özel .venv den ulaşamazsın, global interpret kullan !!!

#cat /boot/firmware/config.txt
#sudo nano /boot/firmware/config.txt
#ls /boot/firmware/overlays/ | grep imx
#rpicam-hello --list-cameras
#dmesg | grep -i imx219  (what does kernel says about imx219)