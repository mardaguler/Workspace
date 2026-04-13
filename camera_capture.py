from picamera2 import Picamera2
import time
from datetime import datetime
import cv2
import numpy as np

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
# RAW kaydet
cv2.imwrite(raw_filename, frame)

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