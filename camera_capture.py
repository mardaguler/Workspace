from picamera2 import Picamera2
import time
from datetime import datetime

# Kamera başlat
camera = Picamera2()

# Çözünürlük ayarla (örnek: 1920x1080)
config = camera.create_still_configuration(
    main={"size": (1920, 1080)},
    controls={"AeEnable": 1, "AwbEnable": 1} # make the camera use auto exposure and auto white balance
)
camera.configure(config)

# Kısa bir bekleme
time.sleep(2)

# Kamerayı başlat
camera.start()

# Tarih ve saat ile dosya adı oluştur
timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
filename = f"Captured/{timestamp}.png"

# Fotoğraf çek ve kaydet
camera.capture_file(filename)

# Kamerayı durdur ve kapat
camera.stop()
camera.close()

print(f"Fotoğraf çekildi ve '{filename}' olarak kaydedildi.")
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