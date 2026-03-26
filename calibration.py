import numpy as np
import cv2 # type: ignore cv2 is in venv, venv de çalıştırmalısın, caapture.py de globalde
""" /home/ozu/Desktop/Workspace/.venv/bin/python /home/ozu/Desktop/Workspace/calibration.py"""
import glob
import os

# --- AYARLAR ---
CHESSBOARD_SIZE = (7, 5)   # SENİN TAHTAN İÇİN DOĞRU
SQUARE_SIZE = 25.0         # mm (ölçerek doğrula)

IMAGE_PATH = '/home/ozu/Desktop/Workspace/Calibration _Photos/*.png'
SAVE_DIR = '/home/ozu/Desktop/Workspace/Calibration_Results'

# klasör yoksa oluştur
os.makedirs(SAVE_DIR, exist_ok=True)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# --- 3D NOKTALAR ---
objp = np.zeros((CHESSBOARD_SIZE[0] * CHESSBOARD_SIZE[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHESSBOARD_SIZE[0], 0:CHESSBOARD_SIZE[1]].T.reshape(-1, 2)
objp *= SQUARE_SIZE

objpoints = []
imgpoints = []

images = glob.glob(IMAGE_PATH)

if not images:
    print("Hata: Görüntü bulunamadı.")
    exit()

print(f"Toplam {len(images)} fotoğraf bulundu. Köşeler aranıyor...")

img_shape = None

# daha sağlam detection için
flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE

for fname in images:
    img = cv2.imread(fname)

    if img is None:
        print(f"Görüntü okunamadı: {fname}")
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if img_shape is None:
        img_shape = gray.shape[::-1]

    # yeni daha güçlü method
    ret, corners = cv2.findChessboardCornersSB(gray, CHESSBOARD_SIZE, flags)

    print(f"{os.path.basename(fname)} -> {'OK' if ret else 'FAIL'}")

    if ret:
        objpoints.append(objp)

        corners = corners.astype(np.float32)
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)

        # debug görüntüsü
        vis = img.copy()
        cv2.drawChessboardCorners(vis, CHESSBOARD_SIZE, corners2, ret)
        cv2.imshow("Corners", vis)
        cv2.waitKey(100)

cv2.destroyAllWindows()

if len(objpoints) == 0:
    print("Hiçbir görüntüde köşe bulunamadı.")
    exit()

print(f"\nBaşarılı! {len(objpoints)} görüntü kullanıldı.")
print("Kalibrasyon hesaplanıyor...")

# --- KALİBRASYON ---
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, img_shape, None, None
)

# --- REPROJECTION ERROR ---
mean_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(
        objpoints[i], rvecs[i], tvecs[i], mtx, dist
    )
    error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
    mean_error += error

mean_error /= len(objpoints)

print("\n" + "="*40)
print("INTRINSIC MATRIX (mtx)")
print("="*40)
print(mtx)

print("\nDISTORTION COEFFS (dist)")
print("="*40)
print(dist)

print(f"\nReprojection Error: {mean_error:.4f} px")

# --- EXTRINSIC (ilk görüntü için) ---
R, _ = cv2.Rodrigues(rvecs[0])
t = tvecs[0]
extrinsic_matrix = np.hstack((R, t))

print("\nExtrinsic Matrix [R|t] (ilk görüntü)")
print("="*40)
print(extrinsic_matrix)

# --- KAYDETME ---

# 1. Tüm kalibrasyon verisi (tek dosya)
np.savez(os.path.join(SAVE_DIR, "camera_calibration.npz"),
         mtx=mtx,
         dist=dist,
         rvecs=rvecs,
         tvecs=tvecs,
         reprojection_error=mean_error,
         img_width=img_shape[0],
         img_height=img_shape[1])

np.save(os.path.join(SAVE_DIR, "mtx.npy"), mtx)
np.save(os.path.join(SAVE_DIR, "dist.npy"), dist)

np.savetxt(os.path.join(SAVE_DIR, "mtx.txt"), mtx)
np.savetxt(os.path.join(SAVE_DIR, "dist.txt"), dist)

np.save(os.path.join(SAVE_DIR, "extrinsic_first.npy"), extrinsic_matrix)

print("\nTüm sonuçlar 'Calibration_Results' klasörüne kaydedildi.")

#unzip camera_calibration.npz    npz dosyasını aç (Calibration_Results/ git)
