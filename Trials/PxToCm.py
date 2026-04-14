import cv2
import numpy as np
import os

# =========================================================
# SETTINGS
# =========================================================
IMAGE_PATH = "/home/ozu/Desktop/Workspace/Annotated Images/Ann2_Chess.png"

PATTERN_SIZE = (7, 5)
SQUARE_SIZE_CM = 3.4

# bunu mouse ile gördüğün GERÇEK köşeye göre güncelle
ORIGIN_PX = (3215, 2415)

OUTPUT_DIR = os.path.dirname(IMAGE_PATH)

LOWER_BLUE = np.array([100, 100, 80], dtype=np.uint8)
UPPER_BLUE = np.array([130, 255, 255], dtype=np.uint8)

LOWER_RED_1 = np.array([0, 150, 150], dtype=np.uint8)
UPPER_RED_1 = np.array([10, 255, 255], dtype=np.uint8)
LOWER_RED_2 = np.array([170, 150, 150], dtype=np.uint8)
UPPER_RED_2 = np.array([180, 255, 255], dtype=np.uint8)

# red point ile blue annotation arasında kabul edilen maksimum mesafe (pixel)
BLUE_ASSOC_MARGIN = 35


# =========================================================
# HELPERS
# =========================================================
def transform_points(points_px, H):
    pts = np.array(points_px, dtype=np.float32).reshape(-1, 1, 2)
    pts_out = cv2.perspectiveTransform(pts, H)
    return pts_out.reshape(-1, 2)


def compute_homography_from_chessboard(image, pattern_size, square_size_cm):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    ret, corners = cv2.findChessboardCorners(gray, pattern_size, flags)
    if not ret:
        raise RuntimeError("Chessboard corners bulunamadı.")

    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        30,
        0.001
    )
    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

    obj_pts = []
    for j in range(pattern_size[1]):
        for i in range(pattern_size[0]):
            obj_pts.append([i * square_size_cm, j * square_size_cm])

    obj_pts = np.array(obj_pts, dtype=np.float32)
    img_pts = corners.reshape(-1, 2).astype(np.float32)

    H, _ = cv2.findHomography(img_pts, obj_pts)
    return H, corners


def build_blue_mask(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, LOWER_BLUE, UPPER_BLUE)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=1)
    return mask


def detect_blue_components(blue_mask):
    """
    Mavi annotation'ları connected component olarak bulur.
    Label + kutu birleşmiş olsa bile component olarak tutulur.
    """
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(blue_mask, connectivity=8)
    comps = []

    for i in range(1, num_labels):
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]

        if area < 40:
            continue
        if w < 8 or h < 8:
            continue

        comps.append((x, y, w, h))

    return comps


def detect_all_red_points(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    mask1 = cv2.inRange(hsv, LOWER_RED_1, UPPER_RED_1)
    mask2 = cv2.inRange(hsv, LOWER_RED_2, UPPER_RED_2)
    red_mask = cv2.bitwise_or(mask1, mask2)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel, iterations=1)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(red_mask, connectivity=8)

    points = []
    kept_mask = np.zeros_like(red_mask)

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]

        if area < 3 or area > 120:
            continue

        cx, cy = centroids[i]
        points.append((cx, cy, area))
        kept_mask[labels == i] = 255

    points = sorted(points, key=lambda p: (int(p[1] // 40), p[0]))
    return points, red_mask, kept_mask


def point_near_component(px, py, comp, margin=35):
    x, y, w, h = comp
    return (
        (x - margin) <= px <= (x + w + margin)
        and
        (y - margin) <= py <= (y + h + margin)
    )


def filter_red_points_by_blue_components(red_points, blue_components, margin=35):
    """
    Her red point için, herhangi bir blue component'e yeterince yakın mı diye bak.
    Böylece kutu çıkarımı eksik olsa bile red point'ler korunur.
    """
    filtered = []

    for px, py, area in red_points:
        keep = False
        for comp in blue_components:
            if point_near_component(px, py, comp, margin):
                keep = True
                break
        if keep:
            filtered.append((px, py, area))

    filtered = sorted(filtered, key=lambda p: (int(p[1] // 40), p[0]))
    return filtered


# =========================================================
# MAIN
# =========================================================
def main():
    img = cv2.imread(IMAGE_PATH)
    if img is None:
        raise FileNotFoundError(f"Görüntü okunamadı: {IMAGE_PATH}")

    # 1) homography
    H_img_to_plane, corners = compute_homography_from_chessboard(
        img, PATTERN_SIZE, SQUARE_SIZE_CM
    )

    # 2) masks
    blue_mask = build_blue_mask(img)
    blue_components = detect_blue_components(blue_mask)

    red_points_all, red_mask_all, red_mask_kept_all = detect_all_red_points(img)
    red_points = filter_red_points_by_blue_components(
        red_points_all,
        blue_components,
        margin=BLUE_ASSOC_MARGIN
    )

    if len(red_points) == 0:
        raise RuntimeError("Mavi annotation yakınında hiç red point bulunamadı.")

    # 3) origin
    origin_px = ORIGIN_PX
    origin_plane = transform_points([origin_px], H_img_to_plane)[0]

    # 4) red points -> plane
    red_points_plane = transform_points(
        [(x, y) for (x, y, area) in red_points],
        H_img_to_plane
    )

    # 5) coordinates
    results = []
    for (px, py, area), p_plane in zip(red_points, red_points_plane):
        X_cm = origin_plane[0] - p_plane[0]
        Y_cm = origin_plane[1] - p_plane[1]
        results.append(((px, py, area), p_plane, X_cm, Y_cm))

    results = sorted(results, key=lambda r: (int(r[0][1] // 40), r[0][0]))

    # 6) visualize
    vis = img.copy()
    cv2.drawChessboardCorners(vis, PATTERN_SIZE, corners, True)

    ox, oy = int(round(origin_px[0])), int(round(origin_px[1]))
    cv2.circle(vis, (ox, oy), 10, (0, 0, 255), -1)
    cv2.putText(
        vis,
        "origin",
        (ox + 10, oy - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2
    )

    # blue components çiz
    for (x, y, w, h) in blue_components:
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 255), 2)

    print("Detected points:")
    for i, ((px, py, area), p_plane, X_cm, Y_cm) in enumerate(results):
        cv2.circle(vis, (int(round(px)), int(round(py))), 5, (0, 255, 0), -1)
        cv2.putText(
            vis,
            f"{i}: ({X_cm:.1f}, {Y_cm:.1f}) cm",
            (int(px) + 8, int(py) - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 0),
            1
        )

        print(
            f"{i}: pixel=({px:.1f},{py:.1f}), "
            f"plane=({p_plane[0]:.2f},{p_plane[1]:.2f}), "
            f"coord=({X_cm:.2f},{Y_cm:.2f}) cm"
        )

    # red filtered mask for visualization
    red_mask_filtered = np.zeros_like(red_mask_all)
    for px, py, area in red_points:
        cv2.circle(red_mask_filtered, (int(round(px)), int(round(py))), 4, 255, -1)

    # save
    out_blue = os.path.join(OUTPUT_DIR, "blue_mask_final.png")
    out_red_all = os.path.join(OUTPUT_DIR, "red_mask_all.png")
    out_red_filtered = os.path.join(OUTPUT_DIR, "red_mask_filtered_by_blue_components.png")
    out_vis = os.path.join(OUTPUT_DIR, "origin_and_red_points_debug.png")

    cv2.imwrite(out_blue, blue_mask)
    cv2.imwrite(out_red_all, red_mask_kept_all)
    cv2.imwrite(out_red_filtered, red_mask_filtered)
    cv2.imwrite(out_vis, vis)

    print("\nSaved:")
    print(f" - {out_blue}")
    print(f" - {out_red_all}")
    print(f" - {out_red_filtered}")
    print(f" - {out_vis}")


if __name__ == "__main__":
    main()