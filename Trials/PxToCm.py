import cv2
import numpy as np
import os

# =========================================================
# SETTINGS
# =========================================================
IMAGE_PATH = "/home/ozu/Desktop/Workspace/Annotated Images/Ann2_Chess.png"

PATTERN_SIZE = (7, 5)   # chessboard internal corners
SQUARE_SIZE_CM = 3.4    # one square side in cm

OUTPUT_DIR = os.path.dirname(IMAGE_PATH)

# Blue annotation range
LOWER_BLUE = np.array([100, 100, 80], dtype=np.uint8)
UPPER_BLUE = np.array([130, 255, 255], dtype=np.uint8)

# Strict red ranges
LOWER_RED_1 = np.array([0, 150, 150], dtype=np.uint8)
UPPER_RED_1 = np.array([10, 255, 255], dtype=np.uint8)
LOWER_RED_2 = np.array([170, 150, 150], dtype=np.uint8)
UPPER_RED_2 = np.array([180, 255, 255], dtype=np.uint8)

# Bright ruler range
LOWER_BRIGHT = np.array([0, 0, 150], dtype=np.uint8)
UPPER_BRIGHT = np.array([180, 80, 255], dtype=np.uint8)


# =========================================================
# HELPERS
# =========================================================
def normalize(v):
    n = np.linalg.norm(v)
    if n < 1e-9:
        raise RuntimeError("Zero-length vector.")
    return v / n


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

    # image pixel -> chessboard plane (cm)
    H, _ = cv2.findHomography(img_pts, obj_pts)
    return H, corners


def build_blue_mask(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, LOWER_BLUE, UPPER_BLUE)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=1)
    return mask


def detect_red_points_near_blue(image, blue_mask):
    """
    Detect all strict-red points globally,
    then keep only those near blue annotations.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    mask1 = cv2.inRange(hsv, LOWER_RED_1, UPPER_RED_1)
    mask2 = cv2.inRange(hsv, LOWER_RED_2, UPPER_RED_2)
    red_mask = cv2.bitwise_or(mask1, mask2)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel, iterations=1)

    # Dilate blue so red points close to blue labels/boxes are kept
    blue_support = cv2.dilate(
        blue_mask,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41)),
        iterations=1
    )

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(red_mask, connectivity=8)

    kept_points = []
    kept_mask = np.zeros_like(red_mask)

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < 3 or area > 120:
            continue

        cx, cy = centroids[i]
        cx_i, cy_i = int(round(cx)), int(round(cy))

        if cx_i < 0 or cy_i < 0 or cx_i >= blue_support.shape[1] or cy_i >= blue_support.shape[0]:
            continue

        if blue_support[cy_i, cx_i] == 0:
            continue

        kept_points.append((cx, cy, area))
        kept_mask[labels == i] = 255

    kept_points = sorted(kept_points, key=lambda p: (int(p[1] // 40), p[0]))
    return kept_points, red_mask, kept_mask, blue_support


def detect_rulers(image):
    """
    Detect bottom ruler and right ruler as bright regions.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    bright_mask = cv2.inRange(hsv, LOWER_BRIGHT, UPPER_BRIGHT)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    bright_mask = cv2.morphologyEx(bright_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    H, W = bright_mask.shape

    # Bottom ruler
    bottom_roi_y = int(H * 0.88)
    bottom_roi = bright_mask[bottom_roi_y:, :]
    contours, _ = cv2.findContours(bottom_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    bottom_candidates = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        area = w * h
        if area < 800:
            continue
        if w < W * 0.35:
            continue
        if h > H * 0.12:
            continue
        bottom_candidates.append((x, y + bottom_roi_y, w, h))

    if not bottom_candidates:
        raise RuntimeError("Bottom ruler bulunamadı.")

    bottom_ruler = max(bottom_candidates, key=lambda r: r[2])

    # Right ruler
    right_roi_x = int(W * 0.94)
    right_roi = bright_mask[:, right_roi_x:]
    contours, _ = cv2.findContours(right_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    right_candidates = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        area = w * h
        if area < 800:
            continue
        if h < H * 0.35:
            continue
        if w > W * 0.10:
            continue
        right_candidates.append((x + right_roi_x, y, w, h))

    if not right_candidates:
        raise RuntimeError("Right ruler bulunamadı.")

    right_ruler = max(right_candidates, key=lambda r: r[3])

    return bottom_ruler, right_ruler, bright_mask


def extract_bottom_ruler_top_edge_points(bright_mask, bottom_ruler, right_ruler, step=3):
    """
    Collect points on the TOP INNER edge of bottom ruler.
    Focus near the bottom-right corner region.
    """
    x, y, w, h = bottom_ruler
    rx, ry, rw, rh = right_ruler

    roi = bright_mask[y:y+h, x:x+w]
    pts = []

    # use region near right side / intersection
    start_x = max(0, (rx - 250) - x)
    end_x = min(w, (rx + 40) - x)

    if end_x <= start_x:
        start_x = 0
        end_x = w

    for xx in range(start_x, end_x, step):
        ys = np.where(roi[:, xx] > 0)[0]
        if len(ys) == 0:
            continue
        top_y = ys[0]
        pts.append((x + xx, y + top_y))

    if len(pts) < 5:
        raise RuntimeError("Bottom ruler top edge points çıkarılamadı.")

    return pts


def extract_right_ruler_left_edge_points(bright_mask, right_ruler, bottom_ruler, step=3):
    """
    Collect points on the LEFT INNER edge of right ruler.
    Focus near the bottom-right corner region.
    """
    x, y, w, h = right_ruler
    bx, by, bw, bh = bottom_ruler

    roi = bright_mask[y:y+h, x:x+w]
    pts = []

    # use region near bottom side / intersection
    start_y = max(0, (by - 250) - y)
    end_y = min(h, (by + 40) - y)

    if end_y <= start_y:
        start_y = 0
        end_y = h

    for yy in range(start_y, end_y, step):
        xs = np.where(roi[yy, :] > 0)[0]
        if len(xs) == 0:
            continue
        left_x = xs[0]
        pts.append((x + left_x, y + yy))

    if len(pts) < 5:
        raise RuntimeError("Right ruler left edge points çıkarılamadı.")

    return pts


def fit_line_pca(points_2d):
    pts = np.asarray(points_2d, dtype=np.float64)
    center = np.mean(pts, axis=0)

    pts0 = pts - center
    _, _, vt = np.linalg.svd(pts0, full_matrices=False)
    direction = normalize(vt[0])

    return center, direction


def line_intersection_2d(p1, d1, p2, d2):
    A = np.column_stack((d1, -d2))
    b = p2 - p1

    if np.linalg.matrix_rank(A) < 2:
        raise RuntimeError("Lines nearly parallel.")

    ts, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    t = ts[0]
    return p1 + t * d1


# =========================================================
# MAIN
# =========================================================
def main():
    img = cv2.imread(IMAGE_PATH)
    if img is None:
        raise FileNotFoundError(f"Görüntü okunamadı: {IMAGE_PATH}")

    # 1) Chessboard homography
    H_img_to_plane, corners = compute_homography_from_chessboard(
        img, PATTERN_SIZE, SQUARE_SIZE_CM
    )

    # 2) Blue mask
    blue_mask = build_blue_mask(img)

    # 3) Red points near blue
    red_points_px, red_mask_all, red_mask_kept, blue_support = detect_red_points_near_blue(img, blue_mask)
    if len(red_points_px) == 0:
        raise RuntimeError("Hiç uygun red point bulunamadı.")

    # 4) Detect rulers
    bottom_ruler, right_ruler, bright_mask = detect_rulers(img)

    # 5) Extract inner edge points for origin
    bottom_edge_px = extract_bottom_ruler_top_edge_points(bright_mask, bottom_ruler, right_ruler, step=3)
    right_edge_px = extract_right_ruler_left_edge_points(bright_mask, right_ruler, bottom_ruler, step=3)

    # 6) Fit lines in IMAGE space to get a better origin pixel
    bottom_line_point_px, bottom_dir_px = fit_line_pca(bottom_edge_px)
    right_line_point_px, right_dir_px = fit_line_pca(right_edge_px)

    origin_px = line_intersection_2d(
        bottom_line_point_px, bottom_dir_px,
        right_line_point_px, right_dir_px
    )

    # 7) Transform ruler edges + origin + red points to plane
    bottom_edge_plane = transform_points(bottom_edge_px, H_img_to_plane)
    right_edge_plane = transform_points(right_edge_px, H_img_to_plane)
    origin_plane = transform_points([origin_px], H_img_to_plane)[0]
    red_points_plane = transform_points([(x, y) for (x, y, area) in red_points_px], H_img_to_plane)

    # 8) Fit ruler axes in PLANE space
    bottom_line_point, bottom_dir = fit_line_pca(bottom_edge_plane)
    right_line_point, right_dir = fit_line_pca(right_edge_plane)

    # desired directions
    # +X = left
    if bottom_dir[0] > 0:
        bottom_dir = -bottom_dir

    # +Y = up
    if right_dir[1] > 0:
        right_dir = -right_dir

    print(f"Dot(bottom_dir, right_dir) = {float(np.dot(bottom_dir, right_dir)):.4f}")
    print(f"Origin px = ({origin_px[0]:.2f}, {origin_px[1]:.2f})")
    print(f"Origin plane = ({origin_plane[0]:.2f}, {origin_plane[1]:.2f})")

    # 9) Coordinates by projection
    results = []
    for (px, py, area), p_plane in zip(red_points_px, red_points_plane):
        vec = p_plane - origin_plane

        X_cm = float(np.dot(vec, bottom_dir))
        Y_cm = float(np.dot(vec, right_dir))

        results.append(((px, py, area), p_plane, X_cm, Y_cm))

    results = sorted(results, key=lambda r: (int(r[0][1] // 40), r[0][0]))

    # 10) Visualization
    vis = img.copy()
    cv2.drawChessboardCorners(vis, PATTERN_SIZE, corners, True)

    bx, by, bw, bh = bottom_ruler
    rx, ry, rw, rh = right_ruler
    cv2.rectangle(vis, (bx, by), (bx + bw, by + bh), (255, 255, 0), 2)
    cv2.rectangle(vis, (rx, ry), (rx + rw, ry + rh), (255, 255, 0), 2)

    # draw ruler edge support points
    for p in bottom_edge_px[::3]:
        cv2.circle(vis, (int(round(p[0])), int(round(p[1]))), 2, (0, 255, 255), -1)

    for p in right_edge_px[::3]:
        cv2.circle(vis, (int(round(p[0])), int(round(p[1]))), 2, (255, 0, 255), -1)

    ox, oy = int(round(origin_px[0])), int(round(origin_px[1]))
    cv2.circle(vis, (ox, oy), 8, (0, 0, 255), -1)
    cv2.putText(vis, "origin", (ox + 10, oy - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    print("Detected points in ruler coordinates:")
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
            f"{i}: pixel=({px:.1f}, {py:.1f}), "
            f"plane=({p_plane[0]:.2f}, {p_plane[1]:.2f}), "
            f"ruler=({X_cm:.2f}, {Y_cm:.2f}) cm"
        )

    # 11) Save outputs
    out_blue = os.path.join(OUTPUT_DIR, "blue_mask_final.png")
    out_blue_support = os.path.join(OUTPUT_DIR, "blue_support_mask.png")
    out_red_all = os.path.join(OUTPUT_DIR, "red_mask_all.png")
    out_red_kept = os.path.join(OUTPUT_DIR, "red_mask_kept_near_blue.png")
    out_ruler = os.path.join(OUTPUT_DIR, "ruler_bright_mask_original.png")
    out_vis = os.path.join(OUTPUT_DIR, "red_points_ruler_coords_no_warp.png")

    cv2.imwrite(out_blue, blue_mask)
    cv2.imwrite(out_blue_support, blue_support)
    cv2.imwrite(out_red_all, red_mask_all)
    cv2.imwrite(out_red_kept, red_mask_kept)
    cv2.imwrite(out_ruler, bright_mask)
    cv2.imwrite(out_vis, vis)

    print("\nSaved:")
    print(f" - {out_blue}")
    print(f" - {out_blue_support}")
    print(f" - {out_red_all}")
    print(f" - {out_red_kept}")
    print(f" - {out_ruler}")
    print(f" - {out_vis}")


if __name__ == "__main__":
    main()