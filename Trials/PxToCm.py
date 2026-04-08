import cv2
import numpy as np
import os

# =========================================================
# SETTINGS
# =========================================================
IMAGE_PATH = "/home/ozu/Desktop/Workspace/Annotated Images/Ann2_Chess.png"

PATTERN_SIZE = (7, 5)   # chessboard internal corners
SQUARE_SIZE_CM = 3.4    # one square side in cm

# Blue annotation range
LOWER_BLUE = np.array([100, 100, 80], dtype=np.uint8)
UPPER_BLUE = np.array([130, 255, 255], dtype=np.uint8)

# Red point ranges
LOWER_RED_1 = np.array([0, 120, 80], dtype=np.uint8)
UPPER_RED_1 = np.array([10, 255, 255], dtype=np.uint8)
LOWER_RED_2 = np.array([170, 120, 80], dtype=np.uint8)
UPPER_RED_2 = np.array([180, 255, 255], dtype=np.uint8)

# Bright ruler range
LOWER_BRIGHT = np.array([0, 0, 150], dtype=np.uint8)
UPPER_BRIGHT = np.array([180, 80, 255], dtype=np.uint8)

OUTPUT_DIR = os.path.dirname(IMAGE_PATH)

# =========================================================
# HELPERS
# =========================================================
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

    # Chessboard plane coordinates in cm
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


def iou_rect(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b

    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh

    ix1 = max(ax, bx)
    iy1 = max(ay, by)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0

    inter = (ix2 - ix1) * (iy2 - iy1)
    union = aw * ah + bw * bh - inter
    return inter / union


def deduplicate_rects(rects, iou_thresh=0.45):
    if not rects:
        return []

    rects = sorted(rects, key=lambda r: r[2] * r[3], reverse=True)
    kept = []

    for r in rects:
        duplicate = False
        for k in kept:
            if iou_rect(r, k) > iou_thresh:
                duplicate = True
                break
        if not duplicate:
            kept.append(r)

    return kept


def detect_blue_boxes(blue_mask):
    """
    Find the actual rectangle part of each blue annotated detection.
    The label text is usually above the box, so search mainly in lower regions
    of each connected component.
    """
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(blue_mask, connectivity=8)
    boxes = []

    for label_id in range(1, num_labels):
        x = stats[label_id, cv2.CC_STAT_LEFT]
        y = stats[label_id, cv2.CC_STAT_TOP]
        w = stats[label_id, cv2.CC_STAT_WIDTH]
        h = stats[label_id, cv2.CC_STAT_HEIGHT]
        area = stats[label_id, cv2.CC_STAT_AREA]

        if area < 40:
            continue
        if w < 8 or h < 8:
            continue

        comp_mask = np.zeros((h, w), dtype=np.uint8)
        comp_mask[labels[y:y+h, x:x+w] == label_id] = 255

        search_regions = [
            (max(0, h // 4), h),
            (max(0, h // 3), h),
            (max(0, h // 2), h),
        ]

        best_rect = None
        best_score = -1.0

        for y0, y1 in search_regions:
            roi = comp_mask[y0:y1, :]
            contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for cnt in contours:
                rx, ry, rw, rh = cv2.boundingRect(cnt)
                rect_area = rw * rh
                cnt_area = cv2.contourArea(cnt)

                if rw < 8 or rh < 8:
                    continue
                if rect_area < 80:
                    continue

                aspect = rw / float(rh)
                if aspect > 6.0 or aspect < 0.15:
                    continue

                fill_ratio = cnt_area / (rect_area + 1e-6)
                score = fill_ratio + 0.001 * rect_area

                if score > best_score:
                    best_score = score
                    best_rect = (x + rx, y + y0 + ry, rw, rh)

        if best_rect is not None:
            boxes.append(best_rect)

    return deduplicate_rects(boxes)


def detect_red_point_in_box(image, box, pad=4):
    x, y, w, h = box

    x0 = max(0, x - pad)
    y0 = max(0, y - pad)
    x1 = min(image.shape[1], x + w + pad)
    y1 = min(image.shape[0], y + h + pad)

    roi = image[y0:y1, x0:x1]

    # Convert to HSV
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # MUCH stricter red threshold
    mask1 = cv2.inRange(hsv, np.array([0, 150, 150]), np.array([10, 255, 255]))
    mask2 = cv2.inRange(hsv, np.array([170, 150, 150]), np.array([180, 255, 255]))
    red_mask = cv2.bitwise_or(mask1, mask2)

    # Remove tiny noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel, iterations=1)

    # Find connected components instead of contours
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(red_mask)

    best_idx = -1
    best_area = 0

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]

        # REAL red dots are small but not 1-2 pixels
        if 5 < area < 200:
            if area > best_area:
                best_area = area
                best_idx = i

    if best_idx == -1:
        return None, red_mask

    cx, cy = centroids[best_idx]

    # Convert to image coordinates
    return (x0 + cx, y0 + cy), red_mask


def detect_ruler_origin(image):
    """
    Detect bottom ruler and right ruler directly in original image.
    Origin is the inner corner:
      - top edge of bottom ruler
      - left edge of right ruler
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    bright_mask = cv2.inRange(hsv, LOWER_BRIGHT, UPPER_BRIGHT)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    bright_mask = cv2.morphologyEx(bright_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    H, W = bright_mask.shape

    # Bottom ruler search
    bottom_roi_y = int(H * 0.85)
    bottom_roi = bright_mask[bottom_roi_y:, :]
    contours, _ = cv2.findContours(bottom_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    bottom_candidates = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        area = w * h
        if area < 500:
            continue
        if w < W * 0.3:
            continue
        if h > H * 0.10:
            continue
        bottom_candidates.append((x, y + bottom_roi_y, w, h))

    if not bottom_candidates:
        raise RuntimeError("Bottom ruler bulunamadı.")

    bottom_ruler = max(bottom_candidates, key=lambda r: r[2])

    # Right ruler search
    right_roi_x = int(W * 0.90)
    right_roi = bright_mask[:, right_roi_x:]
    contours, _ = cv2.findContours(right_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    right_candidates = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        area = w * h
        if area < 500:
            continue
        if h < H * 0.3:
            continue
        if w > W * 0.12:
            continue
        right_candidates.append((x + right_roi_x, y, w, h))

    if not right_candidates:
        raise RuntimeError("Right ruler bulunamadı.")

    right_ruler = max(right_candidates, key=lambda r: r[3])

    bx, by, bw, bh = bottom_ruler
    rx, ry, rw, rh = right_ruler

    origin_x = rx   # left edge of right ruler
    origin_y = by   # top edge of bottom ruler

    return (origin_x, origin_y), bottom_ruler, right_ruler, bright_mask


def transform_points(points_px, H):
    pts = np.array(points_px, dtype=np.float32).reshape(-1, 1, 2)
    pts_out = cv2.perspectiveTransform(pts, H)
    return pts_out.reshape(-1, 2)


def normalize(v):
    n = np.linalg.norm(v)
    if n < 1e-9:
        raise RuntimeError("Direction vector norm is zero.")
    return v / n


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

    # 2) Detect blue boxes
    blue_mask = build_blue_mask(img)
    boxes = detect_blue_boxes(blue_mask)
    if len(boxes) == 0:
        raise RuntimeError("Mavi kutu bulunamadı.")

    # 3) Detect one red point inside each blue box
    red_points_px = []
    vis_red = np.zeros(img.shape[:2], dtype=np.uint8)

    for box in boxes:
        red_pt, red_mask_roi = detect_red_point_in_box(img, box)
        if red_pt is not None:
            red_points_px.append(red_pt)

            x, y, w, h = box
            x0 = max(0, x - 4)
            y0 = max(0, y - 4)
            x1 = min(img.shape[1], x + w + 4)
            y1 = min(img.shape[0], y + h + 4)
            vis_red[y0:y1, x0:x1] = np.maximum(vis_red[y0:y1, x0:x1], red_mask_roi)

    if len(red_points_px) == 0:
        raise RuntimeError("Kutular içinde kırmızı nokta bulunamadı.")

    # 4) Detect ruler origin and ruler rectangles
    origin_px, bottom_ruler, right_ruler, bright_mask = detect_ruler_origin(img)

    # 5) Transform origin, red points, and ruler reference points to plane
    bx, by, bw, bh = bottom_ruler
    rx, ry, rw, rh = right_ruler

    # Use two points on bottom ruler and two points on right ruler
    ruler_ref_points_px = [
        origin_px,          # 0
        *red_points_px,     # 1..N
        (bx, by),           # bottom ruler left/top point
        (bx + bw, by),      # bottom ruler right/top point
        (rx, ry),           # right ruler left/top point
        (rx, ry + rh),      # right ruler left/bottom point
    ]

    transformed = transform_points(ruler_ref_points_px, H_img_to_plane)

    origin_plane = transformed[0]
    red_points_plane = transformed[1:1 + len(red_points_px)]

    bottom_p1_plane = transformed[1 + len(red_points_px) + 0]
    bottom_p2_plane = transformed[1 + len(red_points_px) + 1]
    right_p1_plane = transformed[1 + len(red_points_px) + 2]
    right_p2_plane = transformed[1 + len(red_points_px) + 3]

    # 6) Build ruler-based basis in plane coordinates
    # +X = to the left along bottom ruler
    dir_x = normalize(bottom_p1_plane - bottom_p2_plane)

    # +Y = upward along right ruler
    dir_y = normalize(right_p1_plane - right_p2_plane)

    # Optional orthogonality check
    dot_xy = float(np.dot(dir_x, dir_y))
    print(f"Dot(dir_x, dir_y) = {dot_xy:.4f}")

    # 7) Compute ruler coordinates by projection onto ruler axes
    results = []
    for p_px, p_plane in zip(red_points_px, red_points_plane):
        vec = p_plane - origin_plane

        X_cm = float(np.dot(vec, dir_x))
        Y_cm = float(np.dot(vec, dir_y))

        results.append((p_px, p_plane, X_cm, Y_cm))

    # Sort by image position for stable printing
    results = sorted(results, key=lambda r: (int(r[0][1] // 40), r[0][0]))

    # 8) Visualization
    vis = img.copy()
    cv2.drawChessboardCorners(vis, PATTERN_SIZE, corners, True)

    # rulers
    cv2.rectangle(vis, (bx, by), (bx + bw, by + bh), (255, 255, 0), 2)
    cv2.rectangle(vis, (rx, ry), (rx + rw, ry + rh), (255, 255, 0), 2)

    ox, oy = int(round(origin_px[0])), int(round(origin_px[1]))
    cv2.circle(vis, (ox, oy), 6, (0, 0, 255), -1)
    cv2.putText(vis, "origin", (ox + 8, oy - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    # Draw box candidates
    for x, y, w, h in boxes:
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 255), 2)

    print("Detected points in ruler coordinates:")
    for i, (p_px, p_plane, X_cm, Y_cm) in enumerate(results):
        px, py = p_px

        cv2.circle(vis, (int(round(px)), int(round(py))), 4, (0, 255, 0), -1)
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

    # 9) Save outputs
    out_blue = os.path.join(OUTPUT_DIR, "blue_mask_final.png")
    out_red = os.path.join(OUTPUT_DIR, "red_mask_in_boxes.png")
    out_ruler = os.path.join(OUTPUT_DIR, "ruler_bright_mask_original.png")
    out_vis = os.path.join(OUTPUT_DIR, "red_points_ruler_coords_no_warp.png")

    cv2.imwrite(out_blue, blue_mask)
    cv2.imwrite(out_red, vis_red)
    cv2.imwrite(out_ruler, bright_mask)
    cv2.imwrite(out_vis, vis)

    print("\nSaved:")
    print(f" - {out_blue}")
    print(f" - {out_red}")
    print(f" - {out_ruler}")
    print(f" - {out_vis}")


if __name__ == "__main__":
    main()