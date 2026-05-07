import cv2
import numpy as np
import os
from pathlib import Path

# =========================================================
# SETTINGS
# =========================================================

IMAGE_PATH = "/home/ozu/Desktop/Workspace/Captured/raw_2026-05-07_17-09-13.png"

OUTPUT_DIR = "/home/ozu/Desktop/Workspace/Annotated Images/Chessboard_Homography_Results"
# Chessboard inner corner count
# If physical board has 8 squares x 6 squares,
# inner corner pattern is 7 x 5.
PATTERN_SIZE = (7, 5)

# Physical square size in cm
# 34 mm = 3.4 cm
SQUARE_SIZE_CM = 3.4

# Workspace axis lengths to draw from outer origin
X_AXIS_LENGTH_CM = 96.0
Y_AXIS_LENGTH_CM = 103.0

# =========================================================
# RED TARGET HSV THRESHOLDS
# =========================================================

LOWER_RED_1 = np.array([0, 120, 120], dtype=np.uint8)
UPPER_RED_1 = np.array([12, 255, 255], dtype=np.uint8)

LOWER_RED_2 = np.array([165, 120, 120], dtype=np.uint8)
UPPER_RED_2 = np.array([180, 255, 255], dtype=np.uint8)

MIN_RED_AREA = 50
MAX_RED_AREA = 3000

TEXT_SCALE = 1.0
TEXT_THICKNESS = 2


def resolve_image_path(configured_path):
    """
    Returns a usable image path:
    1) configured path, if it exists
    2) newest image in project directory
    """
    configured = Path(configured_path)
    if configured.exists():
        return str(configured)

    image_candidates = []
    #for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff"):
        #image_candidates.extend(PROJECT_DIR.glob(ext))

    if image_candidates:
        newest = max(image_candidates, key=lambda p: p.stat().st_mtime)
        return str(newest)
    """
    raise FileNotFoundError(
        "No input image found. Set IMAGE_PATH to a valid file or place an image "
        f"in {PROJECT_DIR}."
    )
    """


# =========================================================
# GENERAL HOMOGRAPHY HELPER
# =========================================================

def transform_points(points_px_or_cm, H):
    """
    Applies homography to 2D points.

    If H = image -> plane:
        input  = image pixel points
        output = plane cm points

    If H = plane -> image:
        input  = plane cm points
        output = image pixel points
    """
    pts = np.array(points_px_or_cm, dtype=np.float32).reshape(-1, 1, 2)
    pts_out = cv2.perspectiveTransform(pts, H)
    return pts_out.reshape(-1, 2)


# =========================================================
# RED TARGET DETECTION
# =========================================================

def build_red_mask(image):
    """
    Builds red HSV mask for the large red target point.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    mask1 = cv2.inRange(hsv, LOWER_RED_1, UPPER_RED_1)
    mask2 = cv2.inRange(hsv, LOWER_RED_2, UPPER_RED_2)

    red_mask = cv2.bitwise_or(mask1, mask2)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    red_mask = cv2.morphologyEx(
        red_mask,
        cv2.MORPH_OPEN,
        kernel,
        iterations=1
    )

    red_mask = cv2.morphologyEx(
        red_mask,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=1
    )

    return red_mask


def detect_large_red_point(image):
    """
    Detects the large red point in the image.
    This red point is treated as detected weed center.
    """
    red_mask = build_red_mask(image)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        red_mask,
        connectivity=8
    )

    detected_points = []
    min_area_candidates = []
    kept_mask = np.zeros_like(red_mask)

    print("\n[DEBUG] Connected red components:")

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]

        cx, cy = centroids[i]

        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]

        print(
            f"component {i}: center=({cx:.1f}, {cy:.1f}), "
            f"area={area}, bbox=({x}, {y}, {w}, {h})"
        )

        if area >= MIN_RED_AREA:
            min_area_candidates.append({
                "center": (float(cx), float(cy)),
                "area": int(area),
                "bbox": (int(x), int(y), int(w), int(h)),
                "label": i
            })

        if area < MIN_RED_AREA or area > MAX_RED_AREA:
            print("    -> rejected by area filter")
            continue

        detected_points.append({
            "center": (float(cx), float(cy)),
            "area": int(area),
            "bbox": (int(x), int(y), int(w), int(h)),
            "label": i
        })

        print("    -> accepted")

    if len(detected_points) == 0 and len(min_area_candidates) > 0:
        # Fallback: if upper area bound is too strict for current frame,
        # keep the largest red component above minimum area.
        largest_any = max(min_area_candidates, key=lambda p: p["area"])
        kept_mask[labels == largest_any["label"]] = 255
        print(
            "[DEBUG] No component passed MAX_RED_AREA; using "
            f"largest area >= MIN_RED_AREA ({largest_any['area']})."
        )
        print(f"[DEBUG] red_mask_all nonzero pixels: {cv2.countNonZero(red_mask)}")
        print(f"[DEBUG] red_mask_kept nonzero pixels: {cv2.countNonZero(kept_mask)}")
        return [largest_any], red_mask, kept_mask

    if len(detected_points) == 0:
        print(f"[DEBUG] red_mask_all nonzero pixels: {cv2.countNonZero(red_mask)}")
        print("[DEBUG] red_mask_kept nonzero pixels: 0")
        return [], red_mask, kept_mask

    # Keep only the largest valid red component
    detected_points = sorted(
        detected_points,
        key=lambda p: p["area"],
        reverse=True
    )

    largest = detected_points[0]
    kept_mask[labels == largest["label"]] = 255
    detected_points = [largest]

    print(f"[DEBUG] red_mask_all nonzero pixels: {cv2.countNonZero(red_mask)}")
    print(f"[DEBUG] red_mask_kept nonzero pixels: {cv2.countNonZero(kept_mask)}")

    return detected_points, red_mask, kept_mask


def save_visible_masks(red_mask_all, red_mask_kept):
    """
    Saves normal masks and visually enlarged versions.
    Visible masks are only for human inspection.
    """
    out_red_all = os.path.join(
        OUTPUT_DIR,
        "chessboard_red_mask_all.png"
    )

    out_red_kept = os.path.join(
        OUTPUT_DIR,
        "chessboard_red_mask_kept.png"
    )

    cv2.imwrite(out_red_all, red_mask_all)
    cv2.imwrite(out_red_kept, red_mask_kept)

    kernel_big = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))

    red_mask_all_visible = cv2.dilate(
        red_mask_all,
        kernel_big,
        iterations=1
    )

    red_mask_kept_visible = cv2.dilate(
        red_mask_kept,
        kernel_big,
        iterations=2
    )

    out_red_all_visible = os.path.join(
        OUTPUT_DIR,
        "chessboard_red_mask_all_visible.png"
    )

    out_red_kept_visible = os.path.join(
        OUTPUT_DIR,
        "chessboard_red_mask_kept_visible.png"
    )

    cv2.imwrite(out_red_all_visible, red_mask_all_visible)
    cv2.imwrite(out_red_kept_visible, red_mask_kept_visible)

    print("\nSaved mask outputs:")
    print(out_red_all)
    print(out_red_kept)
    print(out_red_all_visible)
    print(out_red_kept_visible)


# =========================================================
# CHESSBOARD DETECTION AND ORDERING
# =========================================================

def detect_and_order_chessboard_corners(image, pattern_size):
    """
    Detects chessboard inner corners and forces the order:

        grid[0, 0]   = top-left inner corner
        grid[0, -1]  = top-right inner corner
        grid[-1, -1] = bottom-right inner corner
        grid[-1, 0]  = bottom-left inner corner

    Output:
        corrected_corners: OpenCV corner format, ordered
        img_pts: Nx2 image points, ordered
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    ret = False
    corners = None

    # Try stronger SB method first
    try:
        flags_sb = cv2.CALIB_CB_NORMALIZE_IMAGE
        ret, corners = cv2.findChessboardCornersSB(
            gray,
            pattern_size,
            flags_sb
        )
    except Exception:
        ret = False
        corners = None

    # Fallback to classic method
    if not ret:
        flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE

        ret, corners = cv2.findChessboardCorners(
            gray,
            pattern_size,
            flags
        )

        if ret:
            criteria = (
                cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                30,
                0.001
            )

            corners = cv2.cornerSubPix(
                gray,
                corners,
                (11, 11),
                (-1, -1),
                criteria
            )

    if not ret:
        raise RuntimeError(
            "Chessboard corners could not be detected. "
            "Check PATTERN_SIZE, lighting, focus, and board visibility."
        )

    corners = corners.astype(np.float32)

    cols = pattern_size[0]
    rows = pattern_size[1]

    corners_grid = corners.reshape(rows, cols, 2)

    # Force top-left inner corner as grid[0,0]
    if corners_grid[0, :, 1].mean() > corners_grid[-1, :, 1].mean():
        corners_grid = corners_grid[::-1, :, :]

    if corners_grid[:, 0, 0].mean() > corners_grid[:, -1, 0].mean():
        corners_grid = corners_grid[:, ::-1, :]

    img_pts = corners_grid.reshape(-1, 2).astype(np.float32)
    corrected_corners = img_pts.reshape(-1, 1, 2)

    return corrected_corners, img_pts


# =========================================================
# ORIGIN SHIFT USING PIXEL VECTORS
# =========================================================

def compute_outer_origin_from_inner_corner_pixels(img_pts, pattern_size, square_size_cm):
    """
    Computes the outer top-left chessboard origin using pixel vectors.

    First:
        top-left inner corner is taken as temporary origin.

    Then:
        outer origin = inner origin - one square in X pixel vector
                                    - one square in Y pixel vector

    This explicitly uses the fact that:
        one chessboard square = square_size_cm = 3.4 cm

    The pixel/cm values are calculated from the local first square near
    the top-left inner corner.
    """

    cols = pattern_size[0]
    rows = pattern_size[1]

    grid = img_pts.reshape(rows, cols, 2)

    inner_origin_px = grid[0, 0]

    # One chessboard square in pixel vector form near the origin
    x_square_vec_px = grid[0, 1] - grid[0, 0]
    y_square_vec_px = grid[1, 0] - grid[0, 0]

    x_square_px = float(np.linalg.norm(x_square_vec_px))
    y_square_px = float(np.linalg.norm(y_square_vec_px))

    px_per_cm_x = x_square_px / square_size_cm
    px_per_cm_y = y_square_px / square_size_cm

    # Outer top-left corner is one square left and one square up
    outer_origin_px = inner_origin_px - x_square_vec_px - y_square_vec_px

    return {
        "inner_origin_px": inner_origin_px,
        "outer_origin_px": outer_origin_px,
        "x_square_vec_px": x_square_vec_px,
        "y_square_vec_px": y_square_vec_px,
        "x_square_px": x_square_px,
        "y_square_px": y_square_px,
        "px_per_cm_x": px_per_cm_x,
        "px_per_cm_y": px_per_cm_y
    }


def estimate_average_px_per_cm_from_chessboard(img_pts, pattern_size, square_size_cm):
    """
    Estimates average local pixel/cm scale using all adjacent chessboard corners.

    This is only an approximate local value because perspective changes
    pixel/cm across the image.
    """
    cols = pattern_size[0]
    rows = pattern_size[1]

    grid = img_pts.reshape(rows, cols, 2)

    dx_list = []
    for r in range(rows):
        for c in range(cols - 1):
            p1 = grid[r, c]
            p2 = grid[r, c + 1]
            dx_list.append(np.linalg.norm(p2 - p1))

    dy_list = []
    for r in range(rows - 1):
        for c in range(cols):
            p1 = grid[r, c]
            p2 = grid[r + 1, c]
            dy_list.append(np.linalg.norm(p2 - p1))

    px_per_square_x_avg = float(np.mean(dx_list))
    px_per_square_y_avg = float(np.mean(dy_list))

    px_per_cm_x_avg = px_per_square_x_avg / square_size_cm
    px_per_cm_y_avg = px_per_square_y_avg / square_size_cm

    return {
        "px_per_square_x_avg": px_per_square_x_avg,
        "px_per_square_y_avg": px_per_square_y_avg,
        "px_per_cm_x_avg": px_per_cm_x_avg,
        "px_per_cm_y_avg": px_per_cm_y_avg
    }


# =========================================================
# HOMOGRAPHY WITH OUTER ORIGIN
# =========================================================

def build_outer_origin_object_points(pattern_size, square_size_cm):
    """
    Builds chessboard plane coordinates using OUTER top-left chessboard corner as origin.

    Coordinate system:
        outer top-left corner = (0, 0) cm
        +X = right
        +Y = down

    OpenCV detects INNER corners, not outer corners.

    Therefore:
        inner corner (i, j) -> ((i + 1) * S, (j + 1) * S)

    because the first inner corner is one square right and one square down
    from the outer top-left corner.
    """
    cols = pattern_size[0]
    rows = pattern_size[1]

    obj_pts_outer = []

    for j in range(rows):
        for i in range(cols):
            obj_pts_outer.append([
                (i + 1) * square_size_cm,
                (j + 1) * square_size_cm
            ])

    return np.array(obj_pts_outer, dtype=np.float32)


def compute_homography_outer_origin(img_pts, pattern_size, square_size_cm):
    """
    Computes homography directly with OUTER top-left chessboard corner as origin.

    H_outer maps:
        image pixel -> outer-origin chessboard cm coordinate
    """
    obj_pts_outer = build_outer_origin_object_points(
        pattern_size,
        square_size_cm
    )

    H_outer, status = cv2.findHomography(
        img_pts,
        obj_pts_outer
    )

    if H_outer is None:
        raise RuntimeError("Homography could not be computed from chessboard points.")

    return H_outer, obj_pts_outer, status


# =========================================================
# 94 CM X AND 100 CM Y AXIS EXTENSION
# =========================================================

def draw_workspace_axes_from_outer_origin(image, origin_info):
    """
    Draws workspace axes from the outer top-left chessboard origin.

    +X axis length = 94 cm
    +Y axis length = 100 cm

    Uses chessboard pixel vectors:
        x_square_vec_px corresponds to 3.4 cm in +X
        y_square_vec_px corresponds to 3.4 cm in +Y
    """
    output = image.copy()

    outer_origin_px = origin_info["outer_origin_px"]
    x_square_vec_px = origin_info["x_square_vec_px"]
    y_square_vec_px = origin_info["y_square_vec_px"]

    x_scale = X_AXIS_LENGTH_CM / SQUARE_SIZE_CM
    y_scale = Y_AXIS_LENGTH_CM / SQUARE_SIZE_CM

    x_axis_vec_px = x_square_vec_px * x_scale
    y_axis_vec_px = y_square_vec_px * y_scale

    x_axis_end_px = outer_origin_px + x_axis_vec_px
    y_axis_end_px = outer_origin_px + y_axis_vec_px

    origin_i = tuple(np.round(outer_origin_px).astype(int))
    x_end_i = tuple(np.round(x_axis_end_px).astype(int))
    y_end_i = tuple(np.round(y_axis_end_px).astype(int))

    cv2.circle(output, origin_i, 14, (0, 0, 255), -1)

    cv2.arrowedLine(
        output,
        origin_i,
        x_end_i,
        (0, 255, 0),
        4
    )

    cv2.arrowedLine(
        output,
        origin_i,
        y_end_i,
        (255, 0, 0),
        4
    )

    cv2.putText(
        output,
        "Origin (0,0)",
        (origin_i[0] + 15, origin_i[1] - 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 0, 255),
        2
    )

    cv2.putText(
        output,
        f"+X {X_AXIS_LENGTH_CM:.0f} cm",
        (x_end_i[0] + 15, x_end_i[1]),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 0),
        2
    )

    cv2.putText(
        output,
        f"+Y {Y_AXIS_LENGTH_CM:.0f} cm",
        (y_end_i[0] + 15, y_end_i[1]),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 0, 0),
        2
    )

    info = (
        f"{X_AXIS_LENGTH_CM:.0f} cm X = {np.linalg.norm(x_axis_vec_px):.1f}px, "
        f"{Y_AXIS_LENGTH_CM:.0f} cm Y = {np.linalg.norm(y_axis_vec_px):.1f}px"
    )

    cv2.putText(
        output,
        info,
        (30, output.shape[0] - 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2
    )

    out_path = os.path.join(
        OUTPUT_DIR,
        f"outer_origin_{int(X_AXIS_LENGTH_CM)}x{int(Y_AXIS_LENGTH_CM)}cm_axes.png"
    )

    cv2.imwrite(out_path, output)

    print(
        f"[INFO] Saved {X_AXIS_LENGTH_CM:.0f} cm X / "
        f"{Y_AXIS_LENGTH_CM:.0f} cm Y axis debug image: {out_path}"
    )
    print(f"[INFO] X {X_AXIS_LENGTH_CM:.0f} cm vector length = {np.linalg.norm(x_axis_vec_px):.2f} px")
    print(f"[INFO] Y {Y_AXIS_LENGTH_CM:.0f} cm vector length = {np.linalg.norm(y_axis_vec_px):.2f} px")


# =========================================================
# VISUALIZATION
# =========================================================

def draw_inner_origin_debug(image, corners, img_pts, pattern_size):
    """
    Saves debug image showing the top-left inner corner as temporary origin.

    Output:
        leftmost_origin.png
    """
    output = image.copy()

    cv2.drawChessboardCorners(
        output,
        pattern_size,
        corners,
        True
    )

    cols = pattern_size[0]
    rows = pattern_size[1]

    grid = img_pts.reshape(rows, cols, 2)

    inner_origin_px = grid[0, 0]
    x_axis_px = grid[0, -1]
    y_axis_px = grid[-1, 0]

    inner_origin_px_int = tuple(np.round(inner_origin_px).astype(int))
    x_axis_px_int = tuple(np.round(x_axis_px).astype(int))
    y_axis_px_int = tuple(np.round(y_axis_px).astype(int))

    cv2.circle(output, inner_origin_px_int, 12, (0, 255, 255), -1)

    cv2.arrowedLine(
        output,
        inner_origin_px_int,
        x_axis_px_int,
        (0, 255, 0),
        3
    )

    cv2.arrowedLine(
        output,
        inner_origin_px_int,
        y_axis_px_int,
        (255, 0, 0),
        3
    )

    cv2.putText(
        output,
        "TEMP INNER ORIGIN (0,0)",
        (inner_origin_px_int[0] + 10, inner_origin_px_int[1] - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2
    )

    cv2.putText(
        output,
        "+X",
        (x_axis_px_int[0] + 10, x_axis_px_int[1]),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.putText(
        output,
        "+Y",
        (y_axis_px_int[0] + 10, y_axis_px_int[1]),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 0, 0),
        2
    )

    out_path = os.path.join(OUTPUT_DIR, "leftmost_origin.png")
    cv2.imwrite(out_path, output)

    print(f"[INFO] Saved temporary inner-origin debug image: {out_path}")


def draw_outer_origin_shift_debug(image, corners, img_pts, origin_info):
    """
    Saves debug image showing how the origin is shifted from inner corner
    to outer top-left chessboard corner using pixel vectors.

    Output:
        outer_origin_shift_debug.png
    """
    output = image.copy()

    cv2.drawChessboardCorners(
        output,
        PATTERN_SIZE,
        corners,
        True
    )

    inner_origin_px = origin_info["inner_origin_px"]
    outer_origin_px = origin_info["outer_origin_px"]
    x_square_vec_px = origin_info["x_square_vec_px"]
    y_square_vec_px = origin_info["y_square_vec_px"]

    inner_origin_i = tuple(np.round(inner_origin_px).astype(int))
    outer_origin_i = tuple(np.round(outer_origin_px).astype(int))

    # Intermediate points:
    # one square left from inner origin
    one_left_px = inner_origin_px - x_square_vec_px
    one_up_px = inner_origin_px - y_square_vec_px

    one_left_i = tuple(np.round(one_left_px).astype(int))
    one_up_i = tuple(np.round(one_up_px).astype(int))

    cv2.circle(output, inner_origin_i, 12, (0, 255, 255), -1)
    cv2.circle(output, outer_origin_i, 12, (0, 0, 255), -1)

    cv2.circle(output, one_left_i, 8, (255, 0, 255), -1)
    cv2.circle(output, one_up_i, 8, (255, 0, 255), -1)

    cv2.arrowedLine(
        output,
        inner_origin_i,
        one_left_i,
        (255, 0, 255),
        3
    )

    cv2.arrowedLine(
        output,
        one_left_i,
        outer_origin_i,
        (255, 0, 255),
        3
    )

    cv2.putText(
        output,
        "INNER ORIGIN",
        (inner_origin_i[0] + 10, inner_origin_i[1] - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2
    )

    cv2.putText(
        output,
        "OUTER ORIGIN (0,0)",
        (outer_origin_i[0] + 10, outer_origin_i[1] - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 255),
        2
    )

    info1 = f"1 square X = {origin_info['x_square_px']:.2f} px = {SQUARE_SIZE_CM:.1f} cm"
    info2 = f"1 square Y = {origin_info['y_square_px']:.2f} px = {SQUARE_SIZE_CM:.1f} cm"

    cv2.putText(
        output,
        info1,
        (30, output.shape[0] - 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 255, 255),
        2
    )

    cv2.putText(
        output,
        info2,
        (30, output.shape[0] - 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 255, 255),
        2
    )

    out_path = os.path.join(OUTPUT_DIR, "outer_origin_shift_debug.png")
    cv2.imwrite(out_path, output)

    print(f"[INFO] Saved origin-shift debug image: {out_path}")


def draw_outer_origin_final_debug(
    image,
    corners,
    H_outer,
    origin_info,
    detected_points,
    red_points_outer_cm,
    average_scale_info
):
    """
    Draws final debug image using outer top-left chessboard corner as origin.
    """
    output = image.copy()

    cv2.drawChessboardCorners(
        output,
        PATTERN_SIZE,
        corners,
        True
    )

    H_cm_to_img = np.linalg.inv(H_outer)

    # For visual workspace axes:
    outer_origin_cm = np.array([[0, 0]], dtype=np.float32)
    outer_x_axis_cm = np.array([[X_AXIS_LENGTH_CM, 0]], dtype=np.float32)
    outer_y_axis_cm = np.array([[0, Y_AXIS_LENGTH_CM]], dtype=np.float32)

    outer_origin_px = origin_info["outer_origin_px"]

    outer_x_axis_px = transform_points(outer_x_axis_cm, H_cm_to_img)[0]
    outer_y_axis_px = transform_points(outer_y_axis_cm, H_cm_to_img)[0]

    outer_origin_px_int = tuple(np.round(outer_origin_px).astype(int))
    outer_x_axis_px_int = tuple(np.round(outer_x_axis_px).astype(int))
    outer_y_axis_px_int = tuple(np.round(outer_y_axis_px).astype(int))

    cv2.circle(output, outer_origin_px_int, 12, (0, 0, 255), -1)

    cv2.putText(
        output,
        "OUTER ORIGIN (0,0)",
        (outer_origin_px_int[0] + 10, outer_origin_px_int[1] - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 255),
        2
    )

    cv2.arrowedLine(
        output,
        outer_origin_px_int,
        outer_x_axis_px_int,
        (0, 255, 0),
        3
    )

    cv2.arrowedLine(
        output,
        outer_origin_px_int,
        outer_y_axis_px_int,
        (255, 0, 0),
        3
    )

    cv2.putText(
        output,
        f"+X {X_AXIS_LENGTH_CM:.0f} cm",
        (outer_x_axis_px_int[0] + 10, outer_x_axis_px_int[1]),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.putText(
        output,
        f"+Y {Y_AXIS_LENGTH_CM:.0f} cm",
        (outer_y_axis_px_int[0] + 10, outer_y_axis_px_int[1]),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 0, 0),
        2
    )

    # Draw detected red point
    for i, (point, outer_cm) in enumerate(
        zip(detected_points, red_points_outer_cm)
    ):
        px, py = point["center"]
        x, y, w, h = point["bbox"]

        X_outer, Y_outer = outer_cm

        px_i = int(round(px))
        py_i = int(round(py))

        cv2.circle(output, (px_i, py_i), 8, (0, 255, 0), -1)

        cv2.rectangle(
            output,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

        label = f"Weed {i}: ({X_outer:.1f}, {Y_outer:.1f}) cm"

        text_x = px_i + 15
        text_y = py_i - 15

        text_x = max(10, min(text_x, output.shape[1] - 520))
        text_y = max(40, min(text_y, output.shape[0] - 20))

        cv2.putText(
            output,
            label,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            TEXT_SCALE,
            (0, 255, 0),
            TEXT_THICKNESS
        )

        cv2.arrowedLine(
            output,
            (text_x + 150, text_y - 30),
            (px_i, py_i),
            (0, 255, 0),
            2
        )

    info_text_1 = (
        f"Workspace axes: X={X_AXIS_LENGTH_CM:.0f} cm, "
        f"Y={Y_AXIS_LENGTH_CM:.0f} cm"
    )

    info_text_2 = (
        f"Local px/cm: X={origin_info['px_per_cm_x']:.2f}, "
        f"Y={origin_info['px_per_cm_y']:.2f}"
    )

    info_text_3 = (
        f"Avg px/cm: X={average_scale_info['px_per_cm_x_avg']:.2f}, "
        f"Y={average_scale_info['px_per_cm_y_avg']:.2f}"
    )

    cv2.putText(
        output,
        info_text_1,
        (30, output.shape[0] - 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 255),
        2
    )

    cv2.putText(
        output,
        info_text_2,
        (30, output.shape[0] - 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 255),
        2
    )

    cv2.putText(
        output,
        info_text_3,
        (30, output.shape[0] - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 255),
        2
    )

    out_path = os.path.join(
        OUTPUT_DIR,
        "chessboard_outer_origin_final_debug_cm.png"
    )

    cv2.imwrite(out_path, output)

    print(f"[INFO] Saved final outer-origin debug image: {out_path}")


# =========================================================
# MAIN
# =========================================================

def main():
    image_path = resolve_image_path(IMAGE_PATH)
    print(f"[INFO] Using input image: {image_path}")
    image = cv2.imread(image_path)

    if image is None:
        raise FileNotFoundError(f"Image could not be read: {image_path}")

    # -----------------------------------------------------
    # 1) Detect large red weed point first in pixel coordinates
    # -----------------------------------------------------
    detected_points, red_mask_all, red_mask_kept = detect_large_red_point(image)

    save_visible_masks(red_mask_all, red_mask_kept)

    if len(detected_points) == 0:
        raise RuntimeError(
            "No large red point detected. "
            "Check chessboard_red_mask_all.png and chessboard_red_mask_kept_visible.png."
        )

    red_points_px = [point["center"] for point in detected_points]

    print("\nDetected red weed point in pixel coordinates:")
    for i, point in enumerate(detected_points):
        px, py = point["center"]
        area = point["area"]

        print(
            f"{i}: pixel=({px:.2f}, {py:.2f}), area={area}"
        )

    # -----------------------------------------------------
    # 2) Detect and order chessboard inner corners
    # -----------------------------------------------------
    corners, img_pts = detect_and_order_chessboard_corners(
        image,
        PATTERN_SIZE
    )

    # -----------------------------------------------------
    # 3) Save temporary inner-origin debug image
    # -----------------------------------------------------
    draw_inner_origin_debug(
        image,
        corners,
        img_pts,
        PATTERN_SIZE
    )

    # -----------------------------------------------------
    # 4) Compute 3.4 cm -> pixel vectors and move origin
    #    from top-left inner corner to top-left outer corner
    # -----------------------------------------------------
    origin_info = compute_outer_origin_from_inner_corner_pixels(
        img_pts,
        PATTERN_SIZE,
        SQUARE_SIZE_CM
    )

    print("\nOrigin shift using chessboard pixel vectors:")
    print(
        f"Inner top-left corner pixel = "
        f"({origin_info['inner_origin_px'][0]:.2f}, {origin_info['inner_origin_px'][1]:.2f})"
    )
    print(
        f"Outer top-left corner pixel = "
        f"({origin_info['outer_origin_px'][0]:.2f}, {origin_info['outer_origin_px'][1]:.2f})"
    )
    print(f"1 square X direction = {origin_info['x_square_px']:.2f} px")
    print(f"1 square Y direction = {origin_info['y_square_px']:.2f} px")
    print(f"px/cm X = {origin_info['px_per_cm_x']:.2f}")
    print(f"px/cm Y = {origin_info['px_per_cm_y']:.2f}")

    draw_outer_origin_shift_debug(
        image,
        corners,
        img_pts,
        origin_info
    )

    # -----------------------------------------------------
    # 5) Draw 94 cm X and 100 cm Y axis arrows using local px/cm vectors
    # -----------------------------------------------------
    draw_workspace_axes_from_outer_origin(
        image,
        origin_info
    )

    # -----------------------------------------------------
    # 6) Build homography directly with OUTER top-left origin
    # -----------------------------------------------------
    H_outer, obj_pts_outer, status = compute_homography_outer_origin(
        img_pts,
        PATTERN_SIZE,
        SQUARE_SIZE_CM
    )

    print("\nH_outer: image pixel -> outer-origin chessboard cm")
    print(H_outer)

    np.save(
        os.path.join(OUTPUT_DIR, "H_img_to_chessboard_outer_cm.npy"),
        H_outer
    )

    np.savetxt(
        os.path.join(OUTPUT_DIR, "H_img_to_chessboard_outer_cm.txt"),
        H_outer
    )

    # -----------------------------------------------------
    # 7) Convert red weed pixel point to OUTER-origin cm
    # -----------------------------------------------------
    red_points_outer_cm = transform_points(
        red_points_px,
        H_outer
    )

    print("\nDetected weed coordinates relative to OUTER chessboard origin:")
    for i, (point, outer_cm) in enumerate(
        zip(detected_points, red_points_outer_cm)
    ):
        px, py = point["center"]
        X_outer, Y_outer = outer_cm

        print(
            f"{i}: pixel=({px:.2f}, {py:.2f}) "
            f"-> outer_origin_cm=({X_outer:.2f}, {Y_outer:.2f})"
        )

    # -----------------------------------------------------
    # 8) Estimate average px/cm using all chessboard intervals
    # -----------------------------------------------------
    average_scale_info = estimate_average_px_per_cm_from_chessboard(
        img_pts,
        PATTERN_SIZE,
        SQUARE_SIZE_CM
    )

    print("\nAverage approximate scale near chessboard:")
    print(f"avg px_per_square_x = {average_scale_info['px_per_square_x_avg']:.2f} px")
    print(f"avg px_per_square_y = {average_scale_info['px_per_square_y_avg']:.2f} px")
    print(f"avg px_per_cm_x     = {average_scale_info['px_per_cm_x_avg']:.2f} px/cm")
    print(f"avg px_per_cm_y     = {average_scale_info['px_per_cm_y_avg']:.2f} px/cm")

    with open(os.path.join(OUTPUT_DIR, "chessboard_scale_info.txt"), "w") as f:
        f.write("Origin shift using local top-left square vectors\n")
        f.write(f"inner_origin_px = {origin_info['inner_origin_px']}\n")
        f.write(f"outer_origin_px = {origin_info['outer_origin_px']}\n")
        f.write(f"x_square_vec_px = {origin_info['x_square_vec_px']}\n")
        f.write(f"y_square_vec_px = {origin_info['y_square_vec_px']}\n")
        f.write(f"x_square_px = {origin_info['x_square_px']:.4f} px\n")
        f.write(f"y_square_px = {origin_info['y_square_px']:.4f} px\n")
        f.write(f"local px_per_cm_x = {origin_info['px_per_cm_x']:.4f} px/cm\n")
        f.write(f"local px_per_cm_y = {origin_info['px_per_cm_y']:.4f} px/cm\n")
        f.write("\nWorkspace axis lengths\n")
        f.write(f"X_AXIS_LENGTH_CM = {X_AXIS_LENGTH_CM:.4f} cm\n")
        f.write(f"Y_AXIS_LENGTH_CM = {Y_AXIS_LENGTH_CM:.4f} cm\n")
        f.write("\nAverage scale using all adjacent chessboard intervals\n")
        f.write(f"avg px_per_square_x = {average_scale_info['px_per_square_x_avg']:.4f} px\n")
        f.write(f"avg px_per_square_y = {average_scale_info['px_per_square_y_avg']:.4f} px\n")
        f.write(f"avg px_per_cm_x = {average_scale_info['px_per_cm_x_avg']:.4f} px/cm\n")
        f.write(f"avg px_per_cm_y = {average_scale_info['px_per_cm_y_avg']:.4f} px/cm\n")
        f.write("\nNote: px/cm is only an approximate local value because perspective changes scale across the image.\n")

    # -----------------------------------------------------
    # 9) Save final outer-origin debug image
    # -----------------------------------------------------
    draw_outer_origin_final_debug(
        image=image,
        corners=corners,
        H_outer=H_outer,
        origin_info=origin_info,
        detected_points=detected_points,
        red_points_outer_cm=red_points_outer_cm,
        average_scale_info=average_scale_info
    )

    # -----------------------------------------------------
    # 10) Save chessboard corners debug
    # -----------------------------------------------------
    corners_debug = image.copy()

    cv2.drawChessboardCorners(
        corners_debug,
        PATTERN_SIZE,
        corners,
        True
    )

    cv2.imwrite(
        os.path.join(OUTPUT_DIR, "chessboard_corners_debug.png"),
        corners_debug
    )

    print("\nSaved outputs in:")
    print(OUTPUT_DIR)
    print("\nImportant output files:")
    print(os.path.join(OUTPUT_DIR, "leftmost_origin.png"))
    print(os.path.join(OUTPUT_DIR, "outer_origin_shift_debug.png"))
    print(
        os.path.join(
            OUTPUT_DIR,
            f"outer_origin_{int(X_AXIS_LENGTH_CM)}x{int(Y_AXIS_LENGTH_CM)}cm_axes.png"
        )
    )
    print(os.path.join(OUTPUT_DIR, "chessboard_outer_origin_final_debug_cm.png"))
    print(os.path.join(OUTPUT_DIR, "H_img_to_chessboard_outer_cm.npy"))
    print(os.path.join(OUTPUT_DIR, "chessboard_scale_info.txt"))


if __name__ == "__main__":
    main()