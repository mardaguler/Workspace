import cv2
import numpy as np
import os

# =========================================================
# SETTINGS
# =========================================================

IMAGE_PATH = "/home/ozu/Desktop/Workspace/Captured/raw_2026-05-05_11-57-21.png"

OUTPUT_DIR = "/home/ozu/Desktop/Workspace/Annotated Images/RedPoint_Homography_Results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1 m x 1 m working area, in cm
# Coordinate system:
# P0 = top-left corner / origin      -> (0, 0)
# P1 = top-right corner / +X 100 cm  -> (100, 0)
# P2 = bottom-right corner           -> (100, 100)
# P3 = bottom-left corner / +Y 100 cm-> (0, 100)
WORLD_POINTS = np.array([
    [0, 0],        # P0: top-left / origin
    [100, 0],      # P1: top-right / +X 100 cm
    [100, 100],    # P2: bottom-right
    [0, 100],      # P3: bottom-left / +Y 100 cm
], dtype=np.float32)

# Red target HSV thresholds
# These red points represent detected weed centers.
LOWER_RED_1 = np.array([0, 120, 120], dtype=np.uint8)
UPPER_RED_1 = np.array([12, 255, 255], dtype=np.uint8)

LOWER_RED_2 = np.array([165, 120, 120], dtype=np.uint8)
UPPER_RED_2 = np.array([180, 255, 255], dtype=np.uint8)

# Red point area filter
MIN_RED_AREA = 5
MAX_RED_AREA = 3000

# Text drawing settings
TEXT_SCALE = 1.2
TEXT_THICKNESS = 3

clicked_points = []


# =========================================================
# MOUSE CALLBACK
# =========================================================

def mouse_callback(event, x, y, flags, param):
    global clicked_points

    if event == cv2.EVENT_LBUTTONDOWN:
        if len(clicked_points) < 4:
            clicked_points.append((x, y))
            print(f"[INFO] P{len(clicked_points)-1} selected: pixel=({x}, {y})")


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def select_calibration_points(image):
    """
    Manually select 4 calibration points with mouse.

    Selection order:
    P0: top-left corner / origin       -> (0, 0) cm
    P1: top-right corner / +X 100 cm   -> (100, 0) cm
    P2: bottom-right corner            -> (100, 100) cm
    P3: bottom-left corner / +Y 100 cm -> (0, 100) cm
    """
    global clicked_points
    clicked_points = []

    display = image.copy()

    print("\nSelect 4 calibration points in this exact order:")
    print("P0: Top-left corner / origin          -> (0, 0) cm")
    print("P1: Top-right corner / +X 100 cm      -> (100, 0) cm")
    print("P2: Bottom-right corner               -> (100, 100) cm")
    print("P3: Bottom-left corner / +Y 100 cm    -> (0, 100) cm")
    print("\nLeft click on each point.\n")

    window_name = "Select 4 calibration points"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, mouse_callback)

    while True:
        temp = display.copy()

        for i, (x, y) in enumerate(clicked_points):
            cv2.circle(temp, (x, y), 10, (0, 255, 255), -1)
            cv2.putText(
                temp,
                f"P{i}",
                (x + 12, y - 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 255),
                2
            )

        cv2.imshow(window_name, temp)

        key = cv2.waitKey(20) & 0xFF

        if len(clicked_points) == 4:
            print("[INFO] 4 calibration points selected.")
            break

        if key == 27:  # ESC
            break

    cv2.destroyWindow(window_name)

    if len(clicked_points) != 4:
        raise RuntimeError("Exactly 4 calibration points must be selected.")

    return np.array(clicked_points, dtype=np.float32)


def compute_homography(image_points, world_points):
    """
    Computes homography from image pixel coordinates to world cm coordinates.
    """
    image_points = np.array(image_points, dtype=np.float32)

    H, status = cv2.findHomography(image_points, world_points)

    if H is None:
        raise RuntimeError("Homography could not be computed.")

    return H, status


def pixel_to_cm(points_px, H):
    """
    Converts image pixel points to real-world cm coordinates using homography.

    Input:
        points_px: [(x1, y1), (x2, y2), ...]

    Output:
        points_cm: Nx2 array [[X_cm, Y_cm], ...]
    """
    pts = np.array(points_px, dtype=np.float32).reshape(-1, 1, 2)
    pts_cm = cv2.perspectiveTransform(pts, H)
    return pts_cm.reshape(-1, 2)


def build_red_mask(image):
    """
    Builds red HSV mask for fixed red points.
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


def detect_fixed_red_points(image):
    """
    Detects existing red points in the image.
    These red points are treated as detected weed centers.
    """
    red_mask = build_red_mask(image)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        red_mask,
        connectivity=8
    )

    detected_points = []
    kept_mask = np.zeros_like(red_mask)

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]

        if area < MIN_RED_AREA or area > MAX_RED_AREA:
            continue

        cx, cy = centroids[i]

        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]

        detected_points.append({
            "center": (float(cx), float(cy)),
            "area": int(area),
            "bbox": (int(x), int(y), int(w), int(h))
        })

        kept_mask[labels == i] = 255

    # Sort roughly from top-to-bottom, then left-to-right
    detected_points = sorted(
        detected_points,
        key=lambda p: (int(p["center"][1] // 40), p["center"][0])
    )

    return detected_points, red_mask, kept_mask


def draw_calibration_points(image, image_points):
    """
    Draws selected calibration points.
    """
    labels = [
        "P0 origin (0,0)",
        "P1 X100 (100,0)",
        "P2 (100,100)",
        "P3 Y100 (0,100)"
    ]

    for i, (x, y) in enumerate(image_points):
        x_i = int(round(x))
        y_i = int(round(y))

        cv2.circle(image, (x_i, y_i), 12, (0, 255, 255), -1)

        cv2.putText(
            image,
            labels[i],
            (x_i + 15, y_i - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2
        )

    polygon = image_points.reshape(-1, 1, 2).astype(np.int32)
    cv2.polylines(
        image,
        [polygon],
        isClosed=True,
        color=(0, 255, 255),
        thickness=2
    )


def draw_detected_red_results(image, detected_points, detected_points_cm):
    """
    Draws detected red weed-center points and their cm coordinates.
    """
    output = image.copy()

    for i, (point, cm_point) in enumerate(zip(detected_points, detected_points_cm)):
        px, py = point["center"]
        area = point["area"]
        x, y, w, h = point["bbox"]

        X_cm, Y_cm = cm_point

        px_i = int(round(px))
        py_i = int(round(py))

        # Draw detected center
        cv2.circle(output, (px_i, py_i), 7, (0, 255, 0), -1)

        # Draw bounding box around red component
        cv2.rectangle(
            output,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

        label = f"{i}: ({X_cm:.1f}, {Y_cm:.1f}) cm"

        text_x = px_i + 15
        text_y = py_i - 15

        # Keep text inside image
        text_x = max(10, min(text_x, output.shape[1] - 450))
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
            (text_x + 120, text_y - 30),
            (px_i, py_i),
            (0, 255, 0),
            2
        )

        print(
            f"[RED {i}] pixel=({px:.1f}, {py:.1f}), "
            f"area={area}, "
            f"cm=({X_cm:.2f}, {Y_cm:.2f})"
        )

    return output


# =========================================================
# MAIN
# =========================================================

def main():
    image = cv2.imread(IMAGE_PATH)

    if image is None:
        raise FileNotFoundError(f"Image not found: {IMAGE_PATH}")

    # -----------------------------------------------------
    # 1) Select 4 calibration points manually
    # -----------------------------------------------------
    image_points = select_calibration_points(image)

    print("\nSelected calibration points:")
    for i, p in enumerate(image_points):
        print(
            f"P{i}: pixel=({p[0]:.1f}, {p[1]:.1f}) "
            f"-> world=({WORLD_POINTS[i][0]:.1f}, {WORLD_POINTS[i][1]:.1f}) cm"
        )

    # -----------------------------------------------------
    # 2) Compute homography
    # -----------------------------------------------------
    H, status = compute_homography(image_points, WORLD_POINTS)

    print("\nHomography matrix H:")
    print(H)

    np.save(
        os.path.join(OUTPUT_DIR, "H_img_to_cm_fixed_red_points.npy"),
        H
    )

    # -----------------------------------------------------
    # 3) Detect existing fixed red points
    # -----------------------------------------------------
    detected_points, red_mask_all, red_mask_kept = detect_fixed_red_points(image)

    if len(detected_points) == 0:
        cv2.imwrite(
            os.path.join(OUTPUT_DIR, "fixed_red_points_red_mask_all.png"),
            red_mask_all
        )
        raise RuntimeError(
            "No fixed red points detected. Check fixed_red_points_red_mask_all.png."
        )

    print(f"\nDetected fixed red points: {len(detected_points)}")

    for i, point in enumerate(detected_points):
        px, py = point["center"]
        area = point["area"]
        bbox = point["bbox"]

        print(
            f"{i}: pixel=({px:.1f}, {py:.1f}), "
            f"area={area}, bbox={bbox}"
        )

    # -----------------------------------------------------
    # 4) Convert red point pixel centers to cm
    # -----------------------------------------------------
    red_points_px = [point["center"] for point in detected_points]
    red_points_cm = pixel_to_cm(red_points_px, H)

    print("\nDetected red point coordinates in cm:")
    for i, (point, cm_point) in enumerate(zip(detected_points, red_points_cm)):
        px, py = point["center"]
        X_cm, Y_cm = cm_point

        print(
            f"{i}: pixel=({px:.1f}, {py:.1f}) "
            f"-> cm=({X_cm:.2f}, {Y_cm:.2f})"
        )

    # -----------------------------------------------------
    # 5) Debug visualization
    # -----------------------------------------------------
    debug = image.copy()

    draw_calibration_points(debug, image_points)

    debug = draw_detected_red_results(
        debug,
        detected_points,
        red_points_cm
    )

    # -----------------------------------------------------
    # 6) Save outputs
    # -----------------------------------------------------
    cv2.imwrite(
        os.path.join(OUTPUT_DIR, "fixed_red_points_red_mask_all.png"),
        red_mask_all
    )

    cv2.imwrite(
        os.path.join(OUTPUT_DIR, "fixed_red_points_red_mask_kept.png"),
        red_mask_kept
    )

    cv2.imwrite(
        os.path.join(OUTPUT_DIR, "fixed_red_points_final_debug_cm.png"),
        debug
    )

    print("\nSaved outputs:")
    print(os.path.join(OUTPUT_DIR, "H_img_to_cm_fixed_red_points.npy"))
    print(os.path.join(OUTPUT_DIR, "fixed_red_points_red_mask_all.png"))
    print(os.path.join(OUTPUT_DIR, "fixed_red_points_red_mask_kept.png"))
    print(os.path.join(OUTPUT_DIR, "fixed_red_points_final_debug_cm.png"))


if __name__ == "__main__":
    main()