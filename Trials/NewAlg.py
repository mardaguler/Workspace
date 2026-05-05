import cv2
import numpy as np
import random
import os


# =========================================================
# SETTINGS
# =========================================================
IMAGE_PATH = "/home/ozu/Desktop/Workspace/Captured/raw_2026-05-04_21-05-20.png"
OUTPUT_DIR = "/home/ozu/Desktop/Workspace/Annotated Images/RedPoint_Homography_Results.png"

os.makedirs(OUTPUT_DIR, exist_ok=True)

NUM_RED_POINTS = 5
RED_DOT_RADIUS = 10

# 1 m x 1 m working area, in cm
# Setup:
# Origin is ruler intersection.
# Area extends to -X and +Y directions.
WORLD_POINTS = np.array([
    [0, 0],          # P0: origin / ruler intersection
    [100, 0],       # P1: +X direction(relatively), 100 cm point
    [100, 100],     # P2: opposite corner
    [0, 100],        # P3: +Y direction, 100 cm point
], dtype=np.float32)

# Red target HSV thresholds
LOWER_RED_1 = np.array([0, 150, 150], dtype=np.uint8)
UPPER_RED_1 = np.array([10, 255, 255], dtype=np.uint8)

LOWER_RED_2 = np.array([170, 150, 150], dtype=np.uint8)
UPPER_RED_2 = np.array([180, 255, 255], dtype=np.uint8)

MIN_RED_AREA = 20
MAX_RED_AREA = 2000

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

def compute_homography(image_points, world_points):
    """
    Computes homography from image pixel coordinates to world cm coordinates.
    """
    image_points = np.array(image_points, dtype=np.float32)

    H, status = cv2.findHomography(image_points, world_points)

    if H is None:
        raise RuntimeError("Homography could not be computed.")

    return H


def pixel_to_cm(points_px, H):
    """
    Converts image pixel points to world cm coordinates using homography.

    Input:
        points_px: [(x1, y1), (x2, y2), ...]

    Output:
        points_cm: Nx2 array [[X_cm, Y_cm], ...]
    """
    pts = np.array(points_px, dtype=np.float32).reshape(-1, 1, 2)
    pts_cm = cv2.perspectiveTransform(pts, H)
    return pts_cm.reshape(-1, 2)


def select_calibration_points(image):
    """
    Manually select 4 calibration points with mouse.

    Selection order must be:
    P0: origin / ruler intersection       -> (0, 0) cm
    P1: -X 100 cm point                   -> (-100, 0) cm
    P2: opposite corner                   -> (-100, 100) cm
    P3: +Y 100 cm point                   -> (0, 100) cm
    """
    global clicked_points
    clicked_points = []

    display = image.copy()

    print("\nSelect 4 calibration points in this exact order:")
    print("P0: Origin / ruler intersection       -> (0, 0) cm")
    print("P1: -X ruler 100 cm point             -> (-100, 0) cm")
    print("P2: Opposite corner                   -> (-100, 100) cm")
    print("P3: +Y ruler 100 cm point             -> (0, 100) cm")
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


def draw_random_red_targets(image, num_points=5, radius=10):
    """
    Draws random red target points on the image.
    These points simulate weed centers.

    Returns:
        target_points_px: original pixel coordinates of generated red points
        image_with_targets: image with red points drawn
    """
    output = image.copy()
    h, w = output.shape[:2]

    target_points_px = []

    for i in range(num_points):
        x = random.randint(radius, w - radius - 1)
        y = random.randint(radius, h - radius - 1)

        target_points_px.append((x, y))

        cv2.circle(output, (x, y), radius, (0, 0, 255), -1)
        """
        cv2.putText(
            output,
            f"T{i}",
            (x + 12, y - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2
        )
        """
        print(f"[GENERATED TARGET {i}] pixel=({x}, {y})")

    return target_points_px, output


def detect_red_targets(image):
    """
    Detects red target points from the image using HSV thresholding.
    Returns detected center points.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    mask1 = cv2.inRange(hsv, LOWER_RED_1, UPPER_RED_1)
    mask2 = cv2.inRange(hsv, LOWER_RED_2, UPPER_RED_2)
    red_mask = cv2.bitwise_or(mask1, mask2)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel, iterations=1)

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

        detected_points.append((float(cx), float(cy), int(area)))
        kept_mask[labels == i] = 255

    # Sort roughly from top-to-bottom, then left-to-right
    detected_points = sorted(detected_points, key=lambda p: (int(p[1] // 40), p[0]))

    return detected_points, red_mask, kept_mask


def draw_calibration_points(image, image_points):
    labels = [
        "P0 origin (0,0)",
        "P1 -X100 (-100,0)",
        "P2 corner (-100,100)",
        "P3 +Y100 (0,100)"
    ]

    for i, (x, y) in enumerate(image_points):
        x = int(round(x))
        y = int(round(y))

        cv2.circle(image, (x, y), 10, (0, 255, 255), -1)
        cv2.putText(
            image,
            labels[i],
            (x + 12, y - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2
        )


def draw_detected_results(image, detected_points, detected_points_cm):
    """
    Draws detected target centers and their cm coordinates.
    """
    output = image.copy()

    for i, ((px, py, area), (X_cm, Y_cm)) in enumerate(zip(detected_points, detected_points_cm)):
        x = int(round(px))
        y = int(round(py))

        cv2.circle(output, (x, y), 6, (0, 255, 0), -1)

        label = f"{i}: ({X_cm:.1f}, {Y_cm:.1f}) cm"

        text_x = min(x + 15, output.shape[1] - 300)
        text_y = max(y - 15, 25)

        cv2.putText(
            output,
            label,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,               # daha büyük yazı
            (0, 255, 0),
            3                  # daha kalın yazı
        )

        cv2.arrowedLine(
            output,
            (text_x + 90, text_y - 25),
            (x, y),
            (0, 255, 0),
            2
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
    # 1) Select 4 calibration points
    # -----------------------------------------------------
    image_points = select_calibration_points(image)

    print("\nSelected image points:")
    for i, p in enumerate(image_points):
        print(f"P{i}: pixel=({p[0]:.1f}, {p[1]:.1f}) -> world={WORLD_POINTS[i]} cm")

    # -----------------------------------------------------
    # 2) Compute homography
    # -----------------------------------------------------
    H = compute_homography(image_points, WORLD_POINTS)

    print("\nHomography matrix H:")
    print(H)

    np.save(os.path.join(OUTPUT_DIR, "H_img_to_cm.npy"), H)

    # -----------------------------------------------------
    # 3) Draw random red target points
    # -----------------------------------------------------
    generated_points_px, image_with_red = draw_random_red_targets(
        image,
        num_points=NUM_RED_POINTS,
        radius=RED_DOT_RADIUS
    )

    cv2.imwrite(os.path.join(OUTPUT_DIR, "image_with_generated_red_targets.png"), image_with_red)

    # -----------------------------------------------------
    # 4) Detect red targets from the image
    # -----------------------------------------------------
    detected_points, red_mask_all, red_mask_kept = detect_red_targets(image_with_red)

    if len(detected_points) == 0:
        raise RuntimeError("No red target points detected.")

    print("\nDetected red target points:")
    for i, (px, py, area) in enumerate(detected_points):
        print(f"{i}: pixel=({px:.1f}, {py:.1f}), area={area}")

    # -----------------------------------------------------
    # 5) Convert detected red target pixels to cm
    # -----------------------------------------------------
    detected_points_px_only = [(px, py) for (px, py, area) in detected_points]
    detected_points_cm = pixel_to_cm(detected_points_px_only, H)

    print("\nDetected target coordinates in cm:")
    for i, ((px, py, area), (X_cm, Y_cm)) in enumerate(zip(detected_points, detected_points_cm)):
        print(
            f"{i}: pixel=({px:.1f}, {py:.1f}), "
            f"area={area}, "
            f"cm=({X_cm:.2f}, {Y_cm:.2f})"
        )

    # -----------------------------------------------------
    # 6) Save debug outputs
    # -----------------------------------------------------
    debug = image_with_red.copy()
    draw_calibration_points(debug, image_points)
    debug = draw_detected_results(debug, detected_points, detected_points_cm)

    cv2.imwrite(os.path.join(OUTPUT_DIR, "red_mask_all.png"), red_mask_all)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "red_mask_kept.png"), red_mask_kept)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "final_debug_red_targets_cm.png"), debug)

    print("\nSaved outputs:")
    print(os.path.join(OUTPUT_DIR, "H_img_to_cm.npy"))
    print(os.path.join(OUTPUT_DIR, "image_with_generated_red_targets.png"))
    print(os.path.join(OUTPUT_DIR, "red_mask_all.png"))
    print(os.path.join(OUTPUT_DIR, "red_mask_kept.png"))
    print(os.path.join(OUTPUT_DIR, "final_debug_red_targets_cm.png"))


if __name__ == "__main__":
    main()