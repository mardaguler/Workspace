import cv2
import numpy as np
import random
import os

# =========================================================
# SETTINGS
# =========================================================
IMAGE_PATH = "/home/ozu/Desktop/Workspace/Captured/raw_2026-04-22_16-18-11.png"
OUTPUT_PATH = "/home/ozu/Desktop/Workspace/Annotated Images/red_points_cm_debug.png"

NUM_RED_POINTS = 5
RED_DOT_RADIUS = 10

# Real-world coordinates in cm
# Setup: origin at ruler intersection, area extends to -X and +Y
WORLD_POINTS = np.array([
    [0, 0],          # P0: origin / ruler intersection
    [-100, 0],       # P1: -X ruler 100 cm point
    [-100, 100],     # P2: opposite corner
    [0, 100],        # P3: +Y ruler 100 cm point
], dtype=np.float32)

clicked_points = []


# =========================================================
# HELPERS
# =========================================================
def mouse_callback(event, x, y, flags, param):
    global clicked_points

    if event == cv2.EVENT_LBUTTONDOWN:
        if len(clicked_points) < 4:
            clicked_points.append((x, y))
            print(f"[INFO] Point {len(clicked_points)} selected: pixel=({x}, {y})")


def pixel_to_cm(point_px, H):
    """
    Converts one image pixel point to world cm coordinate using homography.
    """
    pt = np.array([[[point_px[0], point_px[1]]]], dtype=np.float32)
    pt_cm = cv2.perspectiveTransform(pt, H)[0][0]
    return float(pt_cm[0]), float(pt_cm[1])


def draw_reference_points(image, image_points):
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


def mark_random_red_point(image, H, index):
    """
    Puts one random red point on image and writes its cm coordinate.
    """
    height, width = image.shape[:2]

    x = random.randint(0, width - 1)
    y = random.randint(0, height - 1)

    X_cm, Y_cm = pixel_to_cm((x, y), H)

    print(
        f"[RED {index}] pixel=({x}, {y}) -> "
        f"cm=({X_cm:.2f}, {Y_cm:.2f})"
    )

    cv2.circle(image, (x, y), RED_DOT_RADIUS, (0, 0, 255), -1)

    label = f"{index}: ({X_cm:.1f}, {Y_cm:.1f}) cm"
    text_x = min(x + 15, width - 260)
    text_y = max(y - 15, 25)

    cv2.putText(
        image,
        label,
        (text_x, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 255),
        2
    )

    cv2.arrowedLine(
        image,
        (text_x + 80, text_y - 25),
        (x, y),
        (0, 0, 255),
        2
    )

    return (x, y), (X_cm, Y_cm)


# =========================================================
# MAIN
# =========================================================
def main():
    image = cv2.imread(IMAGE_PATH)

    if image is None:
        raise FileNotFoundError(f"Image not found: {IMAGE_PATH}")

    display = image.copy()

    print("\nSelect 4 calibration points in this exact order:")
    print("1) Origin / ruler intersection       -> (0, 0) cm")
    print("2) -X ruler 100 cm point             -> (-100, 0) cm")
    print("3) Opposite corner                   -> (-100, 100) cm")
    print("4) +Y ruler 100 cm point             -> (0, 100) cm")
    print("\nLeft click points. Press ESC after selecting 4 points.\n")

    cv2.namedWindow("Select 4 calibration points", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Select 4 calibration points", mouse_callback)

    while True:
        temp = display.copy()

        for i, (x, y) in enumerate(clicked_points):
            cv2.circle(temp, (x, y), 8, (0, 255, 255), -1)
            cv2.putText(
                temp,
                f"P{i}",
                (x + 10, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2
            )

        cv2.imshow("Select 4 calibration points", temp)

        key = cv2.waitKey(20) & 0xFF

        if key == 27:  # ESC
            break

        if len(clicked_points) == 4:
            print("[INFO] 4 points selected.")
            break

    cv2.destroyAllWindows()

    if len(clicked_points) != 4:
        raise RuntimeError("You must select exactly 4 calibration points.")

    image_points = np.array(clicked_points, dtype=np.float32)

    # Homography: image pixel -> world cm
    H, status = cv2.findHomography(image_points, WORLD_POINTS)

    if H is None:
        raise RuntimeError("Homography could not be computed.")

    print("\nHomography matrix H:")
    print(H)

    # Debug visualization
    result = image.copy()

    draw_reference_points(result, image_points)

    print("\nRandom red point results:")
    for i in range(NUM_RED_POINTS):
        mark_random_red_point(result, H, i)

    cv2.imwrite(OUTPUT_PATH, result)

    print(f"\n[INFO] Saved debug image:")
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()