import cv2
import numpy as np

# --------------------------------------------------
# Yardımcı fonksiyonlar
# --------------------------------------------------

def order_boxes_reading_order(boxes):
    # yukarıdan aşağıya, soldan sağa kaba sıralama
    return sorted(boxes, key=lambda b: (b[1] // 50, b[0]))

def find_blue_boxes(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Mavi kutular için yaklaşık HSV aralığı
    lower_blue = np.array([100, 120, 80])
    upper_blue = np.array([130, 255, 255])

    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # Yazıların bazı parçalarını azaltmak için morphology
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h

        # Gürültü eleme
        if area < 150:
            continue
        if w < 10 or h < 10:
            continue

        # Çok uzun yazı satırlarını biraz elemek için
        # ama yine de uzun bbox olabilir, çok sert kesmeyelim
        aspect = w / float(h)
        if aspect > 15:
            continue

        boxes.append((x, y, w, h))

    return order_boxes_reading_order(boxes), mask

def find_red_dot_in_box(image, box):
    x, y, w, h = box

    # Kutu içine biraz margin ver
    pad = 4
    x0 = max(0, x - pad)
    y0 = max(0, y - pad)
    x1 = min(image.shape[1], x + w + pad)
    y1 = min(image.shape[0], y + h + pad)

    roi = image[y0:y1, x0:x1]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # Kırmızı HSV iki aralık ister
    lower_red1 = np.array([0, 120, 80])
    upper_red1 = np.array([10, 255, 255])

    lower_red2 = np.array([170, 120, 80])
    upper_red2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best = None
    best_area = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 3:
            continue
        if area > 300:
            continue

        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue

        cx = int(M["m10"] / M["m00"]) + x0
        cy = int(M["m01"] / M["m00"]) + y0

        if area > best_area:
            best_area = area
            best = (cx, cy)

    return best

def bbox_center(box):
    x, y, w, h = box
    return (int(x + w / 2), int(y + h / 2))

# --------------------------------------------------
# Ana işlem
# --------------------------------------------------

image_path = "annotated.png"   # burayı değiştir
img = cv2.imread(image_path)
vis = img.copy()

boxes, blue_mask = find_blue_boxes(img)

points_px = []

for i, box in enumerate(boxes):
    x, y, w, h = box

    red_pt = find_red_dot_in_box(img, box)
    if red_pt is not None:
        px, py = red_pt
        source_type = "red-dot"
    else:
        px, py = bbox_center(box)
        source_type = "bbox-center"

    points_px.append((px, py))

    # çiz
    cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 255), 2)
    cv2.circle(vis, (px, py), 5, (0, 0, 255), -1)
    cv2.putText(vis, f"{i}: {source_type}", (x, max(20, y - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

print("Bulunan pixel noktaları:")
for i, p in enumerate(points_px):
    print(f"{i}: {p}")

cv2.imwrite("detected_boxes_and_points.png", vis)
cv2.imwrite("blue_mask.png", blue_mask)
print("Kaydedildi: detected_boxes_and_points.png, blue_mask.png")