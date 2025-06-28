import cv2
import numpy as np

# Define a color palette (BGR)
PALETTE = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255),
    (255, 255, 0), (255, 0, 255), (0, 255, 255),
    (128, 0, 0), (0, 128, 0), (0, 0, 128),
    (128, 128, 0), (128, 0, 128), (0, 128, 128),
    (64, 64, 64), (192, 192, 192), (0, 0, 0)
]

def draw_labels_on_image(img, label_path, class_names):
    """Draw closed contours with different colors for each label"""
    try:
        with open(label_path, 'r') as f:
            lines = f.readlines()

        h, w = img.shape[:2]

        for line in lines:
            parts = line.strip().split()
            class_id = int(parts[0])
            color = PALETTE[class_id % len(PALETTE)]

            # Collect all points for this contour
            points = []
            for i in range(1, len(parts), 2):
                x = float(parts[i]) * w
                y = float(parts[i+1]) * h
                points.append([int(x), int(y)])

            if len(points) > 1:
                pts = np.array(points, np.int32).reshape((-1, 1, 2))
                cv2.polylines(img, [pts], isClosed=True, color=color, thickness=2)

                # Draw class name at the first point
                class_name = class_names[class_id] if class_id < len(class_names) else f"Class {class_id}"
                cv2.putText(
                    img, str(class_name), (points[0][0], points[0][1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA
                )

    except Exception as e:
        print(f"Error drawing labels: {e}")

    return img