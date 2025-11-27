# https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11x.pt
# https://github.com/ultralytics/assets/releases/download/v8.3.0/mobile_sam.pt
# pip install ultralytics
from ultralytics.data.annotator import auto_annotate

auto_annotate(data="./images", 
              det_model="./models/yolo11x.pt", 
              sam_model="./models/mobile_sam.pt")

import cv2
from ultralytics import YOLO
import numpy as np

model = YOLO("./models/yolo11x.pt")
class_list = model.names

image_path = "images/image03.jpg"
label_path = "images_auto_annotate_labels/image03.txt"
image = cv2.imread(image_path)
img_height, img_width = image.shape[:2]

colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]

with open(label_path, 'r') as f:
    lines = f.readlines()

for line in lines:
    data = line.strip().split()
    class_id = int(data[0])
    # 解析多边形点坐标
    points = []
    for i in range(1, len(data), 2):
        if i + 1 < len(data):
            x = float(data[i]) * img_width
            y = float(data[i + 1]) * img_height
            points.append([int(x), int(y)])
    
    points = np.array(points, dtype=np.int32)
    color = colors[class_id % len(colors)]
    class_name = class_list[class_id]
    cv2.polylines(image, [points], isClosed=True, color=color, thickness=2)
    overlay = image.copy()
    cv2.fillPoly(overlay, [points], color)
    alpha = 0.3
    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)
    x, y, w, h = cv2.boundingRect(points)
    label = f"{class_name}"
    (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(image, (x, y - text_height - 5), (x + text_width, y), color, -1)
    cv2.putText(image, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

cv2.imshow('YOLO Auto Annotation - Segmentation', image)
cv2.waitKey(0)
cv2.destroyAllWindows()