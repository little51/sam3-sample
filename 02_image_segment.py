import torch
from PIL import Image
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.visualization_utils import draw_box_on_image

# 装载模型
model = build_sam3_image_model(
    checkpoint_path="./models/facebook/sam3/sam3.pt")
processor = Sam3Processor(model)

# 处理图片
image = Image.open("image01.jpg")
inference_state = processor.set_image(image)
output = processor.set_text_prompt(
    state=inference_state, 
    prompt="Segment the soccer players in yellow jerseys in the image")

print(f"检测到 {len(output['boxes'])} 个对象")

def save_boxes(image_path, output_path, output):
    img = Image.open(image_path)
    for box in output["boxes"]:
        x1, y1, x2, y2 = box
        box_xywh = [x1, y1, x2-x1, y2-y1]
        img = draw_box_on_image(img, box_xywh, (0, 255, 0)) 
    img.save(output_path)

save_boxes("image01.jpg", "simple_boxes.jpg", output)