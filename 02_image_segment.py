import torch
from PIL import Image
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.visualization_utils import draw_box_on_image
import numpy as np
import time

# 装载模型
model = build_sam3_image_model(
    checkpoint_path="./models/facebook/sam3/sam3.pt")
processor = Sam3Processor(model)

# 处理图片
image = Image.open("image01.jpg")
inference_state = processor.set_image(image)
start_time = time.time()
output = processor.set_text_prompt(
    state=inference_state, 
    prompt="Segment the soccer players in yellow jerseys in the image")
end_time = time.time()
execution_time_ms = (end_time - start_time) * 1000
print(f"检测用时: {execution_time_ms:.2f} 毫秒")
print(f"检测到 {len(output['boxes'])} 个对象")

def save_boxes(image_path, output_path, output):
    img = Image.open(image_path)
    for box in output["boxes"]:
        x1, y1, x2, y2 = box
        box_xywh = [x1, y1, x2-x1, y2-y1]
        img = draw_box_on_image(img, box_xywh, (0, 255, 0)) 
    img.save(output_path)

def save_masks(image_path, output_path, output):
    img = Image.open(image_path).convert("RGB")
    img_array = np.array(img)
    # 为每个掩码应用不同的颜色
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), 
              (255, 0, 255), (0, 255, 255)]
    # 将掩码移动到CPU并转换为numpy
    masks_cpu = output["masks"].cpu()  # 先移动到CPU
    for i, mask in enumerate(masks_cpu):
        # 将掩码转换为二进制格式
        mask_binary = mask > 0.5  # 阈值化
        mask_np = mask_binary.numpy()  # 现在可以安全转换为numpy
        color = colors[i % len(colors)]
        # 在原始图像上叠加掩码
        for c in range(3):
            img_array[:, :, c] = np.where(
                mask_np,
                img_array[:, :, c] * 0.5 + color[c] * 0.5,  # 半透明叠加
                img_array[:, :, c]
            )
    
    result_img = Image.fromarray(img_array)
    result_img.save(output_path)

# 同时保存边界框和掩码
save_boxes("image01.jpg", "simple_boxes.jpg", output)
save_masks("image01.jpg", "masks_result.jpg", output)