import torch
from PIL import Image
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

# 装载模型
model = build_sam3_image_model(
    checkpoint_path="./models/facebook/sam3/sam3.pt")
processor = Sam3Processor(model)
# 装载图片
image = Image.open("image01.jpg")
inference_state = processor.set_image(image)
# 分割图片
output = processor.set_text_prompt(
    state=inference_state, 
    prompt="Segmenting soccer players in the image")
print(output)

######### 处理分割结果 ########
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

def to_numpy(data):
    """将数据转换为numpy数组，处理GPU tensor的情况"""
    if isinstance(data, torch.Tensor):
        return data.cpu().numpy()
    elif hasattr(data, 'numpy'):
        return data.numpy()
    else:
        return np.array(data)

# 得到掩码、边界和得分，并转换为numpy数组
masks = output["masks"]
boxes = to_numpy(output["boxes"])
scores = to_numpy(output["scores"])

print(f"检测到 {len(masks)} 个对象")
print("边界框:", boxes)

def process_mask(mask):
    """处理掩码形状，确保是2D数组"""
    mask = to_numpy(mask)
    # 如果掩码有批次维度，移除它
    if mask.ndim == 3:
        if mask.shape[0] == 1:
            mask = mask[0]  # 移除批次维度
        else:
            # 如果有多个掩码，取第一个或合并
            mask = mask[0]
    # 确保是2D布尔数组
    if mask.dtype != bool:
        mask = mask.astype(bool)
    return mask

######### 保存分割结果 ########
def save_segmentation_result(image_path, output_path, masks, boxes, scores):
    """保存分割结果到文件"""
    image = Image.open(image_path)
    image_np = np.array(image)  
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.imshow(image_np)
    colors = plt.cm.Set3(np.linspace(0, 1, len(masks)))
    for i, (mask, box, score) in enumerate(zip(masks, boxes, scores)):
        color = colors[i % len(colors)]
        # 处理掩码形状
        processed_mask = process_mask(mask)
        # 调整掩码大小以匹配原图（如果需要）
        if processed_mask.shape != image_np.shape[:2]:
            # 如果掩码大小与图像不匹配，调整掩码大小
            from PIL import Image as PILImage
            mask_pil = PILImage.fromarray(processed_mask.astype(np.uint8) * 255)
            mask_pil = mask_pil.resize((image_np.shape[1], image_np.shape[0]), PILImage.NEAREST)
            processed_mask = np.array(mask_pil).astype(bool)
        # 绘制掩码
        mask_rgba = np.zeros((processed_mask.shape[0], processed_mask.shape[1], 4))
        mask_rgba[processed_mask] = [color[0], color[1], color[2], 0.4]
        ax.imshow(mask_rgba)
        # 绘制边界框
        x1, y1, x2, y2 = box
        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, 
                               linewidth=2, edgecolor=color, facecolor='none')
        ax.add_patch(rect)
        # 添加标签
        ax.text(x1, y1-8, f'{score:.3f}', 
                bbox=dict(boxstyle="round,pad=0.2", facecolor=color, alpha=0.8),
                fontsize=7, color='white')
    
    ax.set_title(f'SAM3 Segmentation - {len(masks)} objects')
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"分割结果已保存到: {output_path}")

# 分割结果保存
save_segmentation_result("image01.jpg", "segmentation_result.jpg", masks, boxes, scores)