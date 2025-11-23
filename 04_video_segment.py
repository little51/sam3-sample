from sam3.model_builder import build_sam3_video_predictor
from sam3.visualization_utils import (
    prepare_masks_for_visualization,
    visualize_formatted_frame_output,
)
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 设置matplotlib后端为Agg，避免显示问题
plt.switch_backend('Agg')

# 初始化预测器
video_predictor = build_sam3_video_predictor(
    checkpoint_path="./models/facebook/sam3/sam3.pt")
video_path = "./sam3/assets/videos/bedroom.mp4" 

# 开始会话
response = video_predictor.handle_request(
    request=dict(
        type="start_session",
        resource_path=video_path,
    )
)
session_id = response["session_id"]

print(f"会话已创建: {session_id}")

# 提取视频帧用于可视化
def extract_video_frames(video_path):
    """提取视频帧用于可视化"""
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # 转换 BGR 到 RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)
    
    cap.release()
    return frames

# 获取视频信息
cap = cv2.VideoCapture(video_path)
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
cap.release()

print(f"视频信息: {width}x{height}, {fps} FPS, 总帧数: {total_frames}")

# 提取视频帧
video_frames_for_vis = extract_video_frames(video_path)
print(f"提取了 {len(video_frames_for_vis)} 帧")

# 逐帧处理：为每一帧添加提示并获取分割结果
print("开始逐帧处理...")
all_outputs = {}

for frame_idx in range(total_frames):
    try:
        # 为每一帧添加文本提示
        response = video_predictor.handle_request(
            request=dict(
                type="add_prompt",
                session_id=session_id,
                frame_index=frame_idx, 
                text="kids",
            )
        )
        
        if "outputs" in response:
            all_outputs[frame_idx] = response["outputs"]
            print(f"成功处理帧 {frame_idx + 1}/{total_frames}")
        else:
            print(f"帧 {frame_idx} 没有输出结果")
            
    except Exception as e:
        print(f"处理帧 {frame_idx} 时出错: {e}")
        break

print(f"成功处理了 {len(all_outputs)} 帧")

# 创建视频写入器
output_video_path = "segmentation_video.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out_video = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

print("开始生成可视化视频...")

# 使用更稳定的方式生成视频帧
def create_visualization_frame(frame_idx, video_frames, outputs, target_size):
    """创建可视化帧，避免libpng错误"""
    try:
        # 创建图形，不使用太大的DPI
        fig = plt.figure(figsize=(target_size[0]/100, target_size[1]/100), dpi=100)
        
        # 可视化分割结果
        visualize_formatted_frame_output(
            frame_idx,
            video_frames,
            outputs_list=[prepare_masks_for_visualization({frame_idx: outputs})],
            titles=["SAM 3 Tracking"],
        )
        
        plt.tight_layout(pad=0)
        plt.axis('off')
        
        # 保存到内存缓冲区
        from io import BytesIO
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', 
                   pad_inches=0, dpi=100, facecolor='black')
        buf.seek(0)
        
        # 使用PIL读取图像
        from PIL import Image
        img = Image.open(buf)
        img = img.convert('RGB')
        
        # 转换为numpy数组
        img_array = np.array(img)
        
        # 转换为BGR格式用于OpenCV
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        plt.close(fig)
        buf.close()
        
        # 调整尺寸
        if img_bgr.shape[1] != target_size[0] or img_bgr.shape[0] != target_size[1]:
            img_bgr = cv2.resize(img_bgr, target_size)
            
        return img_bgr
        
    except Exception as e:
        print(f"创建可视化帧 {frame_idx} 时出错: {e}")
        # 返回黑色帧作为备用
        return np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)

# 生成视频
processed_count = 0
for frame_idx in range(total_frames):
    if frame_idx in all_outputs:
        try:
            # 创建可视化帧
            vis_frame = create_visualization_frame(
                frame_idx, 
                video_frames_for_vis, 
                all_outputs[frame_idx], 
                (width, height)
            )
            
            # 写入视频
            out_video.write(vis_frame)
            processed_count += 1
            print(f"视频生成进度: {frame_idx + 1}/{total_frames}")
            
        except Exception as e:
            print(f"生成帧 {frame_idx} 时出错: {e}")
            # 写入黑色帧作为备用
            black_frame = np.zeros((height, width, 3), dtype=np.uint8)
            out_video.write(black_frame)
    else:
        # 对于没有输出的帧，写入黑色帧
        black_frame = np.zeros((height, width, 3), dtype=np.uint8)
        out_video.write(black_frame)
        print(f"帧 {frame_idx} 无输出，写入黑色帧")

# 释放视频写入器
out_video.release()

print(f"视频生成完成，成功处理 {processed_count} 帧")
print(f"分割视频已保存到: {output_video_path}")

# 结束会话
video_predictor.handle_request(
    request=dict(
        type="close_session",
        session_id=session_id,
    )
)
video_predictor.shutdown()