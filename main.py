import os
os.environ['YOLO_VERBOSE'] = str(False)


import cv2
from ultralytics import YOLO
import numpy as np
from tqdm import tqdm
from pathlib import Path
import concurrent.futures
import torch

from split import split_videos

# 设置环境变量
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 使用第一个GPU

# 加载YOLO模型
MODEL_NAME = "best"
model = YOLO(f"{MODEL_NAME}.pt")

# 导出TensorRT模型（如果不存在）
engine_model_path = Path(f"{MODEL_NAME}.engine")
if not engine_model_path.exists():
    model.export(format="engine", dynamic=True, half=True, device=0)

# 加载TensorRT模型
engine_model = YOLO(f"{MODEL_NAME}.engine")

# 设置视频文件夹和输出文件夹
video_folder = "videos"
split_video_folder = "split_videos"
output_base_folder = "output_frames"

# 分割视频
split_videos(video_folder, split_video_folder)

# 获取视频文件夹中的所有视频文件
video_files = [f for f in os.listdir(split_video_folder) if f.endswith(('.mp4', '.avi', '.mov'))]

def process_batch(frames, frame_indices, output_folder):
    with torch.no_grad():
        results = engine_model(frames, conf=0.2, device=0)
    
    for i, result in enumerate(results):
        if result.boxes.shape[0] > 0:
            output_path = os.path.join(output_folder, f"frame_{frame_indices[i]:04d}.jpg")
            cv2.imwrite(output_path, cv2.cvtColor(frames[i], cv2.COLOR_RGB2BGR))

def process_video(video_file):
    video_path = os.path.join(split_video_folder, video_file)
    output_folder = os.path.join(output_base_folder, os.path.splitext(video_file)[0])
    os.makedirs(output_folder, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    start_x = (width - 640) // 2
    start_y = (height - 640) // 2
    end_x = start_x + 640
    end_y = start_y + 640

    batch_size = 1  # 设置为1以匹配模型期望的输入
    frames = []
    frame_indices = []

    for frame_count in tqdm(range(total_frames), desc=f"处理 {video_file}", unit="帧"):
        ret, frame = cap.read()
        if not ret:
            break
        
        center_frame = frame[start_y:end_y, start_x:end_x]
        center_frame_rgb = cv2.cvtColor(center_frame, cv2.COLOR_BGR2RGB)
        frames.append(center_frame_rgb)
        frame_indices.append(frame_count)

        if len(frames) == batch_size:
            process_batch(frames, frame_indices, output_folder)
            frames = []
            frame_indices = []

    # 处理剩余的帧
    if frames:
        process_batch(frames, frame_indices, output_folder)

    cap.release()
    print(f"处理完成: {video_file}")

# 使用线程池并行处理视频
with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
    executor.map(process_video, video_files)

print("所有视频处理完成")