import os
os.environ['YOLO_VERBOSE'] = str(False)

import cv2
from ultralytics import YOLO
import numpy as np
from tqdm import tqdm
from pathlib import Path
import concurrent.futures
import torch
import logging
import traceback
from PIL import Image

from split import split_videos

# 设置变量
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 使用第一个GPU

MODEL_NAME = "best"
CONF = 0.2
DEBUG = False
video_folder = "videos"
split_video_folder = "split_videos"
output_base_folder = "output_frames"
debug_output_folder = "debug_output"
total_frame_count = 0


# 加载YOLO模型
model = YOLO(f"{MODEL_NAME}.pt")

# 导出TensorRT模型（如果不存在）
engine_model_path = Path(f"{MODEL_NAME}.engine")
if not engine_model_path.exists():
    model.export(format="engine", dynamic=True, half=True, device=0)

# 加载TensorRT模型
engine_model = YOLO(f"{MODEL_NAME}.engine")

# 分割视频
split_videos(video_folder, split_video_folder)

# 获取视频文件夹中的所有视频文件
video_files = [f for f in os.listdir(split_video_folder) if f.endswith(('.mp4', '.avi', '.mov'))]

def process_frame(frame, frame_index, output_folder):
    

    global total_frame_count
    try:
        with torch.no_grad():
            results = engine_model(frame, conf=CONF, device=0)
            if results and len(results) > 0 and len(results[0].boxes.cls) > 0:
                output_path = os.path.join(output_folder, f"frame_{total_frame_count}.jpg")
                # print(f"检测到目标，输出路径: {output_path}")
                
                # 使用PIL保存图像
                cv2.imwrite(output_path, frame)
                
                total_frame_count += 1
            else:
                # print(f"帧 {frame_index} 未检测到目标")
                pass
            
            if DEBUG:   
                debug_path = os.path.join(debug_output_folder, f"frame_{frame_index:04d}.jpg")
                cv2.imwrite(debug_path, frame)

    except Exception as e:
        print(f"处理帧 {frame_index} 时发生错误: {str(e)}")
        import traceback
        print(traceback.format_exc())

def process_video(video_file):
    video_path = os.path.join(split_video_folder, video_file)
    output_folder = os.path.join(output_base_folder)
    os.makedirs(output_folder, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    target_size = 640
    start_x = (width - 640) // 2
    start_y = (height - 640) // 2
    end_x = start_x + 640
    end_y = start_y + 640

    for frame_count in tqdm(range(total_frames), desc=f"处理 {video_file}", unit="帧"):
        try:
            ret, frame = cap.read()
            if not ret:
                logging.warning(f"无法读取帧 {frame_count}")
                break
            
            # 裁剪并调整大小
            center_frame = frame[start_y:end_y, start_x:end_x]           
            process_frame(center_frame, frame_count, output_folder)
        except Exception as e:
            logging.error(f"处理视频 {video_file} 的帧 {frame_count} 时发生错误: {str(e)}")
            logging.error(traceback.format_exc())

    cap.release()
    cv2.destroyAllWindows()
    logging.info(f"处理完成: {video_file}")

# 使用线程池并行处理视频
with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
    executor.map(process_video, video_files)

print("所有视频处理完成")