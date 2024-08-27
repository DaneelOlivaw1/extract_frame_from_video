import os
os.environ["YOLO_DEVICE"] = "gpu"
os.environ["YOLO_VERBOSE"] = "False"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 使用第一个GPU


import cv2
from ultralytics import YOLO
import numpy as np
import tempfile
from tqdm import tqdm

# 加载YOLO模型
model = YOLO("best.pt")
model.to('cuda')  # 将模型移动到GPU

# 导出模型为NCNN格式
model.export(format="ncnn", device=0)


# 加载导出的NCNN模型
ncnn_model = YOLO("./best_ncnn_model")

# 设置视频文件夹和输出文件夹
video_folder = "video"
output_base_folder = "output_frames"

# 获取视频文件夹中的所有视频文件
video_files = [f for f in os.listdir(video_folder) if f.endswith(('.mp4', '.avi', '.mov'))]

for video_file in video_files:
    video_path = os.path.join(video_folder, video_file)
    output_folder = os.path.join(output_base_folder, os.path.splitext(video_file)[0])
    os.makedirs(output_folder, exist_ok=True)

    # 打开视频文件
    cap = cv2.VideoCapture(video_path)

    # 获取视频的总帧数
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 获取视频的宽度和高度
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 计算中心区域的坐标
    start_x = (width - 640) // 2
    start_y = (height - 640) // 2
    end_x = start_x + 640
    end_y = start_y + 640

    with tempfile.TemporaryDirectory() as temp_dir:
        # 使用tqdm创建进度条
        for frame_count in tqdm(range(total_frames), desc=f"处理 {video_file}", unit="帧"):
            ret, frame = cap.read()
            if not ret:
                break
            
            # 裁剪中心区域
            center_frame = frame[start_y:end_y, start_x:end_x]
            
            # 保存为临时文件
            temp_file = os.path.join(temp_dir, f"temp_{frame_count}.jpg")
            cv2.imwrite(temp_file, center_frame)
            
            # 运行推理
            results = ncnn_model(temp_file, conf=0.2, device=0)
            
            # 检查是否有检测到的对象
            if results[0].boxes.shape[0] > 0:
                # 如果检测到对象,保存帧
                output_path = os.path.join(output_folder, f"frame_{frame_count:04d}.jpg")
                cv2.imwrite(output_path, center_frame)
                # tqdm.write(f"保存帧 {frame_count} 到 {output_path}")
                
                # # 打印检测信息
                # for box in results[0].boxes:
                #     tqdm.write(f"检测到目标: 置信度 {box.conf.item():.2f}, 类别 {results[0].names[int(box.cls.item())]}")
                #     tqdm.write(f"边界框坐标 (xyxy): {box.xyxy.tolist()[0]}")
            else:
                # tqdm.write(f"帧 {frame_count}: 未检测到目标")
                pass
            
            # 删除临时文件
            os.remove(temp_file)

    # 释放视频捕获对象
    cap.release()

    print(f"处理完成: {video_file}")

print("所有视频处理完成")