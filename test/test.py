import os
from pathlib import Path
from ultralytics import YOLO
import torch

MODEL_NAME = "best"
conf = 0.05
model = YOLO(f"{MODEL_NAME}.pt")

# 导出TensorRT模型
engine_model_path = Path(f"{MODEL_NAME}.engine")
if not engine_model_path.exists():
    model.export(format="engine", dynamic=True, half=True, device=0)

# 加载TensorRT模型
engine_model = YOLO(f"{MODEL_NAME}.engine")

# 准备测试图像
image_path = r"C:\Users\daneel\Documents\python\extract_frame_from_video\debug_output\frame_1233.jpg"  # 替换为您的图像路径

# 使用OpenVINO模型进行推理
ov_model = YOLO(f"{MODEL_NAME}_openvino_model/")
ov_results = ov_model(image_path, conf=conf)

# 使用NCNN模型进行推理
ncnn_model = YOLO(f"{MODEL_NAME}_ncnn_model/")
ncnn_results = ncnn_model(image_path, conf=conf)

# 使用TensorRT模型进行推理
engine_results = engine_model(image_path, conf=conf)

# 打印结果比较
print("OpenVINO 结果:")
print(ov_results[0].speed)
print(len(ov_results[0].boxes.cls))
print(ov_results[0].boxes)

print("\nNCNN 结果:")
print(ncnn_results[0].speed)
print(len(ncnn_results[0].boxes.cls))
print(ncnn_results[0].boxes)

print("\nTensorRT 结果:")
print(engine_results[0].speed)
print(len(engine_results[0].boxes.cls))
print(engine_results[0].boxes)