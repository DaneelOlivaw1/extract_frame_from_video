import os
from pathlib import Path
from ultralytics import YOLO
import torch

MODEL_NAME = "best"
model = YOLO(f"{MODEL_NAME}.pt")

# 导出TensorRT模型
engine_model_path = Path(f"{MODEL_NAME}.engine")
if not engine_model_path.exists():
    model.export(format="engine", dynamic=True, half=True, device=0)

# 加载TensorRT模型
engine_model = YOLO(f"{MODEL_NAME}.engine")

# 准备测试图像
image_path = "test/1.jpg"  # 替换为您的图像路径

# 使用OpenVINO模型进行推理
ov_model = YOLO(f"{MODEL_NAME}_openvino_model/")
ov_results = ov_model(image_path)

# 使用NCNN模型进行推理
ncnn_model = YOLO(f"{MODEL_NAME}_ncnn_model/")
ncnn_results = ncnn_model(image_path)

# 使用TensorRT模型进行推理
engine_results = engine_model(image_path)

# 打印结果比较
print("OpenVINO 结果:")
print(ov_results[0].speed)

print("\nNCNN 结果:")
print(ncnn_results[0].speed)

print("\nTensorRT 结果:")
print(engine_results[0].speed)