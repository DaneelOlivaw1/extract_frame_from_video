import os
os.environ['YOLO_VERBOSE'] = str(False)

from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO("best.pt")

# Export the model to NCNN format
model.export(format="ncnn")  # creates '/yolov8n_ncnn_model'

# Load the exported NCNN model
ncnn_model = YOLO("./best_ncnn_model")

# Run inference
results = ncnn_model("center_frame.jpg", conf=0.2)
# print(results[0].boxes)
