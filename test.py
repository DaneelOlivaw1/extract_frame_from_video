import torch
print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(torch.cuda.get_device_name(0))

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 使用第一个GPU
print(os.environ.get('CUDA_VISIBLE_DEVICES'))