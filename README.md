# 使用 yolo 提取出来视频中包含目标的帧

## 主要流程

1. 加载 yolo 模型
2. 导出为 TensorRT 模型
3. 把视频进行分段处理，方便后面并行处理（会删除原视频）
4. 并行处理视频，提取包含识别目标的帧到输出文件夹

## 文件是干啥的

**best.pt** 为 yolov8 训练出来的模型

**main.py** 是主程序

**split.py** 使用了ffmpeg进行视频的分段

**test** 文件夹下我测试了3种不同的推理方法，后面有需要的话可以换

**requirements.txt** 是需要安装的依赖

# 需要注意的地方
下面是屏蔽 yolo 乱七八糟的日志的代码，需要在导入 ultralytics 之前运行
```
import os
os.environ['YOLO_VERBOSE'] = str(False)
```

TensorRT 需要手动安装一下，具体的安装方法可以问 GPT（我就是问他，然后他教我安装的）

还需要使用 conda 安装 ffmpeg
```
conda install -c conda-forge ffmpeg
```

# 速度
我的电脑使用ncnn测试下来是 14 帧每秒，~~我觉得是 python 语言的问题，最快就这意思了~~

我尝试把 pytorch 修改为 gpu 版本的，但是速度并没有任何的提升

~~可能改多线程会更快？我没测~~

已经修改成并行处理的了，速度已经优化到极致了！我的电脑跑起来是可以达到210帧每秒的！




