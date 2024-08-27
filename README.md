# 使用 yolo 提取出来视频中包含目标的帧

best.pt 为 yolov8 的模型
main.py 是主程序

这个项目为了推理快一些，我把 yolo 导出为 ncnn 模型，这样推理就快很多了

要不然推理一次得 300ms 实在是太久了


# 需要注意的地方
下面是屏蔽 yolo 乱七八糟的日志的代码，需要在导入 ultralytics 之前运行
```
import os
os.environ['YOLO_VERBOSE'] = str(False)
```


# 速度
我的电脑测试下来是 14 帧每秒，我觉得是 python 语言的问题，最快就这意思了

我尝试把 pytorch 修改为 gpu 版本的，但是速度并没有任何的提升

可能改多线程会更快？我没测
