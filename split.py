import os
import subprocess
from pathlib import Path

def split_videos(input_folder, output_folder):
    # 确保输出文件夹存在
    os.makedirs(output_folder, exist_ok=True)

    # 获取所有视频文件
    video_files = [f for f in os.listdir(input_folder) if f.endswith(('.mp4', '.avi', '.mov', '.mkv'))]

    for video_file in video_files:
        input_path = os.path.join(input_folder, video_file)
        output_base = os.path.join(output_folder, Path(video_file).stem)
        
        # 使用FFmpeg分割视频
        command = [
            "ffmpeg",
            "-i", input_path,
            "-c", "copy",
            "-map", "0",
            "-segment_time", "00:05:00",
            "-f", "segment",
            f"{output_base}_%03d.mp4"
        ]
        
        print(f"正在分割视频: {video_file}")
        subprocess.run(command, check=True)
        print(f"视频 {video_file} 分割完成")

        # 删除原始视频文件
        os.remove(input_path)
        print(f"已删除原始视频: {video_file}")

    print("所有视频分割完成并删除原始文件")

# 使用示例
if __name__ == "__main__":
    input_folder = "videos"
    output_folder = "split_videos"
    split_videos(input_folder, output_folder)