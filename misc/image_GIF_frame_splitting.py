from PIL import Image  # pip install Pillow
import os

def split_gif_into_frames(gif_path, output_folder):
    # 打开 GIF 文件
    with Image.open(gif_path) as im:
        # 获取 GIF 的总帧数
        num_frames = im.n_frames

        # 遍历每一帧
        for i in range(num_frames):
            # 选择当前帧
            im.seek(i)
            # 创建输出文件名
            output_path = f"{output_folder}/frame_{i:03d}.png"
            # 保存当前帧为 PNG 文件
            im.save(output_path, "PNG")
            print(f"Saved frame {i} to {output_path}")


# 确保输出文件夹存在
if not os.path.exists("./frames"):
    os.makedirs("./frames")
    print("Output folder created.")

# 示例用法
gif_path = "aaa.gif"
output_folder = "./frames"
split_gif_into_frames(gif_path, output_folder)
