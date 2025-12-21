from PIL import Image
import os
import sys
import pandas as pd
import csv


def split_and_resize_images(folder_path, positions, output_dir="split_output", target_width=40):
    """
    遍历文件夹，按指定位置分割所有图片，并将每部分宽度统一拉伸到指定像素
    所有分割后的部分都保存在同一个文件夹中

    folder_path: 图片文件夹路径
    positions: 分割位置列表，例如 [0.2, 0.5, 0.8] 表示在20%、50%、80%宽度处分割
    output_dir: 输出文件夹路径
    target_width: 目标宽度（像素），每部分将拉伸到此宽度
    """
    # 验证positions参数
    pos_list = []
    if len(positions) > 0 and 0 < positions[0] < 1:
        pos_list = positions
    else:
        print("注意：positions参数应该是百分比列表（0-1之间的值）")

    # 计算总共会分成多少部分
    num_parts = len(pos_list) + 1
    print(f"每张图片将被分割为 {num_parts} 个部分")
    print(f"每个部分将被拉伸到宽度: {target_width}像素")

    # 创建输出文件夹
    os.makedirs(output_dir, exist_ok=True)
    print(f"创建输出文件夹: {output_dir}")

    # 获取图片文件列表
    extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'}
    files = [f for f in os.listdir(folder_path)
             if any(f.lower().endswith(ext) for ext in extensions)]

    if not files:
        print(f"在 '{folder_path}' 中未找到图片文件")
        return

    print(f"找到 {len(files)} 个图片文件")
    print("开始分割并拉伸...")

    # 处理每个图片
    for idx, filename in enumerate(files, 1):
        img_path = os.path.join(folder_path, filename)

        # 显示进度条
        progress = idx / len(files)
        bar_length = 30
        filled_length = int(bar_length * progress)
        bar = '█' * filled_length + '░' * (bar_length - filled_length)
        sys.stdout.write(f'\r[{bar}] {idx}/{len(files)} ({progress:.0%}) 处理: {filename[:20]:<20}')
        sys.stdout.flush()

        try:
            with Image.open(img_path) as img:
                width, height = img.size

                # 将百分比位置转为像素位置
                pixel_positions = [0]
                for pos in pos_list:
                    pixel_positions.append(int(pos * width))
                pixel_positions.append(width)

                # 确保位置是递增的
                pixel_positions = sorted(set(pixel_positions))

                # 分割图片并保存到同一个文件夹
                base_name = os.path.splitext(filename)[0]
                for i in range(len(pixel_positions) - 1):
                    left, right = pixel_positions[i], pixel_positions[i + 1]
                    if right <= left:
                        continue

                    # 1. 先裁剪
                    crop_img = img.crop((left, 0, right, height))

                    # 2. 再拉伸到目标宽度
                    # 计算目标尺寸：target_width x 原高度
                    resized_img = crop_img.resize((target_width, height), Image.Resampling.LANCZOS)

                    # 确定输出文件名（添加part标识）
                    output_filename = f"{base_name}_{i+1}.png"
                    output_path = os.path.join(output_dir, output_filename)

                    # 保存图片
                    resized_img.save(output_path)

        except Exception as e:
            print(f"\n处理文件 {filename} 时出错: {e}")
            continue

    # 完成提示
    sys.stdout.write(f'\r[{"█" * bar_length}] {len(files)}/{len(files)} (100%) 处理完成！')
    sys.stdout.flush()

    print(f"\n\n所有图片已分割并拉伸完成！")
    print(f"共处理 {len(files)} 个文件")
    print(f"每个部分尺寸: {target_width}×[原高度] 像素")
    print(f"分割结果保存在: {output_dir}")
    print(f"总共生成 {len(files) * num_parts} 个分割图片")


def label_split():
    # 读取CSV文件
    df = pd.read_csv("dataset/train/labels.csv")  # 请将文件名改为你的实际文件名

    # 创建新数据的列表
    new_data = []

    # 遍历每一行数据
    for _, row in df.iterrows():
        filename = row['filename']
        color = row['color']
        all_label = row['all_label']

        # 确保color和all_label长度都是5
        if len(color) != 5 or len(all_label) != 5:
            print(f"Warning: {filename} 的color或all_label长度不是5，跳过该行")
            continue

        # 生成5行新数据
        for i in range(5):
            new_filename = f"{filename.replace('.png', '')}_{i + 1}.png"
            new_color = color[i]
            new_label = all_label[i]

            new_data.append({
                'filename': new_filename,
                'color': new_color,
                'label': new_label
            })

    # 转换为DataFrame
    new_df = pd.DataFrame(new_data, columns=['filename', 'color', 'label'])

    # 保存到新的CSV文件
    new_df.to_csv("dataset/train_final/labels.csv", index=False, quoting=csv.QUOTE_MINIMAL)

    print(f"转换完成！")
    print(f"原始数据行数: {len(df)}")
    print(f"新数据行数: {len(new_df)}")
    print(f"输出文件: output.csv")

    # 显示前几行数据
    print("\n输出文件的前15行数据预览:")
    print(new_df.head(15).to_string(index=False))

# 使用示例
if __name__ == "__main__":
    label_split()
    # 示例1：基本用法
    # split_and_resize_images(
    #     folder_path='dataset/train/images',
    #     positions=[0.23, 0.4, 0.59, 0.76],  # 百分比位置
    #     output_dir='dataset/train_final/images',
    #     target_width=40
    # )
    # split_and_resize_images(
    #     folder_path='dataset/test/images',
    #     positions=[0.23, 0.4, 0.59, 0.76],  # 百分比位置
    #     output_dir='dataset/test_final',
    #     target_width=40
    # )

    #
    # csv_path = "dataset/train/labels.csv"
    # csv_final_path = "dataset/train_final/labels.csv"