import torch
from torchvision.transforms import ToTensor, Compose
from Testmodel import CNN, ResNet18MultiTask
from PIL import Image
import pandas as pd
import os

# 新增：与训练时相同的切割参数
# 注意：这些参数需要和训练时的切割参数保持一致！
# 这里假设训练时使用的参数，如果不同请修改
TARGET_WIDTH = 40  # 单个字符的目标宽度
POS_LIST = [0.23, 0.4, 0.59, 0.76]  # 切割位置列表（百分比）

# 字符类别映射
alphabet = (
    [str(i) for i in range(10)] +
    [chr(i) for i in range(65, 91)]
)
alphabet = ''.join(alphabet)

# 新增：图片切割函数
def split_image(image_path, target_width=TARGET_WIDTH, pos_list=POS_LIST):
    """
    将包含多个字符的验证码图片切割成单个字符
    
    Args:
        image_path: 图片路径
        target_width: 目标宽度
        pos_list: 切割位置列表
        
    Returns:
        list: 包含5个PIL.Image对象的列表
    """
    images = []
    
    try:
        with Image.open(image_path) as img:
            width, height = img.size
            
            # 将百分比位置转为像素位置
            pixel_positions = [0]
            for pos in pos_list:
                pixel_positions.append(int(pos * width))
            pixel_positions.append(width)
            
            # 确保位置是递增的
            pixel_positions = sorted(set(pixel_positions))
            
            # 分割图片
            for i in range(len(pixel_positions) - 1):
                left, right = pixel_positions[i], pixel_positions[i + 1]
                if right <= left:
                    continue
                
                # 1. 先裁剪
                crop_img = img.crop((left, 0, right, height))
                
                # 2. 再拉伸到目标宽度
                resized_img = crop_img.resize((target_width, height), Image.Resampling.LANCZOS)
                
                images.append(resized_img)
                
    except Exception as e:
        print(f"切割图片 {image_path} 时出错: {e}")
    
    return images

# 修改：预测单张图片（现在是单个字符图片）
def predict_single_char(model, img_tensor):
    """
    预测单个字符图片
    
    Args:
        model: 训练好的模型
        img_tensor: 单个字符图片的tensor，形状为(1, C, H, W)
        
    Returns:
        str: 预测的字符
    """
    if torch.cuda.is_available():
        img_tensor = img_tensor.cuda()
    
    model.eval()
    
    is_red = False
    with torch.no_grad():
        # 修改：现在模型只输出字符预测
        char_out, color_out = model(img_tensor)
        
        # 获取预测结果
        char_idx = torch.argmax(char_out, dim=1).item()
        is_red = torch.argmax(color_out, dim=1).item() == 0  # r=[1,0]
        char = alphabet[char_idx]
    
    return char, is_red

# 修改：预测整张验证码图片（包含5个字符）
def predict_image(model, img_path):
    """
    预测包含5个字符的验证码图片
    
    Args:
        model: 训练好的模型
        img_path: 图片路径
        
    Returns:
        str: 预测的5个字符组成的字符串
    """
    # 步骤1：切割图片
    char_images = split_image(img_path)
    
    if len(char_images) != 5:
        print(f"警告: 图片 {img_path} 切割后得到 {len(char_images)} 个字符，期望5个")
        return ""  # 或根据实际情况处理
    
    # 步骤2：对每个字符进行预测
    chars = []
    transform = ToTensor()  # 图片转换为tensor
    
    for i, char_img in enumerate(char_images):
        # 转换为RGB（确保是3通道）
        if char_img.mode != 'RGB':
            char_img = char_img.convert('RGB')
        
        # 转换为tensor并添加batch维度
        img_tensor = transform(char_img).unsqueeze(0)
        
        # 预测单个字符
        char, is_red = predict_single_char(model, img_tensor)
        if is_red:
            chars.append(char)
    
    # 步骤3：返回5个字符组成的字符串
    return ''.join(chars)

# 修改：生成提交文件的函数
def generate_submission():
    # 加载模型
    model = ResNet18MultiTask()
    
    # 修改：根据你的模型实际文件名调整
    # 注意：模型应该是针对单个字符训练的
    model_path = "./checkpoints/model.pth"
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
    else:
        print(f"错误: 找不到模型文件 {model_path}")
        return
    
    if torch.cuda.is_available():
        model = model.cuda()
    
    # 测试集路径
    test_dir = "./dataset/test/images"
    
    # 检查测试集目录是否存在
    if not os.path.exists(test_dir):
        print(f"错误: 测试集目录不存在 {test_dir}")
        return
    
    ids = sorted(os.listdir(test_dir))
    
    # 新增：记录统计信息
    total_images = len(ids)
    processed = 0
    
    rows = []
    for imgname in ids:
        # 忽略非图片文件
        if not imgname.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
        
        img_path = os.path.join(test_dir, imgname)
        
        # 预测整个验证码
        label = predict_image(model, img_path)
        
        # 如果预测失败，记录错误
        # if len(label) != 5:
        #     print(f"警告: 图片 {imgname} 预测结果长度不为5: '{label}'")
        #     # 可以在这里添加默认值或处理逻辑
        #     if len(label) == 0:
        #         label = "XXXXX"  # 默认值
        
        rows.append([imgname, label])
        processed += 1
        
        # 显示进度
        if processed % 100 == 0:
            print(f"已处理 {processed}/{total_images} 张图片")
    
    # 创建DataFrame并保存
    df = pd.DataFrame(rows, columns=["id", "label"])
    df.to_csv("./result/submission5.csv", index=False)
    print(f"\n处理完成！共处理 {processed} 张图片")
    print("submission.csv 已生成！")
    
    # 新增：显示前几行结果
    print("\n前5行预测结果:")
    print(df.head())

# 新增：测试单个图片的函数（用于调试）
def test_single_image(image_path):
    """
    测试单张图片的预测
    
    Args:
        image_path: 图片路径
    """
    # 加载模型
    model = ResNet18MultiTask()
    model_path = "./checkpoints/model.pth"
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
    else:
        print(f"错误: 找不到模型文件 {model_path}")
        return
    
    if torch.cuda.is_available():
        model = model.cuda()
    
    # 预测
    result = predict_image(model, image_path)
    print(f"图片: {os.path.basename(image_path)}")
    print(f"预测结果: {result}")
    
    # 显示切割后的字符图片
    char_images = split_image(image_path)
    print(f"切割为 {len(char_images)} 个字符")
    
    return result

# 新增：批量测试函数
def batch_test(test_dir, num_samples=5):
    """
    批量测试多张图片
    
    Args:
        test_dir: 测试图片目录
        num_samples: 测试的图片数量
    """
    image_files = [f for f in os.listdir(test_dir) 
                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if len(image_files) == 0:
        print("没有找到图片文件")
        return
    
    # 加载模型
    model = ResNet18MultiTask()
    model_path = "./checkpoints/model.pth"
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
    else:
        print(f"错误: 找不到模型文件 {model_path}")
        return
    
    if torch.cuda.is_available():
        model = model.cuda()
    
    print(f"\n开始批量测试 {min(num_samples, len(image_files))} 张图片:")
    print("-" * 50)
    
    for i, imgname in enumerate(image_files[:num_samples]):
        img_path = os.path.join(test_dir, imgname)
        result = predict_image(model, img_path)
        print(f"{i+1:3d}. {imgname:20s} -> {result}")
    
    print("-" * 50)

if __name__ == "__main__":
    # 主程序入口
    print("验证码识别系统 - 测试模式")
    print("=" * 50)
    
    # 选项1: 生成完整的提交文件
    generate_submission()
    
    # 选项2: 测试单张图片（调试用）
    # test_single_image("./dataset/test/images/example.png")
    
    # 选项3: 批量测试几张图片
    # batch_test("./dataset/test/images", num_samples=10)