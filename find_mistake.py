import os
import shutil
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from Testmodel import CNN, ResNet18MultiTask

def analyze_prediction_errors(model, csv_path, img_folder, output_folder='error_results'):
    """
    分析预测错误，将错误图片复制到单独文件夹，并生成错误表格
    
    Args:
        model: 训练好的模型
        csv_path: 包含真实标签的CSV文件路径
        img_folder: 图片文件夹路径
        output_folder: 输出文件夹
    
    Returns:
        DataFrame: 错误预测表格
    """
    # 创建输出文件夹
    os.makedirs(output_folder, exist_ok=True)
    
    # 创建错误图片文件夹
    error_img_folder = os.path.join(output_folder, 'error_images')
    os.makedirs(error_img_folder, exist_ok=True)
    
    # 加载真实标签
    df = pd.read_csv(csv_path)
    
    # 字符类别映射
    alphabet = (
        [str(i) for i in range(10)] +
        [chr(i) for i in range(65, 91)]
    )
    alphabet = ''.join(alphabet)
    
    # 存储错误记录
    error_records = []
    
    # 图片预处理
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    # 逐张图片预测
    for idx, row in tqdm(df.iterrows(), total=len(df), desc='Analyzing predictions'):
        filename = row['filename']
        true_color = row['color']
        true_char = row['label']
        
        # 图片路径
        img_path = os.path.join(img_folder, filename)
        if not os.path.exists(img_path):
            print(f"Warning: Image {img_path} not found")
            continue
        
        # 加载并预处理图片
        try:
            img = Image.open(img_path).convert('RGB')
            img_tensor = transform(img).unsqueeze(0)
            
            if torch.cuda.is_available():
                img_tensor = img_tensor.cuda()
            
            # 预测
            model.eval()
            with torch.no_grad():
                char_out, color_out = model(img_tensor)
                
                # 获取预测结果
                char_idx = torch.argmax(char_out, dim=1).item()
                color_idx = torch.argmax(color_out, dim=1).item()
                
                pred_char = alphabet[char_idx]
                pred_color = 'r' if color_idx == 0 else 'u'
                
                # 检查是否为错误预测
                if pred_char != true_char or pred_color != true_color:
                    # 记录错误预测
                    error_record = {
                        'filename': filename,
                        'pred_color': pred_color,
                        'pred_char': pred_char,
                        'true_color': true_color,
                        'true_char': true_char
                    }
                    error_records.append(error_record)
                    
                    # 复制错误图片
                    error_img_path = os.path.join(error_img_folder, filename)
                    shutil.copy2(img_path, error_img_path)
                    
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            continue
    
    # 转换为DataFrame
    error_df = pd.DataFrame(error_records)
    
    # 保存错误表格
    error_csv_path = os.path.join(output_folder, 'error_predictions.csv')
    if len(error_df) > 0:
        error_df.to_csv(error_csv_path, index=False)
        print(f"\n✓ 已保存错误预测表格: {error_csv_path}")
        print(f"  共 {len(error_df)} 个错误预测")
    else:
        # 创建空表格
        error_df = pd.DataFrame(columns=['filename', 'pred_color', 'pred_char', 'true_color', 'true_char'])
        error_df.to_csv(error_csv_path, index=False)
        print(f"\n✓ 无错误预测，已创建空表格: {error_csv_path}")
    
    # 打印错误表格前20行
    if len(error_records) > 0:
        print(f"\n错误表格预览 (前{min(20, len(error_records))}行):")
        print("="*60)
        print(f"{'文件名':<20} {'预测':<10} {'真实':<10}")
        print("-"*40)
        for i, record in enumerate(error_records[:20]):
            print(f"{record['filename']:<20} {record['pred_char']}({record['pred_color']})  {record['true_char']}({record['true_color']})")
        if len(error_records) > 20:
            print(f"... 还有 {len(error_records)-20} 行")
    
    return error_df


if __name__ == "__main__":
    # # 主程序入口
    # print("验证码识别系统 - 测试模式")
    # print("=" * 50)
    
    # 选项1: 生成完整的提交文件
    model = ResNet18MultiTask()
    model_path = "./checkpoints/model.pth"
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
    else:
        print(f"错误: 找不到模型文件 {model_path}")
        
    if torch.cuda.is_available():
        model = model.cuda()
    csv_path = './dataset/train_final/labels.csv'
    img_folder = './dataset/train_final/images'
    analyze_prediction_errors(model, csv_path, img_folder)