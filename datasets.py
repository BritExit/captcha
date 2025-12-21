from PIL import Image
import torch
from torch.utils.data import Dataset
import os
import pandas as pd
from torchvision.transforms import transforms
import random
import numpy as np

# 62字符集合
# alphabet = (
#     [str(i) for i in range(10)] +
#     [chr(i) for i in range(65, 91)] +
#     [chr(i) for i in range(97, 123)]
# )
alphabet = (
    [str(i) for i in range(10)] +
    [chr(i) for i in range(65, 91)]
)
alphabet = ''.join(alphabet)
use_augmentation = True

save_aug_samples = True
num_samples_to_save = 10
save_dir = "./aug_exampls/images"


class AddGaussianNoise:
    """添加高斯噪声的变换"""
    def __init__(self, mean=0.0, std_range=(0.01, 0.05), p=0.3):
        """
        参数:
            mean: 噪声均值
            std_range: 噪声标准差范围 (min, max)
            p: 应用概率
        """
        self.mean = mean
        self.std_range = std_range
        self.p = p
    
    def __call__(self, img):
        """添加高斯噪声"""
        if random.random() > self.p:
            return img
        
        # 随机选择标准差
        std = random.uniform(self.std_range[0], self.std_range[1])
        
        # 将PIL图像转换为numpy数组
        img_array = np.array(img).astype(np.float32) / 255.0
        
        # 添加高斯噪声
        noise = np.random.normal(self.mean, std, img_array.shape)
        noisy_array = img_array + noise
        
        # 裁剪到[0, 1]范围
        noisy_array = np.clip(noisy_array, 0, 1)
        
        # 转换回PIL图像
        noisy_array = (noisy_array * 255).astype(np.uint8)
        
        return Image.fromarray(noisy_array)
    
    def __repr__(self):
        return f"AddGaussianNoise(mean={self.mean}, std_range={self.std_range}, p={self.p})"

def img_loader(path):
    return Image.open(path).convert("RGB")


def make_dataset(csv_path, img_dir, num_class, num_char):
    df = pd.read_csv(csv_path)
    samples = []

    for idx, row in df.iterrows():
        filename = row["filename"]
        color_str = row["color"]   # e.g. "uuruu"
        label_str = row["label"]

        img_path = os.path.join(img_dir, filename)

        ## 字符 one-hot
        char_vec = []
        for ch in label_str:
            v = [0]*num_class
            v[alphabet.find(ch)] = 1
            char_vec += v

        ## 颜色 one-hot（r=1,u=0）
        color_vec = []
        for c in color_str:
            if c == "r":    color_vec += [1,0]
            else:           color_vec += [0,1]

        samples.append((img_path, char_vec, color_vec))
    return samples


class CaptchaData(Dataset):
    def __init__(self, img_dir, csv_path,
                 num_class=36, num_char=1,
                 transform=None,
                 use_augmentation=use_augmentation):

        self.samples = make_dataset(csv_path, img_dir,
                                    num_class, num_char)
        self.transform = transform
        self.use_augmentation = use_augmentation  # 保存增强标志

        if use_augmentation:
            self.augmentation_transforms = transforms.Compose([
                transforms.RandomRotation(degrees=5),  # 随机旋转 ±5度
                transforms.RandomAffine(degrees=0, translate=(0.04, 0.04)),  # 随机平移
                # transforms.RandomPerspective(distortion_scale=0.2, p=0.5),  # 随机透视变换
                # transforms.ColorJitter(brightness=0.1, contrast=0.1),  # 颜色抖动
                # transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),  # 高斯模糊
                AddGaussianNoise(mean=0.0, std_range=(0.01, 0.05), p=0.3),  # 高斯噪声
            ])
        else:
            self.augmentation_transforms = None

       

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        img_path, char_vec, color_vec = self.samples[index]
        img = img_loader(img_path)

        # 添加：在应用transform前先应用数据增强
        if self.use_augmentation and random.random() > 0.5:  # 50%的概率应用增强
            img = self.augmentation_transforms(img)

        if self.transform:
            img = self.transform(img)

        return img, torch.tensor(char_vec).float(), torch.tensor(color_vec).float()

