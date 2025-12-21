import torch.nn as nn
import torch
import torch.nn.functional as F
from typing import Optional, Tuple

class CNN(nn.Module):
    def __init__(self, num_class=36, num_char=1, input_shape=[3, 40, 60]):
        super(CNN, self).__init__()
        self.num_class = num_class
        self.num_char = num_char



        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            # nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(32),
            # nn.ReLU(inplace=True),
            nn.LeakyReLU(0.1, inplace=True),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            # nn.ReLU(inplace=True),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2, 2),



            nn.Conv2d(64, 128, 3, padding=1),
            # nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(128),
            # nn.ReLU(inplace=True),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(128, 256, 3, padding=1),
            # nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(256),
            # nn.ReLU(inplace=True),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2, 2),


            nn.Conv2d(256, 256, 3, padding=1),
            # nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(256),
            # nn.ReLU(inplace=True),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            # nn.ReLU(inplace=True),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2, 2),


            nn.Conv2d(256, 512, 3, padding=1),
            # nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(512),
            # nn.ReLU(inplace=True),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            # nn.ReLU(inplace=True),
            nn.LeakyReLU(0.1, inplace=True),


            # nn.Dropout2d(0.25)  # 防止过拟合

        )

        # 全局平均池化替代展平
        # self.global_pool = nn.AdaptiveAvgPool2d((1, 1))


        with torch.no_grad():
            dummy_input = torch.zeros(1, *input_shape)
            dummy_output = self.conv(dummy_input)
            self.flatten_size = dummy_output.numel() // dummy_output.shape[0]
            # print(f"自动计算展平尺寸: {self.flatten_size}")


        flat = self.flatten_size



        # 字符识别分支
        self.fc_char = nn.Sequential(
            nn.Linear(flat, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_class * num_char)
        )

        # 颜色识别分支
        self.fc_color = nn.Sequential(
            nn.Linear(flat, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 2 * num_char)
        )

        # self.fc_char = nn.Linear(64, num_class * num_char)   # 字符预测
        # self.fc_color = nn.Linear(64, 2 * num_char)          # 颜色预测 r/u

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        # x = self.fc_layers(x)

        char_out = self.fc_char(x)
        color_out = self.fc_color(x)

        return char_out, color_out



# class BasicBlock(nn.Module):
#     """基础残差块，用于ResNet"""
#     def __init__(self, in_channels, out_channels, stride=1, downsample=None):
#         super(BasicBlock, self).__init__()
#         self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
#                               stride=stride, padding=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(out_channels)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
#                               stride=1, padding=1, bias=False)
#         self.bn2 = nn.BatchNorm2d(out_channels)
#         self.downsample = downsample
#         self.stride = stride

#     def forward(self, x):
#         identity = x

#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)

#         out = self.conv2(out)
#         out = self.bn2(out)

#         if self.downsample is not None:
#             identity = self.downsample(x)

#         out += identity
#         out = self.relu(out)

#         return out

# class ResNet(nn.Module):
#     def __init__(self, num_class=36, num_char=1, input_shape=[3, 40, 60]):
#         super(ResNet, self).__init__()
#         self.num_class = num_class
#         self.num_char = num_char
        
#         # 初始卷积层
#         self.conv1 = nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3, bias=False)
#         self.bn1 = nn.BatchNorm2d(32)
#         self.relu = nn.ReLU(inplace=True)
#         self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
#         # 残差块层
#         self.layer1 = self._make_layer(32, 32, 2, stride=1)
#         self.layer2 = self._make_layer(32, 64, 2, stride=2)
#         self.layer3 = self._make_layer(64, 128, 2, stride=2)
#         self.layer4 = self._make_layer(128, 256, 2, stride=2)
        
#         # 自适应平均池化，将特征图统一到固定尺寸
#         self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
#         # 计算全连接层输入维度
#         with torch.no_grad():
#             dummy_input = torch.zeros(1, *input_shape)
#             dummy_output = self._forward_features(dummy_input)
#             self.flatten_size = dummy_output.numel() // dummy_output.shape[0]
#             print(f"自动计算展平尺寸: {self.flatten_size}")
        
#         flat = self.flatten_size
        
#         # 字符识别分支
#         self.fc_char = nn.Sequential(
#             nn.Linear(flat, 1024),
#             nn.ReLU(inplace=True),
#             nn.Dropout(0.5),
#             nn.Linear(1024, 512),
#             nn.ReLU(inplace=True),
#             nn.Dropout(0.3),
#             nn.Linear(512, num_class * num_char)
#         )

#         # 颜色识别分支
#         self.fc_color = nn.Sequential(
#             nn.Linear(flat, 128),
#             nn.ReLU(inplace=True),
#             nn.Dropout(0.3),
#             nn.Linear(128, 2 * num_char)
#         )
    
#     def _make_layer(self, in_channels, out_channels, blocks, stride=1):
#         """创建残差层"""
#         downsample = None
#         if stride != 1 or in_channels != out_channels:
#             downsample = nn.Sequential(
#                 nn.Conv2d(in_channels, out_channels, kernel_size=1, 
#                          stride=stride, bias=False),
#                 nn.BatchNorm2d(out_channels),
#             )
        
#         layers = []
#         layers.append(BasicBlock(in_channels, out_channels, stride, downsample))
        
#         for _ in range(1, blocks):
#             layers.append(BasicBlock(out_channels, out_channels))
        
#         return nn.Sequential(*layers)
    
#     def _forward_features(self, x):
#         """提取特征"""
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = self.maxpool(x)
        
#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
#         x = self.layer4(x)
        
#         x = self.avgpool(x)
#         x = torch.flatten(x, 1)
#         return x
    
#     def forward(self, x):
#         features = self._forward_features(x)
        
#         char_out = self.fc_char(features)
#         color_out = self.fc_color(features)
        
#         return char_out, color_out




class BasicBlock(nn.Module):
    """ResNet基础残差块"""
    expansion = 1
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3,
            stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3,
            padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)
        
        return out


class ResNet18MultiTask(nn.Module):
    """多任务ResNet-18，用于字符和颜色预测"""
    def __init__(self, num_chars: int = 36, num_colors: int = 2):
        super().__init__()
        self.in_channels = 64
        self.num_chars = num_chars
        self.num_colors = num_colors
        
        # 初始卷积层 - 适应40x60输入
        # 原始ResNet使用7x7 stride=2，这里使用3x3 stride=1保持较大特征图
        self.conv1 = nn.Conv2d(
            3, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=1, padding=0)
        
        # 4个残差层 (ResNet-18: [2, 2, 2, 2])
        self.layer1 = self._make_layer(BasicBlock, 64, 2, stride=1)
        self.layer2 = self._make_layer(BasicBlock, 128, 2, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 256, 2, stride=2)
        self.layer4 = self._make_layer(BasicBlock, 512, 2, stride=2)
        
        # 全局平均池化
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 共享特征提取层
        self.shared_fc = nn.Sequential(
            nn.Linear(512 * BasicBlock.expansion, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3)
        )
        
        # 任务特定头部
        # 字符预测头 (36分类)
        self.char_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, num_chars)
        )
        
        # 颜色预测头 (2分类)
        self.color_head = nn.Sequential(
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, num_colors)
        )
        
        # 权重初始化
        self._initialize_weights()
    
    def _make_layer(
        self,
        block: nn.Module,
        out_channels: int,
        blocks: int,
        stride: int = 1
    ) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channels,
                    out_channels * block.expansion,
                    kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels * block.expansion),
            )
        
        layers = []
        layers.append(block(
            self.in_channels, out_channels, stride, downsample
        ))
        
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(
                self.in_channels, out_channels, stride=1
            ))
        
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        """初始化网络权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu'
                )
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            x: 输入张量，形状为 (batch_size, 3, 40, 60)
            
        Returns:
            tuple: (字符预测, 颜色预测)
        """
        # 骨干网络
        x = self.conv1(x)  # (batch, 64, 40, 60)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)  # (batch, 64, 20, 30)
        
        x = self.layer1(x)  # (batch, 64, 20, 30)
        x = self.layer2(x)  # (batch, 128, 10, 15)
        x = self.layer3(x)  # (batch, 256, 5, 8)
        x = self.layer4(x)  # (batch, 512, 3, 4)
        
        # 全局平均池化
        x = self.avgpool(x)  # (batch, 512, 1, 1)
        x = torch.flatten(x, 1)  # (batch, 512)
        
        # 共享特征
        shared_features = self.shared_fc(x)  # (batch, 256)
        
        # 任务特定预测
        char_pred = self.char_head(shared_features)  # (batch, 36)
        color_pred = self.color_head(shared_features)  # (batch, 2)
        
        return char_pred, color_pred

class EfficientCharNet(nn.Module):
    """针对40×60字符识别的高效网络"""
    
    def __init__(self, num_chars=36, num_colors=2):
        super().__init__()
        
        # 关键策略：减少下采样，保持空间分辨率
        self.features = nn.Sequential(
            # 第一层：提取基础特征
            nn.Conv2d(3, 32, kernel_size=3, padding=1),  # 40×60×32
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),  # 20×30×32
            
            # 第二层：增加通道，轻微下采样
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # 20×30×64
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),  # 10×15×64
            
            # 第三层：深度特征，不下采样
            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # 10×15×128
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            # 注意：这里不下采样！
            
            # 全局平均池化
            nn.AdaptiveAvgPool2d((1, 1))  # 1×1×128
        )
        
        # 分类头
        self.char_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, num_chars)
        )
        
        self.color_head = nn.Sequential(
            nn.Linear(128, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(32, num_colors)
        )
        
    def forward(self, x):
        features = self.features(x)
        features = features.view(features.size(0), -1)
        
        char_pred = self.char_head(features)
        color_pred = self.color_head(features)
        
        return char_pred, color_pred


class CNN_color(nn.Module):
    def __init__(self, num_char=1, input_shape=[3, 40, 60]):
        super(CNN, self).__init__()
        self.num_class = num_class
        self.num_char = num_char



        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),



            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

        )


        with torch.no_grad():
            dummy_input = torch.zeros(1, *input_shape)
            dummy_output = self.conv(dummy_input)
            self.flatten_size = dummy_output.numel() // dummy_output.shape[0]
            # print(f"自动计算展平尺寸: {self.flatten_size}")


        flat = self.flatten_size

        # 颜色识别分支
        self.fc_color = nn.Sequential(
            nn.Linear(flat, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, 2 * num_char)
        )

        # self.fc_char = nn.Linear(64, num_class * num_char)   # 字符预测
        # self.fc_color = nn.Linear(64, 2 * num_char)          # 颜色预测 r/u

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        # x = self.fc_layers(x)

        color_out = self.fc_color(x)

        return color_out