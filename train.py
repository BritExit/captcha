import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import ToTensor, Compose
from Testmodel import CNN, ResNet18MultiTask, EfficientCharNet
from datasets import CaptchaData
import time
import sys
from torchvision.transforms import RandomRotation, RandomAffine, ColorJitter, RandomPerspective
import math
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, SequentialLR, LambdaLR

# batch_size = 1024
# batch_size = 64
# batch_size = 64
# lr = 0.0005 * math.sqrt(batch_size / 64)
# batch_size = 1024; lr = 0.001; MAX_TOTAL_SAMPLES = 200000
batch_size = 512; lr = 0.001; MAX_TOTAL_SAMPLES = 200000
# batch_size = 512, lr = 0.004
# batch_size = 256; lr = 0.001; MAX_TOTAL_SAMPLES = 100000
# batch_size = 256; lr = 0.002; MAX_TOTAL_SAMPLES = 200000
max_epoch = 100
model_path = "./checkpoints/model.pth"
char_weight, color_weight = 1, 0.1 #
# 21号记得试一下0.1的这个参数 
# char_weight, color_weight = 0, 1 # 只关注颜色
# color_weight = 0.0
use_scheduler = True
use_color_only = False
use_evaluate_fast = False

val_ratio = 0.2  # 20%的数据用作验证集

# 比较多的错误 0 O 1 I
# F
# T
# J
# L
# 5
# E
num_classes = 36
class_weights = torch.ones(num_classes)  
class_weights[0] = 2.0
class_weights[1] = 2.0
class_weights[14] = 2.0
class_weights[18] = 2.0
class_weights[24] = 3.0
class_weights = class_weights.to('cuda')

def print_header(text):
    """打印标题"""
    print("\n" + "=" * 60)
    print(f"  {text}")
    print("=" * 60)


def print_subheader(text):
    """打印子标题"""
    print(f"\n{'─' * 40}")
    print(f"  {text}")
    print(f"{'─' * 40}")


def evaluate_fast_no_color(model, data_loader, loss_fn, device, max_samples=10000):
    """快速评估版本"""
    model.eval()
    total_loss = 0.0
    char_correct = 0
    char_total = 0
    
    with torch.no_grad():
        for img, char_gt, _ in data_loader:
            img = img.to(device)
            char_gt = char_gt.to(device)
            
            char_out, color_out = model(img)
            loss = loss_fn(char_out, char_gt)
            total_loss += loss.item() * img.size(0)
            
            char_pred = char_out.argmax(dim=1)
            char_correct += (char_pred == char_gt).sum().item()
            char_total += img.size(0)
            
            if char_total >= max_samples:
                break
    
    if char_total == 0:
        return 0.0, 0.0, 0.0, 0.0
    
    avg_loss = total_loss / char_total
    char_acc = char_correct / char_total * 100
    
    return avg_loss, char_acc, 1, char_acc

# 添加：评估函数
def evaluate(model, data_loader, loss_char_fn, loss_color_fn, device, max_samples=10000):
    # if use_evaluate_fast:
    #     return evaluate_fast(model, data_loader, loss_fn, device=device, max_samples=10000)

    """评估模型在数据集上的性能"""
    model.eval()
    total_loss = 0.0
    char_correct = 0
    color_correct = 0
    char_total = 0
    color_total = 0
    sample_correct = 0
    total_samples = 0

    with torch.no_grad():
        for img, char_gt, color_gt in data_loader:
            img = img.to(device)
            char_gt = char_gt.to(device)
            color_gt = color_gt.to(device)

            # 前向传播
            char_out, color_out = model(img)

            # 计算损失
            loss_char = loss_char_fn(char_out, char_gt)
            loss_color = loss_color_fn(color_out, color_gt)
            loss = char_weight * loss_char + color_weight * loss_color
            total_loss += loss.item() * img.size(0)

            # 计算准确率
            batch_size = img.size(0)
            char_total += batch_size * 1  # 5个字符
            color_total += batch_size * 1  # 5个颜色位置
            total_samples += batch_size

            # 字符准确率
            char_pred = char_out.view(batch_size, 1, 36).argmax(dim=2)
            char_target = char_gt.view(batch_size, 1, 36).argmax(dim=2)
            char_correct += (char_pred == char_target).sum().item()

            # 颜色准确率
            color_pred = color_out.view(batch_size, 1, 2).argmax(dim=2)
            color_target = color_gt.view(batch_size, 1, 2).argmax(dim=2)
            color_correct += (color_pred == color_target).sum().item()

            # 样本准确率（所有字符和颜色都正确）
            char_correct_all = (char_pred == char_target).all(dim=1)
            color_correct_all = (color_pred == color_target).all(dim=1)
            # sample_correct += (char_correct_all).sum().item()
            if use_color_only:
                sample_correct += (color_correct_all).sum().item()
            else:
                sample_correct += (char_correct_all & color_correct_all).sum().item()
            

            if char_total >= max_samples:
                break

    avg_loss = total_loss / total_samples
    char_acc = char_correct / char_total * 100
    color_acc = color_correct / color_total * 100
    sample_acc = sample_correct / total_samples * 100

    return avg_loss, char_acc, color_acc, sample_acc


def evaluate_fast(model, data_loader, loss_char_fn, loss_color_fn, device="cuda", max_samples=10000):
    """评估模型，随机抽取max_samples个样本"""
    model.eval()

    # 先收集所有数据
    all_imgs = []
    all_char_gt = []
    all_color_gt = []

    print(f"  正在收集数据样本...", end='')
    with torch.no_grad():
        for img, char_gt, color_gt in data_loader:
            all_imgs.append(img)
            all_char_gt.append(char_gt)
            all_color_gt.append(color_gt)

    # 将所有批次数据拼接
    all_imgs = torch.cat(all_imgs, dim=0)
    all_char_gt = torch.cat(all_char_gt, dim=0)
    all_color_gt = torch.cat(all_color_gt, dim=0)

    total_data_size = len(all_imgs)

    # 随机选择max_samples个索引
    if total_data_size > max_samples:
        indices = torch.randperm(total_data_size)[:max_samples]
        selected_imgs = all_imgs[indices]
        selected_char_gt = all_char_gt[indices]
        selected_color_gt = all_color_gt[indices]
    else:
        selected_imgs = all_imgs
        selected_char_gt = all_char_gt
        selected_color_gt = all_color_gt
        max_samples = total_data_size

    print(f" 总数据: {total_data_size}, 抽样: {len(selected_imgs)}")

    # 创建小批量处理
    eval_batch_size = data_loader.batch_size
    total_loss = 0.0
    char_correct = 0
    color_correct = 0
    char_total = 0
    color_total = 0
    sample_correct = 0

    for i in range(0, len(selected_imgs), eval_batch_size):
        end_idx = min(i + eval_batch_size, len(selected_imgs))

        batch_imgs = selected_imgs[i:end_idx].to(device)
        batch_char_gt = selected_char_gt[i:end_idx].to(device)
        batch_color_gt = selected_color_gt[i:end_idx].to(device)

        # 前向传播
        char_out, color_out = model(batch_imgs)

        # 计算损失
        loss_char = loss_char_fn(char_out, batch_char_gt)
        loss_color = loss_color_fn(color_out, batch_color_gt)
        loss = char_weight * loss_char + color_weight * loss_color
        total_loss += loss.item() * len(batch_imgs)

        # 计算准确率
        batch_size = len(batch_imgs)
        char_total += batch_size * 1
        color_total += batch_size * 1

        # 字符准确率
        char_pred = char_out.view(batch_size, 1, 36).argmax(dim=2)
        char_target = batch_char_gt.view(batch_size, 1, 36).argmax(dim=2)
        char_correct += (char_pred == char_target).sum().item()

        # 颜色准确率
        color_pred = color_out.view(batch_size, 1, 2).argmax(dim=2)
        color_target = batch_color_gt.view(batch_size, 1, 2).argmax(dim=2)
        color_correct += (color_pred == color_target).sum().item()

        # 样本准确率
        char_correct_all = (char_pred == char_target).all(dim=1)
        color_correct_all = (color_pred == color_target).all(dim=1)
        sample_correct += (char_correct_all & color_correct_all).sum().item()

    avg_loss = total_loss / len(selected_imgs)
    char_acc = char_correct / char_total * 100
    color_acc = color_correct / color_total * 100
    sample_acc = sample_correct / len(selected_imgs) * 100

    return avg_loss, char_acc, color_acc, sample_acc

def train():
    print_header("🚀 验证码识别模型训练开始")

    # 1. 数据准备阶段
    print_subheader("📁 数据准备")
    transform = Compose([ToTensor()])

    print("正在加载训练数据集...")

    # full_dataset = CaptchaData(
    #     img_dir="./dataset/train/images",
    #     csv_path="./dataset/train/labels.csv",
    #     transform=transform,
    #     # use_augmentation=True
    # )

    full_dataset = CaptchaData(
        img_dir="./dataset/train_final/images",
        csv_path="./dataset/train_final/labels.csv",
        transform=transform,
        # use_augmentation=True
    )



    
    

    # 如果数据集太大，先随机抽样一部分
    if len(full_dataset) > MAX_TOTAL_SAMPLES:
        print(f"  数据集过大 ({len(full_dataset)} 样本)，进行随机抽样...")
        indices = torch.randperm(len(full_dataset))[:MAX_TOTAL_SAMPLES]
        full_dataset = torch.utils.data.Subset(full_dataset, indices)
        print(f"  抽样后数据集: {len(full_dataset)} 样本")

    # 修改：划分训练集和验证集
    train_size = int((1 - val_ratio) * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    print(f"✅ 数据集加载完成")
    print(f"  总数据集大小: {len(full_dataset)} 张图片")
    print(f"  训练集大小: {len(train_dataset)} 张图片")
    print(f"  验证集大小: {len(val_dataset)} 张图片")
    print(f"  验证集比例: {val_ratio * 100}%")



    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              drop_last=True)

    val_loader = DataLoader(val_dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            drop_last=False)
    print(f"  Batch Size: {batch_size}")
    print(f"  每个Epoch的Batch数: {len(train_loader)}")
    print(f"  总训练步数: {max_epoch * len(train_loader)} 步")

    show_freq = math.ceil(len(train_loader) / 10)


    # 2. 模型准备阶段
    print_subheader("🤖 模型准备")
    # model = CNN()
    # model = ResNet()
    model = ResNet18MultiTask()
    # model = EfficientCharNet()

    # 计算模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  模型: CNN")
    print(f"  总参数量: {total_params:,}")
    print(f"  可训练参数量: {trainable_params:,}")

    # GPU设置
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        model = model.cuda()
        print(f"  🎮 使用GPU: {device_name}")
    else:
        print(f"  💻 使用CPU")

    # 3. 训练配置
    print_subheader("⚙️ 训练配置")
    # opt = torch.optim.Adam(model.parameters(), lr=lr)
    opt = torch.optim.AdamW(model.parameters(), 
                       lr=lr,  # 降低学习率
                       betas=(0.9, 0.999),
                       eps=1e-8,
                       weight_decay=0.01)  # 添加权重衰减
                    #    weight_decay=0.01)  # 添加权重衰减
    # 损失函数
    # loss_fn = nn.CrossEntropyLoss()
    # loss_fn = 
    loss_char_fn = nn.CrossEntropyLoss(weight=class_weights)
    loss_color_fn = nn.CrossEntropyLoss()
    # loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
    # loss_fn = nn.MultiLabelSoftMarginLoss()

    
    scheduler = ReduceLROnPlateau(
        opt, 
        mode='min',           # 监控指标越小越好
        factor=2/3,          # 学习率衰减因子
        patience=5,          # 容忍多少个epoch没有改善
        verbose=True,        # 打印调整信息
        threshold=0.0001,    # 改善阈值
        threshold_mode='rel', # 相对改善
        cooldown=5,          # 调整后的冷却期
        min_lr=0.0001          # 最小学习率
    )


    print(f"  优化器: AdamW (lr={lr})")
    print(f"  损失函数: CrossEntropyLoss")
    print(f"  最大Epoch数: {max_epoch}")
    print(f"  模型保存路径: {model_path}")

    # 添加：记录最佳验证损失
    best_val_loss = float('inf')
    best_epoch = 0
    best_val_sample_acc = 0

    # 4. 开始训练
    print_header("🏃 开始训练")

    # 记录开始时间
    start_time = time.time()

    for epoch in range(max_epoch):
        print(f"\n📅 Epoch {epoch + 1}/{max_epoch}")
        print(f"{'─' * 40}")

        # 打印当前学习率
        current_lr = opt.param_groups[0]['lr']
        print(f"  当前学习率: {current_lr:.6f}")

        epoch_start_time = time.time()
        model.train()
        total_loss = 0
        batch_count = 0

        # 进度条初始化
        total_batches = len(train_loader)

        print("  训练阶段:")
        for batch_idx, (img, char_gt, color_gt) in enumerate(train_loader):
            batch_count += 1

            # 显示进度
            progress = (batch_idx + 1) / total_batches * 100
            sys.stdout.write(f"\r  Batch {batch_idx + 1}/{total_batches} [{progress:.1f}%]")
            sys.stdout.flush()

            # 数据移动到GPU
            if torch.cuda.is_available():
                img = img.cuda()
                char_gt = char_gt.cuda()
                color_gt = color_gt.cuda()

            # 前向传播
            char_out, color_out = model(img)

            # 计算损失
            loss_char = loss_char_fn(char_out, char_gt)
            loss_color = loss_color_fn(color_out, color_gt)
            loss = char_weight * loss_char + color_weight * loss_color

            # 反向传播
            opt.zero_grad()
            loss.backward()
            opt.step()

            # 累加损失
            total_loss += loss.item()

            # 每10个batch显示一次详细损失
            if (batch_idx + 1) % show_freq == 0 or (batch_idx + 1) == total_batches:
                print(f"\r  Batch {batch_idx + 1}/{total_batches} - "
                      f"Loss: {loss.item():.6f} "
                      f"(字符: {loss_char.item():.6f}, "
                      f"颜色: {loss_color.item():.6f})")

        # 计算epoch统计
        epoch_time = time.time() - epoch_start_time
        avg_loss = total_loss / len(train_loader)

      

        print(f"\n  📊 Epoch {epoch + 1} 统计:")
        print(f"    平均损失: {avg_loss:.6f}")
        print(f"    总损失: {total_loss:.6f}")
        print(f"    处理批次: {batch_count}")
        print(f"    耗时: {epoch_time:.2f}秒")
        print(f"    每批次平均: {epoch_time / batch_count:.3f}秒")

        # 修改：验证阶段
        print("  验证阶段:")
        val_loss, val_char_acc, val_color_acc, val_sample_acc = evaluate(
            model, val_loader, loss_char_fn, loss_color_fn, "cuda"
        )
        print(f"  📊 验证统计:")
        print(f"    平均损失: {val_loss:.6f}")
        print(f"    字符准确率: {val_char_acc:.2f}%")
        print(f"    颜色准确率: {val_color_acc:.2f}%")
        print(f"    样本准确率: {val_sample_acc:.2f}%")

        
        print("  训练集评估:")
        train_loss, train_char_acc, train_color_acc, train_sample_acc = evaluate(
            model, train_loader, loss_char_fn, loss_color_fn, "cuda"
        )
        print(f"  📊 训练统计:")
        print(f"    平均损失: {train_loss:.6f}")
        print(f"    字符准确率: {train_char_acc:.2f}%")
        print(f"    颜色准确率: {train_color_acc:.2f}%")
        print(f"    样本准确率: {train_sample_acc:.2f}%")


        # 使用验证损失来调整学习率
        if use_scheduler:
            scheduler.step(train_loss)  # 关键：传入验证损失
        
            # 检查学习率是否变化
            new_lr = opt.param_groups[0]['lr']
            if new_lr != current_lr:
                print(f"  🔧 学习率已调整: {current_lr:.6f} -> {new_lr:.6f}")
                current_lr = new_lr


        # 保存最佳模型，依据准确率
        # if val_loss < best_val_loss:
        if val_sample_acc > best_val_sample_acc:
            best_val_loss = val_loss
            best_val_sample_acc = val_sample_acc
            best_epoch = epoch + 1
            torch.save(model.state_dict(), model_path)
            print(f"  💾 最佳模型已保存 (Epoch {best_epoch}, Loss: {best_val_loss:.6f})")

        # 保存模型
        # torch.save(model.state_dict(), model_path)
        # print(f"  💾 模型已保存到: {model_path}")

        # 显示训练进度预估
        elapsed_time = time.time() - start_time
        avg_epoch_time = elapsed_time / (epoch + 1)
        remaining_epochs = max_epoch - (epoch + 1)
        remaining_time = avg_epoch_time * remaining_epochs

        hours = int(remaining_time // 3600)
        minutes = int((remaining_time % 3600) // 60)
        seconds = int(remaining_time % 60)

        print(f"  ⏳ 剩余时间: {hours:02d}:{minutes:02d}:{seconds:02d}")

    # 5. 训练完成
    print_header("✅ 训练完成")
    total_time = time.time() - start_time

    hours = int(total_time // 3600)
    minutes = int((total_time % 3600) // 60)
    seconds = int(total_time % 60)

    print(f"  总训练时间: {hours:02d}:{minutes:02d}:{seconds:02d}")
    print(f"  总Epoch数: {max_epoch}")
    print(f"  总Batch数: {max_epoch * len(train_loader)}")
    print(f"  最佳Epoch: {best_epoch} (验证损失: {best_val_loss:.6f})")
    print(f"  最终模型: {model_path}")

    # 6：最终评估
    print_subheader("📈 最终模型评估")

    # 加载最佳模型
    model.load_state_dict(torch.load(model_path, map_location="cuda"))

    # 评估训练集
    print("  训练集评估:")
    train_loss, train_char_acc, train_color_acc, train_sample_acc = evaluate(
        model, train_loader, loss_char_fn, loss_color_fn, "cuda"
    )
    print(f"    平均损失: {train_loss:.6f}")
    print(f"    字符准确率: {train_char_acc:.2f}%")
    print(f"    颜色准确率: {train_color_acc:.2f}%")
    print(f"    样本准确率: {train_sample_acc:.2f}%")

    # 评估验证集
    print("  验证集评估:")
    val_loss, val_char_acc, val_color_acc, val_sample_acc = evaluate(
        model, val_loader, loss_char_fn, loss_color_fn, "cuda"
    )
    print(f"    平均损失: {val_loss:.6f}")
    print(f"    字符准确率: {val_char_acc:.2f}%")
    print(f"    颜色准确率: {val_color_acc:.2f}%")
    print(f"    样本准确率: {val_sample_acc:.2f}%")

    # 7. 模型信息总结
    print_subheader("📈 模型总结")

    # 获取模型结构信息
    print("模型结构:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            # print(f"  {name:25s} | 形状: {tuple(param.shape):20s} | 参数量: {param.numel():,}")
            print(f"  {name:25s} | 形状: {str(tuple(param.shape)):20s} | 参数量: {param.numel():,}")

    print(f"\n🎉 训练完成！可以使用模型进行验证码识别了。")


if __name__ == "__main__":
    train()