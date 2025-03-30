#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import time
import math
import traceback
from collections import defaultdict, OrderedDict
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchdiffeq
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import glob
from torchmetrics.functional import structural_similarity_index_measure as ssim_fn
from skimage.metrics import structural_similarity as sk_ssim

from data_loaders import loaders
from model.model import DiffUNet
from torchvision import datasets, transforms
from torchvision.transforms import ToPILImage
from torchvision.utils import make_grid

#########################
# 定义 SSIM 计算函数
#########################
def gaussian(window_size, sigma):
    """生成一维高斯核"""
    gauss = torch.Tensor([math.exp(-(x - window_size//2)**2 / float(2 * sigma**2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    """生成二维高斯核窗口，用于 SSIM 计算"""
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    """内部使用的 SSIM 计算函数"""
    mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channel)

    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size//2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size//2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size//2, groups=channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

def ssim(img1, img2, window_size=11, size_average=True):
    """
    计算图像 SSIM 值，输入张量形状为 (N, C, H, W)，要求图像取值范围在 [0, 1] 内
    """
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel).to(img1.device)
    return _ssim(img1, img2, window, window_size, channel, size_average)

#########################
# 评估函数
#########################
def evaluate_val(model, val_loader, device, use_torch_diffeq=True):
    """
    遍历整个 Radio_val 数据集，对每个批次采用 torchdiffeq 或直接调用模型得到预测（取最后一个时间步骤），
    并计算 MSE、NMSE、MAE、PSNR、RMSE 和 SSIM 评估指标。
    """
    model.eval()
    metrics = defaultdict(float)
    total_samples = 0

    # 确保保存图像的目录存在
    fig_dir = "fig_DRM"
    os.makedirs(fig_dir, exist_ok=True)

    with torch.no_grad():
        for idx, batch in tqdm(enumerate(val_loader), desc="Evaluating Radio_val"):
            inputs, targets, _ = batch
            inputs = inputs.to(device)
            targets = targets.to(device)
            batch_size = inputs.size(0)
            total_samples += batch_size

            if use_torch_diffeq:
                # 使用 torchdiffeq.odeint 进行轨迹积分，调用方式与 Trainer 中一致
                traj = torchdiffeq.odeint(
                    lambda t, x: model(image=inputs, x=x, pred_type="denoise", step=t),
                    torch.randn(batch_size, 1, 256, 256, device=device),
                    torch.linspace(0, 1, 2, device=device),
                    atol=1e-4,
                    rtol=1e-4,
                    method="dopri5",
                )
                # 取最后一个时间步骤作为预测结果
                pred = traj[-1]
            else:
                pred = model(image=inputs, x=torch.randn(batch_size, 1, 256, 256, device=device),
                             pred_type="denoise", step=1.0)

            mse_loss = F.mse_loss(pred, targets)
            target_energy = F.mse_loss(targets, torch.zeros_like(targets))
            # 避免除零
            if target_energy == 0:
                nmse_loss = float('inf')
            else:
                nmse_loss = mse_loss / target_energy
            mae_loss = F.l1_loss(pred, targets)
            mse_val = mse_loss.item()
            psnr = 20 * np.log10(1.0 / np.sqrt(mse_val)) if mse_val > 0 else float("inf")

            # 新增 RMSE 和 SSIM 计算
            rmse_loss = torch.sqrt(mse_loss)
            ssim_val = ssim(pred, targets, window_size=11, size_average=True)

            # 累加各个指标
            metrics["MSE"] += mse_loss.item() * batch_size
            metrics["NMSE"] += (nmse_loss.item() if isinstance(nmse_loss, torch.Tensor) else nmse_loss) * batch_size
            metrics["MAE"] += mae_loss.item() * batch_size
            metrics["PSNR"] += psnr * batch_size
            metrics["RMSE"] += rmse_loss.item() * batch_size
            metrics["SSIM"] += ssim_val.item() * batch_size

            print(f"Batch {idx} -- PSNR: {psnr:.4f}")
            print(f"Batch {idx} -- NMSE: {nmse_loss}")
            print(f"Batch {idx} -- RMSE: {rmse_loss.item():.4f}")
            print(f"Batch {idx} -- SSIM: {ssim_val.item():.4f}")

            
            numpy_array = pred.cpu().numpy().squeeze(1)  # shape: (batch_size, H, W)
            num_images = numpy_array.shape[0]
            rows = int(math.ceil(math.sqrt(num_images)))
            cols = int(math.ceil(num_images / rows))

            fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
            
            if rows == 1 or cols == 1:
                axes = np.array(axes).reshape(rows, cols)

            index = 0
            for i in range(rows):
                for j in range(cols):
                    if index < num_images:
                        axes[i, j].imshow(numpy_array[index], cmap='gray')
                        axes[i, j].axis('off')
                        index += 1
                    else:
                        axes[i, j].axis('off')
            plt.suptitle(f"NMSE: {nmse_loss}")
            plt.subplots_adjust(wspace=0.05, hspace=0.05)
            plt.savefig(os.path.join(fig_dir, f"traj_{idx}.png"), bbox_inches='tight', dpi=300)
            plt.close()

    averaged_metrics = {k: v / total_samples for k, v in metrics.items()}
    return averaged_metrics

def main():

    # 数据加载（请确保数据集路径正确）
    dataset_path = "/home/user/dxc/motion/MedSegDiff/RadioUNet/RadioMapSeer/"
    print(f"加载数据集: {dataset_path}")

    try:
        Radio_test = loaders.RadioUNet_c(phase="test", carsSimul="yes", carsInput="yes", dir_dataset="/home/user/dxc/motion/MedSegDiff/RadioUNet/RadioMapSeer/")
    except Exception as e:
        print(f"加载数据集时出错: {e}")
        traceback.print_exc()
        sys.exit(1)

    batch_size = 16
    
    val_loader = torch.utils.data.DataLoader(Radio_test, batch_size=batch_size, shuffle=False, num_workers=0)

    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    
    model = DiffUNet(con_channels=3)
    model.to(device)

    
    use_ema = True
    if use_ema:
        ema_checkpoint_path = '/home/user/dxc/radio/Diff-UNet/checkpoints/cfm_DRM_2/cfg_model_ema_step_600000.pt'
        if os.path.exists(ema_checkpoint_path):
            try:
                state_dict = torch.load(ema_checkpoint_path, map_location=device)
                model.load_state_dict(state_dict)
                print(f"成功加载 EMA 模型: {ema_checkpoint_path}")
            except Exception as e:
                print(f"加载 EMA 模型失败: {e}")
                traceback.print_exc()
                print("将使用普通模型进行测试")
        else:
            print(f"EMA 模型文件不存在: {ema_checkpoint_path}. 将使用普通模型")
    else:
        checkpoint_path = '/home/user/dxc/radio/Diff-UNet/checkpoints/cfm_DRM_2/cfg_model_ema_step_285000.pt'
        try:
            if os.path.exists(checkpoint_path):
                state_dict = torch.load(checkpoint_path, map_location=device)
                model.load_state_dict(state_dict)
                print(f"成功加载模型: {checkpoint_path}")
            else:
                print(f"模型文件不存在: {checkpoint_path}")
                print("将使用未训练的模型继续")
        except Exception as e:
            print(f"无法加载模型: {e}")
            traceback.print_exc()
            print("将使用未训练的模型继续")

    
    print("获取验证数据示例用于轨迹展示...")
    try:
        for batch in val_loader:
            inputs, targets, _ = batch
            break
        inputs = inputs.to(device)
        targets = targets.to(device)
        print(f"输入形状: {inputs.shape}, 目标形状: {targets.shape}")
    except Exception as e:
        print(f"获取数据时出错: {e}")
        traceback.print_exc()
        sys.exit(1)

    USE_TORCH_DIFFEQ = True
    with torch.no_grad():
        if USE_TORCH_DIFFEQ:
            traj = torchdiffeq.odeint(
                lambda t, x: model(image=inputs, x=x, pred_type="denoise", step=t),
                torch.randn(batch_size, 1, 256, 256, device=device),
                torch.linspace(0, 1, 5, device=device),
                atol=1e-4,
                rtol=1e-4,
                method="dopri5",
            )
        else:
            traj = model(image=inputs, x=torch.randn(batch_size, 1, 256, 256, device=device),
                         pred_type="denoise", step=1.0)

    
    final_traj = traj[-1]  # shape: (batch_size, 1, 256, 256)
    numpy_array = final_traj.cpu().numpy().squeeze(1)

    
    num_images = numpy_array.shape[0]
    rows = int(math.ceil(math.sqrt(num_images)))
    cols = int(math.ceil(num_images / rows))

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
    if rows == 1 or cols == 1:
        axes = np.array(axes).reshape(rows, cols)

    index = 0
    for i in range(rows):
        for j in range(cols):
            if index < num_images:
                axes[i, j].imshow(numpy_array[index], cmap='gray')
                axes[i, j].axis('off')
                index += 1
            else:
                axes[i, j].axis('off')
    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    plt.savefig("traj_final.png", bbox_inches='tight', dpi=300)
    plt.show()
    print("轨迹图已保存为 traj_final.png")

    # 在整个 Radio_val 数据集上进行评估
    print("开始在整个 Radio_val 数据集上评估...")
    metrics = evaluate_val(model, val_loader, device, use_torch_diffeq=USE_TORCH_DIFFEQ)
    print("在整个 Radio_val 数据集上的评估指标:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

if __name__ == "__main__":
    main()