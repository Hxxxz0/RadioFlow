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
import argparse
import glob
import shutil
from PIL import Image
from data_loaders import loaders
from model.model import DiffUNet
from torchvision import datasets, transforms
from torchvision.transforms import ToPILImage
from torchvision.utils import make_grid
from torchdyn.core import NeuralODE
from torchcfm.conditional_flow_matching import ConditionalFlowMatcher

def evaluate_val(model, val_loader, device, use_torch_diffeq=True):
    """
    Traverse the entire Radio_val dataset, using torchdiffq for each batch or calling the model directly to get the prediction (take the last time step),
And calculate MSE, NMSE, MAE and PSNR evaluation indicators.
    """
    model.eval()
    metrics = defaultdict(float)
    total_samples = 0

    # 确保保存图像的目录存在
    fig_dir = "fig"
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

            metrics["MSE"] += mse_loss.item() * batch_size
            metrics["NMSE"] += (nmse_loss.item() if isinstance(nmse_loss, torch.Tensor) else nmse_loss) * batch_size
            metrics["MAE"] += mae_loss.item() * batch_size
            metrics["PSNR"] += psnr * batch_size
            print(f"Batch {idx} -- PSNR: {psnr:.4f}")
            print(f"Batch {idx} -- NMSE: {nmse_loss}")

            # 绘制并保存当前批次的预测图
            numpy_array = pred.cpu().numpy().squeeze(1)  # shape: (batch_size, H, W)
            num_images = numpy_array.shape[0]
            rows = int(math.ceil(math.sqrt(num_images)))
            cols = int(math.ceil(num_images / rows))

            fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
            # 如果只有一行或一列，保证 axes 是二维数组
            if rows == 1 or cols == 1:
                axes = np.array(axes).reshape(rows, cols)

            index = 0
            for i in range(rows):
                for j in range(cols):
                    if index < num_images:
                        axes[i, j].imshow(numpy_array[index])
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

def run_trajectory_inference(model, image, steps=100, x_t=None, device=None):
    # Ensure image directory exists for saving images
    os.makedirs("val_result", exist_ok=True)
    os.makedirs("val_result/Trajectory", exist_ok=True)

    conditioner = image.to(device)  # Use original image for conditioning
    # Initial noise (can be random or provided)
    if x_t is None:
        x_t = torch.randn((1, 1, image.shape[2], image.shape[3])).to(device)
    
    def ode_func(t, x):
        """Vector field function for the ODE solver."""
        with torch.no_grad():
            # Convert scalar t to tensor with batch dimension 
            t_tensor = torch.ones((x.shape[0],), device=device) * t
            
            # Use torchdiffeq.odeint for trajectory integration, calling in the same way as in the Trainer
            score = model(image=conditioner, x=x, pred_type="denoise", step=t_tensor)
            return score
    
    trajectory = []
    t_span = torch.linspace(0, 1.0, steps).to(device)
    
    # Record start time
    start_time = time.time()
    
    # Solve ODE to get trajectory
    solution = torchdiffeq.odeint(
        ode_func, 
        x_t.reshape(1, -1), 
        t_span,
        method='dopri5',
        atol=1e-4,
        rtol=1e-4,
    )
    
    # Get the last time step as prediction result
    solution = solution.reshape(steps, 1, 1, image.shape[2], image.shape[3])
    trajectory = solution.detach().cpu()  # Store full trajectory for visualization
    
    end_time = time.time()
    print(f"Inference time: {end_time - start_time:.2f}s")
    
    return trajectory

def calculate_metrics(pred, target):
    # Avoid division by zero
    epsilon = 1e-10
    
    # Convert tensors to numpy arrays
    pred_np = pred.detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()
    
    # Calculate MSE
    mse = np.mean((pred_np - target_np) ** 2)
    
    # Calculate PSNR
    max_i = 1.0  # Assuming normalized to [0, 1]
    psnr = 20 * np.log10(max_i / (np.sqrt(mse) + epsilon))
    
    return mse, psnr

def plot_trajectory(trajectory, inputs, targets, save_path, n_steps_display=10):
    # Plot and save current batch predictions
    trajectory_numpy = trajectory.numpy()
    inputs_numpy = inputs.detach().cpu().numpy()
    targets_numpy = targets.detach().cpu().numpy()
    
    n_samples = trajectory_numpy.shape[1]
    
    # If there's only one row or one column, ensure axes is a 2D array
    fig, axes = plt.subplots(n_samples, n_steps_display + 2, figsize=(3 * (n_steps_display + 2), 3 * n_samples))
    
    if n_samples == 1:
        axes = axes.reshape(1, -1)
    
    display_indices = np.linspace(0, trajectory_numpy.shape[0] - 1, n_steps_display, dtype=int)
    
    for i in range(n_samples):
        # Plot input
        axes[i, 0].imshow(inputs_numpy[i, 0], cmap='gray')
        axes[i, 0].set_title("Input")
        axes[i, 0].axis('off')
        
        # Plot trajectory steps
        for j, idx in enumerate(display_indices):
            axes[i, j + 1].imshow(trajectory_numpy[idx, i, 0], cmap='gray')
            axes[i, j + 1].set_title(f"Step {idx}")
            axes[i, j + 1].axis('off')
        
        # Plot target and final prediction
        axes[i, -1].imshow(targets_numpy[i, 0], cmap='gray')
        axes[i, -1].set_title("Target")
        axes[i, -1].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved trajectory plot to {save_path}")

def main():
    # Directory for saving model and results
    base_dir = os.path.dirname(os.path.abspath(__file__))
    save_dir = os.path.join(base_dir, "checkpoints", "test_output")
    os.makedirs(save_dir, exist_ok=True)
    
    # Data loading (make sure dataset paths are correct)
    data_root_1 = os.path.join(base_dir, "data_loaders/MR2CT/test")
    val_radiomic = loaders.Dataset_PairedImages_NoLabel(data_root_1)
    
    # Here we choose to use the test dataset for evaluation, can change to val if needed
    val_dataset = val_radiomic
    print(f"Validation samples: {len(val_dataset)}")
    
    # Create a data loader for validation
    batch_size = 1  # For visualization clarity
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,  # To ensure evaluation order, set shuffle=False
        num_workers=4
    )
    
    # Device setup (modify device number as needed)
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create lightweight model
    model = DiffUNet(con_channels=1).to(device)
    
    # Check for pre-trained model, prioritize loading EMA model (if exists)
    path_list = glob.glob('checkpoints/*/models/cfg_model_ema_best.pt')
    if path_list:
        # Sort by modification time (get the latest)
        path_list.sort(key=os.path.getmtime, reverse=True)
        ema_path = path_list[0]
        print(f"Loading EMA model from: {ema_path}")
        model.load_state_dict(torch.load(ema_path, map_location=device))
    else:
        # Try to load regular model if EMA not found
        path_list = glob.glob('checkpoints/*/models/cfg_model_best.pt')
        if path_list:
            path_list.sort(key=os.path.getmtime, reverse=True)
            model_path = path_list[0]
            print(f"Loading model from: {model_path}")
            model.load_state_dict(torch.load(model_path, map_location=device))
        else:
            print("No pre-trained model found. Starting from scratch.")
    
    model.eval()
    print(f"Model parameter count: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Sample for trajectory visualization
    sample_inputs = []
    sample_targets = []
    
    # Save model outputs for all validation samples
    all_metrics = []
    with torch.no_grad():
        for i, (inputs, targets, _) in enumerate(tqdm(val_loader, desc="Evaluation")):
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            # Save some samples for trajectory visualization
            if i < 5 and i == 0:  # Save first 5 samples
                sample_inputs.append(inputs.cpu())
                sample_targets.append(targets.cpu())
                
                # Visualize trajectory for these samples
                trajectory = run_trajectory_inference(model, inputs, steps=100, device=device)
                
                # Only take the last time step as final prediction result
                pred = trajectory[-1]
                
                # Automatically calculate grid layout based on sample count
                fig_path = os.path.join(save_dir, f"sample_{i}_trajectory.png")
                plot_trajectory(trajectory, inputs.cpu(), targets.cpu(), fig_path)
                
                # Calculate metrics for this sample
                mse, psnr = calculate_metrics(pred, targets.cpu())
                print(f"Sample {i}: MSE = {mse:.4f}, PSNR = {psnr:.2f} dB")
                all_metrics.append((mse, psnr))
            
            # For all samples, calculate metrics
            trajectory = run_trajectory_inference(model, inputs, steps=100, device=device)
            pred = trajectory[-1]
            mse, psnr = calculate_metrics(pred, targets.cpu())
            all_metrics.append((mse, psnr))
            
            # Save the prediction
            save_path = os.path.join(save_dir, f"sample_{i}_pred.png")
            plt.figure(figsize=(15, 5))
            plt.subplot(131)
            plt.imshow(inputs.cpu().numpy()[0, 0], cmap='gray')
            plt.title("Input (MR)")
            plt.axis('off')
            
            plt.subplot(132)
            plt.imshow(pred.numpy()[0, 0], cmap='gray')
            plt.title("Prediction (CT)")
            plt.axis('off')
            
            plt.subplot(133)
            plt.imshow(targets.cpu().numpy()[0, 0], cmap='gray')
            plt.title("Ground Truth (CT)")
            plt.axis('off')
            
            plt.tight_layout()
            plt.savefig(save_path)
            plt.close()
    
    # Calculate and report average metrics
    mse_values, psnr_values = zip(*all_metrics)
    avg_mse = np.mean(mse_values)
    avg_psnr = np.mean(psnr_values)
    
    print("\n==== Evaluation on Radio_val dataset ====")
    print(f"Average MSE: {avg_mse:.4f}")
    print(f"Average PSNR: {avg_psnr:.2f} dB")
    
    # Save metrics to file
    with open(os.path.join(save_dir, "metrics.txt"), "w") as f:
        f.write(f"Average MSE: {avg_mse:.4f}\n")
        f.write(f"Average PSNR: {avg_psnr:.2f} dB\n")
        f.write("\nIndividual samples:\n")
        for i, (mse, psnr) in enumerate(all_metrics):
            f.write(f"Sample {i}: MSE = {mse:.4f}, PSNR = {psnr:.2f} dB\n")

if __name__ == "__main__":
    main()