"""
Unified evaluation script for both DRM and SRM tasks.
Usage:
    python test.py --task drm   # for DRM evaluation
    python test.py --task srm   # for SRM evaluation
Parameters (with defaults in config.py) can be overridden via CLI.
"""
import os
import sys
import math
import argparse
import traceback
from collections import defaultdict

import torch
import torch.nn.functional as F
import torchdiffeq
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from skimage.metrics import structural_similarity as sk_ssim

from config import Config
from data_loaders.loaders import RadioUNet_c
from model.model import DiffUNet


def build_ssim_window(window_size: int, channel: int) -> torch.Tensor:
    """
    Create Gaussian window for SSIM calculation.
    """
    def gauss_1d(k, sigma=1.5):
        center = k // 2
        arr = torch.tensor([math.exp(-((i - center)**2) / (2 * sigma**2)) for i in range(k)])
        return arr / arr.sum()

    _1d = gauss_1d(window_size)
    _2d = _1d.unsqueeze(1) @ _1d.unsqueeze(0)
    window = _2d.expand(channel, 1, window_size, window_size).contiguous()
    return window


def ssim_torch(img1: torch.Tensor, img2: torch.Tensor, window: torch.Tensor) -> torch.Tensor:
    """
    Compute SSIM using convolution with precomputed Gaussian window.
    """
    c = img1.size(1)
    mu1 = F.conv2d(img1, window, padding=window.size(-1)//2, groups=c)
    mu2 = F.conv2d(img2, window, padding=window.size(-1)//2, groups=c)
    mu1_sq, mu2_sq = mu1**2, mu2**2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding=window.size(-1)//2, groups=c) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding=window.size(-1)//2, groups=c) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding=window.size(-1)//2, groups=c) - mu1_mu2

    C1, C2 = 0.01**2, 0.03**2
    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2)) / ((mu1_sq+mu2_sq+C1)*(sigma1_sq+sigma2_sq+C2))
    return ssim_map.mean()


def evaluate(model, loader, device, use_ode: bool, save_dir: str):
    """
    Run inference and compute MSE, NMSE, MAE, PSNR, RMSE, SSIM.
    Save sample trajectories as images.
    """
    model.eval()
    metrics = defaultdict(float)
    count = 0

    # prepare SSIM window
    window = build_ssim_window(Config.ssim_window, 1).to(device)

    os.makedirs(save_dir, exist_ok=True)

    with torch.no_grad():
        for batch_i, batch in enumerate(tqdm(loader, desc="Evaluating")):
            inputs, targets, _ = batch
            inputs, targets = inputs.to(device), targets.to(device)
            bsz = inputs.size(0)
            count += bsz

            # generate trajectory or direct output
            if use_ode:
                x0 = torch.randn(bsz, 1, Config.img_size, Config.img_size, device=device)
                t_steps = torch.linspace(0, 1, Config.ode_steps, device=device)
                traj = torchdiffeq.odeint(
                    lambda t, x: model(image=inputs, x=x, pred_type="denoise", step=t),
                    x0, t_steps,
                    atol=Config.ode_tol, rtol=Config.ode_tol
                )
                pred = traj[-1]
            else:
                pred = model(image=inputs,
                             x=torch.randn(bsz, 1, Config.img_size, Config.img_size, device=device),
                             pred_type="denoise", step=1.0)

            # compute losses
            mse = F.mse_loss(pred, targets)
            nmse = mse / (F.mse_loss(targets, torch.zeros_like(targets)) + 1e-12)
            mae = F.l1_loss(pred, targets)
            psnr = 20 * torch.log10(1.0 / torch.sqrt(mse + 1e-12))
            rmse = torch.sqrt(mse)
            ssim_val = ssim_torch(pred, targets, window)

            for name, val in zip(
                ["MSE", "NMSE", "MAE", "PSNR", "RMSE", "SSIM"],
                [mse, nmse, mae, psnr, rmse, ssim_val]
            ):
                metrics[name] += val.item() * bsz

            # save few sample traj images
            samples = pred.cpu().numpy().squeeze(1)
            grid_rows = math.ceil(math.sqrt(bsz))
            fig, axes = plt.subplots(grid_rows, grid_rows, figsize=(grid_rows*2,grid_rows*2))
            axes = axes.flatten()
            for i in range(bsz):
                axes[i].imshow(samples[i], cmap='gray')
                axes[i].axis('off')
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f"traj_{batch_i}.png"))
            plt.close()

    # average metrics
    return {k: v/count for k, v in metrics.items()}


def load_model_from_ckpt(model, path, device):
    """
    Safely load a checkpoint into the model.
    """
    if os.path.exists(path):
        try:
            state = torch.load(path, map_location=device)
            model.load_state_dict(state)
            print(f"Loaded checkpoint: {path}")
        except Exception:
            traceback.print_exc()
            print("Failed to load, using random init.")
    else:
        print(f"Checkpoint not found: {path}, using random init.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', choices=['drm','srm'], required=True,
                        help='Which task to evaluate')
    parser.add_argument('--use_ode', action='store_true', help='Use ODE solver')
    parser.add_argument('--checkpoint', type=str, help='Path to model checkpoint')
    parser.add_argument('--output_dir', type=str, default='results', help='Where to save outputs')
    args = parser.parse_args()

    # pick config based on task
    cfg = Config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # dataset
    dataset = RadioUNet_c(
        phase='test',
        carsSimul=(cfg.cars_simul if args.task=='drm' else 'no'),
        carsInput=(cfg.cars_input if args.task=='drm' else 'no'),
        dir_dataset=cfg.data_dir
    )
    loader = torch.utils.data.DataLoader(dataset,
                                         batch_size=cfg.batch_size,
                                         shuffle=False,
                                         num_workers=cfg.num_workers)

    # model
    model = DiffUNet(con_channels=(3 if args.task=='drm' else 2))
    model.to(device)

    # load weights
    ckpt = args.checkpoint or (cfg.drm_ckpt if args.task=='drm' else cfg.srm_ckpt)
    load_model_from_ckpt(model, ckpt, device)

    # eval
    out_dir = os.path.join(args.output_dir, args.task)
    metrics = evaluate(model, loader, device, args.use_ode, out_dir)
    print("Results:")
    for k,v in metrics.items():
        print(f"{k}: {v:.4f}")

if __name__ == '__main__':
    main()
