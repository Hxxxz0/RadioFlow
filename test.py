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


def euler_integrate(model, inputs, x0, steps: int, device: torch.device) -> torch.Tensor:
    """
    Fixed-step forward Euler ODE integration from t=0 to t=1.
    
    Args:
        model: The diffusion model
        inputs: Input conditions (buildings, transmitter, etc.)
        x0: Initial noise
        steps: Number of Euler integration steps
        device: Torch device
    
    Returns:
        Final denoised prediction after integration
    """
    x = x0
    dt = 1.0 / steps
    # Pre-compute conditional embeddings once to avoid redundant encoding at each step
    embeddings = model.embed_model(inputs)
    for k in range(steps):
        t = torch.tensor(k / steps, device=device)
        v = model(image=inputs, x=x, pred_type="denoise", step=t, embedding=embeddings)
        x = x + dt * v
    return x


def evaluate(model, loader, device, use_ode: bool, save_dir: str, 
             solver: str = "euler", euler_steps: int = 2):
    """
    Run inference and compute MSE, NMSE, MAE, PSNR, RMSE, SSIM.
    Save sample trajectories as images.
    
    Args:
        model: The diffusion model
        loader: DataLoader for test set
        device: Torch device
        use_ode: Whether to use ODE solver (vs single-step inference)
        save_dir: Directory to save output images
        solver: ODE solver type ('euler' or 'odeint')
        euler_steps: Number of Euler steps (only used when solver='euler')
    
    Returns:
        Dictionary of averaged metrics
    """
    model.eval()
    metrics = defaultdict(float)
    count = 0

    # Prepare SSIM window
    window = build_ssim_window(Config.ssim_window, 1).to(device)

    os.makedirs(save_dir, exist_ok=True)

    with torch.no_grad():
        for batch_i, batch in enumerate(tqdm(loader, desc="Evaluating")):
            inputs, targets, _ = batch
            inputs, targets = inputs.to(device), targets.to(device)
            bsz = inputs.size(0)
            count += bsz

            # Generate trajectory or direct output
            if use_ode:
                x0 = torch.randn(bsz, 1, Config.img_size, Config.img_size, device=device)
                if solver == "euler":
                    # Fast fixed-step Euler ODE integration
                    pred = euler_integrate(model, inputs, x0, euler_steps, device)
                else:
                    # Adaptive high-order ODE solver (slower but more accurate)
                    t_steps = torch.linspace(0, 1, Config.ode_steps, device=device)
                    traj = torchdiffeq.odeint(
                        lambda t, x: model(image=inputs, x=x, pred_type="denoise", step=t),
                        x0, t_steps,
                        atol=Config.ode_tol, rtol=Config.ode_tol
                    )
                    pred = traj[-1]
            else:
                # Single-step direct inference (no ODE)
                pred = model(image=inputs,
                             x=torch.randn(bsz, 1, Config.img_size, Config.img_size, device=device),
                             pred_type="denoise", step=torch.tensor(1.0, device=device))

            # Compute metrics
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

            # Save sample trajectory images
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

    # Average metrics across all samples
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
    parser = argparse.ArgumentParser(description='RadioFlow evaluation script')
    parser.add_argument('--task', choices=['drm','srm'], default=Config.task,
                        help='Task to evaluate: drm or srm')
    parser.add_argument('--model_size', type=str, default=None, 
                        choices=['lite', 'large'],
                        help='Model size: lite or large (default: use config value)')
    parser.add_argument('--use_ode', action='store_true', default=Config.use_ode, 
                        help='Use ODE solver for inference')
    parser.add_argument('--solver', choices=['odeint','euler'], default=Config.solver, 
                        help='ODE solver type: euler (fast) or odeint (accurate)')
    parser.add_argument('--euler_steps', type=int, default=Config.euler_steps, 
                        help='Number of Euler integration steps')
    parser.add_argument('--checkpoint', type=str, default=Config.checkpoint, 
                        help='Path to model checkpoint')
    parser.add_argument('--output_dir', type=str, default=Config.output_dir, 
                        help='Output directory for results')
    args = parser.parse_args()

    # Load configuration
    cfg = Config()
    
    # Override model_size if provided via command line
    if args.model_size:
        cfg.model_size = args.model_size
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Task: {args.task}")
    print(f"Model size: {cfg.model_size}")
    print(f"Solver: {args.solver} (Euler steps: {args.euler_steps})" if args.solver == 'euler' else f"Solver: {args.solver}")

    # Prepare dataset
    dataset = RadioUNet_c(
        phase='test',
        carsSimul=(cfg.cars_simul if args.task=='drm' else 'no'),
        carsInput=(cfg.cars_input if args.task=='drm' else 'no'),
        dir_dataset=cfg.data_dir,
        simulation=cfg.simulation
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers
    )

    # Initialize model
    model = DiffUNet(con_channels=(3 if args.task=='drm' else 2), model_size=cfg.model_size)
    model.to(device)

    # Load checkpoint
    load_model_from_ckpt(model, args.checkpoint, device)

    # Run evaluation
    out_dir = os.path.join(args.output_dir, args.task)
    metrics = evaluate(
        model, loader, device, args.use_ode, out_dir, 
        solver=args.solver, euler_steps=args.euler_steps
    )
    
    # Print results
    print("\n" + "="*50)
    print("Evaluation Results:")
    print("="*50)
    for k, v in metrics.items():
        print(f"{k:6s}: {v:.4f}")
    print("="*50)

if __name__ == '__main__':
    main()
