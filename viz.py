"""
Unified RadioFlow CFG-scale comparison for SRM and DRM tasks.
Usage:
    python viz.py --task [srm|drm] [--solver euler] [--euler_steps N]
"""
import os
import argparse
import math
import numpy as np
import torch
import torchdiffeq
import matplotlib.pyplot as plt
from data_loaders.loaders import RadioUNet_c, RadioUNet_s
from model.model import DiffUNet
from config import VizConfig


def euler_integrate_with_cfg(model, inputs, x0, steps: int, cfg_scale: float, device: torch.device) -> torch.Tensor:
    """
    Fixed-step forward Euler ODE integration with CFG from t=0 to t=1.
    
    Args:
        model: The diffusion model
        inputs: Input conditions
        x0: Initial noise
        steps: Number of Euler integration steps
        cfg_scale: Classifier-free guidance scale
        device: Torch device
    
    Returns:
        Final denoised prediction after integration
    """
    x = x0
    dt = 1.0 / steps
    # Pre-compute conditional embeddings once
    embeddings = model.embed_model(inputs)
    for k in range(steps):
        t = torch.tensor(k / steps, device=device)
        v = model.forward_with_cfg(image=inputs, x=x, step=t, cfg_scale=cfg_scale, embedding=embeddings)
        x = x + dt * v
    return x


def select_samples(maps_inds, cfg: VizConfig):
    """
    Select a fixed number of random sample indices within the specified range.
    """
    start, end = cfg.sample_range
    subset = maps_inds[start:end]
    chosen = np.random.choice(len(subset), size=cfg.sample_count, replace=False)
    return [subset[i] for i in chosen], [start + i for i in chosen]


def run_comparison(task: str, cfg: VizConfig, solver: str = 'euler', euler_steps: int = 2, model_size: str = None):
    """
    Run CFG-scale comparison visualization.
    
    Args:
        task: Task type ('srm' or 'drm')
        cfg: VizConfig instance
        solver: ODE solver type ('euler' or 'odeint')
        euler_steps: Number of Euler steps (only used when solver='euler')
        model_size: Model size ('lite' or 'large'), if None uses cfg.model_size
    """
    # Use provided model_size or default from cfg
    if model_size is None:
        model_size = cfg.model_size
    # Determine parameters per task
    if task == 'srm':
        loader_cls = RadioUNet_c
        ckpt_path = os.path.join(cfg.output_dir, cfg.srm_ckpt)
        con_channels = 2
        input_channels = [0, 1]
    else:
        loader_cls = RadioUNet_s
        ckpt_path = os.path.join(cfg.output_dir, cfg.drm_ckpt)
        con_channels = 3
        input_channels = [0, 1, 3]

    # Prepare device
    device = torch.device(cfg.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Task: {task}")
    print(f"Model size: {model_size}")
    print(f"Solver: {solver}" + (f" (Euler steps: {euler_steps})" if solver == 'euler' else ""))

    # Load dataset
    dataset = loader_cls(
        phase='test',
        dir_dataset=cfg.data_dir,
        carsSimul='yes' if task=='drm' else 'no',
        carsInput='yes' if task=='drm' else 'no',
    )
    maps_inds = dataset.maps_inds

    # Random samples
    samples, dataset_ids = select_samples(maps_inds, cfg)

    # Load model
    model = DiffUNet(con_channels=con_channels, model_size=model_size)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.to(device).eval()

    # Prepare plot grid
    cols = len(cfg.cfg_scales)
    rows = cfg.sample_count
    fig, axes = plt.subplots(rows, cols, figsize=(cols*3, rows*3))
    titles = [f'cfg={scale}' for scale in cfg.cfg_scales]

    # Iterate samples
    with torch.no_grad():
        for r, idx in enumerate(dataset_ids):
            inp, target, _ = dataset[idx]
            # Input channels selection
            inp_flow = inp[input_channels].unsqueeze(0).to(device)

            for c, scale in enumerate(cfg.cfg_scales):
                # Solve ODE with selected solver
                x0 = torch.randn(1, 1, 256, 256, device=device)
                if solver == 'euler':
                    # Fast fixed-step Euler ODE integration
                    pred_tensor = euler_integrate_with_cfg(model, inp_flow, x0, euler_steps, scale, device)
                    pred = pred_tensor[0, 0].cpu().numpy()
                else:
                    # Adaptive high-order ODE solver
                    traj = torchdiffeq.odeint(
                        lambda t, x: model.forward_with_cfg(
                            image=inp_flow, x=x, step=t, cfg_scale=scale
                        ),
                        x0,
                        torch.linspace(0, 1, cfg.ode_steps, device=device),
                        atol=cfg.ode_tol, rtol=cfg.ode_tol, method='dopri5'
                    )
                    pred = traj[-1][0, 0].cpu().numpy()

                ax = axes[r, c]
                ax.imshow(pred, cmap='gray')
                ax.axis('off')
                if r == 0:
                    ax.set_title(titles[c], fontsize=12)

    plt.tight_layout(pad=0.4)
    os.makedirs(cfg.output_dir, exist_ok=True)
    out_file = os.path.join(cfg.output_dir, f'RadioFlow_cfg_{task}_{solver}_comparison.png')
    plt.savefig(out_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved visualization to: {out_file}")

def main():
    parser = argparse.ArgumentParser(description='RadioFlow CFG-scale visualization')
    parser.add_argument('--task', choices=['srm', 'drm'], required=True,
                        help='Task type: srm or drm')
    parser.add_argument('--model_size', type=str, default=None, 
                        choices=['lite', 'large'],
                        help='Model size: lite or large (default: use config value)')
    parser.add_argument('--solver', choices=['euler', 'odeint'], default=VizConfig.solver,
                        help='ODE solver type: euler (fast) or odeint (accurate)')
    parser.add_argument('--euler_steps', type=int, default=VizConfig.euler_steps,
                        help='Number of Euler integration steps')
    args = parser.parse_args()
    
    run_comparison(args.task, VizConfig, solver=args.solver, euler_steps=args.euler_steps, model_size=args.model_size)

if __name__ == '__main__':
    main()