"""
Unified RadioFlow CFG-scale comparison for SRM and DRM tasks.
Usage:
    python viz.py --task [srm|drm]
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


def select_samples(maps_inds, cfg: VizConfig):
    """
    Select a fixed number of random sample indices within the specified range.
    """
    start, end = cfg.sample_range
    subset = maps_inds[start:end]
    chosen = np.random.choice(len(subset), size=cfg.sample_count, replace=False)
    return [subset[i] for i in chosen], [start + i for i in chosen]


def run_comparison(task: str, cfg: VizConfig):
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

    # Load dataset
    dataset = loader_cls(
        phase='test',
        dir_dataset=cfg.data_dir,
        carsSimul='yes' if task=='drm' else None,
        carsInput='yes' if task=='drm' else None,
    )
    maps_inds = dataset.maps_inds

    # Random samples
    samples, dataset_ids = select_samples(maps_inds, cfg)

    # Load model
    model = DiffUNet(con_channels=con_channels)
    state = torch.load(os.path.join(cfg.data_dir, ckpt_path), map_location=device)
    model.load_state_dict(state)
    model.to(device).eval()

    # Prepare plot grid
    cols = len(cfg.cfg_scales)
    rows = cfg.sample_count
    fig, axes = plt.subplots(rows, cols, figsize=(cols*3, rows*3))
    titles = [f'cfg={scale}' for scale in cfg.cfg_scales]

    # Iterate samples
    for r, idx in enumerate(dataset_ids):
        inp, target = dataset[idx]
        target_np = target.unsqueeze(0)[0, 0].cpu().numpy()
        # input channels selection
        inp_flow = inp[input_channels].unsqueeze(0).to(device)

        for c, scale in enumerate(cfg.cfg_scales):
            # solve ODE
            traj = torchdiffeq.odeint(
                lambda t, x: model.forward_with_cfg(
                    image=inp_flow, x=x, step=t, cfg_scale=scale
                ),
                torch.randn(1, 1, 256, 256, device=device),
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
    out_file = os.path.join(cfg.output_dir, f'RadioFlow_cfg_{task}_comparison.png')
    plt.savefig(out_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved visualization to: {out_file}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', choices=['srm', 'drm'], required=True,
                        help='Task type: srm or drm')
    args = parser.parse_args()
    run_comparison(args.task, VizConfig)

if __name__ == '__main__':
    main()