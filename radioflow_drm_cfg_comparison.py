import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from data_loaders import loaders
from model.model import DiffUNet
import torchdiffeq

# ----------- Configuration -----------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
radioflow_model_path = '/ssd2/project/RadioFlow/checkpoints/DRM_Large.pt'
dataset_path = '/ssd2/project/RadioFlow/RadioMapSeer/'
output_path = '/ssd2/project/RadioFlow/Compare_Results/RadioFlow_cfg_drm_comparison.png'
os.makedirs(os.path.dirname(output_path), exist_ok=True)

# ----------- Load Test Set -----------
Radio_test = loaders.RadioUNet_s(phase="test", carsSimul="yes", carsInput="yes", dir_dataset=dataset_path)
maps_inds = Radio_test.maps_inds
test_map_ids = maps_inds[601:700]

# Randomly select samples
np.random.seed(None)
chosen_indices = np.random.choice(len(test_map_ids), size=5, replace=False)
random_map_ids = test_map_ids[chosen_indices]
dataset_indices = chosen_indices

# ----------- Load Model -----------
radioflow = DiffUNet(con_channels=3)  # Using building, tx, and cars input channels
radioflow.load_state_dict(torch.load(radioflow_model_path, map_location=device))
radioflow.to(device).eval()

# ----------- Visualization Setup -----------
cfg_scales = [1.5, 2.5, 3.5, 4.5, 5.5]
fig, axes = plt.subplots(5, len(cfg_scales), figsize=(15, 14))  # 5 rows, columns based on cfg_scale values
titles = ['Ground Truth'] + [f'RadioFlow (cfg={cfg})' for cfg in cfg_scales]

for row, dataset_id in enumerate(dataset_indices):
    map_id = random_map_ids[row]
    inputs, targets = Radio_test[dataset_id]  # [C, H, W]

    targets = targets.unsqueeze(0).to(device)

    # Separate channels (building, tx, cars)
    input_flow = inputs[[0, 1, 3]].unsqueeze(0).to(device)

    with torch.no_grad():
        target_np = targets[0, 0].cpu().numpy()

        # Calculate RadioFlow results for different cfg_scales
        for col, cfg in enumerate(cfg_scales):
            traj = torchdiffeq.odeint(
                lambda t, x: radioflow.forward_with_cfg(image=input_flow, x=x, step=t, cfg_scale=cfg),
                torch.randn(1, 1, 256, 256, device=device),
                torch.linspace(0, 1, 2, device=device),
                atol=1e-4, rtol=1e-4, method="dopri5"
            )
            pred_flow = traj[-1][0, 0].cpu().numpy()

            # Plot results
            ax = axes[row, col]
            if row == 0:
                ax.set_title(titles[col], fontsize=12)
            ax.imshow(pred_flow)
            ax.axis('off')

# ----------- Save Results -----------
plt.tight_layout(pad=0.5, w_pad=0.1, h_pad=0.2)
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.close()
print("✅ RadioFlow visualization comparison saved to:", output_path)
