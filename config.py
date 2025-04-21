"""
Configuration file for training scripts.
All paths and hyperparameters are defined here.
"""

class Config:
    # Dataset settings
    data_dir = "/ssd2/zhanghongbo04/project/RadioFlow/RadioMapSeer/"
    batch_size = 64
    num_workers = 4
    cars_simul = "yes"        # "yes" or "no" flag for car simulation
    cars_input = "yes"        # "yes" or "no" flag for car input

    # Training settings
    n_epochs = 1000
    lr = 1e-3
    weight_decay = 1e-5
    warmup_ratio = 0.1         # fraction of epochs for warmup
    ema_decay = 0.999          # EMA decay factor

    # Logging and checkpointing intervals (in training steps)
    log_interval = 100
    val_interval = 1000
    save_interval = 5000

    # Model settings
    con_channels = 3           # input channel dimension
    save_dir = "checkpoints/cfm_DRM"
    device = "cuda"

class VizConfig:
    # Device
    device = 'cuda'  # or 'cpu'

    # Paths
    data_dir = '/ssd2/project/RadioFlow/RadioMapSeer/'
    output_dir = '/ssd2/project/RadioFlow/Compare_Results/'

    # Model checkpoints
    srm_ckpt = 'checkpoints/SRM_Large.pt'
    drm_ckpt = 'checkpoints/DRM_Large.pt'

    # Visualization settings
    sample_range = (601, 700)   # indices range for test maps
    sample_count = 5            # number of random samples
    cfg_scales = [1.5, 2.5, 3.5, 4.5, 5.5]
    ode_steps = 2               # number of ODE steps (start and end)
    ode_tol = 1e-4              # ODE solver tolerance
