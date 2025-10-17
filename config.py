"""
Configuration file for training and evaluation scripts.
All paths and hyperparameters are defined here.
"""

# Model architecture feature configurations
MODEL_FEATURES = {
    'lite': (32, 32, 64, 128, 256, 32),
    'large': (128, 128, 256, 512, 1024, 128)
}

class Config:
    # Dataset settings
    data_dir = "./RadioMapSeer/"  # Local dataset path
    task = 'srm'  # Default task: 'srm' or 'drm'
    
    batch_size = 64
    num_workers = 64
    cars_simul = "yes"  # "yes" or "no" flag for car simulation
    cars_input = "yes"  # "yes" or "no" flag for car input
    simulation = "DPM"  # Dataset simulation type

    # Training settings
    n_epochs = 1000
    lr = 1e-3
    weight_decay = 1e-5
    warmup_ratio = 0.1  # Fraction of epochs for warmup
    ema_decay = 0.999   # EMA decay factor

    # Logging and checkpointing intervals (in training steps)
    log_interval = 100
    val_interval = 1000
    save_interval = 5000
    
    # Model and checkpoint settings
    model_size = 'lite'  # Model size: 'lite' or 'large'
    con_channels = 3  # Input channel dimension
    save_dir = "checkpoints/cfm_DRM"
    output_dir = 'results'  # Default output directory
    checkpoint = 'SRM_Lite.pt'  # Default checkpoint path
    device = "cuda"

    # Image settings
    img_size = 256       # Image size (height and width)
    ssim_window = 11     # SSIM calculation window size

    # ODE solver settings (for inference)
    use_ode = True       # Use ODE solver for inference
    solver = 'euler'     # ODE solver type: 'euler' (fast) or 'odeint' (accurate)
    euler_steps = 2      # Number of Euler integration steps (used when solver='euler')
    ode_steps = 2        # Number of time steps for odeint (used when solver='odeint')
    ode_tol = 1e-5       # ODE solver tolerance for odeint

    # Flow Matching settings (for training)
    sigma = 0.0  # Noise std for flow matching


class VizConfig:
    # Device
    device = 'cuda'  # Device: 'cuda' or 'cpu'

    # Paths
    data_dir = "./RadioMapSeer/"  # Dataset path
    output_dir = "./"  # Output directory for visualizations

    # Model settings
    model_size = 'lite'  # Model size: 'lite' or 'large'
    
    # Model checkpoints
    srm_ckpt = 'SRM_Lite.pt'
    drm_ckpt = 'DRM_Lite.pt'

    # Visualization settings
    sample_range = (601, 700)  # Index range for test maps
    sample_count = 5           # Number of random samples to visualize
    cfg_scales = [1.5, 2.5, 3.5, 4.5, 5.5]  # CFG scale values to compare
    
    # ODE solver settings
    solver = 'euler'     # ODE solver type: 'euler' or 'odeint'
    euler_steps = 2      # Number of Euler steps (used when solver='euler')
    ode_steps = 2        # Number of time steps for odeint (used when solver='odeint')
    ode_tol = 1e-4       # ODE solver tolerance for odeint
