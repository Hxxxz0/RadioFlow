"""
Configuration file for training scripts.
All paths and hyperparameters are defined here.
"""

"""
Configuration file for training scripts.
All paths and hyperparameters are defined here.
"""

class Config:
    # Dataset settings
    data_dir = "./RadioMapSeer/"  # 本地数据集路径

    batch_size = 64
    num_workers = 4
    cars_simul = "yes"        # "yes" or "no" flag for car simulation
    cars_input = "yes"        # "yes" or "no" flag for car input
    simulation = "DPM"  # 数据集仿真类型
    task = 'srm'  # 默认任务为drm，可通过命令行参数覆盖

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
    output_dir = 'results'  # 默认输出目录
    # Model settings
    con_channels = 3           # input channel dimension
    save_dir = "checkpoints/cfm_DRM"
    device = "cuda"
    checkpoint = './checkpoints/SRM_Lite.pt'  # 默认checkpoint路径
    use_ode = True  # 默认使用ODE求解器


    # 图像设置
    img_size = 256  # 图像尺寸
    ssim_window = 11  # SSIM计算窗口大小

    # Flow Matching设置
    sigma = 0.0  # 噪声标准差，提供必要的随机性
    ode_method = 'euler'  # ODE求解器方法
    ode_steps = 2  # ODE求解步数
    ode_tol = 1e-5  # ODE求解容差


class VizConfig:
    # Device
    device = 'cuda'  # or 'cpu'

    # Paths
    srm_ckpt = 'checkpoints/DDPM_SRM_Large.pt'  # SRM模型检查点路径
    drm_ckpt = 'checkpoints/DDPM_DRM_Large.pt'  # DRM模型检查点路径

    # Model checkpoints
    srm_ckpt = 'checkpoints/SRM_Large.pt'
    drm_ckpt = 'checkpoints/DRM_Large.pt'

    # Visualization settings
    sample_range = (601, 700)   # indices range for test maps
    sample_count = 5            # number of random samples
    cfg_scales = [1.5, 2.5, 3.5, 4.5, 5.5]
    ode_steps = 2               # number of ODE steps (start and end)
    ode_tol = 1e-4              # ODE solver tolerance
