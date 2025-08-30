"""
Configuration file for training scripts.
All paths and hyperparameters are defined here.
"""

class Config:
    # 数据集设置
    data_dir = "./RadioMapSeer/"  # 本地数据集路径
    batch_size = 32  # 批次大小
    num_workers = 0  # 数据加载的工作线程数，避免多进程加载问题
    cars_simul = "no"  # SRM任务不使用汽车仿真
    cars_input = "no"  # SRM任务不使用汽车输入
    simulation = "DPM"  # 数据集仿真类型

    # 训练设置
    n_epochs = 120  # 训练轮数
    lr = 3e-4  # 学习率
    weight_decay = 1e-4  # 权重衰减
    warmup_ratio = 0.1  # 预热比例，预热阶段占总轮数的比例
    ema_decay = 0.998  # 指数移动平均衰减率
    time_sampling_power = 2.0  # 幂函数时间采样的指数，值大于1使样本更集中于t=1

    # 日志和检查点保存间隔（以训练步骤计）
    log_interval = 100  # 日志记录间隔
    val_interval = 1000  # 验证间隔
    save_interval = 5000  # 检查点保存间隔

    # 模型设置
    con_channels = 2  # SRM任务使用2个通道
    save_dir = "checkpoints/flow_matching_SRM_v4"  # 检查点保存目录
    device = "cuda"  # 计算设备
    
    # DDPM设置
    num_timesteps = 1000  # DDPM时间步数
    beta_start = 1e-4  # 噪声调度起始值
    beta_end = 2e-2  # 噪声调度结束值
    beta_schedule = 'cosine'  # 噪声调度类型，使用余弦调度
    
    # 推理设置
    use_ddim = True  # 是否使用DDIM加速采样
    ddim_steps = 100  # DDIM采样步数
    cfg_scale = 1.5  # CFG引导强度
    
    # 图像设置
    img_size = 256  # 图像尺寸
    ssim_window = 11  # SSIM计算窗口大小

    # Flow Matching设置
    sigma = 0.0  # 噪声标准差，提供必要的随机性
    ode_method = 'dopri5'  # ODE求解器方法
    ode_steps = 100  # ODE求解步数
    ode_tol = 1e-5  # ODE求解容差
    
    # CFG参数
    cfg_drop_prob = 0.15  # 条件丢弃概率，增强CFG效果

    # 测试设置
    task = 'srm'  # 默认任务为drm，可通过命令行参数覆盖
    use_ode = False  # 默认不使用ODE求解器
    checkpoint = './checkpoint/SRM_Lite.pt'  # 默认checkpoint路径为空
    output_dir = 'results'  # 默认输出目录

class VizConfig:
    # 设备设置
    device = 'cuda'  # 计算设备，可选 'cpu'

    # 路径设置
    data_dir = './RadioMapSeer/'  # 本地数据集路径
    output_dir = './Compare_Results/'  # 输出结果目录

    # 模型检查点
    srm_ckpt = 'checkpoints/DDPM_SRM_Large.pt'  # SRM模型检查点路径
    drm_ckpt = 'checkpoints/DDPM_DRM_Large.pt'  # DRM模型检查点路径

    # 可视化设置
    sample_range = (601, 700)  # 测试地图的索引范围
    sample_count = 5  # 随机样本数量
    cfg_scales = [1.5, 2.5, 3.5, 4.5, 5.5]  # CFG引导强度列表
    ddim_steps = 50  # DDIM采样步数
    
    # DDPM设置
    num_timesteps = 1000  # DDPM时间步数
    use_ddim = True  # 使用DDIM加速采样
