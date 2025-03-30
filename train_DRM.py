import os
import time
import copy
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm
from data_loaders import loaders
from model.model import DiffUNet
import torchdiffeq
from torchdyn.core import NeuralODE
from torchcfm.conditional_flow_matching import ConditionalFlowMatcher
from torchcfm.conditional_flow_matching import ConditionalFlowMatcher,SchrodingerBridgeConditionalFlowMatcher,ExactOptimalTransportConditionalFlowMatcher

########################################
# 学习率调度函数 (Warmup + Cosine)
########################################
def build_lr_lambda(total_epochs, warmup_ratio=0.1):
    """
    total_epochs : 总的训练轮次
    warmup_ratio : warmup 阶段占比，例如 0.1 表示 1/10 的 epoch 用于 warmup
    """
    warmup_epochs = int(total_epochs * warmup_ratio)

    def lr_lambda(epoch):
        # 在 warmup_epochs 之前进行线性增加
        if epoch < warmup_epochs:
            return float(epoch) / float(warmup_epochs)
        # 之后进行余弦退火
        progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
        return 0.5 * (1.0 + np.cos(np.pi * progress))
    return lr_lambda

########################################
#  模型EMA类
########################################
class ModelEMA:
    """
    用于维护一个模型的滑动平均（Exponential Moving Average）。
    ema_model 与原始模型具有相同结构，但参数是平滑更新的。
    """
    def __init__(self, model, decay=0.999):
        # 创建一份与 model 同结构的模型，并复制参数
        self.ema_model = copy.deepcopy(model)
        # 确保不计算梯度
        for param in self.ema_model.parameters():
            param.requires_grad_(False)
        self.decay = decay

    def update(self, model):
        """
        在每个训练 step 后调用，用于更新 EMA 模型参数
        """
        with torch.no_grad():
            msd = model.state_dict()       
            esd = self.ema_model.state_dict()
            for k in esd.keys():
                if k in msd:
                    esd[k].mul_(self.decay).add_(msd[k], alpha=1 - self.decay)
            self.ema_model.load_state_dict(esd)

########################################
#  绘制损失曲线函数
########################################
def plot_loss_curve(losses, val_losses, save_dir):
    plt.figure(figsize=(10, 5))
    plt.plot(losses, label='训练损失')
    if val_losses:
        # 为验证曲线创建对应的x轴坐标
        val_x = np.linspace(0, len(losses), len(val_losses))
        plt.plot(val_x, val_losses, label='验证损失')
    plt.xlabel('训练步骤')
    plt.ylabel('损失')
    plt.title('模型训练损失曲线')
    plt.legend()
    path = os.path.join(save_dir, "loss_curve.png")
    plt.savefig(path)
    plt.close()
    print(f"损失曲线已保存到 {path}")

########################################
#  高效Trainer类
########################################
class AdvancedTrainer:
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        device,
        save_dir,
        n_epochs=1000,
        lr=1e-3,
        weight_decay=1e-5,
        warmup_ratio=0.1,
        ema_decay=0.999,
        log_interval=100,
        val_interval=1000,
        save_interval=5000,
    ):
        """
        参数说明:
          model         : PyTorch 模型 (e.g. DiffUNet 实例)
          train_loader  : 训练集 DataLoader
          val_loader    : 验证集 DataLoader
          device        : 设备 (cpu / cuda)
          save_dir      : 模型与可视化结果的保存目录
          n_epochs      : 训练轮数
          lr            : 初始学习率
          weight_decay  : AdamW 的 weight decay
          warmup_ratio  : warmup 占比
          ema_decay     : EMA 的衰减因子
          log_interval  : 打印日志的间隔 (steps)
          val_interval  : 验证的间隔 (steps)
          save_interval : 存储 checkpoint 的间隔 (steps)
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

        self.n_epochs = n_epochs
        self.log_interval = log_interval
        self.val_interval = val_interval
        self.save_interval = save_interval
        
        # 优化器: AdamW
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)

        # 学习率调度器: Warmup + Cosine
        lr_lambda = build_lr_lambda(total_epochs=self.n_epochs, warmup_ratio=warmup_ratio)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_lambda)
        
        # 混合精度
        self.scaler = torch.cuda.amp.GradScaler()

        # EMA
        self.ema_decay = ema_decay
        self.model_ema = ModelEMA(model, decay=self.ema_decay)

        # 训练过程统计
        self.start_time = time.time()
        self.step_count = 0
        self.losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')

        # 其他的必要设置
        self.FM = ConditionalFlowMatcher(sigma=0.0)  # 只需一个简单FM示例
        #self.FM = ExactOptimalTransportConditionalFlowMatcher(sigma=0.0)
        # 如果你需要 NeuralODE 或其他，就自行扩展
        self.node = NeuralODE(self.model, solver="dopri5", sensitivity="adjoint", atol=1e-4, rtol=1e-4)

    def train_one_epoch(self, epoch_idx):
        """
        单个 epoch 的训练过程
        """
        self.model.train()
        epoch_loss = 0.0
        
        for i, batch_data in enumerate(tqdm(self.train_loader, desc=f"Epoch [{epoch_idx+1}/{self.n_epochs}]")):
            inputs, targets, _ = batch_data
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            self.optimizer.zero_grad()

            # 生成随机初始噪声
            x0 = torch.randn_like(targets)
            # 采样时间和流
            t, xt, ut = self.FM.sample_location_and_conditional_flow(x0, targets)
            
            with torch.cuda.amp.autocast():
                vt = self.model(image=inputs, x=xt, pred_type="denoise", step=t)
                loss = torch.mean((vt - ut)**2)
            
            # 混合精度
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            # 更新 EMA
            self.model_ema.update(self.model)

            # 统计
            epoch_loss += loss.item()
            self.losses.append(loss.item())
            self.step_count += 1

            # 日志
            if self.step_count % self.log_interval == 0:
                elapsed = time.time() - self.start_time
                current_lr = self.optimizer.param_groups[0]['lr']
                print(f"[Epoch {epoch_idx+1}/{self.n_epochs} | Step {self.step_count}] "
                      f"Loss = {loss.item():.4f}, LR = {current_lr:.6f}, Elapsed = {elapsed:.2f}s")

            # 验证
            if self.step_count % self.val_interval == 0:
                self.validate_and_plot()

            # 保存模型
            if self.step_count % self.save_interval == 0:
                self.save_checkpoint(step_tag=f"step_{self.step_count}")

        epoch_loss /= len(self.train_loader)
        print(f"=> Epoch [{epoch_idx+1}/{self.n_epochs}] Finished. Avg Train Loss: {epoch_loss:.4f}")
        return epoch_loss

    def validate_and_plot(self):
        """
        在 val_loader 上进行验证，并绘制损失曲线
        """
        self.model.eval()
        val_loss_total = 0.0
        
        with torch.no_grad():
            for val_data in self.val_loader:
                val_inputs, val_targets, _ = val_data
                val_inputs = val_inputs.to(self.device)
                val_targets = val_targets.to(self.device)

                x0_val = torch.randn_like(val_targets)
                t_val, xt_val, ut_val = self.FM.sample_location_and_conditional_flow(x0_val, val_targets)
                # 可以选择用 self.model 或 self.model_ema.ema_model 来验证
                vt_val = self.model(image=val_inputs, x=xt_val, pred_type="denoise", step=t_val)
                v_loss = torch.mean((vt_val - ut_val)**2)
                val_loss_total += v_loss.item()
        
        avg_val_loss = val_loss_total / len(self.val_loader)
        self.val_losses.append(avg_val_loss)
        print(f"=> Validation @ Step {self.step_count}, Val Loss: {avg_val_loss:.4f}")

        # 更新最佳模型
        if avg_val_loss < self.best_val_loss:
            self.best_val_loss = avg_val_loss
            self.save_checkpoint(step_tag="best")

        # 绘图
        plot_loss_curve(self.losses, self.val_losses, self.save_dir)

        self.model.train()  # 切回 train 状态
        return avg_val_loss

    def save_checkpoint(self, step_tag="latest"):
        """
        保存当前模型和EMA模型
        """
        model_path = os.path.join(self.save_dir, f"cfg_model_{step_tag}.pt")
        ema_path   = os.path.join(self.save_dir, f"cfg_model_ema_{step_tag}.pt")

        if isinstance(self.model, torch.nn.DataParallel):
            torch.save(self.model.module.state_dict(), model_path)
            torch.save(self.model_ema.ema_model.module.state_dict(), ema_path)
        else:
            torch.save(self.model.state_dict(), model_path)
            torch.save(self.model_ema.ema_model.state_dict(), ema_path)
        print(f"=> 模型已保存到 {model_path} 和 {ema_path}")

    def fit(self):
        """
        整个训练流程入口
        """
        print("==> 开始训练...")
        for epoch_idx in range(self.n_epochs):
            # 训练一个epoch
            train_loss = self.train_one_epoch(epoch_idx)
            # 每个 epoch 完成后做一次 lr_scheduler 更新
            self.scheduler.step(epoch_idx + 1)

        total_time = time.time() - self.start_time
        print(f"==> 训练完成! 总耗时: {total_time:.2f}s")

        # 训练完成后再保存一次
        self.save_checkpoint(step_tag="final")


######################################################
#   下面示例如何使用该 Trainer
######################################################
if __name__ == "__main__":
    # 1. 创建数据集与 DataLoader
    save_dir = "checkpoints/cfm_DRM"
    os.makedirs(save_dir, exist_ok=True)

    Radio_train = loaders.RadioUNet_c(phase="train", carsSimul="yes", carsInput="yes", dir_dataset="/home/user/dxc/motion/MedSegDiff/RadioUNet/RadioMapSeer/")
    Radio_val = loaders.RadioUNet_c(phase="val", carsSimul="yes", carsInput="yes", dir_dataset="/home/user/dxc/motion/MedSegDiff/RadioUNet/RadioMapSeer/")
    Radio_test = loaders.RadioUNet_c(phase="test", carsSimul="yes", carsInput="yes", dir_dataset="/home/user/dxc/motion/MedSegDiff/RadioUNet/RadioMapSeer/")


    train_loader = DataLoader(Radio_train, batch_size=64, shuffle=True, num_workers=0)
    val_loader   = DataLoader(Radio_val,   batch_size=64, shuffle=False, num_workers=0)

        
    
    

    # 2. 准备设备 & 模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    model = DiffUNet(con_channels=3)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"轻量级模型参数数量: {total_params:,}")

    #checkpoint_path = '/home/user/dxc/radio/Diff-UNet/checkpoints/ema/cfg_model_ema_step_135000.pt'
    # state_dict = torch.load(checkpoint_path, map_location=device)
    # model.load_state_dict(state_dict)

    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        print(f"使用 {torch.cuda.device_count()} 个GPU!")
        model = torch.nn.DataParallel(model)
    model.to(device)

    # 3. 初始化 Trainer 并开始训练
    trainer = AdvancedTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        save_dir=save_dir,
        n_epochs=1000,        # 根据需求可调
        lr=1e-3,              # 初始学习率
        weight_decay=1e-5,    # AdamW权重衰减
        warmup_ratio=0.1,     # 前10%epoch作为warmup
        ema_decay=0.999,      # EMA衰减系数
        log_interval=100,
        val_interval=1000,
        save_interval=5000,
    )
    trainer.fit()