"""
Unified training script for both DRM and SRM tasks.
"""
import os
import time
import copy
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm

# Flow matching and neural ODE modules
import torchdiffeq
from torchdyn.core import NeuralODE
from torchcfm.conditional_flow_matching import ConditionalFlowMatcher

# Project modules
from data_loaders.loaders import RadioUNet_c, RadioMap3Dset
from model.model import DiffUNet
from config import Config


def build_lr_lambda(total_epochs, warmup_ratio=0.1):
    """
    Warmup + Cosine annealing learning rate schedule.
    """
    warmup_epochs = int(total_epochs * warmup_ratio)

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return float(epoch) / float(warmup_epochs)
        progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    return lr_lambda


class ModelEMA:
    """
    Maintains an exponential moving average of model parameters.
    """
    def __init__(self, model, decay=0.999):
        self.ema_model = copy.deepcopy(model)
        for p in self.ema_model.parameters():
            p.requires_grad_(False)
        self.decay = decay

    def update(self, model):
        with torch.no_grad():
            msd = model.state_dict()
            esd = self.ema_model.state_dict()
            for k in esd.keys():
                if k in msd:
                    esd[k].mul_(self.decay).add_(msd[k], alpha=1 - self.decay)
            self.ema_model.load_state_dict(esd)


def plot_loss_curve(losses, val_losses, save_dir):
    """
    Save a figure of training and validation loss over steps.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(losses, label='Train Loss')
    if val_losses:
        val_x = np.linspace(0, len(losses), len(val_losses))
        plt.plot(val_x, val_losses, label='Val Loss')
    plt.xlabel('Training Step')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.legend()
    path = os.path.join(save_dir, "loss_curve.png")
    plt.savefig(path)
    plt.close()
    print(f"Saved loss curve to {path}")


class AdvancedTrainer:
    """
    Trainer encapsulating training, validation, EMA, logging, and checkpointing.
    """
    def __init__(self, cfg: Config, model, train_loader, val_loader, device):
        self.cfg = cfg
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        os.makedirs(cfg.save_dir, exist_ok=True)

        # Optimizer and scheduler
        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=build_lr_lambda(cfg.n_epochs, cfg.warmup_ratio)
        )

        # Mixed precision scaler
        self.scaler = torch.cuda.amp.GradScaler()

        # EMA model
        self.ema = ModelEMA(model, decay=cfg.ema_decay)

        # Flow matcher and neural ODE wrapper
        self.flow_matcher = ConditionalFlowMatcher(sigma=0.0)
        self.neural_ode = NeuralODE(
            self.model, solver='dopri5', sensitivity='adjoint', atol=1e-4, rtol=1e-4
        )

        # Training bookkeeping
        self.step = 0
        self.start_time = time.time()
        self.train_losses = []
        self.val_losses = []
        self.best_val = float('inf')

    def train_one_epoch(self, epoch):
        self.model.train()
        epoch_loss = 0.0

        for batch in tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.cfg.n_epochs}"):
            inputs, targets, _ = batch
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            # Sample random noise and conditional flow
            x0 = torch.randn_like(targets)
            t, xt, ut = self.flow_matcher.sample_location_and_conditional_flow(x0, targets)

            self.optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                vt = self.model(image=inputs, x=xt, pred_type='denoise', step=t)
                loss = torch.mean((vt - ut) ** 2)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            # EMA update
            self.ema.update(self.model)

            # Logging
            epoch_loss += loss.item()
            self.train_losses.append(loss.item())
            self.step += 1

            if self.step % self.cfg.log_interval == 0:
                lr = self.optimizer.param_groups[0]['lr']
                elapsed = time.time() - self.start_time
                print(f"Step {self.step}: loss={loss.item():.4f}, lr={lr:.6f}, time={elapsed:.1f}s")

            if self.step % self.cfg.val_interval == 0:
                self.validate()

            if self.step % self.cfg.save_interval == 0:
                self.save_checkpoint(f'step_{self.step}')

        avg_loss = epoch_loss / len(self.train_loader)
        print(f"Epoch {epoch+1} done. Avg loss: {avg_loss:.4f}")
        return avg_loss

    def validate(self):
        self.model.eval()
        total_val = 0.0
        with torch.no_grad():
            for batch in self.val_loader:
                inputs, targets, _ = batch
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                x0 = torch.randn_like(targets)
                t, xt, ut = self.flow_matcher.sample_location_and_conditional_flow(x0, targets)
                vt = self.model(image=inputs, x=xt, pred_type='denoise', step=t)
                total_val += torch.mean((vt - ut) ** 2).item()

        avg_val = total_val / len(self.val_loader)
        self.val_losses.append(avg_val)
        print(f"Validation @ step {self.step}: val_loss={avg_val:.4f}")

        # Save best
        if avg_val < self.best_val:
            self.best_val = avg_val
            self.save_checkpoint('best')

        plot_loss_curve(self.train_losses, self.val_losses, self.cfg.save_dir)
        self.model.train()

    def save_checkpoint(self, tag):
        model_path = os.path.join(self.cfg.save_dir, f"model_{tag}.pt")
        ema_path = os.path.join(self.cfg.save_dir, f"model_ema_{tag}.pt")
        torch.save(self.model.state_dict(), model_path)
        torch.save(self.ema.ema_model.state_dict(), ema_path)
        print(f"Saved checkpoints: {model_path}, {ema_path}")

    def fit(self):
        print("Starting training...")
        for epoch in range(self.cfg.n_epochs):
            self.train_one_epoch(epoch)
            self.scheduler.step()
        self.save_checkpoint('final')
        total = time.time() - self.start_time
        print(f"Training complete in {total:.1f}s")


if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train RadioFlow model')
    parser.add_argument('--task', choices=['drm','srm'], default=Config.task,
                        help='Task to train: drm or srm')
    parser.add_argument('--model_size', type=str, default=None,
                        choices=['lite', 'large'],
                        help='Model size: lite or large (default: use config value)')
    args = parser.parse_args()
    
    cfg = Config()

    # Override task and model_size if provided via command line
    if args.task:
        cfg.task = args.task
        print(f"Using task: {cfg.task}")
    if args.model_size:
        cfg.model_size = args.model_size
        print(f"Using model size: {cfg.model_size}")

    # Set task-specific configurations
    if cfg.task == 'drm':
        cfg.con_channels = 3
        cfg.cars_simul = "yes"
        cfg.cars_input = "yes"
        cfg.save_dir = f"checkpoints/cfm_DRM_{cfg.model_size.title()}"
        cfg.checkpoint = f'DRM_{cfg.model_size.title()}.pt'
    else:  # srm
        cfg.con_channels = 2
        cfg.cars_simul = "no"
        cfg.cars_input = "no"
        cfg.save_dir = f"checkpoints/cfm_SRM_{cfg.model_size.title()}"
        cfg.checkpoint = f'SRM_{cfg.model_size.title()}.pt'

    print(f"Input channels: {cfg.con_channels}")
    print(f"Save directory: {cfg.save_dir}")
    print(f"Checkpoint: {cfg.checkpoint}")

    os.makedirs(cfg.save_dir, exist_ok=True)

    # Prepare datasets and loaders
    train_ds = RadioUNet_c(
        phase='train', carsSimul=cfg.cars_simul,
        carsInput=cfg.cars_input, dir_dataset=cfg.data_dir,
        simulation=cfg.simulation
    )
    val_ds = RadioUNet_c(
        phase='val', carsSimul=cfg.cars_simul,
        carsInput=cfg.cars_input, dir_dataset=cfg.data_dir,
        simulation=cfg.simulation
    )
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)

    # Device and model setup
    device = torch.device(cfg.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Task: {cfg.task}")
    model = DiffUNet(con_channels=cfg.con_channels, model_size=cfg.model_size)
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    model.to(device)

    # Start training
    trainer = AdvancedTrainer(cfg, model, train_loader, val_loader, device)
    trainer.fit()
