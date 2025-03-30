import numpy as np

import torch 
import torch.nn as nn 

from monai.utils import set_determinism
import sys
import os



from model.unet.basic_unet import BasicUNetEncoder
from model.unet.basic_unet_denose import BasicUNetDe

set_determinism(123)

# 修改数据目录路径
data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

# 修改日志目录
logdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs_btcv/diffunet_transunet_datasettings/")
if not os.path.exists(logdir):
    os.makedirs(logdir)

model_save_path = os.path.join(logdir, "model")
if not os.path.exists(model_save_path):
    os.makedirs(model_save_path)

# 调整训练参数
max_epoch = 1000  # 减少训练轮数，方便测试
batch_size = 1
val_every = 50
env = "pytorch"  # 改为单GPU训练
num_gpus = 1

device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

class DiffUNet(nn.Module):
    def __init__(self,con_channels=2) -> None:
        super().__init__()
        self.embed_model = BasicUNetEncoder(
            spatial_dims=2,           # 2D 图像
            in_channels=con_channels,            # 输入通道数为2
            out_channels=32,          # 输出通道数为32
            features=[128, 128, 256,512 , 1024, 128],#(32, 32, 64, 128, 256, 32),
            #features=(32, 32, 64, 128, 256, 32),
            act=("LeakyReLU", {"negative_slope": 0.1, "inplace": True}),
            norm=("instance", {"affine": True}),
            bias=True
        )

        self.model = BasicUNetDe(
            spatial_dims=2,           # 2D 图像
            in_channels=con_channels+1,            # 修改输入通道数为2
            out_channels=1,           # 输出通道数为1
            features=[128, 128, 256,512 , 1024, 128],#(32, 32, 64, 128, 256, 32),
            #features=(32, 32, 64, 128, 256, 32),
            act=("LeakyReLU", {"negative_slope": 0.1, "inplace": False}),
            norm=("instance", {"affine": True}),
            bias=True,
            use_cross_attention=True
        )
        self.cfg_drop_prob = 0.25

        

    def forward(self, image=None, x=None, pred_type='denoise', step=None, embedding=None):
        """
        Args:
            image: 输入图像，形状为 [B, 2, H, W]
            x: 输入噪声，形状为 [B, 1, H, W]
            pred_type: 预测类型，默认为 "denoise"
            step: 时间步
            embedding: 额外的嵌入特征（list），如果为 None，则通过 embed_model 从 image 计算得到
        """
        # 确保时间步的形状正确
        if step.dim() == 0:
            step = step.repeat(x.shape[0])
        
        if pred_type == "denoise":
            # 如果没有传入 embedding，则通过 embed_model 计算，注意返回的是一个 list
            embeddings = self.embed_model(image) if embedding is None else embedding

            # 在训练阶段，根据 cfg 掉落概率对每个样本的所有嵌入进行 mask（即置为全 0）
            if self.training and hasattr(self, 'cfg_drop_prob'):
    # 生成 [B] 的随机 mask（True 表示需要 mask）
                mask = torch.rand(embeddings[0].shape[0], device=embeddings[0].device) < self.cfg_drop_prob
                if mask.any():
                    # 将 mask reshape 成可以广播到 embeddings[i] 的形状（例如 [B, 1, 1, 1]）
                    mask_reshape = mask.view(-1, *[1]*(embeddings[0].dim()-1))
                    for i in range(len(embeddings)):
                        # 使用 masked_fill 返回新 tensor，避免 inplace 修改
                        embeddings[i] = embeddings[i].masked_fill(mask_reshape, 0.0)

            # 确保时间步是 tensor 格式
            if isinstance(step, int):
                step = torch.tensor([step]).to(x.device)
            elif isinstance(step, torch.Tensor) and step.dim() == 0:
                step = step.unsqueeze(0)
            
            # 返回去噪结果，同时传入 embedding list
            return self.model(x, t=step, image=image, embeddings=embeddings)

    def forward_with_cfg(self, image=None, x=None, step=None, embedding=None, cfg_scale=2.5):
        """
        使用 classifier-free guidance (CFG) 的 forward 方法。
        
        Args:
            image: 输入图像，形状为 [B, 2, H, W]
            x: 输入噪声，形状为 [B, 1, H, W]
            step: 时间步
            embedding: 额外的嵌入特征（list），如果为 None，则通过 embed_model 从 image 计算得到
            cfg_scale: CFG 指导尺度参数，决定条件预测与无条件预测的组合程度
        
        Returns:
            使用 CFG 组合后的去噪结果，计算方式为：
                pred = pred_uncond + cfg_scale * (pred_cond - pred_uncond)
        """
        # 确保时间步的形状正确
        if step.dim() == 0:
            step = step.repeat(x.shape[0])
        
        # 获取条件嵌入（list），如果没有传入则计算
        if embedding is None:
            embedding = self.embed_model(image)
        cond_embedding = embedding

        # 构造无条件嵌入：对于 list 中的每个 tensor，创建一个全 0 的 tensor，形状保持一致
        uncond_embedding = [torch.zeros_like(e) for e in embedding]

        # 计算无条件预测
        pred_uncond = self.model(x, t=step, image=image, embeddings=uncond_embedding)
        # 计算条件预测
        pred_cond = self.model(x, t=step, image=image, embeddings=cond_embedding)

        # 根据 CFG 公式组合预测结果
        return pred_uncond + cfg_scale * (pred_cond - pred_uncond)



def test_diffunet():
    print("开始测试 DiffUNet 模型...")
    try:
        # 创建随机输入数据
        image = torch.randn(10, 2, 256, 256).to(device)  # 批量大小为1，通道数为2
        noise = torch.randn(10, 1, 256, 256).to(device)  # 修改噪声输入的通道数为1
        print("输入图像形状:", image.shape)
        print("输入噪声形状:", noise.shape)

        # 实例化模型
        model = DiffUNet().to(device)
        print("模型已创建并移至设备:", device)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"轻量级模型参数数量: {total_params:,}")
        from torchsummary import summary
        #summary(model, input_size=(2, 256, 256))
        # 执行前向传播
        import time
        with torch.no_grad():
            start_time = time.time()
            output = model(image=image, x=noise, pred_type="denoise", step=torch.tensor([0]).to(device))
            print("前向传播成功")
            end_time = time.time()
            print("time:",end_time-start_time)
            output = model.forward_with_cfg(image=image, x=noise, step=torch.tensor([0]).to(device))
            print("前向传播成功")
            print("输出形状:", output.shape)
            print("输出最大值:", output.max().item())
            print("输出最小值:", output.min().item())
            print("输出均值:", output.mean().item())

        print("测试完成！")
        return True
    except Exception as e:
        print("测试过程中出现错误:", str(e))
        return False

if __name__ == "__main__":
    # 首先运行测试
    test_success = test_diffunet()
    
    