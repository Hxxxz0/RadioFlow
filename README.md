# RadioFlow 🚀📡  
*Flow‑Matching for Lightning‑Fast, High‑Fidelity Radio‑Map Generation*

![banner](docs/RadioFlow_model.png)

<p align="center">
  <img src="https://img.shields.io/badge/Flow‑Matching-%F0%9F%94%A5-red">
  <img src="https://img.shields.io/badge/One‑Step%20Sampling-%E2%9C%85-00b300">
  <img src="https://img.shields.io/badge/Edge‑Ready-%F0%9F%92%AA-blue">
</p>

---
## ✨ Why RadioFlow?

RadioFlow is a lightweight, ultra-fast generative model tailored for high-fidelity radio map construction. Compared to existing baselines like diffusion-based and UNet-based methods, it delivers significantly better visual quality, drastically reduced inference time, and an exceptionally compact model size—especially with the edge-friendly **RadioFlow-Lite** variant. Powered by **Conditional Flow Matching**, **spatial attention UNet**, and **classifier-free guidance**, it achieves state-of-the-art performance with a single-step ODE solver, completely bypassing the costly iterative denoising used in diffusion models.

The framework features a modular design with:
- 🧱 Flexible UNet-based architecture and attention modules  
- 🧠 A training pipeline supporting mixed precision, EMA, and real-time visualization  
- ⚙️ RadioFlow can be seamlessly scaled down to a lightweight version for edge and embedded devices

▶️ **[Download Pretrained Checkpoints (BaiduNetDisk)](https://pan.baidu.com/s/1uuIglmtNukc6_RjFsE7Z_w?pwd=n8f4)**

> *From noise to signal map in just one deterministic step.* 🚀
---

## 🚀 Quick Start



## 📝 Reproducing Paper Results

| 🧪 Task | 📉 NMSE ↓ | 🔊 PSNR ↑ | 📏 RMSE ↓ | 🧠 SSIM ↑ |
|--------|-----------|-----------|-----------|------------|
| **SRM** | **0.0023** | **39.83 dB** | **0.0103** | **0.9249** |
| **DRM** | **0.0028** | **39.37 dB** | **0.0108** | **0.9236** |



## 📊 Visual Gallery
| DRM Flow (ours) vs RadioUNet | SRM Flow (ours) vs RadioUNet |
|:----------------------------:|:----------------------------:|
| ![DRM](Compare_Results/DRM_flow_unet_comparison.png) | ![SRM](Compare_Results/SRM_flow_unet_comparison.png) |
| *Fig. 1: DRM Flow comparison* | *Fig. 2: SRM Flow comparison* |

| DRM Task: CFG Scale Comparison                                          | SRM Task: CFG Scale Comparison                                          |
|:------:|:-------:|
| ![DRM Ablation](Compare_Results/RadioFlow_cfg_drm_comparison.png)       | ![SRM Ablation](Compare_Results/RadioFlow_cfg_srm_comparison.png)       |
| *Fig. 3: DRM map outputs under different CFG scale settings*            | *Fig. 4: SRM map outputs under different CFG scale settings*            |


![Model Performance Comparison](Compare_Results/Model_Performance_Comparison.png)

*Fig. 5: Quantitative comparison of NMSE, PSNR, RMSE, Time,and Params for RadioFlow against other methods.*  
