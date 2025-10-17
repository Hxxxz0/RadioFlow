# RadioFlow 🚀📡  
*Flow‑Matching for Lightning‑Fast, High‑Fidelity Radio‑Map Generation*

![banner](docs/RadioFlow_model.png)

<p align="center">
  <img src="https://img.shields.io/badge/Flow‑Matching-%F0%9F%94%A5-red">
  <img src="https://img.shields.io/badge/One‑Step%20Sampling-%E2%9C%85-00b300">
  <img src="https://img.shields.io/badge/Edge‑Ready-%F0%9F%92%AA-blue">
</p>

<p align="center">
  <a href="https://arxiv.org/pdf/2510.09314"><img src="https://img.shields.io/badge/arXiv-2510.09314-b31b1b.svg" alt="arXiv"></a>
</p>

---
## ✨ Why RadioFlow?

RadioFlow is a lightweight, ultra-fast generative model tailored for high-fidelity radio map construction. Compared to existing baselines like diffusion-based and UNet-based methods, it delivers significantly better visual quality, drastically reduced inference time, and an exceptionally compact model size—especially with the edge-friendly **RadioFlow-Lite** variant. Powered by **Conditional Flow Matching**, **spatial attention UNet**, and **classifier-free guidance**, it achieves state-of-the-art performance with a single-step ODE solver, completely bypassing the costly iterative denoising used in diffusion models.

The framework features a modular design with:
- 🧱 Flexible UNet-based architecture and attention modules  
- 🧠 A training pipeline supporting mixed precision, EMA, and real-time visualization  
- ⚙️ RadioFlow can be seamlessly scaled down to a lightweight version for edge and embedded devices

> *From noise to signal map in just one deterministic step.* 🚀

📄 **Paper:** [arXiv:2510.09314](https://arxiv.org/pdf/2510.09314)
---
## 🚀 Quick Start

### 1. Environment Setup

Install dependencies:
```bash
pip install -r requirements.txt
```

### 2. Dataset

- **RadioMapSeer**  
  [Download link](https://radiomapseer.github.io/)

- **RadioMap3DSeer**  
  [Download link](https://drive.google.com/file/d/1YW3RyM9KYBe110CXC5aZJJ0MAIti65bY/view)

**Pretrained checkpoints:** [BaiduNetDisk](https://pan.baidu.com/s/1uuIglmtNukc6_RjFsE7Z_w?pwd=n8f4)

### 3. Training

1. Open `config.py` and set:
   - `data_dir`: path to your dataset
   - `model_size`: `'lite'` or `'large'` (default: `'lite'`)
   - training hyperparameters (e.g., learning rate, batch size, number of epochs)
2. Choose the appropriate data loader:
   - `RadioUNet_c` for the RadioMapSeer dataset  
   - `RadioMap3Dset` for the RadioMap3DSeer dataset
3. Launch training:
   ```bash
   # Train with Lite model (default)
   python train.py
   
   # Train with Large model
   python train.py --model_size large
   ```

### 4. Testing

- **SRM evaluation (Lite model):**
  ```bash
  python test.py --checkpoint SRM_Lite.pt --task srm
  ```
- **SRM evaluation (Large model):**
  ```bash
  python test.py --checkpoint SRM_Large.pt --task srm --model_size large
  ```
- **DRM evaluation (Lite model):**
  ```bash
  python test.py --checkpoint DRM_Lite.pt --task drm
  ```
- **DRM evaluation (Large model):**
  ```bash
  python test.py --checkpoint DRM_Large.pt --task drm --model_size large
  ```

**⚠️ Important:** Ensure the `--model_size` parameter matches your checkpoint. Use `--model_size large` for `*_Large.pt` checkpoints.

### 5. Visualization

1. In `config.py`, configure the `VizConfig` class to specify visualization options.
2. Run the visualization script:
   ```bash
   # Visualize SRM with Lite model
   python viz.py --task srm
   
   # Visualize SRM with Large model
   python viz.py --task srm --model_size large
   
   # Visualize DRM
   python viz.py --task drm
   ```
## 📝 Reproducing Paper Results

| 🧪 Task | 📦 Dataset        | 📉 NMSE ↓  | 🔊 PSNR ↑   | 📏 RMSE ↓  | 🧠 SSIM ↑  |
|--------|-------------------|------------|-------------|------------|------------|
| **SRM** | RadioMapSeer      | **0.0023** | **39.83 dB** | **0.0103** | **0.9249** |
| **DRM** | RadioMapSeer      | **0.0028** | **39.37 dB** | **0.0108** | **0.9236** |
| **SRM** | RadioMap3DSeer    | **0.0496** | **26.87 dB** | **0.0458** | **0.7377** |



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
