# RadioFlow 🚀📡  
*Flow‑Matching for Lightning‑Fast, High‑Fidelity Radio‑Map Generation*

![banner](docs/teaser_radioflow.png)

<p align="center">
  <img src="https://img.shields.io/badge/Flow‑Matching-%F0%9F%94%A5-red">
  <img src="https://img.shields.io/badge/One‑Step%20Sampling-%E2%9C%85-00b300">
  <img src="https://img.shields.io/badge/Edge‑Ready-%F0%9F%92%AA-blue">
</p>

---

## ✨ Why RadioFlow?

| ⚡ Metric | **RadioFlow** | RadioDiff | RadioUNet |
|-----------|---------------|-----------|-----------|
| **Params** | **3 M (Lite) / 53 M (Large)** | 111 M | 27 M |
| **Inference** | **0.06 s (Lite) / 0.13 s (Large)** | 0.60 s | 0.06 s |
| **PSNR (SRM)** | **39.8 dB** | 35.1 dB | 32.0 dB |

* RadioFlow hits **state‑of‑the‑art accuracy while slashing parameters by ≈ 30× and inference time by ≈ 10×** versus diffusion baselines Radio_Flow-4(1).pdf](file-service://file-NicsRkHMaxWavkMsFcrezf).  
* Powered by **Conditional Flow‑Matching (CFM)** – deterministic ODE sampling means **single‑step generation**, no iterative denoising loops Radio_Flow-4(1).pdf](file-service://file-NicsRkHMaxWavkMsFcrezf).  
* **Spatial‑attention UNet + Classifier‑Free Guidance** for crystal‑clear details and robust generalization Radio_Flow-4(1).pdf](file-service://file-NicsRkHMaxWavkMsFcrezf).  
* **RadioFlow‑Lite** squeezes into **3.9 M parameters** for edge devices with only a marginal drop in fidelity Radio_Flow-4(1).pdf](file-service://file-NicsRkHMaxWavkMsFcrezf).  

---

## 🗺️ What’s Inside?

* **`model/`** Lightweight UNet back‑bone with conditional embeddings & CBAM attention.  
* **`data_loaders/`** Flexible loaders for static (SRM) & dynamic (DRM) RadioMapSeer splits.  
* **`train.py`** One‑command training harness with mixed‑precision & EMA.  
* **`test.py` / `viz.py`** Evaluation + gorgeous heat‑map visualizations.  
* **`Compare_Results/`** Pre‑computed reconstructions & metrics table.  

---

## 🚀 Quick Start



## 📝 Reproducing Paper Results

| 🧪 Task | 📉 NMSE ↓ | 🔊 PSNR ↑ | 📏 RMSE ↓ | 🧠 SSIM ↑ |
|--------|-----------|-----------|-----------|------------|
| **SRM** | **0.0023** | **39.83 dB** | **0.0103** | **0.9249** |
| **DRM** | **0.0028** | **39.37 dB** | **0.0108** | **0.9236** |



## 📊 Visual Gallery


