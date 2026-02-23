---
title: P2 ETF Hybrid Wavelet PPO
emoji: 📈
colorFrom: blue
colorTo: indigo
sdk: streamlit
python_version: "3.10"
sdk_version: 1.31.0
app_file: app.py
pinned: false
license: mit
---

# P2 ETF Hybrid Strategy Engine (2026 Edition)

A high-performance quantitative dashboard featuring 11 intelligence engines. This system utilizes Multi-Resolution Analysis (MODWT) to isolate market signals from noise, processed through a hierarchy of SVR, Reinforcement Learning, and Deep Fusion models.

### 🧠 Core Methodology

* **MODWT Wavelet Denoising**: Unlike standard moving averages, Maximal Overlap Discrete Wavelet Transform removes high-frequency "jitter" without introducing phase lag, preserving perfect time-alignment for signal entry.
* **High-Penalty RBF-SVR**: Options A, G, and H now utilize a Radial Basis Function kernel with $C=2000.0$. This specific configuration is engineered to capture **sharp market reversals** and "V-shaped" recoveries that smoother polynomial models often miss.
* **PPO/A2C Agents**: Reinforcement Learning layers (Options B, C, D) treat portfolio allocation as a dynamic policy, optimizing for the Sharpe Ratio while using SVR outputs as denoised state observations.
* **Bayesian Confidence Gating**: Options E and H apply a 60% recursive Bayesian threshold. If the signal conviction is low, the engine automatically rotates to **CASH** to protect capital during regime transitions.
* **Dual-Stream Cloud Fusion**: Options I, J, and K utilize parallel neural networks trained on 18 years of data (2008–2026) to fuse price action with macro pillars like the VIX and Credit Spreads.

### 🕒 Wall Street Synchronization
The engine is locked to the **America/New_York** timezone. The prediction date dynamically flips at **9:30 AM EST**:
* **Pre-Market**: Displays the target date for the session opening today.
* **During/Post-Market**: Flips to the next valid trading day (skipping weekends) to represent the next active signal.

### 🛠 Intelligence Engine Map

| Option | Engine Type | Focus |
| :--- | :--- | :--- |
| **A** | Wavelet-SVR | Trend Following & Reversals |
| **B** | SVR-PPO | RL-Guided Execution |
| **C/D** | A2C / Ensemble | Allocation Optimization |
| **E/F** | Bayesian / HMM | Regime Detection |
| **G/H** | Hybrid Models | Denoised Prediction |
| **I/J/K** | CNN-LSTM-Attention | Long-term Macro Fusion |

---
*Developed for the 2026 trading environment, integrating stable RL seeds and localized exchange timing.*
