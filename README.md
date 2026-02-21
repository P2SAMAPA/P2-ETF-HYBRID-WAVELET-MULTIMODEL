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

# P2 ETF Hybrid Strategy Engine
This is a Hybrid SVR-PPO model for ETF rotation using MODWT Wavelet denoising.

### Methodology
- **DWT Wavelet**: Removes high-frequency noise while maintaining time-alignment.
- **SVR**: Predicts next-day returns based on macro signals.
- **PPO Agent**: Reinforcement Learning for optimal portfolio execution.
