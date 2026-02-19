---
title: P2 ETF Hybrid Wavelet PPO
emoji: 📈
colorFrom: blue
colorTo: indigo
sdk: streamlit
app_file: app.py
pinned: false
license: mit
---

# P2 ETF Hybrid Strategy Engine
This is a Hybrid SVR-PPO model for ETF rotation using MODWT Wavelet denoising.

### Methodology
- **MODWT Wavelet**: Removes high-frequency noise while maintaining time-alignment.
- **SVR**: Predicts next-day returns based on macro signals.
- **PPO Agent**: Reinforcement Learning for optimal portfolio execution net of transaction costs.
