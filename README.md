---
title: P2 ETF WAVELET SVR MULTI MODEL
emoji: 🦅
colorFrom: blue
colorTo: indigo
sdk: streamlit
sdk_version: 1.31.0
app_file: app.py
pinned: false
license: mit
python_version: 3.10
---

# P2 ETF Hybrid Strategy Engine
This is a Hybrid SVR-PPO model for ETF rotation using MODWT Wavelet denoising.

### Methodology
- **DWT Wavelet**: Removes high-frequency noise while maintaining time-alignment.
- **SVR**: Predicts next-day returns based on macro signals.
- **PPO Agent**: Reinforcement Learning for optimal portfolio execution net of transaction costs.

### Environment Configuration
- **Python**: 3.10 (Locked for model compatibility)
- **Storage**: Git LFS for binary model weight hydration (.h5)
