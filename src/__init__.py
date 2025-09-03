"""
Diffusion Models for Stock Prediction
====================================

This package implements diffusion models for financial time series prediction and portfolio management.

Main modules:
- diffusion: Core diffusion model implementation
- model_unet1d: UNet1D architecture for time series  
- portfolio_engine: Portfolio management and backtesting
- backtesting: Backtesting framework
- train: Training utilities
- run: Main demo script
"""

__version__ = "0.1.0"
__author__ = "Diffusion Finance Team"

# Import main classes for easy access
try:
    from .diffusion import GaussianDiffusion1D, DiffusionConfig
    from .model_unet1d import UNet1D
    from .portfolio_engine import PortfolioConfig, WalkForwardBacktester
    from .backtesting import DiffusionBacktester, BacktestConfig
except ImportError:
    # Allow import errors during development
    pass