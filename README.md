# Diffusion Models for Stock Analysis

Testing diffusion models for stock prediction. Diffusion models have been successfully implemented in NLP, so I wanted to explore their potency in stock returns. I got inspired by this [arXiv paper](https://arxiv.org/html/2402.06656v1).

## Setup

1. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```

2. **Optional:** Download full S&P 500 dataset from [Kaggle](https://www.kaggle.com/datasets/camnugent/sandp500) if you want to test this on other stocks.
   - The repository includes 3 sample stocks (NVIDIA, Microsoft, Google) in the `stocks/` directory

## Project

This project implements a **1D Diffusion Model** (DDPM) for stock price prediction using cross-stock correlations.

- **UNet1D**: 1D convolutional neural network with skip connections for time series modeling
- **Diffusion Process**: Implements DDPM with cosine noise schedule and v-parameterization  
- **Cross-Stock Prediction**: Uses GOOG+MSFT correlations (tech sector) to infer NVDA behavior
- **Training Pipeline**: Includes EMA (Exponential Moving Average) and mixed precision training

## Usage

Run the model from the root directory:
```bash
python3 -m src.run
```

## Results

![NVDA vs MSFT Inpainting Results](imgs/nvda_msft_inpaint.png)

The example shows NVDA prediction during a period with different market conditions than the training data. The model uses inpainting; treating missing NVDA data as "blank channels" and filling them based on observed GOOG+MSFT patterns.

## Observations

The model maintains some realistic properties of stock data; preserving fat-tailed return distributions and temporal autocorrelations where recent movements are more predictive than distant ones.

Compared to GANs I tested earlier, diffusion models seem more stable and **avoid mode collapse** when generating financial time series.

This was just an initial experiment inspired by the diffusion transformers paper mentioned above, and next steps would be implementing it.
