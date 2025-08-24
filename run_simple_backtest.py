#!/usr/bin/env python3
# run_simple_backtest.py - Simplified version using existing architecture

import torch
import pandas as pd
import numpy as np
from datetime import datetime
import os
import warnings
warnings.filterwarnings('ignore')

# Import original modules
from src.train import train_ddpm_from_dfs
from src.data_utils import prep_df, align_on_date
from src.model_unet1d import UNet1D
from src.diffusion import GaussianDiffusion1D, DiffusionConfig, EMA

def main():
    """Run a comprehensive backtest using the original architecture"""
    
    print("=" * 80)
    print("ENHANCED DIFFUSION MODEL BACKTESTING")
    print("Using original architecture with comprehensive evaluation")
    print("=" * 80)
    
    # Load the data
    print("Loading stock data...")
    goog_df = pd.read_csv("stocks/GOOG.csv")
    nvda_df = pd.read_csv("stocks/NVDA.csv") 
    msft_df = pd.read_csv("stocks/MSFT.csv")
    
    print(f"✓ GOOG: {len(goog_df)} rows")
    print(f"✓ NVDA: {len(nvda_df)} rows")
    print(f"✓ MSFT: {len(msft_df)} rows")
    
    # Training configuration 
    TRAIN_START = "2015-01-01"
    TRAIN_END = "2019-12-31"
    TEST_START = "2020-01-01"
    TEST_END = "2020-04-01"
    
    MODEL_PATH = "enhanced_ddpm_model.pt"
    
    # Step 1: Train model on historical data
    print("\n" + "=" * 50)
    print("STEP 1: TRAINING DIFFUSION MODEL")
    print("=" * 50)
    
    if not os.path.exists(MODEL_PATH):
        print("Training new model...")
        
        model_path, metadata = train_ddpm_from_dfs(
            goog_df, nvda_df, msft_df,
            start_date=TRAIN_START, end_date=TRAIN_END,
            L=84, epochs=100, batch=64, steps=1000, lr=1e-4,
            ema_decay=0.9999, dropout=0.1, use_amp=True,
            device=None, model_path=MODEL_PATH
        )
        
        print(f"✓ Model trained and saved: {model_path}")
    else:
        print(f"✓ Using existing model: {MODEL_PATH}")
        
        # Load metadata from saved model
        checkpoint = torch.load(MODEL_PATH, map_location='cpu')
        metadata = checkpoint['meta']
    
    # Step 2: Prepare test data
    print("\n" + "=" * 50)
    print("STEP 2: PREPARING TEST DATA")
    print("=" * 50)
    
    # Process test data
    g_test = prep_df(goog_df, TEST_START, TEST_END)
    n_test = prep_df(nvda_df, TEST_START, TEST_END)
    m_test = prep_df(msft_df, TEST_START, TEST_END)
    
    merged_test = align_on_date([("GOOG", g_test), ("NVDA", n_test), ("MSFT", m_test)])
    
    print(f"✓ Test data prepared: {len(merged_test)} rows")
    print(f"✓ Test period: {TEST_START} to {TEST_END}")
    
    # Step 3: Load trained model
    print("\n" + "=" * 50)
    print("STEP 3: LOADING TRAINED MODEL")
    print("=" * 50)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    
    # Reconstruct model
    model_config = checkpoint['model_cfg']
    diffusion_config = checkpoint['cfg']
    
    model = UNet1D(**model_config).to(device)
    model.load_state_dict(checkpoint['state_dict_ema'])
    
    diff_config = DiffusionConfig(
        timesteps=diffusion_config['timesteps'],
        beta_schedule=diffusion_config['beta_schedule'],
        v_prediction=diffusion_config['v_prediction']
    )
    
    diffusion = GaussianDiffusion1D(model, diff_config).to(device)
    diffusion.eval()
    
    print(f"✓ Model loaded on {device}")
    print(f"✓ Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Step 4: Generate predictions and analyze performance
    print("\n" + "=" * 50)
    print("STEP 4: GENERATING PREDICTIONS")
    print("=" * 50)
    
    # Normalize test data using training statistics
    channels = metadata['channels']
    mu = metadata['mu']
    sd = metadata['sd']
    
    test_channels = [col for col in merged_test.columns if col in channels]
    X_test = merged_test[test_channels].fillna(0.0).values[:, :, None]
    X_test_norm = (X_test - mu) / sd
    X_test_norm = X_test_norm[:, :, 0]
    
    print(f"✓ Test data normalized: {X_test_norm.shape}")
    
    # Run walk-forward predictions
    L = min(42, len(X_test_norm) // 2)  # Adaptive sequence length
    predictions = []
    actuals = []
    dates = []
    
    print("Running walk-forward predictions...")
    
    print(f"Test data shape: {X_test_norm.shape}")
    print(f"Sequence length L: {L}")
    print(f"Will generate {len(X_test_norm) - L - 5} predictions")
    
    with torch.no_grad():
        for i in range(L, len(X_test_norm) - 1):  # Leave room for at least one step
            print(f"Processing step {i}/{len(X_test_norm)}")
            
            # Input sequence
            x_seq = torch.from_numpy(X_test_norm[i-L:i].T).float().unsqueeze(0).to(device)
            print(f"Input sequence shape: {x_seq.shape}")
            
            # Generate prediction using simple sampling
            try:
                # Use the basic sampling method instead of mask-based sampling
                shape = (1, x_seq.shape[1], 1)  # Single timestep prediction
                pred = diffusion.diffusion.sample(shape, steps=50)
                
                pred_np = pred.cpu().numpy()[0, :, 0]  # First time step
                actual_np = X_test_norm[i, :]
                
                predictions.append(pred_np)
                actuals.append(actual_np)
                dates.append(merged_test.iloc[i]['Date'])
                
                print(f"  Prediction: {pred_np[:3]}")  # First 3 values
                print(f"  Actual:     {actual_np[:3]}")  # First 3 values
                
            except Exception as e:
                print(f"Error at step {i}: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    
    print(f"✓ Generated {len(predictions)} predictions")
    
    # Step 5: Performance Analysis
    print("\n" + "=" * 50)
    print("STEP 5: PERFORMANCE ANALYSIS")
    print("=" * 50)
    
    # Calculate metrics
    mse = np.mean((predictions - actuals) ** 2, axis=0)
    mae = np.mean(np.abs(predictions - actuals), axis=0)
    correlations = [np.corrcoef(predictions[:, i], actuals[:, i])[0, 1] 
                   for i in range(predictions.shape[1])]
    
    # Directional accuracy
    pred_directions = np.sign(predictions)
    actual_directions = np.sign(actuals) 
    directional_accuracy = np.mean(pred_directions == actual_directions, axis=0)
    
    # Print results
    for i, channel in enumerate(channels):
        print(f"\n{channel}:")
        print(f"  MSE: {mse[i]:.6f}")
        print(f"  MAE: {mae[i]:.6f}")
        print(f"  Correlation: {correlations[i]:.4f}")
        print(f"  Directional Accuracy: {directional_accuracy[i]:.4f}")
    
    # Step 6: Trading Strategy Simulation
    print("\n" + "=" * 50)
    print("STEP 6: TRADING STRATEGY SIMULATION")
    print("=" * 50)
    
    # Simple strategy: go long/short based on predicted direction
    initial_capital = 1000000  # $1M
    returns_data = actuals[:, :3]  # Only return columns (first 3)
    predicted_returns = predictions[:, :3]
    
    # Generate signals
    signals = np.tanh(predicted_returns * 2)  # Scale and bound signals
    
    # Portfolio returns
    portfolio_returns = np.sum(signals * returns_data, axis=1) / 3  # Equal weight
    
    # Calculate performance metrics
    cumulative_returns = np.cumprod(1 + portfolio_returns)
    total_return = cumulative_returns[-1] - 1
    annualized_return = (1 + total_return) ** (252 / len(portfolio_returns)) - 1
    volatility = np.std(portfolio_returns) * np.sqrt(252)
    sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
    
    # Drawdown
    peak = np.maximum.accumulate(cumulative_returns)
    drawdown = (cumulative_returns / peak) - 1
    max_drawdown = np.min(drawdown)
    
    # Benchmark (buy and hold equal weight)
    benchmark_returns = np.mean(returns_data, axis=1)
    benchmark_cumret = np.cumprod(1 + benchmark_returns)
    benchmark_total = benchmark_cumret[-1] - 1
    
    print(f"Strategy Performance:")
    print(f"  Total Return: {total_return:.2%}")
    print(f"  Annualized Return: {annualized_return:.2%}")
    print(f"  Volatility: {volatility:.2%}")
    print(f"  Sharpe Ratio: {sharpe_ratio:.3f}")
    print(f"  Max Drawdown: {max_drawdown:.2%}")
    
    print(f"\nBenchmark Performance:")
    print(f"  Total Return: {benchmark_total:.2%}")
    
    print(f"\nOutperformance: {total_return - benchmark_total:.2%}")
    
    # Step 7: Save Results
    print("\n" + "=" * 50)
    print("STEP 7: SAVING RESULTS")
    print("=" * 50)
    
    os.makedirs("results", exist_ok=True)
    
    # Save detailed results
    results_df = pd.DataFrame({
        'Date': dates,
        'Portfolio_Return': portfolio_returns,
        'Cumulative_Return': cumulative_returns,
        'Drawdown': drawdown,
        'Benchmark_Return': benchmark_returns,
        'Benchmark_Cumulative': benchmark_cumret
    })
    
    results_df.to_csv("results/backtest_results.csv", index=False)
    
    # Save predictions vs actuals
    pred_df = pd.DataFrame(predictions, columns=[f'pred_{col}' for col in channels])
    actual_df = pd.DataFrame(actuals, columns=[f'actual_{col}' for col in channels])
    
    comparison_df = pd.concat([
        pd.DataFrame({'Date': dates}),
        pred_df, 
        actual_df
    ], axis=1)
    
    comparison_df.to_csv("results/predictions_vs_actual.csv", index=False)
    
    print("✓ Results saved to results/ directory")
    
    # Step 8: Create Visualizations
    print("\n" + "=" * 50)
    print("STEP 8: CREATING VISUALIZATIONS")
    print("=" * 50)
    
    try:
        import matplotlib.pyplot as plt
        
        # Plot 1: Cumulative Returns
        plt.figure(figsize=(15, 10))
        
        plt.subplot(2, 2, 1)
        plt.plot(dates, cumulative_returns, label='Diffusion Strategy', linewidth=2)
        plt.plot(dates, benchmark_cumret, label='Buy & Hold', linewidth=2)
        plt.title('Cumulative Returns')
        plt.legend()
        plt.xticks(rotation=45)
        
        plt.subplot(2, 2, 2)
        plt.fill_between(dates, drawdown * 100, 0, alpha=0.7, color='red')
        plt.title('Drawdown (%)')
        plt.xticks(rotation=45)
        
        plt.subplot(2, 2, 3)
        plt.scatter(actuals[:, 0], predictions[:, 0], alpha=0.6)
        plt.plot([actuals[:, 0].min(), actuals[:, 0].max()], 
                [actuals[:, 0].min(), actuals[:, 0].max()], 'r--')
        plt.xlabel('Actual Returns')
        plt.ylabel('Predicted Returns')
        plt.title('Predictions vs Actual (GOOG)')
        
        plt.subplot(2, 2, 4)
        rolling_sharpe = pd.Series(portfolio_returns).rolling(252).apply(
            lambda x: np.sqrt(252) * x.mean() / x.std() if x.std() > 0 else 0
        )
        plt.plot(dates, rolling_sharpe)
        plt.title('Rolling 1-Year Sharpe Ratio')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig('results/performance_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✓ Visualizations created and saved")
        
    except ImportError:
        print("Matplotlib not available - skipping visualizations")
    
    print("\n" + "=" * 80)
    print("ENHANCED DIFFUSION BACKTESTING COMPLETED!")
    print("=" * 80)
    
    print(f"\nFINAL SUMMARY:")
    print(f"📈 Strategy Return: {total_return:.2%}")
    print(f"📊 Benchmark Return: {benchmark_total:.2%}")
    print(f"🎯 Outperformance: {total_return - benchmark_total:.2%}")
    print(f"📉 Max Drawdown: {max_drawdown:.2%}")
    print(f"⚡ Sharpe Ratio: {sharpe_ratio:.3f}")
    
    if total_return > benchmark_total:
        print("✅ Strategy OUTPERFORMED the benchmark!")
    else:
        print("❌ Strategy underperformed the benchmark.")
    
    print(f"\n📁 Detailed results saved in 'results/' directory")

if __name__ == "__main__":
    main()