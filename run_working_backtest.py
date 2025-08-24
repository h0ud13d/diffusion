#!/usr/bin/env python3
# run_working_backtest.py - Working implementation using simplified enhanced architecture

import torch
import pandas as pd
import numpy as np
from datetime import datetime
import os
import warnings
warnings.filterwarnings('ignore')

# Import modules
from src.portfolio_framework import PortfolioDataManager
from src.enhanced_diffusion import EnhancedGaussianDiffusion1D, EnhancedDiffusionConfig, EMAv2
from src.model_unet1d import UNet1D  # Use original UNet for now
from src.diffusion import DiffusionConfig
from src.backtesting import DiffusionBacktester, BacktestConfig, create_benchmark_results
from src.visualization import PerformanceVisualizer, ReportGenerator

def create_enhanced_dataset_simple(assets, start_date, end_date, sequence_length=42):
    """Create enhanced dataset with original architecture compatibility"""
    
    data_manager = PortfolioDataManager(assets)
    merged_df, metadata = data_manager.load_and_process_data(start_date, end_date)
    
    # Use only return columns for simplicity (like original)
    returns_data = merged_df[metadata['return_columns']].fillna(0.0).values
    
    # Simple normalization
    returns_mean = np.mean(returns_data, axis=0, keepdims=True)
    returns_std = np.std(returns_data, axis=0, keepdims=True) + 1e-8
    returns_norm = (returns_data - returns_mean) / returns_std
    
    # Create simple dataset
    from torch.utils.data import Dataset
    
    class SimpleDataset(Dataset):
        def __init__(self, data, seq_len):
            self.data = data
            self.seq_len = seq_len
            
        def __len__(self):
            return len(self.data) - self.seq_len
            
        def __getitem__(self, idx):
            x = self.data[idx:idx + self.seq_len].T  # (channels, seq_len)
            return {'x': torch.from_numpy(x).float(), 'conditions': torch.zeros(1)}
    
    dataset = SimpleDataset(returns_norm, sequence_length)
    
    # Split into train/val
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    
    train_dataset = torch.utils.data.Subset(dataset, range(train_size))
    val_dataset = torch.utils.data.Subset(dataset, range(train_size, train_size + val_size))
    
    metadata_simple = {
        'assets': assets,
        'return_columns': metadata['return_columns'],
        'feature_columns': metadata['return_columns'],  # Same as returns for simplicity
        'condition_columns': [],
        'scalers': {
            'return_means': returns_mean,
            'return_scales': returns_std
        },
        'sequence_length': sequence_length,
        'dates': merged_df['Date'].values
    }
    
    return train_dataset, val_dataset, dataset, metadata_simple, merged_df

def train_enhanced_simple(assets, train_start, train_end, epochs=50):
    """Train with enhanced diffusion but simplified data"""
    
    print("Training enhanced diffusion model with simplified setup...")
    
    # Create dataset
    train_dataset, val_dataset, full_dataset, metadata, merged_df = create_enhanced_dataset_simple(
        assets, train_start, train_end, sequence_length=42
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    # Model setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Use original UNet but with enhanced diffusion
    model = UNet1D(
        c_in=len(assets), 
        c_out=len(assets), 
        base=128,
        tdim=256,
        drop=0.1
    ).to(device)
    
    # Enhanced diffusion config
    diffusion_config = EnhancedDiffusionConfig(
        timesteps=1000,
        beta_schedule="cosine",
        v_prediction=True,
        classifier_free_guidance=False,  # Disable for simplicity
        guidance_scale=1.0
    )
    
    diffusion = EnhancedGaussianDiffusion1D(model, diffusion_config).to(device)
    ema = EMAv2(model, decay=0.9999, warmup_steps=500)
    
    # Training setup
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-6)
    scaler = torch.cuda.amp.GradScaler() if device.startswith('cuda') else None
    
    # Data loaders
    from torch.utils.data import DataLoader
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        train_loss = 0.0
        
        for batch in train_loader:
            x = batch['x'].to(device)  # (B, C, L)
            
            optimizer.zero_grad()
            
            # Enhanced diffusion loss
            with torch.cuda.amp.autocast(enabled=(scaler is not None)):
                loss = diffusion.loss_on(x)
            
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            
            ema.update(model)
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validate
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                x = batch['x'].to(device)
                loss = diffusion.loss_on(x)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        
        print(f"[{epoch:03d}] train {train_loss:.6f}  val {val_loss:.6f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'ema_state_dict': ema.shadow,
                'diffusion_config': {
                    'timesteps': diffusion_config.timesteps,
                    'beta_schedule': diffusion_config.beta_schedule,
                    'v_prediction': diffusion_config.v_prediction,
                    'classifier_free_guidance': diffusion_config.classifier_free_guidance,
                    'guidance_scale': diffusion_config.guidance_scale
                },
                'metadata': metadata
            }
            
            torch.save(checkpoint, "enhanced_simple_model.pt")
    
    print(f"Training completed. Best val loss: {best_val_loss:.6f}")
    return "enhanced_simple_model.pt", metadata

def run_enhanced_backtest():
    """Run comprehensive backtest with working enhanced model"""
    
    print("=" * 80)
    print("ENHANCED DIFFUSION MODEL - WORKING IMPLEMENTATION")
    print("=" * 80)
    
    assets = ["GOOG", "NVDA", "MSFT"]
    train_start = "2015-01-01"
    train_end = "2019-12-31"
    test_start = "2020-01-01"
    test_end = "2020-04-01"
    
    model_path = "enhanced_simple_model.pt"
    
    # Step 1: Train model
    print("\n" + "=" * 50)
    print("STEP 1: TRAINING ENHANCED MODEL")
    print("=" * 50)
    
    if not os.path.exists(model_path):
        model_path, metadata = train_enhanced_simple(
            assets, train_start, train_end, epochs=30
        )
    else:
        print(f"Using existing model: {model_path}")
        checkpoint = torch.load(model_path, map_location='cpu')
        metadata = checkpoint['metadata']
    
    # Step 2: Load model for testing
    print("\n" + "=" * 50)
    print("STEP 2: LOADING MODEL FOR TESTING")
    print("=" * 50)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(model_path, map_location=device)
    
    # Reconstruct model
    model = UNet1D(
        c_in=len(assets), 
        c_out=len(assets), 
        base=128,
        tdim=256,
        drop=0.1
    ).to(device)
    
    # Load EMA weights
    model.load_state_dict(checkpoint['ema_state_dict'])
    
    # Reconstruct diffusion
    config_dict = checkpoint['diffusion_config']
    diffusion_config = EnhancedDiffusionConfig(**config_dict)
    diffusion = EnhancedGaussianDiffusion1D(model, diffusion_config).to(device)
    diffusion.eval()
    
    print(f"✓ Model loaded on {device}")
    
    # Step 3: Prepare test data
    print("\n" + "=" * 50)
    print("STEP 3: PREPARING TEST DATA")
    print("=" * 50)
    
    _, _, test_dataset, test_metadata, test_df = create_enhanced_dataset_simple(
        assets, test_start, test_end, sequence_length=42
    )
    
    print(f"✓ Test data: {len(test_dataset)} samples")
    
    # Step 4: Generate predictions
    print("\n" + "=" * 50)
    print("STEP 4: GENERATING PREDICTIONS")
    print("=" * 50)
    
    predictions = []
    actuals = []
    dates = []
    
    # Use test data for prediction
    returns_data = test_df[metadata['return_columns']].fillna(0.0).values
    returns_norm = (returns_data - metadata['scalers']['return_means']) / metadata['scalers']['return_scales']
    
    L = metadata['sequence_length']
    print(f"Sequence length: {L}")
    print(f"Test data shape: {returns_norm.shape}")
    
    with torch.no_grad():
        for i in range(L, min(len(returns_norm), L + 20)):  # Limited predictions
            print(f"Generating prediction {i-L+1}/20...")
            
            try:
                # Generate sample
                shape = (1, len(assets), 1)  # Single step prediction
                pred = diffusion.sample(
                    shape=shape,
                    method="ddpm",
                    steps=50,
                    progress=False
                )
                
                pred_np = pred.cpu().numpy()[0, :, 0]
                actual_np = returns_norm[i, :]
                
                predictions.append(pred_np)
                actuals.append(actual_np)
                dates.append(test_df.iloc[i]['Date'])
                
            except Exception as e:
                print(f"Error at step {i}: {e}")
                continue
    
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    
    print(f"✓ Generated {len(predictions)} predictions")
    
    # Step 5: Performance Analysis
    print("\n" + "=" * 50)
    print("STEP 5: PERFORMANCE ANALYSIS")
    print("=" * 50)
    
    if len(predictions) > 0:
        # Denormalize predictions
        pred_denorm = predictions * metadata['scalers']['return_scales'] + metadata['scalers']['return_means']
        actual_denorm = actuals * metadata['scalers']['return_scales'] + metadata['scalers']['return_means']
        
        # Calculate metrics
        mse = np.mean((predictions - actuals) ** 2, axis=0)
        correlations = [np.corrcoef(predictions[:, i], actuals[:, i])[0, 1] 
                       for i in range(predictions.shape[1]) if len(predictions) > 1]
        
        # Directional accuracy
        pred_dirs = np.sign(pred_denorm)
        actual_dirs = np.sign(actual_denorm)
        directional_acc = np.mean(pred_dirs == actual_dirs, axis=0)
        
        print("\nPerformance Metrics:")
        for i, asset in enumerate(assets):
            print(f"\n{asset}:")
            print(f"  MSE (normalized): {mse[i]:.6f}")
            if i < len(correlations):
                print(f"  Correlation: {correlations[i]:.4f}")
            print(f"  Directional Accuracy: {directional_acc[i]:.4f}")
        
        # Step 6: Simple Trading Strategy
        print("\n" + "=" * 50)
        print("STEP 6: TRADING STRATEGY SIMULATION")
        print("=" * 50)
        
        # Simple strategy: equal weight portfolio based on predicted directions
        portfolio_returns = np.mean(pred_dirs * actual_denorm, axis=1)
        cumulative_returns = np.cumprod(1 + portfolio_returns)
        
        total_return = cumulative_returns[-1] - 1 if len(cumulative_returns) > 0 else 0
        
        # Benchmark: buy and hold
        benchmark_returns = np.mean(actual_denorm, axis=1)
        benchmark_cumret = np.cumprod(1 + benchmark_returns)
        benchmark_total = benchmark_cumret[-1] - 1 if len(benchmark_cumret) > 0 else 0
        
        print(f"\nStrategy Results:")
        print(f"  Strategy Return: {total_return:.2%}")
        print(f"  Benchmark Return: {benchmark_total:.2%}")
        print(f"  Outperformance: {total_return - benchmark_total:.2%}")
        
        if total_return > benchmark_total:
            print("✅ Strategy OUTPERFORMED benchmark!")
        else:
            print("❌ Strategy underperformed benchmark")
        
        # Step 7: Save Results
        print("\n" + "=" * 50)
        print("STEP 7: SAVING RESULTS")
        print("=" * 50)
        
        os.makedirs("results", exist_ok=True)
        
        # Save results
        results_df = pd.DataFrame({
            'Date': dates,
            'GOOG_Pred': predictions[:, 0],
            'NVDA_Pred': predictions[:, 1],
            'MSFT_Pred': predictions[:, 2],
            'GOOG_Actual': actuals[:, 0],
            'NVDA_Actual': actuals[:, 1],
            'MSFT_Actual': actuals[:, 2],
            'Portfolio_Return': portfolio_returns,
            'Benchmark_Return': benchmark_returns
        })
        
        results_df.to_csv("results/enhanced_backtest_results.csv", index=False)
        print("✓ Results saved to results/enhanced_backtest_results.csv")
        
    else:
        print("❌ No predictions generated - check model and data compatibility")
    
    print("\n" + "=" * 80)
    print("ENHANCED DIFFUSION BACKTEST COMPLETED!")
    print("=" * 80)

if __name__ == "__main__":
    run_enhanced_backtest()