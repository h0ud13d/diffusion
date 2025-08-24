#!/usr/bin/env python3
# run_enhanced_backtest.py - Main script to run the enhanced diffusion model with comprehensive backtesting

import torch
import pandas as pd
import numpy as np
from datetime import datetime
import os
import warnings
warnings.filterwarnings('ignore')

# Import our enhanced modules
from src.enhanced_model import EnhancedUNet1D, ConditionEncoder
from src.enhanced_diffusion import EnhancedGaussianDiffusion1D, EnhancedDiffusionConfig
from src.enhanced_train import train_enhanced_diffusion_model, EnhancedTrainer
from src.portfolio_framework import PortfolioDataManager
from src.backtesting import DiffusionBacktester, BacktestConfig, create_benchmark_results
from src.visualization import PerformanceVisualizer, ReportGenerator

def load_enhanced_model(checkpoint_path: str, device: str = "auto"):
    """Load the enhanced diffusion model from checkpoint"""
    
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Loading model from {checkpoint_path} on {device}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Extract configs
    model_config = checkpoint['model_config']
    diffusion_config_dict = checkpoint['diffusion_config']
    metadata = checkpoint['metadata']
    
    # Reconstruct diffusion config
    diffusion_config = EnhancedDiffusionConfig(
        timesteps=diffusion_config_dict['timesteps'],
        beta_schedule=diffusion_config_dict['beta_schedule'],
        v_prediction=diffusion_config_dict['v_prediction'],
        classifier_free_guidance=diffusion_config_dict.get('classifier_free_guidance', True),
        guidance_scale=diffusion_config_dict.get('guidance_scale', 7.5),
        drop_prob=diffusion_config_dict.get('drop_prob', 0.1)
    )
    
    # Build models
    feature_dim = len(metadata['feature_columns'])
    condition_dim = len(metadata['condition_columns'])
    return_dim = len(metadata['return_columns'])
    
    # Condition encoder
    condition_encoder = None
    if condition_dim > 0:
        condition_encoder = ConditionEncoder(
            input_dim=condition_dim,
            output_dim=model_config.get('context_dim', 128),
            hidden_dim=model_config.get('condition_hidden_dim', 256)
        ).to(device)
        condition_encoder.load_state_dict(checkpoint['condition_encoder_state_dict'])
    
    # Main model
    model = EnhancedUNet1D(
        c_in=feature_dim,
        c_out=return_dim,
        base_channels=model_config.get('base_channels', 64),
        channel_multipliers=model_config.get('channel_multipliers', [1, 2, 4, 8]),
        time_dim=model_config.get('time_dim', 256),
        context_dim=model_config.get('context_dim', 128) if condition_dim > 0 else None,
        dropout=model_config.get('dropout', 0.1),
        attention_resolutions=model_config.get('attention_resolutions', [1, 2]),
        num_heads=model_config.get('num_heads', 8)
    ).to(device)
    
    # Load EMA weights for best performance
    model.load_state_dict(checkpoint['ema_state_dict'])
    
    # Diffusion wrapper
    diffusion = EnhancedGaussianDiffusion1D(model, diffusion_config).to(device)
    
    return diffusion, condition_encoder, metadata

def create_enhanced_diffusion_wrapper(diffusion_model, condition_encoder, metadata, device):
    """Create a wrapper that matches the backtesting interface"""
    
    class DiffusionModelWrapper:
        def __init__(self, diffusion, condition_encoder, metadata, device):
            self.diffusion = diffusion
            self.condition_encoder = condition_encoder
            self.metadata = metadata
            self.device = device
            
        def parameters(self):
            return self.diffusion.parameters()
        
        def eval(self):
            self.diffusion.eval()
            if self.condition_encoder:
                self.condition_encoder.eval()
        
        def sample(self, shape, context=None, method="ddim", steps=50, progress=False):
            """Generate samples from the model"""
            
            # Encode context if provided
            if context is not None and self.condition_encoder is not None:
                with torch.no_grad():
                    encoded_context = self.condition_encoder(context)
            else:
                encoded_context = None
            
            # Generate samples
            return self.diffusion.sample(
                shape=shape,
                context=encoded_context,
                method=method,
                steps=steps,
                progress=progress
            )
    
    return DiffusionModelWrapper(diffusion_model, condition_encoder, metadata, device)

def main():
    """Main execution function"""
    
    print("=" * 80)
    print("ENHANCED DIFFUSION MODEL FOR QUANTITATIVE TRADING")
    print("=" * 80)
    
    # Configuration - use only available assets
    ASSETS = ["GOOG", "NVDA", "MSFT"]
    TRAIN_START = "2018-01-01"
    TRAIN_END = "2023-06-30"
    TEST_START = "2023-07-01" 
    TEST_END = "2024-12-31"
    
    MODEL_PATH = "enhanced_diffusion_model.pt"
    SEQUENCE_LENGTH = 42  # Reduced sequence length for limited data
    EPOCHS = 50  # Reduced epochs for faster training
    
    # Step 1: Train the enhanced model (if not already trained)
    if not os.path.exists(MODEL_PATH):
        print("\n" + "=" * 50)
        print("STEP 1: TRAINING ENHANCED DIFFUSION MODEL")
        print("=" * 50)
        
        model_config = {
            'base_channels': 128,
            'channel_multipliers': [1, 2, 4, 8],
            'time_dim': 512,
            'context_dim': 256,
            'condition_hidden_dim': 512,
            'dropout': 0.1,
            'attention_resolutions': [1, 2, 3],
            'num_heads': 8,
        }
        
        diffusion_config = EnhancedDiffusionConfig(
            timesteps=1000,
            beta_schedule="cosine",
            v_prediction=True,
            classifier_free_guidance=True,
            guidance_scale=7.5,
            drop_prob=0.1
        )
        
        training_config = {
            'batch_size': 32,
            'learning_rate': 1e-4,
            'weight_decay': 1e-6,
            'ema_decay': 0.9999,
            'ema_warmup': 2000,
            'use_amp': True,
            'use_scheduler': True,
            'grad_clip': 1.0,
            'patience': 25,
            'num_workers': 4
        }
        
        model_path, metadata = train_enhanced_diffusion_model(
            assets=ASSETS,
            start_date=TRAIN_START,
            end_date=TRAIN_END,
            sequence_length=SEQUENCE_LENGTH,
            epochs=EPOCHS,
            model_config=model_config,
            diffusion_config=diffusion_config,
            training_config=training_config,
            save_path=MODEL_PATH
        )
        
        print(f"\n✓ Model training completed: {model_path}")
        
    else:
        print(f"\n✓ Using existing model: {MODEL_PATH}")
    
    # Step 2: Load the trained model
    print("\n" + "=" * 50)
    print("STEP 2: LOADING TRAINED MODEL")
    print("=" * 50)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    diffusion_model, condition_encoder, metadata = load_enhanced_model(MODEL_PATH, device)
    model_wrapper = create_enhanced_diffusion_wrapper(diffusion_model, condition_encoder, metadata, device)
    
    print(f"✓ Model loaded successfully on {device}")
    print(f"✓ Assets: {metadata['assets']}")
    print(f"✓ Features: {len(metadata['feature_columns'])}")
    print(f"✓ Conditions: {len(metadata['condition_columns'])}")
    
    # Step 3: Prepare test data
    print("\n" + "=" * 50)
    print("STEP 3: PREPARING TEST DATA")
    print("=" * 50)
    
    data_manager = PortfolioDataManager(ASSETS)
    test_df, test_metadata = data_manager.load_and_process_data(TEST_START, TEST_END)
    
    print(f"✓ Test data loaded: {len(test_df)} rows")
    print(f"✓ Test period: {TEST_START} to {TEST_END}")
    
    # Normalize test data using training statistics
    scalers = metadata['scalers']
    features_norm = (test_df[metadata['feature_columns']].fillna(0.0).values - scalers['feature_means']) / scalers['feature_scales']
    conditions_norm = (test_df[metadata['condition_columns']].fillna(0.0).values - scalers['condition_means']) / scalers['condition_scales']
    
    # Add normalized data back to test_df
    test_df_normalized = test_df.copy()
    for i, col in enumerate(metadata['feature_columns']):
        test_df_normalized[col] = features_norm[:, i]
    for i, col in enumerate(metadata['condition_columns']):
        test_df_normalized[f"{col}_norm"] = conditions_norm[:, i]
    
    # Step 4: Run comprehensive backtesting
    print("\n" + "=" * 50)
    print("STEP 4: COMPREHENSIVE BACKTESTING")
    print("=" * 50)
    
    # Backtest configuration
    backtest_config = BacktestConfig(
        initial_capital=1000000.0,  # $1M
        rebalance_frequency="daily",
        transaction_cost=0.001,  # 0.1%
        max_leverage=1.0,
        risk_free_rate=0.03,  # 3%
        max_position_size=0.4,  # 40% max per asset
        walk_forward=True,
        training_window=252,
        retraining_frequency=63,
        prediction_horizon=5
    )
    
    # Initialize backtester
    backtester = DiffusionBacktester(backtest_config)
    
    # Run backtest
    results = backtester.run_backtest(
        model_wrapper, data_manager, test_df_normalized, metadata
    )
    
    print(f"\n✓ Backtest completed!")
    print(f"✓ Total Return: {results.total_return:.2%}")
    print(f"✓ Sharpe Ratio: {results.sharpe_ratio:.3f}")
    print(f"✓ Max Drawdown: {results.max_drawdown:.2%}")
    
    # Step 5: Create benchmark comparison
    print("\n" + "=" * 50)
    print("STEP 5: BENCHMARK COMPARISON")
    print("=" * 50)
    
    # Simple buy-and-hold benchmark (equal weight)
    returns_data = test_df[metadata['return_columns']].fillna(0.0).values
    equal_weight_returns = returns_data.mean(axis=1)
    benchmark_results = create_benchmark_results(equal_weight_returns, backtest_config.initial_capital)
    
    print(f"✓ Benchmark Total Return: {benchmark_results.total_return:.2%}")
    print(f"✓ Benchmark Sharpe Ratio: {benchmark_results.sharpe_ratio:.3f}")
    print(f"✓ Benchmark Max Drawdown: {benchmark_results.max_drawdown:.2%}")
    
    # Step 6: Generate comprehensive reports
    print("\n" + "=" * 50)
    print("STEP 6: GENERATING REPORTS")
    print("=" * 50)
    
    # Results dictionary for comparison
    results_dict = {
        "Enhanced Diffusion": results,
        "Equal Weight Benchmark": benchmark_results
    }
    
    # Initialize visualization and reporting
    visualizer = PerformanceVisualizer()
    report_generator = ReportGenerator()
    
    # Create results directory
    os.makedirs("results", exist_ok=True)
    
    # Generate performance plots
    print("Creating performance visualizations...")
    visualizer.plot_portfolio_performance(
        results_dict, 
        save_path="results/portfolio_performance.png"
    )
    
    visualizer.plot_risk_metrics(
        results_dict,
        save_path="results/risk_metrics.png"
    )
    
    # Generate position heatmap (if positions available)
    if not results.positions.empty:
        visualizer.plot_position_heatmap(
            results.positions,
            title="Enhanced Diffusion Model - Position Weights",
            save_path="results/position_heatmap.png"
        )
    
    # Generate HTML report
    print("Generating comprehensive HTML report...")
    report_generator.generate_html_report(
        results_dict, 
        metadata, 
        output_path="results/backtest_report.html"
    )
    
    # Save detailed results to CSV
    print("Saving detailed results...")
    report_generator.save_results_to_csv(results_dict, "results/")
    
    print("\n" + "=" * 50)
    print("FINAL SUMMARY")
    print("=" * 50)
    
    print(f"Enhanced Diffusion Model Performance:")
    print(f"  • Total Return:       {results.total_return:.2%}")
    print(f"  • Annualized Return:  {results.annualized_return:.2%}")
    print(f"  • Volatility:         {results.volatility:.2%}")
    print(f"  • Sharpe Ratio:       {results.sharpe_ratio:.3f}")
    print(f"  • Max Drawdown:       {results.max_drawdown:.2%}")
    print(f"  • Calmar Ratio:       {results.calmar_ratio:.3f}")
    print(f"  • VaR (95%):         {results.var_95:.2%}")
    
    print(f"\nBenchmark Performance:")
    print(f"  • Total Return:       {benchmark_results.total_return:.2%}")
    print(f"  • Annualized Return:  {benchmark_results.annualized_return:.2%}")
    print(f"  • Volatility:         {benchmark_results.volatility:.2%}")
    print(f"  • Sharpe Ratio:       {benchmark_results.sharpe_ratio:.3f}")
    print(f"  • Max Drawdown:       {benchmark_results.max_drawdown:.2%}")
    
    # Performance comparison
    outperformance = results.total_return - benchmark_results.total_return
    print(f"\n🎯 Outperformance: {outperformance:.2%}")
    
    if outperformance > 0:
        print("✅ Enhanced Diffusion model outperformed the benchmark!")
    else:
        print("❌ Enhanced Diffusion model underperformed the benchmark.")
    
    print(f"\n📁 All results saved to 'results/' directory")
    print(f"📊 View the comprehensive report: results/backtest_report.html")
    
    print("\n" + "=" * 80)
    print("ENHANCED DIFFUSION BACKTESTING COMPLETED SUCCESSFULLY!")
    print("=" * 80)

if __name__ == "__main__":
    main()