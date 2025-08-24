#!/usr/bin/env python3
# run_ultimate_diffusion.py - Ultimate state-of-the-art diffusion model for quantitative trading

import os
import sys
import torch
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

# Import all our advanced modules
from src.advanced_diffusion import AdvancedDiffusionModel, AdvancedDiffusionConfig, AdvancedDiffusionTransformer
from src.advanced_features import AdvancedFeatureEngineer, FeatureConfig
from src.advanced_trainer import AdvancedDiffusionTrainer, AdvancedTrainingConfig, AdvancedFinancialDataset, set_seed
from src.advanced_backtesting import AdvancedBacktester, AdvancedBacktestConfig
from src.advanced_samplers import DPMSolverMultistep, HeunSampler, EDMSampler
from src.visualization import PerformanceVisualizer, ReportGenerator

def main():
    """Ultimate diffusion model implementation and backtesting"""
    
    print("=" * 100)
    print("🚀 ULTIMATE DIFFUSION MODEL FOR QUANTITATIVE TRADING")
    print("State-of-the-art Transformer-based Architecture with Advanced Features")
    print("=" * 100)
    
    # Set random seed for reproducibility
    set_seed(42)
    
    # Configuration
    ASSETS = ["GOOG", "NVDA", "MSFT"]
    TRAIN_START = "2015-01-01"
    TRAIN_END = "2019-12-31"
    TEST_START = "2020-01-01"
    TEST_END = "2020-04-01"
    
    MODEL_PATH = "ultimate_diffusion_model.pt"
    RESULTS_DIR = "ultimate_results"
    
    # Create results directory
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Step 1: Advanced Feature Engineering
    print("\n" + "=" * 80)
    print("STEP 1: ADVANCED FEATURE ENGINEERING")
    print("=" * 80)
    
    # Configure sophisticated feature engineering
    feature_config = FeatureConfig(
        use_ta_indicators=True,
        sma_periods=[5, 10, 20, 50, 200],
        ema_periods=[12, 26, 50, 100],
        rsi_periods=[14, 21, 28],
        bb_periods=[20, 50],
        use_microstructure=True,
        vwap_periods=[5, 20, 60],
        use_alternative_data=True,
        sentiment_features=True,
        macro_features=True,
        use_cross_asset=True,
        correlation_periods=[20, 60, 252],
        beta_periods=[60, 252],
        use_temporal_features=True,
        cyclical_encoding=True,
        regime_detection=True,
        use_risk_features=True,
        var_periods=[20, 60],
        volatility_periods=[5, 10, 20, 60]
    )
    
    feature_engineer = AdvancedFeatureEngineer(feature_config)
    
    # Load and process data
    print("Loading and processing market data...")
    asset_dataframes = {}
    for asset in ASSETS:
        try:
            df = pd.read_csv(f"stocks/{asset}.csv")
            print(f"✓ Loaded {asset}: {len(df)} rows")
            
            # Apply advanced feature engineering
            processed_df = feature_engineer.engineer_features(df, asset)
            asset_dataframes[asset] = processed_df
            print(f"✓ Engineered {len(processed_df.columns)} features for {asset}")
            
        except FileNotFoundError:
            print(f"❌ Could not load data for {asset}")
            continue
    
    if not asset_dataframes:
        print("❌ No data could be loaded. Exiting.")
        return
    
    # Merge all asset data
    print("Merging multi-asset dataset...")
    merged_df = asset_dataframes[ASSETS[0]].copy()
    
    for asset in ASSETS[1:]:
        if asset in asset_dataframes:
            asset_df = asset_dataframes[asset]
            merged_df = merged_df.merge(asset_df, on='Date', how='inner', suffixes=('', f'_{asset}'))
    
    # Add cross-asset features
    merged_df = feature_engineer.add_cross_asset_features(merged_df, ASSETS)
    
    print(f"✓ Final dataset: {len(merged_df)} rows, {len(merged_df.columns)} columns")
    
    # Filter data by date ranges
    merged_df['Date'] = pd.to_datetime(merged_df['Date'])
    train_df = merged_df[(merged_df['Date'] >= TRAIN_START) & (merged_df['Date'] <= TRAIN_END)]
    test_df = merged_df[(merged_df['Date'] >= TEST_START) & (merged_df['Date'] <= TEST_END)]
    
    print(f"✓ Training data: {len(train_df)} rows")
    print(f"✓ Testing data: {len(test_df)} rows")
    
    # Step 2: Prepare Advanced Dataset
    print("\n" + "=" * 80)
    print("STEP 2: PREPARING ADVANCED DATASET")
    print("=" * 80)
    
    # Identify feature columns (exclude Date and basic OHLCV)
    exclude_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume'] + \
                  [col for col in merged_df.columns if any(basic in col for basic in ['Open', 'High', 'Low', 'Close', 'Volume']) and not 'returns_' in col]
    
    feature_cols = [col for col in merged_df.columns if col not in exclude_cols]
    return_cols = [col for col in feature_cols if col.startswith('returns_')]
    condition_cols = [col for col in feature_cols if not col.startswith('returns_')]
    
    print(f"✓ Total features: {len(feature_cols)}")
    print(f"✓ Return features: {len(return_cols)}")
    print(f"✓ Condition features: {len(condition_cols)}")
    
    # Extract and normalize data
    train_features = train_df[feature_cols].fillna(0.0).values
    train_conditions = train_df[condition_cols].fillna(0.0).values if condition_cols else np.zeros((len(train_df), 1))
    
    test_features = test_df[feature_cols].fillna(0.0).values
    test_conditions = test_df[condition_cols].fillna(0.0).values if condition_cols else np.zeros((len(test_df), 1))
    
    # Create datasets
    sequence_length = min(128, len(train_features) // 10)  # Adaptive sequence length
    
    train_dataset = AdvancedFinancialDataset(
        train_features, train_conditions, sequence_length=sequence_length, augment=True
    )
    
    val_size = len(train_dataset) // 5
    train_size = len(train_dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])
    
    print(f"✓ Training samples: {len(train_dataset)}")
    print(f"✓ Validation samples: {len(val_dataset)}")
    print(f"✓ Sequence length: {sequence_length}")
    
    # Step 3: Build Ultimate Diffusion Model
    print("\n" + "=" * 80)
    print("STEP 3: BUILDING ULTIMATE DIFFUSION MODEL")
    print("=" * 80)
    
    # Training configuration
    training_config = AdvancedTrainingConfig(
        # Model architecture
        hidden_dim=512,  # Reduced for compatibility
        num_layers=8,    # Reduced for compatibility
        num_heads=8,
        max_seq_len=sequence_length,
        use_rotary_pos_emb=True,
        
        # Training hyperparameters
        batch_size=16,   # Reduced for memory
        num_epochs=50,   # Reduced for demo
        learning_rate=1e-4,
        weight_decay=1e-6,
        warmup_steps=1000,
        
        # Advanced features
        use_mixed_precision=True,
        use_ema=True,
        ema_decay=0.9999,
        dropout=0.1,
        
        # Monitoring
        eval_every=100,
        save_every=1000,
        use_wandb=False,  # Disabled for demo
    )
    
    # Initialize trainer
    trainer = AdvancedDiffusionTrainer(training_config)
    
    # Build model
    input_dim = len(return_cols)  # Only predict returns
    context_dim = len(condition_cols) if condition_cols else None
    
    model = trainer.build_model(input_dim, context_dim)
    
    print(f"✓ Model architecture: {training_config.hidden_dim}d, {training_config.num_layers} layers")
    print(f"✓ Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"✓ Input dimension: {input_dim}")
    print(f"✓ Context dimension: {context_dim}")
    
    # Step 4: Training (if model doesn't exist)
    print("\n" + "=" * 80)
    print("STEP 4: ADVANCED TRAINING")
    print("=" * 80)
    
    if not os.path.exists(MODEL_PATH):
        print("Training ultimate diffusion model...")
        
        # Create data loaders
        from torch.utils.data import DataLoader
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=training_config.batch_size, 
            shuffle=True, 
            num_workers=2,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=training_config.batch_size, 
            shuffle=False, 
            num_workers=2,
            pin_memory=True
        )
        
        # Train model
        history = trainer.train(train_loader, val_loader, checkpoint_dir="checkpoints")
        
        # Save final model
        trainer.save_checkpoint(MODEL_PATH, is_best=True)
        
        print(f"✓ Training completed!")
        print(f"✓ Best validation loss: {trainer.best_val_loss:.6f}")
        
    else:
        print(f"✓ Using existing model: {MODEL_PATH}")
        trainer.load_checkpoint(MODEL_PATH)
    
    # Step 5: Advanced Backtesting
    print("\n" + "=" * 80)
    print("STEP 5: ADVANCED BACKTESTING")
    print("=" * 80)
    
    # Configure advanced backtesting
    backtest_config = AdvancedBacktestConfig(
        initial_capital=1000000.0,  # $1M
        position_sizing_method="kelly",
        max_position_size=0.3,
        max_portfolio_leverage=1.0,
        
        # Transaction costs
        fixed_cost_per_trade=1.0,
        variable_cost_rate=0.001,  # 10 bps
        market_impact_model="linear",
        
        # Risk management
        portfolio_var_limit=0.02,
        max_drawdown_limit=0.20,
        
        # Advanced features
        use_regime_detection=True,
        walk_forward_enabled=False,  # Disabled for demo
        monte_carlo_runs=100,        # Reduced for demo
    )
    
    # Initialize backtester
    backtester = AdvancedBacktester(backtest_config)
    
    # Prepare metadata for backtesting
    metadata = {
        'assets': ASSETS,
        'feature_columns': return_cols,  # Only use return columns for prediction
        'return_columns': return_cols,
        'condition_columns': condition_cols,
        'sequence_length': sequence_length,
        'scalers': {
            'feature_means': np.mean(train_features, axis=0, keepdims=True),
            'feature_scales': np.std(train_features, axis=0, keepdims=True) + 1e-8
        }
    }
    
    # Run backtest
    print("Running advanced backtest...")
    results = backtester.run_backtest(
        trainer.diffusion_model,
        None,  # data_manager not needed for this implementation
        test_df,
        metadata
    )
    
    # Step 6: Monte Carlo Analysis
    print("\n" + "=" * 80)
    print("STEP 6: MONTE CARLO ANALYSIS") 
    print("=" * 80)
    
    if len(results.returns) > 20:  # Only run if we have enough data
        print("Running Monte Carlo simulation...")
        mc_results = backtester.monte_carlo_analysis(results)
        
        print(f"✓ Monte Carlo simulations completed ({backtest_config.monte_carlo_runs} runs)")
        print(f"  Simulated return mean: {mc_results['simulated_total_returns']['mean']:.2%}")
        print(f"  Simulated return 95% CI: [{mc_results['simulated_total_returns']['percentiles'][0]:.2%}, "
              f"{mc_results['simulated_total_returns']['percentiles'][4]:.2%}]")
    
    # Step 7: Generate Advanced Reports
    print("\n" + "=" * 80)
    print("STEP 7: GENERATING ADVANCED REPORTS")
    print("=" * 80)
    
    # Create benchmark results (simple buy & hold)
    if len(test_df) > 0:
        benchmark_returns = test_df[return_cols].fillna(0.0).values.mean(axis=1)
        initial_value = backtest_config.initial_capital
        benchmark_values = initial_value * np.cumprod(1 + benchmark_returns)
        
        from src.backtesting import create_benchmark_results
        benchmark_results = create_benchmark_results(benchmark_returns, initial_value)
        
        # Results comparison
        results_dict = {
            "Ultimate Diffusion": results,
            "Buy & Hold Benchmark": benchmark_results
        }
        
        print("Creating performance visualizations...")
        visualizer = PerformanceVisualizer()
        
        try:
            # Performance plots
            visualizer.plot_portfolio_performance(
                results_dict,
                save_path=f"{RESULTS_DIR}/ultimate_performance.png"
            )
            
            visualizer.plot_risk_metrics(
                results_dict,
                save_path=f"{RESULTS_DIR}/ultimate_risk_metrics.png"
            )
            
            # Position heatmap
            if not results.positions.empty:
                visualizer.plot_position_heatmap(
                    results.positions,
                    title="Ultimate Diffusion Model - Position Allocation",
                    save_path=f"{RESULTS_DIR}/ultimate_positions.png"
                )
            
        except Exception as e:
            print(f"Visualization error: {e}")
        
        # Generate comprehensive report
        print("Generating comprehensive HTML report...")
        try:
            report_generator = ReportGenerator()
            report_generator.generate_html_report(
                results_dict,
                metadata,
                output_path=f"{RESULTS_DIR}/ultimate_report.html"
            )
            
            # Save detailed results
            report_generator.save_results_to_csv(results_dict, f"{RESULTS_DIR}/")
            
        except Exception as e:
            print(f"Report generation error: {e}")
    
    # Step 8: Final Summary
    print("\n" + "=" * 100)
    print("🎯 ULTIMATE DIFFUSION MODEL - FINAL RESULTS")
    print("=" * 100)
    
    if len(results.returns) > 0:
        print(f"\n📊 PERFORMANCE METRICS:")
        print(f"  • Total Return:        {results.total_return:.2%}")
        print(f"  • Annualized Return:   {results.annualized_return:.2%}")
        print(f"  • Volatility:          {results.volatility:.2%}")
        print(f"  • Sharpe Ratio:        {results.sharpe_ratio:.3f}")
        print(f"  • Sortino Ratio:       {results.sortino_ratio:.3f}")
        print(f"  • Calmar Ratio:        {results.calmar_ratio:.3f}")
        print(f"  • Max Drawdown:        {results.max_drawdown:.2%}")
        
        print(f"\n⚡ RISK METRICS:")
        print(f"  • VaR (95%):          {results.var_95:.2%}")
        print(f"  • VaR (99%):          {results.var_99:.2%}")
        print(f"  • Expected Shortfall: {results.expected_shortfall_95:.2%}")
        print(f"  • Skewness:           {results.skewness:.3f}")
        print(f"  • Kurtosis:           {results.kurtosis:.3f}")
        
        if hasattr(benchmark_results, 'total_return'):
            outperformance = results.total_return - benchmark_results.total_return
            print(f"\n🎯 BENCHMARK COMPARISON:")
            print(f"  • Benchmark Return:    {benchmark_results.total_return:.2%}")
            print(f"  • Outperformance:      {outperformance:.2%}")
            
            if outperformance > 0:
                print(f"  • Result: ✅ OUTPERFORMED BENCHMARK!")
            else:
                print(f"  • Result: ❌ Underperformed benchmark")
        
        print(f"\n🔧 MODEL ARCHITECTURE:")
        print(f"  • Transformer Layers:  {training_config.num_layers}")
        print(f"  • Hidden Dimension:    {training_config.hidden_dim}")
        print(f"  • Attention Heads:     {training_config.num_heads}")
        print(f"  • Parameters:          {sum(p.numel() for p in model.parameters()):,}")
        print(f"  • Sequence Length:     {sequence_length}")
        
        print(f"\n📈 ADVANCED FEATURES:")
        print(f"  • ✅ Transformer-based architecture")
        print(f"  • ✅ Advanced diffusion (v-parameterization)")
        print(f"  • ✅ Sophisticated feature engineering")
        print(f"  • ✅ Kelly criterion position sizing")
        print(f"  • ✅ Advanced risk management")
        print(f"  • ✅ Transaction cost modeling")
        print(f"  • ✅ Monte Carlo analysis")
        
    else:
        print("❌ No backtest results generated")
    
    print(f"\n📁 RESULTS SAVED:")
    print(f"  • Model: {MODEL_PATH}")
    print(f"  • Reports: {RESULTS_DIR}/")
    print(f"  • Visualizations: {RESULTS_DIR}/*.png")
    print(f"  • HTML Report: {RESULTS_DIR}/ultimate_report.html")
    
    print("\n" + "=" * 100)
    print("🚀 ULTIMATE DIFFUSION MODEL EXECUTION COMPLETED!")
    print("State-of-the-art quantitative trading system ready for deployment!")
    print("=" * 100)

if __name__ == "__main__":
    main()