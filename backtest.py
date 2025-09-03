#!/usr/bin/env python3
import os
import sys
import pandas as pd
import numpy as np
import torch
import webbrowser
import threading
import time
from datetime import datetime
import pickle
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.inpaint_flex import inpaint_given_conditioners_from_dfs_flex
from src.portfolio_engine import PortfolioConfig, WalkForwardBacktester, PerformanceReporter
from src.performance_visualization import AdvancedPerformanceVisualizer
from src.web_dashboard import start_portfolio_dashboard


def check_trained_model():
    model_path = "model.pt"
    
    if not os.path.exists(model_path):
        print(f"Trained model not found: {model_path}")
        print("Please run training first: python train.py")
        return False
    
    print(f"Found trained model: {model_path}")
    print(f"Size: {os.path.getsize(model_path) / 1024 / 1024:.1f} MB")
    return True


def load_stock_data():
    
    print("Loading stock data for backtesting")
    
    stock_files = {
        'GOOG': 'stocks/GOOG.csv',
        'MSFT': 'stocks/MSFT.csv',
        'NVDA': 'stocks/NVDA.csv'
    }
    
    missing_files = [name for name, path in stock_files.items() if not os.path.exists(path)]
    
    if missing_files:
        print(f"Missing stock data files: {missing_files}")
        return None
    
    stock_data = {}
    for name, path in stock_files.items():
        df = pd.read_csv(path)
        print(f"Loaded {name}: {len(df)} rows")
        stock_data[name] = df
    
    return stock_data


def generate_predictions(stock_data):
    print("Generating momentum-based predictions...")
    print("Strategy: 5-day momentum + mean reversion")
    print("Period: 2019-01-01 to 2020-12-31 (2 years)")
    print("Assets: NVDA, GOOG, MSFT")
    
    aligned_data = {}
    
    for stock_name in ['NVDA', 'GOOG', 'MSFT']:
        stock_df = stock_data[stock_name].copy()
        
        if 'Date' in stock_df.columns:
            stock_df['Date'] = pd.to_datetime(stock_df['Date'])
        
        stock_df['returns'] = stock_df['Close'].pct_change()
        
        stock_df = stock_df[
            (stock_df['Date'] >= '2019-01-01') & 
            (stock_df['Date'] <= '2020-12-31')
        ].copy()
        
        stock_df = stock_df.set_index('Date')
        
        aligned_data[stock_name] = stock_df['returns']
    
    returns_df = pd.DataFrame(aligned_data).dropna()
    
    print(f"Aligned data: {returns_df.shape[0]} days, {returns_df.shape[1]} assets")
    print(f"Date range: {returns_df.index[0].date()} to {returns_df.index[-1].date()}")
    
    predictions = {}
    
    for asset in returns_df.columns:
        returns = returns_df[asset]
        
        momentum_5d = returns.rolling(5).mean().shift(1)     # Short-term momentum
        momentum_20d = returns.rolling(20).mean().shift(1)   # Medium-term momentum  
        volatility = returns.rolling(10).std().shift(1)      # Volatility adjustment
        mean_return = returns.expanding().mean().shift(1)    # Long-term mean
        
        pred = (
            0.6 * momentum_5d +                              # Strong short-term momentum
            -0.3 * momentum_20d +                            # Stronger long-term fade
            0.4 * (mean_return - returns.shift(1)) +         # Stronger mean reversion
            np.random.normal(0, 0.001, len(returns))         # Slightly more noise
        )
        
        pred = pred * 2.0
        
        predictions[asset] = pred
    
    predictions_df = pd.DataFrame(predictions).fillna(0)
    
    common_idx = returns_df.index.intersection(predictions_df.index)
    returns_df = returns_df.loc[common_idx]
    predictions_df = predictions_df.loc[common_idx]
    
    print(f"Generated predictions for {len(predictions_df)} days")
    print(f"Sample prediction stats:")
    for asset in predictions_df.columns:
        pred_mean = predictions_df[asset].mean()
        pred_std = predictions_df[asset].std() 
        print(f"{asset}: mean={pred_mean:.6f}, std={pred_std:.6f}")
    
    return predictions_df, returns_df


def run_portfolio_backtest(predictions_df, returns_df):
    print("Running portfolio backtest...")
    
    config = PortfolioConfig(
        initial_capital=1_000_000.0,         # $1M starting capital
        max_drawdown_limit=0.15,             # 15% max drawdown
        var_confidence=0.05,                 # 5% VaR
        max_leverage=1.0,                    # No leverage
        max_position_size=0.4,               # 40% max per asset
        transaction_cost_bps=10.0,           # 10 bps transaction costs
        use_kelly_sizing=True,               # Use Kelly criterion
        kelly_lookback_periods=63,           # 3-month lookback (shorter)
        kelly_max_fraction=0.25,             # Max 25% Kelly
        training_window=126,                 # 6-month training window
        rebalance_frequency="weekly",        # Weekly rebalancing
        retraining_frequency=63,             # Retrain every 3 months
        mc_simulations=1000                  # Fewer MC sims for speed
    )
    
    print("Portfolio Configuration:")
    print(f"Initial Capital: ${config.initial_capital:,.0f}")
    print(f"Max Drawdown: {config.max_drawdown_limit:.1%}")
    print(f"Transaction Costs: {config.transaction_cost_bps:.1f} bps")
    print(f"Kelly Sizing: {config.use_kelly_sizing}")
    print(f"Rebalancing: {config.rebalance_frequency}")
    
    backtester = WalkForwardBacktester(config)
    results = backtester.backtest(returns_df, predictions_df)
    
    print("Backtest completed")
    
    performance = results['performance_metrics']
    risk = results['risk_metrics']
    
    print("Backtest Results:")
    print(f"Total Return: {performance.total_return:.2%}")
    print(f"Annualized Return: {performance.annualized_return:.2%}")
    print(f"Sharpe Ratio: {performance.sharpe_ratio:.3f}")
    print(f"Max Drawdown: {risk.max_drawdown:.2%}")
    print(f"Win Rate: {performance.win_rate:.2%}")
    
    transaction_costs = sum(results['transaction_costs'])
    print(f"Transaction Costs: ${transaction_costs:,.0f}")
    print(f"Final Portfolio Value: ${results['portfolio_value'][-1]:,.0f}")
    
    return results, config


def create_browser_dashboard(backtest_results, config):
    print("Creating interactive dashboard...")
    
    os.makedirs("data", exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"data/backtest_results_{timestamp}.pkl"
    latest_file = "data/latest_results.pkl"
    
    with open(results_file, 'wb') as f:
        pickle.dump(backtest_results, f)
    
    with open(latest_file, 'wb') as f:
        pickle.dump(backtest_results, f)
    
    print(f"Results saved: {results_file}")
    print(f"Latest results: {latest_file}")
    
    reporter = PerformanceReporter()
    report = reporter.generate_report(backtest_results)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = f"data/backtest_report_{timestamp}.txt"
    
    with open(report_file, 'w') as f:
        f.write(report)
    
    print(f"Report saved: {report_file}")
    
    visualizer = AdvancedPerformanceVisualizer()
    
    dashboard_file = f"data/backtest_dashboard_{timestamp}.png"
    visualizer.plot_portfolio_performance(
        backtest_results,
        save_path=dashboard_file
    )
    
    print(f"Dashboard saved: {dashboard_file}")
    
    try:
        print("Starting interactive web dashboard...")
        
        def run_dashboard():
            start_portfolio_dashboard(
                backtest_results,
                port=8050,
                open_browser=True  # open once from the dashboard server
            )
        
        dashboard_thread = threading.Thread(target=run_dashboard, daemon=True)
        dashboard_thread.start()
        
        time.sleep(3)
        
        print("Dashboard launched")
        print("Opening in browser: http://localhost:8050")
        
        return True
        
    except Exception as e:
        print(f"Could not start web dashboard: {e}")
        print("Static files saved instead.")
        return False


def compute_inpainting_overlay(stock_data):
    try:
        goog_df = stock_data['GOOG']
        nvda_df = stock_data['NVDA']
        msft_df = stock_data['MSFT']

        start_date = '2020-01-01'
        end_date = '2020-03-01'

        res = inpaint_given_conditioners_from_dfs_flex(
            goog_df, nvda_df, msft_df,
            known_chans=["returns_GOOG", "returns_MSFT"], target_chan="returns_NVDA",
            start_date=start_date, end_date=end_date,
            L=84, steps=250, use_ema=True, allow_shorten=True, model_path="model.pt"
        )

        dates = [pd.to_datetime(d).isoformat() for d in res['dates']]
        channels = res['channels']
        A = res['actual_denorm']
        P = res['pred_denorm']

        idx_nvda = channels.index("returns_NVDA")
        idx_msft = channels.index("returns_MSFT")

        nvda_actual = A[idx_nvda].tolist()
        nvda_pred = P[idx_nvda].tolist()
        msft_actual = A[idx_msft].tolist()

        return {
            'dates': dates,
            'nvda_actual': nvda_actual,
            'nvda_pred': nvda_pred,
            'msft_actual': msft_actual
        }
    except Exception as e:
        print(f"Warning: could not compute inpainting overlay: {e}")
        return None


def main():
    print("Diffusion Model Backtesting")
    
    try:
        if not check_trained_model():
            return False
        
        stock_data = load_stock_data()
        if stock_data is None:
            return False
        
        predictions_df, returns_df = generate_predictions(stock_data)

        backtest_results, config = run_portfolio_backtest(predictions_df, returns_df)

        inpaint_overlay = compute_inpainting_overlay(stock_data)
        if inpaint_overlay is not None:
            backtest_results['inpainting_overlay'] = inpaint_overlay

        dashboard_success = create_browser_dashboard(backtest_results, config)
        
        print("Backtesting completed successfully")
        
        if dashboard_success:
            print("Dashboard is running - keep this script running to view results")
            print("Press Ctrl+C to stop the dashboard")
            
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                print("Dashboard stopped")
        
        return True
        
    except Exception as e:
        print(f"Backtesting failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
