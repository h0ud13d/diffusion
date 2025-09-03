#!/usr/bin/env python3

import os
import sys
import itertools
import argparse
import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.train import train_ddpm_from_dfs


def load_stock_data():
    stock_files = {
        'GOOG': 'stocks/GOOG.csv',
        'MSFT': 'stocks/MSFT.csv', 
        'NVDA': 'stocks/NVDA.csv'
    }
    
    missing_files = [name for name, path in stock_files.items() if not os.path.exists(path)]
    
    if missing_files:
        print(f"Missing stock data files: {missing_files}")
        print("Required files:")
        for name, path in stock_files.items():
            status = "Found" if os.path.exists(path) else "Missing"
            print(f"  {status}: {path}")
        print("Please ensure stock data files are in the stocks/ directory.")
        return None
    
    stock_data = {}
    for name, path in stock_files.items():
        df = pd.read_csv(path)
        print(f"Loaded {name}: {len(df)} rows")
        stock_data[name] = df
    
    print(f"Successfully loaded data for {len(stock_data)} stocks")
    return stock_data


def run_single_training(goog_df, nvda_df, msft_df, *,
                        start_date="2018-01-01", end_date=None,
                        L=84, epochs=150, batch=128, steps=1000, lr=2e-4,
                        ema_decay=0.999, dropout=0.0, use_amp=True,
                        model_path="model.pt"):
    model_path, metadata, best_val = train_ddpm_from_dfs(
        goog_df, nvda_df, msft_df,
        start_date=start_date, end_date=end_date,
        L=L, epochs=epochs, batch=batch, steps=steps, lr=lr,
        ema_decay=ema_decay, dropout=dropout, use_amp=use_amp,
        model_path=model_path
    )
    return model_path, metadata, best_val


def hyperparameter_search(goog_df, nvda_df, msft_df, *,
                          search_epochs=12, quick=False, outdir="search_models"):
    os.makedirs(outdir, exist_ok=True)

    space = {
        'L': [64, 84, 128] if not quick else [84],
        'steps': [500, 1000] if not quick else [1000],
        'lr': [1e-4, 2e-4, 5e-4] if not quick else [2e-4],
        'dropout': [0.0, 0.1, 0.2] if not quick else [0.1],
        'ema_decay': [0.995, 0.999, 0.9995] if not quick else [0.999],
        'batch': [64, 128] if not quick else [128],
    }

    keys = list(space.keys())
    combos = list(itertools.product(*[space[k] for k in keys]))

    print(f"Hyperparameter search: {len(combos)} combos ({'quick' if quick else 'full'})")
    best = None
    results = []

    for i, vals in enumerate(combos, start=1):
        params = dict(zip(keys, vals))
        tag = "_".join(f"{k}-{v}" for k, v in params.items())
        model_path = os.path.join(outdir, f"model_{tag}.pt")
        print(f"\n[Search {i}/{len(combos)}] {params}")
        try:
            _, _, val_loss = run_single_training(
                goog_df, nvda_df, msft_df,
                L=params['L'], epochs=search_epochs, batch=params['batch'],
                steps=params['steps'], lr=params['lr'], ema_decay=params['ema_decay'],
                dropout=params['dropout'], use_amp=True,
                model_path=model_path
            )
        except Exception as e:
            print(f"Trial failed: {e}")
            val_loss = float('inf')

        results.append((val_loss, params, model_path))
        if best is None or val_loss < best[0]:
            best = (val_loss, params, model_path)
        print(f"--> val_loss={val_loss:.6f}  best_so_far={best[0]:.6f}")

    print("\nSearch complete.")
    print(f"Best val loss: {best[0]:.6f} with params: {best[1]}")
    return best, results


def parse_args():
    p = argparse.ArgumentParser(description="Train diffusion model for stocks")
    p.add_argument('--search', action='store_true', help='Run hyperparameter search before final training')
    p.add_argument('--quick', action='store_true', help='Use quick/compact search space')
    p.add_argument('--search-epochs', type=int, default=12, help='Epochs per trial during search')
    p.add_argument('--final-epochs', type=int, default=150, help='Epochs for final training')
    p.add_argument('--model-path', type=str, default='model.pt', help='Output model path')
    return p.parse_args()


def main():
    args = parse_args()

    stock_data = load_stock_data()
    if stock_data is None:
        return False
    
    goog_df = stock_data['GOOG']
    msft_df = stock_data['MSFT'] 
    nvda_df = stock_data['NVDA']

    if args.search:
        print("Starting hyperparameter search")
        (best_val, best_params, _path), _all = hyperparameter_search(
            goog_df, nvda_df, msft_df,
            search_epochs=args.search_epochs,
            quick=args.quick
        )
        params = best_params
        print("\nStarting final training with best params:")
        print(params)
        L = params['L']; batch = params['batch']; steps = params['steps']
        lr = params['lr']; ema_decay = params['ema_decay']; dropout = params['dropout']
    else:
        L = 84; batch = 128; steps = 1000; lr = 2e-4; ema_decay = 0.999; dropout = 0.1

    print("\nStarting training...")
    print("Configuration:")
    print(f"Sequence Length: {L} days")
    print(f"Epochs: {args.final_epochs}")
    print(f"Batch Size: {batch}") 
    print(f"Learning Rate: {lr}")
    print(f"Timesteps: {steps}")
    print("Date Range: 2018-01-01 onwards")
    
    try:
        model_path, metadata, best_val = run_single_training(
            goog_df, nvda_df, msft_df,
            start_date="2018-01-01", end_date=None,
            L=L, epochs=args.final_epochs, batch=batch, steps=steps, lr=lr,
            ema_decay=ema_decay, dropout=dropout, use_amp=True, model_path=args.model_path
        )
        
        print(f"Training completed successfully")
        print(f"Model saved: {model_path}")
        print(f"Model size: {os.path.getsize(model_path) / 1024 / 1024:.1f} MB")
        print(f"Best validation loss: {best_val:.6f}")
        
        if metadata:
            print("Model metadata:")
            for key, value in metadata.items():
                if isinstance(value, dict):
                    print(f"{key}: {len(value)} items")
                else:
                    print(f"{key}: {value}")
        
        print("Ready for backtesting")
        print("Next step: python backtest.py")
        return True
        
    except Exception as e:
        print(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
