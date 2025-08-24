# portfolio_framework.py - Multi-asset portfolio modeling and advanced data processing
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Tuple, Optional, Union
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class AdvancedDataProcessor:
    """Enhanced data processing with technical indicators and multi-timeframe features"""
    
    def __init__(self, assets: List[str], lookback_periods: List[int] = [5, 10, 20, 50]):
        self.assets = assets
        self.lookback_periods = lookback_periods
        
    def calculate_technical_indicators(self, df: pd.DataFrame, asset: str) -> pd.DataFrame:
        """Calculate technical indicators for a single asset"""
        df = df.copy()
        close_col = 'Adj Close'
        volume_col = 'Volume'
        
        # Returns
        df[f'returns_{asset}'] = np.log(df[close_col]).diff().fillna(0.0)
        
        # Volatility (rolling standard deviation of returns)
        for period in self.lookback_periods:
            df[f'volatility_{period}d_{asset}'] = df[f'returns_{asset}'].rolling(period).std().fillna(0.0)
        
        # Moving averages
        for period in self.lookback_periods:
            df[f'sma_{period}d_{asset}'] = df[close_col].rolling(period).mean()
            df[f'price_to_sma_{period}d_{asset}'] = df[close_col] / df[f'sma_{period}d_{asset}'] - 1.0
        
        # Exponential moving averages
        for period in self.lookback_periods[:3]:  # Only shorter periods for EMA
            df[f'ema_{period}d_{asset}'] = df[close_col].ewm(span=period).mean()
            df[f'price_to_ema_{period}d_{asset}'] = df[close_col] / df[f'ema_{period}d_{asset}'] - 1.0
        
        # RSI (Relative Strength Index)
        delta = df[close_col].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df[f'rsi_{asset}'] = 100 - (100 / (1 + rs))
        df[f'rsi_{asset}'] = (df[f'rsi_{asset}'] - 50) / 50  # Normalize to [-1, 1]
        
        # Bollinger Bands
        for period in [20]:  # Standard Bollinger period
            rolling_mean = df[close_col].rolling(period).mean()
            rolling_std = df[close_col].rolling(period).std()
            df[f'bb_upper_{period}d_{asset}'] = rolling_mean + (2 * rolling_std)
            df[f'bb_lower_{period}d_{asset}'] = rolling_mean - (2 * rolling_std)
            df[f'bb_position_{period}d_{asset}'] = (df[close_col] - rolling_mean) / (2 * rolling_std)
        
        # MACD
        ema_12 = df[close_col].ewm(span=12).mean()
        ema_26 = df[close_col].ewm(span=26).mean()
        df[f'macd_{asset}'] = ema_12 - ema_26
        df[f'macd_signal_{asset}'] = df[f'macd_{asset}'].ewm(span=9).mean()
        df[f'macd_histogram_{asset}'] = df[f'macd_{asset}'] - df[f'macd_signal_{asset}']
        
        # Normalize MACD features by price
        price_scale = df[close_col].rolling(50).mean().fillna(df[close_col])
        df[f'macd_{asset}'] /= price_scale
        df[f'macd_signal_{asset}'] /= price_scale
        df[f'macd_histogram_{asset}'] /= price_scale
        
        # Volume indicators
        if volume_col in df.columns:
            df[f'volume_sma_{asset}'] = df[volume_col].rolling(20).mean()
            df[f'volume_ratio_{asset}'] = df[volume_col] / df[f'volume_sma_{asset}'] - 1.0
            
            # Price-Volume Trend
            df[f'pvt_{asset}'] = ((df[close_col].diff() / df[close_col].shift()) * df[volume_col]).cumsum()
            df[f'pvt_{asset}'] = df[f'pvt_{asset}'].diff().fillna(0.0)  # Use changes in PVT
        
        # Momentum indicators
        for period in self.lookback_periods:
            df[f'momentum_{period}d_{asset}'] = df[close_col].pct_change(period).fillna(0.0)
        
        # Support/Resistance levels (simplified)
        for period in [20, 50]:
            df[f'high_{period}d_{asset}'] = df['High'].rolling(period).max()
            df[f'low_{period}d_{asset}'] = df['Low'].rolling(period).min()
            df[f'price_to_high_{period}d_{asset}'] = df[close_col] / df[f'high_{period}d_{asset}'] - 1.0
            df[f'price_to_low_{period}d_{asset}'] = df[close_col] / df[f'low_{period}d_{asset}'] - 1.0
        
        return df
    
    def calculate_cross_asset_features(self, merged_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate cross-asset correlation and beta features"""
        df = merged_df.copy()
        
        # Get return columns
        return_cols = [col for col in df.columns if col.startswith('returns_')]
        
        if len(return_cols) < 2:
            return df
        
        # Rolling correlations between assets
        for i, asset1 in enumerate(return_cols):
            for j, asset2 in enumerate(return_cols[i+1:], i+1):
                for period in [20, 50]:
                    corr_col = f'corr_{period}d_{asset1}_{asset2}'
                    df[corr_col] = df[asset1].rolling(period).corr(df[asset2]).fillna(0.0)
        
        # Rolling betas (assuming first asset is market proxy)
        if len(return_cols) > 1:
            market_returns = df[return_cols[0]]  # First asset as market
            for asset_col in return_cols[1:]:
                for period in [20, 50]:
                    covariance = df[asset_col].rolling(period).cov(market_returns)
                    market_variance = market_returns.rolling(period).var()
                    beta_col = f'beta_{period}d_{asset_col}_to_{return_cols[0]}'
                    df[beta_col] = (covariance / market_variance).fillna(0.0)
        
        # Cross-asset momentum spreads
        for i, asset1 in enumerate(return_cols):
            for j, asset2 in enumerate(return_cols[i+1:], i+1):
                for period in [5, 20]:
                    momentum1 = df[asset1].rolling(period).mean()
                    momentum2 = df[asset2].rolling(period).mean()
                    spread_col = f'momentum_spread_{period}d_{asset1}_{asset2}'
                    df[spread_col] = (momentum1 - momentum2).fillna(0.0)
        
        return df
    
    def add_market_regime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add market regime and macro environment features"""
        df = df.copy()
        
        # Get return columns
        return_cols = [col for col in df.columns if col.startswith('returns_')]
        
        if not return_cols:
            return df
        
        # Market-wide volatility (VIX proxy)
        all_returns = df[return_cols].fillna(0)
        df['market_volatility'] = all_returns.std(axis=1)
        df['market_volatility_ma'] = df['market_volatility'].rolling(20).mean().fillna(df['market_volatility'])
        df['volatility_regime'] = (df['market_volatility'] > df['market_volatility_ma']).astype(float)
        
        # Trend regime
        market_return = all_returns.mean(axis=1)
        df['market_trend_5d'] = market_return.rolling(5).mean()
        df['market_trend_20d'] = market_return.rolling(20).mean()
        df['trend_regime'] = (df['market_trend_5d'] > df['market_trend_20d']).astype(float)
        
        # Drawdown features
        market_cumret = (1 + market_return).cumprod()
        market_peak = market_cumret.expanding().max()
        df['market_drawdown'] = (market_cumret / market_peak - 1.0).fillna(0.0)
        df['drawdown_regime'] = (df['market_drawdown'] < -0.05).astype(float)  # 5% drawdown threshold
        
        return df

class PortfolioDataset(Dataset):
    """Enhanced dataset for multi-asset portfolio modeling"""
    
    def __init__(self, 
                 X: np.ndarray, 
                 conditions: np.ndarray,
                 sequence_length: int,
                 prediction_horizon: int = 1,
                 stride: int = 1):
        """
        Args:
            X: Features array (N, num_features)
            conditions: Conditioning features (N, condition_dim)
            sequence_length: Length of input sequences
            prediction_horizon: How many steps ahead to predict
            stride: Stride between sequences
        """
        self.X = X
        self.conditions = conditions
        self.L = sequence_length
        self.H = prediction_horizon
        self.stride = stride
        self.N = X.shape[0]
        self.num_features = X.shape[1]
        
        # Calculate valid starting positions
        max_start = self.N - self.L - self.H + 1
        if max_start <= 0:
            raise ValueError(f"Not enough data points. Need at least {self.L + self.H}, got {self.N}")
        
        self.starts = np.arange(0, max_start, self.stride)
    
    def __len__(self):
        return len(self.starts)
    
    def __getitem__(self, idx):
        start = self.starts[idx]
        
        # Input sequence
        x_seq = self.X[start:start + self.L]  # (L, num_features)
        
        # Target sequence (for multi-step prediction)
        y_seq = self.X[start + self.L:start + self.L + self.H]  # (H, num_features)
        
        # Conditioning information (use last available)
        conditions = self.conditions[start + self.L - 1]  # (condition_dim,)
        
        return {
            'x': torch.from_numpy(x_seq.T).float(),  # (num_features, L)
            'y': torch.from_numpy(y_seq.T).float(),  # (num_features, H)
            'conditions': torch.from_numpy(conditions).float()
        }

class PortfolioDataManager:
    """Comprehensive data management for portfolio modeling"""
    
    def __init__(self, 
                 assets: List[str],
                 data_dir: str = "stocks/",
                 lookback_periods: List[int] = [5, 10, 20, 50]):
        
        self.assets = assets
        self.data_dir = data_dir
        self.processor = AdvancedDataProcessor(assets, lookback_periods)
        self.scalers = {}
    
    def load_and_process_data(self, 
                            start_date: Optional[str] = None,
                            end_date: Optional[str] = None) -> Tuple[pd.DataFrame, Dict]:
        """Load and process all asset data"""
        
        # Load individual asset data
        asset_dfs = {}
        for asset in self.assets:
            file_path = f"{self.data_dir}/{asset}.csv"
            try:
                df = pd.read_csv(file_path)
                df['Date'] = pd.to_datetime(df['Date'])
                
                # Filter by date range
                if start_date:
                    df = df[df['Date'] >= pd.Timestamp(start_date)]
                if end_date:
                    df = df[df['Date'] <= pd.Timestamp(end_date)]
                
                # Calculate technical indicators
                df = self.processor.calculate_technical_indicators(df, asset)
                
                # Debug: print columns to identify duplicates
                if 'Date' in df.columns.tolist() and df.columns.tolist().count('Date') > 1:
                    print(f"Warning: Multiple 'Date' columns found in {asset} data")
                    print(f"Columns: {df.columns.tolist()}")
                    # Keep only the first Date column
                    date_cols = [i for i, col in enumerate(df.columns) if col == 'Date']
                    if len(date_cols) > 1:
                        df = df.iloc[:, [i for i in range(len(df.columns)) if i != date_cols[1]]]
                
                asset_dfs[asset] = df
                
            except FileNotFoundError:
                print(f"Warning: Data file for {asset} not found at {file_path}")
                continue
        
        if not asset_dfs:
            raise ValueError("No asset data could be loaded")
        
        # Merge all assets on date
        merged_df = None
        for asset, df in asset_dfs.items():
            # Select relevant columns (exclude OHLCV raw data to avoid redundancy)
            keep_cols = ['Date'] + [col for col in df.columns 
                                   if not col in ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume'] and col != 'Date']
            # Ensure we only have one Date column
            df_subset = df[keep_cols].copy()
            
            # Remove any duplicate Date columns
            if 'Date' in df_subset.columns and df_subset.columns.tolist().count('Date') > 1:
                df_subset = df_subset.loc[:, ~df_subset.columns.duplicated()]
            
            if merged_df is None:
                merged_df = df_subset
            else:
                merged_df = merged_df.merge(df_subset, on='Date', how='inner')
        
        # Calculate cross-asset features
        merged_df = self.processor.calculate_cross_asset_features(merged_df)
        
        # Add market regime features
        merged_df = self.processor.add_market_regime_features(merged_df)
        
        # Sort by date and reset index
        merged_df = merged_df.sort_values('Date').reset_index(drop=True)
        
        # Create metadata
        feature_columns = [col for col in merged_df.columns if col != 'Date']
        return_columns = [col for col in feature_columns if col.startswith('returns_')]
        condition_columns = [col for col in feature_columns if not col.startswith('returns_')]
        
        metadata = {
            'assets': self.assets,
            'feature_columns': feature_columns,
            'return_columns': return_columns,
            'condition_columns': condition_columns,
            'dates': merged_df['Date'].values,
            'start_date': start_date,
            'end_date': end_date
        }
        
        return merged_df, metadata
    
    def create_normalized_datasets(self,
                                 merged_df: pd.DataFrame,
                                 metadata: Dict,
                                 sequence_length: int = 84,
                                 prediction_horizon: int = 1,
                                 train_split: float = 0.8,
                                 val_split: float = 0.1) -> Tuple[PortfolioDataset, PortfolioDataset, PortfolioDataset, Dict]:
        """Create normalized train/val/test datasets"""
        
        # Extract features
        feature_data = merged_df[metadata['feature_columns']].fillna(0.0).values
        return_data = merged_df[metadata['return_columns']].fillna(0.0).values
        condition_data = merged_df[metadata['condition_columns']].fillna(0.0).values
        
        # Split data temporally
        n_total = len(feature_data)
        n_train = int(n_total * train_split)
        n_val = int(n_total * val_split)
        n_test = n_total - n_train - n_val
        
        # Ensure minimum data for each split
        min_data_per_split = sequence_length + prediction_horizon + 10
        if n_val < min_data_per_split:
            n_val = min(min_data_per_split, n_total // 4)
            n_train = n_total - n_val - max(min_data_per_split, n_total // 6)
            n_test = n_total - n_train - n_val
        
        train_features = feature_data[:n_train]
        train_returns = return_data[:n_train]
        train_conditions = condition_data[:n_train]
        
        val_features = feature_data[n_train:n_train + n_val]
        val_returns = return_data[n_train:n_train + n_val]
        val_conditions = condition_data[n_train:n_train + n_val]
        
        test_features = feature_data[n_train + n_val:]
        test_returns = return_data[n_train + n_val:]
        test_conditions = condition_data[n_train + n_val:]
        
        # Normalize data using training statistics
        # Features (robust normalization)
        feature_means = np.median(train_features, axis=0, keepdims=True)
        feature_scales = np.percentile(np.abs(train_features - feature_means), 90, axis=0, keepdims=True)
        feature_scales = np.maximum(feature_scales, 1e-8)
        
        train_features_norm = (train_features - feature_means) / feature_scales
        val_features_norm = (val_features - feature_means) / feature_scales
        test_features_norm = (test_features - feature_means) / feature_scales
        
        # Conditions (robust normalization)
        condition_means = np.median(train_conditions, axis=0, keepdims=True)
        condition_scales = np.percentile(np.abs(train_conditions - condition_means), 90, axis=0, keepdims=True)
        condition_scales = np.maximum(condition_scales, 1e-8)
        
        train_conditions_norm = (train_conditions - condition_means) / condition_scales
        val_conditions_norm = (val_conditions - condition_means) / condition_scales
        test_conditions_norm = (test_conditions - condition_means) / condition_scales
        
        # Store normalization parameters
        self.scalers = {
            'feature_means': feature_means,
            'feature_scales': feature_scales,
            'condition_means': condition_means,
            'condition_scales': condition_scales,
            'return_means': np.zeros((1, len(metadata['return_columns']))),  # Keep returns uncentered
            'return_scales': np.ones((1, len(metadata['return_columns'])))   # Keep returns unscaled
        }
        
        # Create datasets
        try:
            train_dataset = PortfolioDataset(train_features_norm, train_conditions_norm, 
                                           sequence_length, prediction_horizon)
            val_dataset = PortfolioDataset(val_features_norm, val_conditions_norm, 
                                         sequence_length, prediction_horizon)
            test_dataset = PortfolioDataset(test_features_norm, test_conditions_norm, 
                                          sequence_length, prediction_horizon)
        except ValueError as e:
            print(f"Error creating datasets: {e}")
            # Try with shorter sequence length
            sequence_length = min(sequence_length, n_train // 4)
            print(f"Retrying with sequence_length={sequence_length}")
            
            train_dataset = PortfolioDataset(train_features_norm, train_conditions_norm, 
                                           sequence_length, prediction_horizon)
            val_dataset = PortfolioDataset(val_features_norm, val_conditions_norm, 
                                         sequence_length, prediction_horizon)
            test_dataset = PortfolioDataset(test_features_norm, test_conditions_norm, 
                                          sequence_length, prediction_horizon)
        
        # Update metadata with normalization info
        updated_metadata = metadata.copy()
        updated_metadata.update({
            'scalers': self.scalers,
            'sequence_length': sequence_length,
            'prediction_horizon': prediction_horizon,
            'n_train': len(train_dataset),
            'n_val': len(val_dataset),
            'n_test': len(test_dataset),
            'feature_dim': len(metadata['feature_columns']),
            'condition_dim': len(metadata['condition_columns']),
            'return_dim': len(metadata['return_columns'])
        })
        
        return train_dataset, val_dataset, test_dataset, updated_metadata
    
    def denormalize_features(self, features_norm: np.ndarray) -> np.ndarray:
        """Convert normalized features back to original scale"""
        return features_norm * self.scalers['feature_scales'] + self.scalers['feature_means']
    
    def denormalize_conditions(self, conditions_norm: np.ndarray) -> np.ndarray:
        """Convert normalized conditions back to original scale"""
        return conditions_norm * self.scalers['condition_scales'] + self.scalers['condition_means']