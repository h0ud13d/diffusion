# backtesting.py - Comprehensive backtesting framework for diffusion models
import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Tuple, Optional, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import warnings
warnings.filterwarnings('ignore')

@dataclass
class BacktestConfig:
    """Configuration for backtesting"""
    initial_capital: float = 100000.0
    rebalance_frequency: str = "daily"  # daily, weekly, monthly
    transaction_cost: float = 0.001  # 0.1% transaction costs
    max_leverage: float = 1.0
    risk_free_rate: float = 0.02  # 2% annual risk-free rate
    benchmark_returns: Optional[np.ndarray] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    
    # Risk management
    max_position_size: float = 0.3  # 30% max position per asset
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    
    # Walk-forward parameters
    walk_forward: bool = True
    training_window: int = 252  # 1 year
    retraining_frequency: int = 63  # 3 months
    prediction_horizon: int = 5  # 5-day predictions

@dataclass 
class BacktestResults:
    """Container for backtest results"""
    # Performance metrics
    total_return: float = 0.0
    annualized_return: float = 0.0
    volatility: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    calmar_ratio: float = 0.0
    
    # Risk metrics
    var_95: float = 0.0
    var_99: float = 0.0
    expected_shortfall_95: float = 0.0
    downside_deviation: float = 0.0
    sortino_ratio: float = 0.0
    
    # Trade statistics
    num_trades: int = 0
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 0.0
    
    # Time series data
    portfolio_value: pd.Series = field(default_factory=pd.Series)
    positions: pd.DataFrame = field(default_factory=pd.DataFrame)
    returns: pd.Series = field(default_factory=pd.Series)
    drawdowns: pd.Series = field(default_factory=pd.Series)
    trades: pd.DataFrame = field(default_factory=pd.DataFrame)
    
    # Model-specific metrics
    prediction_accuracy: float = 0.0
    directional_accuracy: float = 0.0
    correlation_with_actual: float = 0.0

class RiskManager:
    """Risk management utilities"""
    
    def __init__(self, config: BacktestConfig):
        self.config = config
    
    def apply_position_limits(self, weights: np.ndarray) -> np.ndarray:
        """Apply position size limits"""
        # Clip individual positions
        weights = np.clip(weights, -self.config.max_position_size, self.config.max_position_size)
        
        # Apply leverage constraint
        total_leverage = np.sum(np.abs(weights))
        if total_leverage > self.config.max_leverage:
            weights = weights * (self.config.max_leverage / total_leverage)
        
        return weights
    
    def calculate_var(self, returns: np.ndarray, confidence: float = 0.95) -> float:
        """Calculate Value at Risk"""
        return np.percentile(returns, (1 - confidence) * 100)
    
    def calculate_expected_shortfall(self, returns: np.ndarray, confidence: float = 0.95) -> float:
        """Calculate Expected Shortfall (Conditional VaR)"""
        var = self.calculate_var(returns, confidence)
        return returns[returns <= var].mean()

class PerformanceAnalyzer:
    """Performance and risk analysis utilities"""
    
    def __init__(self, risk_free_rate: float = 0.02):
        self.risk_free_rate = risk_free_rate
    
    def calculate_returns(self, prices: pd.Series) -> pd.Series:
        """Calculate returns from prices"""
        return prices.pct_change().fillna(0.0)
    
    def calculate_sharpe_ratio(self, returns: pd.Series, risk_free_rate: Optional[float] = None) -> float:
        """Calculate Sharpe ratio"""
        rf = risk_free_rate if risk_free_rate is not None else self.risk_free_rate
        excess_returns = returns - rf / 252  # Daily risk-free rate
        return np.sqrt(252) * excess_returns.mean() / returns.std() if returns.std() > 0 else 0.0
    
    def calculate_sortino_ratio(self, returns: pd.Series, risk_free_rate: Optional[float] = None) -> float:
        """Calculate Sortino ratio (downside deviation)"""
        rf = risk_free_rate if risk_free_rate is not None else self.risk_free_rate
        excess_returns = returns - rf / 252
        downside_returns = excess_returns[excess_returns < 0]
        downside_dev = np.sqrt(252) * downside_returns.std() if len(downside_returns) > 0 else 0.0
        return np.sqrt(252) * excess_returns.mean() / downside_dev if downside_dev > 0 else 0.0
    
    def calculate_max_drawdown(self, prices: pd.Series) -> Tuple[float, pd.Series]:
        """Calculate maximum drawdown and drawdown series"""
        cumulative = prices / prices.iloc[0]
        peak = cumulative.expanding().max()
        drawdown = (cumulative / peak - 1.0)
        max_dd = drawdown.min()
        return max_dd, drawdown
    
    def calculate_calmar_ratio(self, returns: pd.Series, max_drawdown: float) -> float:
        """Calculate Calmar ratio"""
        annual_return = (1 + returns).prod() ** (252 / len(returns)) - 1
        return annual_return / abs(max_drawdown) if max_drawdown != 0 else 0.0
    
    def analyze_trades(self, trades: pd.DataFrame) -> Dict:
        """Analyze trade statistics"""
        if len(trades) == 0:
            return {}
        
        trades['pnl'] = trades['exit_price'] - trades['entry_price']
        wins = trades[trades['pnl'] > 0]
        losses = trades[trades['pnl'] < 0]
        
        return {
            'num_trades': len(trades),
            'win_rate': len(wins) / len(trades) if len(trades) > 0 else 0.0,
            'avg_win': wins['pnl'].mean() if len(wins) > 0 else 0.0,
            'avg_loss': losses['pnl'].mean() if len(losses) > 0 else 0.0,
            'profit_factor': abs(wins['pnl'].sum() / losses['pnl'].sum()) if len(losses) > 0 and losses['pnl'].sum() != 0 else 0.0
        }

class DiffusionBacktester:
    """Main backtesting engine for diffusion models"""
    
    def __init__(self, config: BacktestConfig):
        self.config = config
        self.risk_manager = RiskManager(config)
        self.analyzer = PerformanceAnalyzer(config.risk_free_rate)
    
    def generate_signals(self, 
                        model,
                        data: pd.DataFrame,
                        features: np.ndarray,
                        conditions: np.ndarray,
                        metadata: Dict,
                        current_idx: int,
                        lookback_window: int) -> np.ndarray:
        """Generate trading signals using diffusion model predictions"""
        
        # Extract recent data for prediction
        if current_idx < lookback_window:
            return np.zeros(metadata['return_dim'])
        
        # Get input sequence
        start_idx = max(0, current_idx - lookback_window)
        input_features = features[start_idx:current_idx]  # (L, num_features)
        input_conditions = conditions[current_idx - 1]  # Most recent conditions
        
        # Prepare model inputs
        device = next(model.parameters()).device
        x = torch.from_numpy(input_features.T).float().unsqueeze(0).to(device)  # (1, num_features, L)
        cond = torch.from_numpy(input_conditions).float().unsqueeze(0).to(device)  # (1, condition_dim)
        
        # Generate predictions
        model.eval()
        with torch.no_grad():
            # Sample multiple predictions for uncertainty estimation
            predictions = []
            for _ in range(5):  # Generate 5 samples
                pred = model.sample(
                    shape=(1, metadata['return_dim'], self.config.prediction_horizon),
                    context=cond,
                    method="ddim",
                    steps=50,
                    progress=False
                )
                predictions.append(pred.cpu().numpy())
            
            predictions = np.array(predictions)  # (5, 1, return_dim, H)
            mean_prediction = predictions.mean(axis=0)[0, :, 0]  # Use first time step
            std_prediction = predictions.std(axis=0)[0, :, 0]
        
        # Convert predictions to signals
        # Simple strategy: go long on positive predictions, short on negative
        signals = np.tanh(mean_prediction * 2)  # Scale and bound signals
        
        # Adjust signals based on uncertainty (lower confidence = smaller positions)
        confidence = 1.0 / (1.0 + std_prediction)
        signals = signals * confidence
        
        return signals
    
    def execute_rebalance(self,
                         current_positions: np.ndarray,
                         target_positions: np.ndarray,
                         current_prices: np.ndarray,
                         portfolio_value: float) -> Tuple[np.ndarray, List[Dict], float]:
        """Execute portfolio rebalancing"""
        
        trades = []
        transaction_costs = 0.0
        
        # Calculate position changes
        position_changes = target_positions - current_positions
        
        for i, change in enumerate(position_changes):
            if abs(change) > 1e-6:  # Only trade if significant change
                trade_value = change * portfolio_value
                cost = abs(trade_value) * self.config.transaction_cost
                transaction_costs += cost
                
                trades.append({
                    'asset_idx': i,
                    'change': change,
                    'trade_value': trade_value,
                    'cost': cost,
                    'price': current_prices[i]
                })
        
        new_positions = target_positions.copy()
        new_portfolio_value = portfolio_value - transaction_costs
        
        return new_positions, trades, new_portfolio_value
    
    def run_backtest(self,
                    model,
                    data_manager,
                    test_data: pd.DataFrame,
                    metadata: Dict) -> BacktestResults:
        """Run comprehensive backtest"""
        
        print("Starting backtest...")
        
        # Extract data
        features = test_data[metadata['feature_columns']].fillna(0.0).values
        conditions = test_data[metadata['condition_columns']].fillna(0.0).values
        returns_data = test_data[metadata['return_columns']].fillna(0.0).values
        dates = test_data['Date'].values
        
        # Normalize features and conditions
        features_norm = (features - metadata['scalers']['feature_means']) / metadata['scalers']['feature_scales']
        conditions_norm = (conditions - metadata['scalers']['condition_means']) / metadata['scalers']['condition_scales']
        
        # Initialize portfolio
        num_assets = metadata['return_dim']
        portfolio_values = [self.config.initial_capital]
        positions = np.zeros(num_assets)
        all_positions = [positions.copy()]
        all_trades = []
        
        # Get price data for position sizing
        prices = np.exp(np.cumsum(returns_data, axis=0))
        prices = prices / prices[0]  # Normalize to start at 1
        
        lookback_window = metadata.get('sequence_length', 84)
        
        # Main backtest loop
        for t in range(lookback_window, len(features_norm)):
            current_date = dates[t]
            current_prices = prices[t]
            current_returns = returns_data[t]
            
            # Update portfolio value based on previous positions
            if t > lookback_window:
                position_returns = np.dot(positions, current_returns)
                portfolio_values.append(portfolio_values[-1] * (1 + position_returns))
            
            # Generate new signals
            try:
                signals = self.generate_signals(
                    model, test_data, features_norm, conditions_norm,
                    metadata, t, lookback_window
                )
                
                # Apply risk management
                target_positions = self.risk_manager.apply_position_limits(signals)
                
            except Exception as e:
                print(f"Error generating signals at t={t}: {e}")
                target_positions = positions  # Keep current positions
            
            # Execute rebalancing
            if np.any(np.abs(target_positions - positions) > 0.01):  # 1% threshold for rebalancing
                positions, trades, portfolio_value = self.execute_rebalance(
                    positions, target_positions, current_prices, portfolio_values[-1]
                )
                portfolio_values[-1] = portfolio_value
                all_trades.extend([{**trade, 'date': current_date, 'portfolio_value': portfolio_value} 
                                 for trade in trades])
            
            all_positions.append(positions.copy())
        
        # Convert to pandas objects
        portfolio_series = pd.Series(portfolio_values, index=dates[:len(portfolio_values)])
        returns_series = portfolio_series.pct_change().fillna(0.0)
        positions_df = pd.DataFrame(all_positions, 
                                   index=dates[:len(all_positions)], 
                                   columns=[f"pos_{asset}" for asset in metadata['assets']])
        trades_df = pd.DataFrame(all_trades)
        
        # Calculate performance metrics
        results = self._calculate_results(portfolio_series, returns_series, positions_df, trades_df)
        
        # Add model-specific metrics
        self._add_model_metrics(results, model, test_data, metadata)
        
        print(f"Backtest completed. Total return: {results.total_return:.2%}")
        return results
    
    def _calculate_results(self, 
                          portfolio_series: pd.Series,
                          returns_series: pd.Series, 
                          positions_df: pd.DataFrame,
                          trades_df: pd.DataFrame) -> BacktestResults:
        """Calculate comprehensive backtest results"""
        
        results = BacktestResults()
        
        # Basic performance
        results.total_return = (portfolio_series.iloc[-1] / portfolio_series.iloc[0]) - 1.0
        days = len(returns_series)
        results.annualized_return = (1 + results.total_return) ** (252 / days) - 1.0
        results.volatility = returns_series.std() * np.sqrt(252)
        results.sharpe_ratio = self.analyzer.calculate_sharpe_ratio(returns_series)
        
        # Drawdown analysis
        results.max_drawdown, results.drawdowns = self.analyzer.calculate_max_drawdown(portfolio_series)
        results.calmar_ratio = self.analyzer.calculate_calmar_ratio(returns_series, results.max_drawdown)
        
        # Risk metrics
        results.var_95 = self.risk_manager.calculate_var(returns_series.values, 0.95)
        results.var_99 = self.risk_manager.calculate_var(returns_series.values, 0.99)
        results.expected_shortfall_95 = self.risk_manager.calculate_expected_shortfall(returns_series.values, 0.95)
        results.sortino_ratio = self.analyzer.calculate_sortino_ratio(returns_series)
        
        downside_returns = returns_series[returns_series < 0]
        results.downside_deviation = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0.0
        
        # Trade analysis
        trade_stats = self.analyzer.analyze_trades(trades_df)
        results.num_trades = trade_stats.get('num_trades', 0)
        results.win_rate = trade_stats.get('win_rate', 0.0)
        results.avg_win = trade_stats.get('avg_win', 0.0)
        results.avg_loss = trade_stats.get('avg_loss', 0.0)
        results.profit_factor = trade_stats.get('profit_factor', 0.0)
        
        # Store time series
        results.portfolio_value = portfolio_series
        results.returns = returns_series
        results.positions = positions_df
        results.trades = trades_df
        
        return results
    
    def _add_model_metrics(self, results: BacktestResults, model, test_data: pd.DataFrame, metadata: Dict):
        """Add model-specific performance metrics"""
        try:
            # This would require generating predictions and comparing to actual returns
            # Simplified version here
            results.prediction_accuracy = 0.0  # Placeholder
            results.directional_accuracy = 0.0  # Placeholder
            results.correlation_with_actual = 0.0  # Placeholder
        except Exception as e:
            print(f"Could not calculate model metrics: {e}")

class WalkForwardBacktester(DiffusionBacktester):
    """Walk-forward backtesting with model retraining"""
    
    def __init__(self, config: BacktestConfig):
        super().__init__(config)
        self.retrain_dates = []
        self.model_performances = []
    
    def run_walk_forward_backtest(self,
                                 model_class,
                                 model_config,
                                 data_manager,
                                 full_data: pd.DataFrame,
                                 metadata: Dict) -> BacktestResults:
        """Run walk-forward backtest with periodic retraining"""
        
        print("Starting walk-forward backtest...")
        
        # Implementation would be more complex, involving:
        # 1. Split data into training/testing windows
        # 2. Train model on each window
        # 3. Test on out-of-sample period
        # 4. Roll forward and repeat
        
        # For now, use the simpler backtest
        return self.run_backtest(model_class, data_manager, full_data, metadata)

def compare_strategies(results_dict: Dict[str, BacktestResults]) -> pd.DataFrame:
    """Compare multiple backtest results"""
    
    metrics = ['total_return', 'annualized_return', 'volatility', 'sharpe_ratio', 
               'max_drawdown', 'calmar_ratio', 'sortino_ratio', 'var_95']
    
    comparison = []
    for strategy_name, results in results_dict.items():
        row = {'strategy': strategy_name}
        for metric in metrics:
            row[metric] = getattr(results, metric, 0.0)
        comparison.append(row)
    
    return pd.DataFrame(comparison)

def create_benchmark_results(benchmark_returns: np.ndarray, 
                           initial_capital: float = 100000.0) -> BacktestResults:
    """Create BacktestResults for a benchmark strategy"""
    
    portfolio_value = initial_capital * np.cumprod(1 + benchmark_returns)
    portfolio_series = pd.Series(portfolio_value)
    returns_series = pd.Series(benchmark_returns)
    
    analyzer = PerformanceAnalyzer()
    results = BacktestResults()
    
    results.total_return = portfolio_value[-1] / initial_capital - 1.0
    results.annualized_return = (1 + results.total_return) ** (252 / len(benchmark_returns)) - 1.0
    results.volatility = returns_series.std() * np.sqrt(252)
    results.sharpe_ratio = analyzer.calculate_sharpe_ratio(returns_series)
    results.max_drawdown, results.drawdowns = analyzer.calculate_max_drawdown(portfolio_series)
    results.portfolio_value = portfolio_series
    results.returns = returns_series
    
    return results