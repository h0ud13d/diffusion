# advanced_backtesting.py - Sophisticated backtesting framework for advanced diffusion models
import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

warnings.filterwarnings('ignore')

@dataclass
class AdvancedBacktestConfig:
    """Configuration for advanced backtesting"""
    # Capital and sizing
    initial_capital: float = 1000000.0
    position_sizing_method: str = "kelly"  # fixed, kelly, risk_parity, mean_variance
    max_position_size: float = 0.25
    max_portfolio_leverage: float = 1.0
    
    # Transaction costs
    fixed_cost_per_trade: float = 1.0
    variable_cost_rate: float = 0.0005  # 5 bps
    market_impact_model: str = "linear"  # none, linear, square_root
    market_impact_coeff: float = 0.001
    
    # Risk management
    portfolio_var_limit: float = 0.02  # 2% daily VaR limit
    position_var_limit: float = 0.01   # 1% daily position VaR limit
    max_drawdown_limit: float = 0.15   # 15% max drawdown before stopping
    stop_loss_pct: Optional[float] = None
    take_profit_pct: Optional[float] = None
    
    # Rebalancing
    rebalance_frequency: str = "daily"  # daily, weekly, monthly, signal_based
    rebalance_threshold: float = 0.05   # 5% drift threshold for signal-based
    
    # Advanced features
    use_regime_detection: bool = True
    regime_lookback: int = 252
    use_dynamic_hedging: bool = True
    hedge_instruments: List[str] = field(default_factory=lambda: ["SPY", "VIX"])
    
    # Walk-forward analysis
    walk_forward_enabled: bool = True
    training_window: int = 504  # 2 years
    retrain_frequency: int = 63  # 3 months
    validation_split: float = 0.2
    
    # Monte Carlo simulation
    monte_carlo_runs: int = 1000
    bootstrap_block_length: int = 20
    
    # Performance attribution
    factor_attribution: bool = True
    benchmark_symbols: List[str] = field(default_factory=lambda: ["SPY", "QQQ", "IWM"])

@dataclass
class BacktestResults:
    """Comprehensive backtest results"""
    # Performance metrics
    total_return: float = 0.0
    annualized_return: float = 0.0
    volatility: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    max_drawdown: float = 0.0
    
    # Risk metrics
    var_95: float = 0.0
    var_99: float = 0.0
    expected_shortfall_95: float = 0.0
    expected_shortfall_99: float = 0.0
    skewness: float = 0.0
    kurtosis: float = 0.0
    
    # Trade statistics
    num_trades: int = 0
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 0.0
    expectancy: float = 0.0
    
    # Advanced metrics
    information_ratio: float = 0.0
    tracking_error: float = 0.0
    upside_capture: float = 0.0
    downside_capture: float = 0.0
    beta: float = 0.0
    alpha: float = 0.0
    
    # Time series
    portfolio_values: pd.Series = field(default_factory=pd.Series)
    returns: pd.Series = field(default_factory=pd.Series)
    positions: pd.DataFrame = field(default_factory=pd.DataFrame)
    trades: pd.DataFrame = field(default_factory=pd.DataFrame)
    drawdowns: pd.Series = field(default_factory=pd.Series)
    
    # Attribution results
    factor_attribution: Dict[str, float] = field(default_factory=dict)
    sector_attribution: Dict[str, float] = field(default_factory=dict)

class AdvancedRiskManager:
    """Sophisticated risk management system"""
    
    def __init__(self, config: AdvancedBacktestConfig):
        self.config = config
        self.position_history = []
        self.return_history = []
        
    def calculate_position_sizes(self, 
                               expected_returns: np.ndarray,
                               covariance_matrix: np.ndarray,
                               current_prices: np.ndarray,
                               portfolio_value: float) -> np.ndarray:
        """Calculate optimal position sizes based on the chosen method"""
        
        if self.config.position_sizing_method == "fixed":
            return self._fixed_position_sizing(expected_returns, portfolio_value)
        elif self.config.position_sizing_method == "kelly":
            return self._kelly_position_sizing(expected_returns, covariance_matrix, portfolio_value)
        elif self.config.position_sizing_method == "risk_parity":
            return self._risk_parity_sizing(covariance_matrix, portfolio_value)
        elif self.config.position_sizing_method == "mean_variance":
            return self._mean_variance_sizing(expected_returns, covariance_matrix, portfolio_value)
        else:
            raise ValueError(f"Unknown position sizing method: {self.config.position_sizing_method}")
    
    def _fixed_position_sizing(self, expected_returns: np.ndarray, portfolio_value: float) -> np.ndarray:
        """Fixed position sizing"""
        num_assets = len(expected_returns)
        max_position_value = self.config.max_position_size * portfolio_value
        
        positions = np.zeros(num_assets)
        positions[expected_returns > 0] = max_position_value
        positions[expected_returns < 0] = -max_position_value
        
        return positions
    
    def _kelly_position_sizing(self, 
                              expected_returns: np.ndarray,
                              covariance_matrix: np.ndarray,
                              portfolio_value: float) -> np.ndarray:
        """Kelly Criterion position sizing"""
        try:
            # Kelly formula: f = (μ - r) / σ² where r is risk-free rate (assumed 0)
            inv_cov = np.linalg.pinv(covariance_matrix)
            kelly_weights = inv_cov @ expected_returns
            
            # Apply leverage constraint
            total_leverage = np.sum(np.abs(kelly_weights))
            if total_leverage > self.config.max_portfolio_leverage:
                kelly_weights *= self.config.max_portfolio_leverage / total_leverage
            
            # Apply individual position limits
            kelly_weights = np.clip(kelly_weights, 
                                  -self.config.max_position_size, 
                                  self.config.max_position_size)
            
            return kelly_weights * portfolio_value
            
        except np.linalg.LinAlgError:
            # Fallback to equal weight if covariance matrix is singular
            return self._fixed_position_sizing(expected_returns, portfolio_value)
    
    def _risk_parity_sizing(self, covariance_matrix: np.ndarray, portfolio_value: float) -> np.ndarray:
        """Risk parity position sizing"""
        # Simplified risk parity - equal risk contribution
        volatilities = np.sqrt(np.diag(covariance_matrix))
        inv_vol_weights = 1 / volatilities
        weights = inv_vol_weights / np.sum(inv_vol_weights)
        
        # Apply position limits
        weights = np.clip(weights, 0, self.config.max_position_size)
        weights = weights / np.sum(weights)  # Renormalize
        
        return weights * portfolio_value
    
    def _mean_variance_sizing(self,
                            expected_returns: np.ndarray,
                            covariance_matrix: np.ndarray,
                            portfolio_value: float) -> np.ndarray:
        """Mean-variance optimization"""
        try:
            # Markowitz optimization
            inv_cov = np.linalg.pinv(covariance_matrix)
            ones = np.ones(len(expected_returns))
            
            # Optimal weights
            A = ones.T @ inv_cov @ ones
            B = ones.T @ inv_cov @ expected_returns
            C = expected_returns.T @ inv_cov @ expected_returns
            
            weights = inv_cov @ expected_returns / A
            
            # Apply constraints
            total_leverage = np.sum(np.abs(weights))
            if total_leverage > self.config.max_portfolio_leverage:
                weights *= self.config.max_portfolio_leverage / total_leverage
                
            weights = np.clip(weights, 
                            -self.config.max_position_size,
                            self.config.max_position_size)
            
            return weights * portfolio_value
            
        except (np.linalg.LinAlgError, ZeroDivisionError):
            return self._kelly_position_sizing(expected_returns, covariance_matrix, portfolio_value)
    
    def calculate_portfolio_risk(self, positions: np.ndarray, 
                               covariance_matrix: np.ndarray) -> Dict[str, float]:
        """Calculate portfolio risk metrics"""
        weights = positions / np.sum(np.abs(positions)) if np.sum(np.abs(positions)) > 0 else positions
        
        # Portfolio variance
        portfolio_var = weights.T @ covariance_matrix @ weights
        portfolio_vol = np.sqrt(portfolio_var)
        
        # VaR estimates (assuming normal distribution)
        var_95 = 1.645 * portfolio_vol  # 95% VaR
        var_99 = 2.326 * portfolio_vol  # 99% VaR
        
        return {
            'portfolio_volatility': portfolio_vol,
            'portfolio_var_95': var_95,
            'portfolio_var_99': var_99
        }
    
    def apply_risk_limits(self, positions: np.ndarray,
                         covariance_matrix: np.ndarray,
                         portfolio_value: float) -> np.ndarray:
        """Apply risk management limits"""
        # Calculate current risk
        risk_metrics = self.calculate_portfolio_risk(positions, covariance_matrix)
        
        # Scale down positions if they exceed risk limits
        if risk_metrics['portfolio_var_95'] > self.config.portfolio_var_limit:
            scale_factor = self.config.portfolio_var_limit / risk_metrics['portfolio_var_95']
            positions = positions * scale_factor
        
        return positions

class AdvancedTransactionCostModel:
    """Advanced transaction cost modeling"""
    
    def __init__(self, config: AdvancedBacktestConfig):
        self.config = config
    
    def calculate_transaction_costs(self,
                                  trade_values: np.ndarray,
                                  volumes: np.ndarray,
                                  prices: np.ndarray) -> np.ndarray:
        """Calculate comprehensive transaction costs"""
        abs_trade_values = np.abs(trade_values)
        
        # Fixed costs
        fixed_costs = np.where(abs_trade_values > 0, self.config.fixed_cost_per_trade, 0)
        
        # Variable costs (spread and fees)
        variable_costs = abs_trade_values * self.config.variable_cost_rate
        
        # Market impact
        if self.config.market_impact_model == "linear":
            market_impact = abs_trade_values * self.config.market_impact_coeff
        elif self.config.market_impact_model == "square_root":
            market_impact = np.sqrt(abs_trade_values) * self.config.market_impact_coeff
        else:
            market_impact = np.zeros_like(abs_trade_values)
        
        total_costs = fixed_costs + variable_costs + market_impact
        return total_costs

class WalkForwardAnalyzer:
    """Walk-forward analysis implementation"""
    
    def __init__(self, config: AdvancedBacktestConfig):
        self.config = config
    
    def create_walk_forward_windows(self, 
                                  total_length: int) -> List[Tuple[int, int, int, int]]:
        """Create overlapping windows for walk-forward analysis"""
        windows = []
        
        current_start = 0
        while current_start + self.config.training_window + 63 < total_length:  # Need at least 63 days for testing
            train_end = current_start + self.config.training_window
            val_split = int(self.config.training_window * (1 - self.config.validation_split))
            val_start = current_start + val_split
            
            test_start = train_end
            test_end = min(test_start + self.config.retrain_frequency, total_length)
            
            windows.append((current_start, val_start, train_end, test_end))
            current_start += self.config.retrain_frequency
        
        return windows

class AdvancedBacktester:
    """Main advanced backtesting engine"""
    
    def __init__(self, config: AdvancedBacktestConfig):
        self.config = config
        self.risk_manager = AdvancedRiskManager(config)
        self.cost_model = AdvancedTransactionCostModel(config)
        self.walk_forward = WalkForwardAnalyzer(config)
    
    def run_backtest(self,
                    model,
                    data_manager,
                    test_data: pd.DataFrame,
                    metadata: Dict) -> BacktestResults:
        """Run comprehensive backtest"""
        
        print("Running advanced backtest...")
        
        # Prepare data
        features = test_data[metadata['feature_columns']].fillna(0.0).values
        returns_data = test_data[metadata['return_columns']].fillna(0.0).values
        dates = pd.to_datetime(test_data['Date'])
        
        # Normalize features
        if 'scalers' in metadata:
            features = (features - metadata['scalers']['feature_means']) / metadata['scalers']['feature_scales']
        
        # Initialize portfolio
        portfolio_values = [self.config.initial_capital]
        positions = np.zeros(len(metadata['return_columns']))
        all_positions = [positions.copy()]
        all_trades = []
        
        sequence_length = metadata.get('sequence_length', 64)
        prediction_horizon = 5  # Predict 5 days ahead
        
        print(f"Sequence length: {sequence_length}")
        print(f"Data shape: {features.shape}")
        print(f"Returns shape: {returns_data.shape}")
        
        # Main backtest loop
        for t in range(sequence_length, len(features) - prediction_horizon):
            current_date = dates.iloc[t]
            current_portfolio_value = portfolio_values[-1]
            
            # Update portfolio value based on returns
            if t > sequence_length:
                period_returns = returns_data[t-1]
                position_returns = np.dot(positions, period_returns)
                new_portfolio_value = current_portfolio_value * (1 + position_returns)
                portfolio_values.append(new_portfolio_value)
                current_portfolio_value = new_portfolio_value
            
            # Generate predictions using the model
            try:
                predictions = self._generate_model_predictions(
                    model, features, t, sequence_length, metadata
                )
                
                if predictions is None:
                    # Keep current positions if prediction fails
                    all_positions.append(positions.copy())
                    continue
                
                # Estimate covariance matrix for recent returns
                lookback_window = min(60, t - sequence_length)
                if lookback_window > 10:
                    recent_returns = returns_data[t-lookback_window:t]
                    cov_matrix = np.cov(recent_returns.T)
                else:
                    # Use identity matrix if not enough data
                    cov_matrix = np.eye(len(metadata['return_columns'])) * 0.01
                
                # Calculate new position sizes
                target_positions = self.risk_manager.calculate_position_sizes(
                    predictions, cov_matrix, 
                    np.ones(len(predictions)),  # Dummy prices
                    current_portfolio_value
                )
                
                # Apply risk limits
                target_positions = self.risk_manager.apply_risk_limits(
                    target_positions, cov_matrix, current_portfolio_value
                )
                
                # Calculate trades needed
                position_changes = target_positions - positions
                
                # Calculate transaction costs
                trade_costs = self.cost_model.calculate_transaction_costs(
                    position_changes,
                    np.ones_like(position_changes) * 1000000,  # Dummy volume
                    np.ones_like(position_changes)  # Dummy prices
                )
                
                total_cost = np.sum(trade_costs)
                
                # Execute trades if profitable after costs
                if np.sum(predictions * position_changes) > total_cost:
                    positions = target_positions
                    current_portfolio_value -= total_cost
                    portfolio_values[-1] = current_portfolio_value
                    
                    # Record trades
                    for i, change in enumerate(position_changes):
                        if abs(change) > 1e-6:
                            all_trades.append({
                                'date': current_date,
                                'asset_idx': i,
                                'position_change': change,
                                'cost': trade_costs[i]
                            })
                
            except Exception as e:
                print(f"Error at step {t}: {e}")
                # Keep current positions on error
                
            all_positions.append(positions.copy())
        
        # Convert to pandas structures
        portfolio_series = pd.Series(portfolio_values, index=dates[:len(portfolio_values)])
        returns_series = portfolio_series.pct_change().fillna(0.0)
        
        positions_df = pd.DataFrame(
            all_positions[:len(dates)], 
            index=dates[:len(all_positions)],
            columns=[f'pos_{asset}' for asset in metadata['assets']]
        )
        
        trades_df = pd.DataFrame(all_trades)
        
        # Calculate comprehensive results
        results = self._calculate_advanced_results(
            portfolio_series, returns_series, positions_df, trades_df, returns_data[:len(portfolio_values)]
        )
        
        print(f"Backtest completed. Total return: {results.total_return:.2%}")
        return results
    
    def _generate_model_predictions(self,
                                  model,
                                  features: np.ndarray,
                                  current_idx: int,
                                  sequence_length: int,
                                  metadata: Dict) -> Optional[np.ndarray]:
        """Generate predictions using the diffusion model"""
        
        try:
            device = next(model.parameters()).device
            
            # Prepare input sequence
            start_idx = current_idx - sequence_length
            input_seq = features[start_idx:current_idx]  # (seq_len, features)
            
            # Convert to tensor and add batch dimension
            x = torch.from_numpy(input_seq.T).float().unsqueeze(0).to(device)  # (1, features, seq_len)
            
            # Generate sample
            with torch.no_grad():
                model.eval()
                
                # Simple unconditional generation
                shape = (1, len(metadata['return_columns']), 1)
                sample = model.sample_ddim(shape, context=None, num_steps=20, progress=False)
                
                prediction = sample.cpu().numpy()[0, :, 0]  # (return_dim,)
                
                # Apply some post-processing
                prediction = np.tanh(prediction * 2)  # Scale and bound predictions
                
                return prediction
                
        except Exception as e:
            print(f"Model prediction error: {e}")
            return None
    
    def _calculate_advanced_results(self,
                                  portfolio_values: pd.Series,
                                  returns: pd.Series,
                                  positions: pd.DataFrame,
                                  trades: pd.DataFrame,
                                  benchmark_returns: np.ndarray) -> BacktestResults:
        """Calculate comprehensive performance metrics"""
        
        results = BacktestResults()
        
        # Basic performance metrics
        results.total_return = (portfolio_values.iloc[-1] / portfolio_values.iloc[0]) - 1
        
        days_elapsed = len(returns)
        results.annualized_return = (1 + results.total_return) ** (252 / days_elapsed) - 1
        results.volatility = returns.std() * np.sqrt(252)
        
        # Sharpe ratio
        if results.volatility > 0:
            results.sharpe_ratio = results.annualized_return / results.volatility
        
        # Drawdown analysis
        cumulative = portfolio_values / portfolio_values.iloc[0]
        peak = cumulative.expanding().max()
        drawdown = (cumulative / peak - 1)
        results.max_drawdown = drawdown.min()
        results.drawdowns = drawdown
        
        # Sortino ratio
        downside_returns = returns[returns < 0]
        if len(downside_returns) > 0:
            downside_deviation = downside_returns.std() * np.sqrt(252)
            if downside_deviation > 0:
                results.sortino_ratio = results.annualized_return / downside_deviation
        
        # Calmar ratio
        if results.max_drawdown < 0:
            results.calmar_ratio = results.annualized_return / abs(results.max_drawdown)
        
        # Risk metrics
        results.var_95 = np.percentile(returns, 5)
        results.var_99 = np.percentile(returns, 1)
        
        var_95_returns = returns[returns <= results.var_95]
        var_99_returns = returns[returns <= results.var_99]
        
        if len(var_95_returns) > 0:
            results.expected_shortfall_95 = var_95_returns.mean()
        if len(var_99_returns) > 0:
            results.expected_shortfall_99 = var_99_returns.mean()
        
        results.skewness = returns.skew()
        results.kurtosis = returns.kurt()
        
        # Trade analysis
        if not trades.empty:
            results.num_trades = len(trades)
            
            # Simplified trade analysis
            if 'position_change' in trades.columns:
                profitable_trades = trades[trades['position_change'] > 0]
                losing_trades = trades[trades['position_change'] < 0]
                
                if len(trades) > 0:
                    results.win_rate = len(profitable_trades) / len(trades)
                if len(profitable_trades) > 0:
                    results.avg_win = profitable_trades['position_change'].mean()
                if len(losing_trades) > 0:
                    results.avg_loss = losing_trades['position_change'].mean()
        
        # Store time series
        results.portfolio_values = portfolio_values
        results.returns = returns
        results.positions = positions
        results.trades = trades
        
        # Benchmark comparison (if available)
        if benchmark_returns.shape[0] >= len(returns):
            benchmark_ret = pd.Series(benchmark_returns[:len(returns)].mean(axis=1))
            
            # Beta and Alpha
            covariance = np.cov(returns, benchmark_ret)[0, 1]
            benchmark_var = benchmark_ret.var()
            
            if benchmark_var > 0:
                results.beta = covariance / benchmark_var
                results.alpha = results.annualized_return - results.beta * benchmark_ret.mean() * 252
            
            # Information ratio
            active_returns = returns - benchmark_ret
            results.tracking_error = active_returns.std() * np.sqrt(252)
            if results.tracking_error > 0:
                results.information_ratio = active_returns.mean() * 252 / results.tracking_error
        
        return results
    
    def monte_carlo_analysis(self, base_results: BacktestResults) -> Dict[str, Any]:
        """Run Monte Carlo simulation on backtest results"""
        
        returns = base_results.returns.values
        n_periods = len(returns)
        
        # Bootstrap simulation
        simulated_returns = []
        
        for _ in range(self.config.monte_carlo_runs):
            # Block bootstrap to preserve serial correlation
            sim_returns = []
            while len(sim_returns) < n_periods:
                block_start = np.random.randint(0, len(returns) - self.config.bootstrap_block_length)
                block = returns[block_start:block_start + self.config.bootstrap_block_length]
                sim_returns.extend(block)
            
            sim_returns = sim_returns[:n_periods]
            simulated_returns.append(sim_returns)
        
        # Calculate statistics
        sim_cumulative_returns = []
        sim_sharpe_ratios = []
        sim_max_drawdowns = []
        
        for sim_ret in simulated_returns:
            sim_ret_series = pd.Series(sim_ret)
            cum_ret = (1 + sim_ret_series).prod() - 1
            sim_cumulative_returns.append(cum_ret)
            
            if sim_ret_series.std() > 0:
                sharpe = sim_ret_series.mean() / sim_ret_series.std() * np.sqrt(252)
                sim_sharpe_ratios.append(sharpe)
            
            cum_series = (1 + sim_ret_series).cumprod()
            peak = cum_series.expanding().max()
            dd = (cum_series / peak - 1).min()
            sim_max_drawdowns.append(dd)
        
        return {
            'simulated_total_returns': {
                'mean': np.mean(sim_cumulative_returns),
                'std': np.std(sim_cumulative_returns),
                'percentiles': np.percentile(sim_cumulative_returns, [5, 25, 50, 75, 95])
            },
            'simulated_sharpe_ratios': {
                'mean': np.mean(sim_sharpe_ratios),
                'std': np.std(sim_sharpe_ratios),
                'percentiles': np.percentile(sim_sharpe_ratios, [5, 25, 50, 75, 95])
            },
            'simulated_max_drawdowns': {
                'mean': np.mean(sim_max_drawdowns),
                'std': np.std(sim_max_drawdowns),
                'percentiles': np.percentile(sim_max_drawdowns, [5, 25, 50, 75, 95])
            }
        }