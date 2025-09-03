import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Tuple, Optional, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import warnings
from scipy import stats
from scipy.optimize import minimize_scalar
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
warnings.filterwarnings('ignore')


@dataclass
class PortfolioConfig:
    initial_capital: float = 1_000_000.0
    
    # Risk management
    max_drawdown_limit: float = 0.15  # 15% max drawdown
    var_confidence: float = 0.05  # 5% VaR
    max_leverage: float = 1.0
    max_position_size: float = 0.25  # 25% max per asset
    
    # Transaction costs
    transaction_cost_bps: float = 10.0  # 10 bps
    market_impact_model: str = "square_root"  # linear, square_root, power_law
    market_impact_coeff: float = 0.01
    
    # Kelly sizing
    use_kelly_sizing: bool = True
    kelly_lookback_periods: int = 252  # 1 year
    kelly_max_fraction: float = 0.25  # Max 25% Kelly fraction
    
    # Walk-forward settings
    training_window: int = 504  # 2 years
    rebalance_frequency: str = "weekly"  # daily, weekly, monthly
    retraining_frequency: int = 63  # 3 months
    
    # Monte Carlo stress testing
    mc_simulations: int = 10000
    stress_test_scenarios: int = 1000
    confidence_intervals: List[float] = field(default_factory=lambda: [0.05, 0.25, 0.75, 0.95])


@dataclass
class RiskMetrics:
    var_1d: float = 0.0
    var_5d: float = 0.0
    expected_shortfall: float = 0.0
    max_drawdown: float = 0.0
    drawdown_duration: int = 0
    volatility: float = 0.0
    downside_deviation: float = 0.0
    beta: float = 0.0
    tracking_error: float = 0.0


@dataclass
class PerformanceMetrics:
    total_return: float = 0.0
    annualized_return: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    information_ratio: float = 0.0
    alpha: float = 0.0
    win_rate: float = 0.0


class KellySizer:
    
    def __init__(self, lookback_periods: int = 252, max_fraction: float = 0.25):
        self.lookback_periods = lookback_periods
        self.max_fraction = max_fraction
        
    def calculate_kelly_fractions(self, returns: pd.DataFrame, predictions: pd.DataFrame) -> pd.Series:
        kelly_fractions = {}
        
        for asset in returns.columns:
            if asset in predictions.columns:
                # Get recent returns for covariance estimation
                recent_returns = returns[asset].tail(self.lookback_periods)
                
                # Calculate expected return from predictions
                expected_return = predictions[asset].mean()
                
                # Calculate variance
                variance = recent_returns.var()
                
                if variance > 0 and not np.isnan(expected_return):
                    # Kelly fraction = (expected_return) / variance
                    kelly_frac = expected_return / variance
                    
                    # Apply maximum fraction constraint
                    kelly_frac = np.clip(kelly_frac, -self.max_fraction, self.max_fraction)
                else:
                    kelly_frac = 0.0
                    
                kelly_fractions[asset] = kelly_frac
            else:
                kelly_fractions[asset] = 0.0
                
        return pd.Series(kelly_fractions)


class RiskManager:
    def __init__(self, config: PortfolioConfig):
        self.config = config
        
    def calculate_var(self, returns: np.ndarray, confidence: float = 0.05) -> float:
        if len(returns) == 0:
            return 0.0
        return np.percentile(returns, confidence * 100)
    
    def calculate_expected_shortfall(self, returns: np.ndarray, confidence: float = 0.05) -> float:
        if len(returns) == 0:
            return 0.0
        var = self.calculate_var(returns, confidence)
        return returns[returns <= var].mean()
    
    def calculate_drawdown(self, equity_curve: np.ndarray) -> Tuple[float, int]:
        if len(equity_curve) == 0:
            return 0.0, 0
            
        peak = np.maximum.accumulate(equity_curve)
        drawdown = (equity_curve - peak) / peak
        
        max_drawdown = drawdown.min()
        
        drawdown_duration = 0
        current_duration = 0
        for dd in drawdown:
            if dd < 0:
                current_duration += 1
                drawdown_duration = max(drawdown_duration, current_duration)
            else:
                current_duration = 0
                
        return abs(max_drawdown), drawdown_duration
    
    def check_risk_limits(self, portfolio_value: float, max_portfolio_value: float, 
                         recent_returns: np.ndarray) -> Dict[str, bool]:
        checks = {}
        
        current_drawdown = (portfolio_value - max_portfolio_value) / max_portfolio_value
        checks['drawdown_ok'] = current_drawdown > -self.config.max_drawdown_limit
        
        if len(recent_returns) >= 10:
            var_1d = self.calculate_var(recent_returns, self.config.var_confidence)
            expected_loss = abs(var_1d) * portfolio_value
            checks['var_ok'] = expected_loss < portfolio_value * 0.1  # Max 10% daily VaR
        else:
            checks['var_ok'] = True
            
        return checks


class TransactionCostModel:
    def __init__(self, config: PortfolioConfig):
        self.config = config
        
    def calculate_transaction_costs(self, trade_value: float, volume_profile: Optional[float] = None) -> float:
        
        # Base transaction cost (bps)
        base_cost = abs(trade_value) * (self.config.transaction_cost_bps / 10000)
        
        # Market impact
        market_impact = self._calculate_market_impact(trade_value, volume_profile)
        
        return base_cost + market_impact
    
    def _calculate_market_impact(self, trade_value: float, volume_profile: Optional[float] = None) -> float:
        if volume_profile is None:
            volume_profile = 1.0  # Default volume profile
            
        trade_size_ratio = abs(trade_value) / (volume_profile * 1e6)  # Normalize by volume
        
        if self.config.market_impact_model == "linear":
            impact = self.config.market_impact_coeff * trade_size_ratio
        elif self.config.market_impact_model == "square_root":
            impact = self.config.market_impact_coeff * np.sqrt(trade_size_ratio)
        elif self.config.market_impact_model == "power_law":
            impact = self.config.market_impact_coeff * (trade_size_ratio ** 0.6)
        else:
            impact = 0.0
            
        return abs(trade_value) * impact


class MonteCarloStressTester:
    def __init__(self, config: PortfolioConfig):
        self.config = config
        
    def run_stress_tests(self, returns: pd.DataFrame, weights: pd.DataFrame,
                        scenarios: Optional[Dict[str, np.ndarray]] = None) -> Dict[str, float]:
        results = {}
        
        # Historical simulation
        results.update(self._historical_simulation(returns, weights))
        
        # Parametric simulation
        results.update(self._parametric_simulation(returns, weights))
        
        # Custom scenario tests
        if scenarios:
            results.update(self._scenario_tests(scenarios, weights))
            
        return results
    
    def _historical_simulation(self, returns: pd.DataFrame, weights: pd.DataFrame) -> Dict[str, float]:
        portfolio_returns = []
        
        for _ in range(self.config.mc_simulations):
            # Bootstrap historical returns
            sampled_returns = returns.sample(n=len(weights), replace=True)
            sampled_weights = weights.sample(n=len(sampled_returns), replace=True)
            
            # Calculate portfolio return
            portfolio_return = (sampled_returns.values * sampled_weights.values).sum(axis=1).mean()
            portfolio_returns.append(portfolio_return)
            
        portfolio_returns = np.array(portfolio_returns)
        
        return {
            'hist_sim_var_95': np.percentile(portfolio_returns, 5),
            'hist_sim_var_99': np.percentile(portfolio_returns, 1),
            'hist_sim_expected_shortfall': portfolio_returns[portfolio_returns <= np.percentile(portfolio_returns, 5)].mean(),
            'hist_sim_worst_case': portfolio_returns.min()
        }
    
    def _parametric_simulation(self, returns: pd.DataFrame, weights: pd.DataFrame) -> Dict[str, float]:
        # Estimate covariance matrix
        cov_matrix = returns.cov().values
        mean_returns = returns.mean().values
        
        portfolio_returns = []
        
        for _ in range(self.config.mc_simulations):
            # Generate random returns
            random_returns = np.random.multivariate_normal(mean_returns, cov_matrix)
            
            # Sample random weights
            weight_idx = np.random.randint(0, len(weights))
            portfolio_weights = weights.iloc[weight_idx].values
            
            # Calculate portfolio return
            portfolio_return = np.dot(random_returns, portfolio_weights)
            portfolio_returns.append(portfolio_return)
            
        portfolio_returns = np.array(portfolio_returns)
        
        return {
            'param_sim_var_95': np.percentile(portfolio_returns, 5),
            'param_sim_var_99': np.percentile(portfolio_returns, 1),
            'param_sim_expected_shortfall': portfolio_returns[portfolio_returns <= np.percentile(portfolio_returns, 5)].mean()
        }
    
    def _scenario_tests(self, scenarios: Dict[str, np.ndarray], weights: pd.DataFrame) -> Dict[str, float]:
        results = {}
        
        for scenario_name, scenario_returns in scenarios.items():
            portfolio_returns = []
            
            for weight_row in weights.iterrows():
                portfolio_weights = weight_row[1].values
                portfolio_return = np.dot(scenario_returns, portfolio_weights)
                portfolio_returns.append(portfolio_return)
                
            results[f'{scenario_name}_mean'] = np.mean(portfolio_returns)
            results[f'{scenario_name}_worst'] = np.min(portfolio_returns)
            
        return results


class WalkForwardBacktester:
    def __init__(self, config: PortfolioConfig):
        self.config = config
        self.kelly_sizer = KellySizer(config.kelly_lookback_periods, config.kelly_max_fraction)
        self.risk_manager = RiskManager(config)
        self.cost_model = TransactionCostModel(config)
        self.stress_tester = MonteCarloStressTester(config)
        
    def backtest(self, returns: pd.DataFrame, predictions: pd.DataFrame,
                 model_retrain_func: Optional[Callable] = None) -> Dict:
        """
        Run comprehensive walk-forward backtest
        
        Args:
            returns: Historical returns DataFrame
            predictions: Model predictions DataFrame  
            model_retrain_func: Function to retrain model
            
        Returns:
            Dictionary with backtest results
        """
        
        # Initialize tracking variables
        portfolio_value = [self.config.initial_capital]
        portfolio_weights = []
        transaction_costs = []
        risk_metrics = []
        rebalance_dates = []
        
        # Get rebalancing dates
        rebalance_freq = self._get_rebalance_frequency()
        dates = returns.index
        
        start_idx = self.config.training_window
        end_idx = len(returns)
        
        max_portfolio_value = self.config.initial_capital
        
        for i in range(start_idx, end_idx, rebalance_freq):
            current_date = dates[i]
            rebalance_dates.append(current_date)
            
            # Get training data
            train_returns = returns.iloc[i-self.config.training_window:i]
            train_predictions = predictions.iloc[i-self.config.training_window:i]
            
            # Retrain model if needed
            if (i - start_idx) % self.config.retraining_frequency == 0 and model_retrain_func:
                model_retrain_func(train_returns, train_predictions)
            
            # Calculate Kelly fractions if enabled
            if self.config.use_kelly_sizing:
                kelly_fractions = self.kelly_sizer.calculate_kelly_fractions(
                    train_returns, train_predictions
                )
            else:
                # Equal weight fallback
                kelly_fractions = pd.Series(
                    1.0 / len(returns.columns), 
                    index=returns.columns
                )
            
            # Apply position size limits
            kelly_fractions = kelly_fractions.clip(-self.config.max_position_size, 
                                                 self.config.max_position_size)
            
            # Normalize weights
            total_weight = abs(kelly_fractions).sum()
            if total_weight > self.config.max_leverage:
                kelly_fractions = kelly_fractions * (self.config.max_leverage / total_weight)
            
            portfolio_weights.append(kelly_fractions)
            
            # Calculate period returns and portfolio value
            if i < len(returns) - 1:
                period_returns = returns.iloc[i+1:min(i+rebalance_freq+1, len(returns))]
                
                for j, period_return in period_returns.iterrows():
                    portfolio_return = (kelly_fractions * period_return).sum()
                    
                    # Calculate transaction costs (only on rebalancing)
                    if j == period_returns.index[0]:  # First day of period
                        trade_values = kelly_fractions * portfolio_value[-1]
                        period_costs = sum(self.cost_model.calculate_transaction_costs(tv) 
                                         for tv in trade_values)
                        transaction_costs.append(period_costs)
                    else:
                        transaction_costs.append(0.0)
                    
                    # Update portfolio value
                    new_value = portfolio_value[-1] * (1 + portfolio_return) - transaction_costs[-1]
                    portfolio_value.append(new_value)
                    
                    # Update max portfolio value for drawdown calculation
                    max_portfolio_value = max(max_portfolio_value, new_value)
                    
                    # Check risk limits
                    recent_returns = np.array([pv / portfolio_value[max(0, len(portfolio_value)-22)] - 1 
                                             for pv in portfolio_value[-21:]])  # 21-day returns
                    
                    risk_checks = self.risk_manager.check_risk_limits(
                        new_value, max_portfolio_value, recent_returns
                    )
                    
                    # Risk limit breach handling
                    if not all(risk_checks.values()):
                        print(f"Risk limit breached on {j}: {risk_checks}")
                        # Reduce positions by 50% if limits breached
                        kelly_fractions *= 0.5
        
        # Calculate final metrics
        portfolio_value = np.array(portfolio_value)
        portfolio_returns = np.diff(portfolio_value) / portfolio_value[:-1]
        
        # Performance metrics
        performance = self._calculate_performance_metrics(portfolio_returns)
        
        # Risk metrics  
        risk = self._calculate_risk_metrics(portfolio_returns, portfolio_value)
        
        # Stress test results
        if len(portfolio_weights) > 0:
            weights_df = pd.DataFrame(portfolio_weights)
            stress_results = self.stress_tester.run_stress_tests(returns, weights_df)
        else:
            stress_results = {}
            
        return {
            'portfolio_value': portfolio_value,
            'portfolio_returns': portfolio_returns,
            'portfolio_weights': portfolio_weights,
            'transaction_costs': transaction_costs,
            'rebalance_dates': rebalance_dates,
            'performance_metrics': performance,
            'risk_metrics': risk,
            'stress_test_results': stress_results,
            'config': self.config
        }
    
    def _get_rebalance_frequency(self) -> int:
        if self.config.rebalance_frequency == "daily":
            return 1
        elif self.config.rebalance_frequency == "weekly":
            return 5
        elif self.config.rebalance_frequency == "monthly":
            return 21
        else:
            return 5  # Default to weekly
    
    def _calculate_performance_metrics(self, returns: np.ndarray) -> PerformanceMetrics:
        
        if len(returns) == 0:
            return PerformanceMetrics()
            
        total_return = (1 + returns).prod() - 1
        annualized_return = (1 + returns).prod() ** (252 / len(returns)) - 1
        volatility = returns.std() * np.sqrt(252)
        
        # Sharpe ratio (assuming 2% risk-free rate)
        excess_returns = returns - (0.02 / 252)
        sharpe_ratio = excess_returns.mean() / excess_returns.std() * np.sqrt(252) if excess_returns.std() > 0 else 0
        
        # Sortino ratio
        downside_returns = returns[returns < 0]
        downside_deviation = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else volatility
        sortino_ratio = (annualized_return - 0.02) / downside_deviation if downside_deviation > 0 else 0
        
        # Win rate
        win_rate = (returns > 0).mean()
        
        return PerformanceMetrics(
            total_return=total_return,
            annualized_return=annualized_return,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            win_rate=win_rate
        )
    
    def _calculate_risk_metrics(self, returns: np.ndarray, portfolio_value: np.ndarray) -> RiskMetrics:
        if len(returns) == 0:
            return RiskMetrics()
            
        # VaR calculations
        var_1d = self.risk_manager.calculate_var(returns, 0.05)
        var_5d = self.risk_manager.calculate_var(returns, 0.05) * np.sqrt(5)  # 5-day VaR
        
        # Expected shortfall
        expected_shortfall = self.risk_manager.calculate_expected_shortfall(returns, 0.05)
        
        # Drawdown metrics
        max_drawdown, drawdown_duration = self.risk_manager.calculate_drawdown(portfolio_value)
        
        # Volatility metrics
        volatility = returns.std() * np.sqrt(252)
        downside_returns = returns[returns < 0]
        downside_deviation = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
        
        return RiskMetrics(
            var_1d=var_1d,
            var_5d=var_5d,
            expected_shortfall=expected_shortfall,
            max_drawdown=max_drawdown,
            drawdown_duration=drawdown_duration,
            volatility=volatility,
            downside_deviation=downside_deviation
        )


class PerformanceReporter:
    def __init__(self):
        pass
        
    def generate_report(self, backtest_results: Dict) -> str:
        """Generate detailed performance report"""
        
        performance = backtest_results['performance_metrics']
        risk = backtest_results['risk_metrics']
        stress = backtest_results['stress_test_results']
        config = backtest_results['config']
        
        report = f"""
CONFIGURATION:
- Initial Capital: ${config.initial_capital:,.0f}
- Max Drawdown Limit: {config.max_drawdown_limit:.1%}
- Max Position Size: {config.max_position_size:.1%}
- Transaction Cost: {config.transaction_cost_bps:.1f} bps
- Kelly Sizing: {config.use_kelly_sizing}
- Rebalancing: {config.rebalance_frequency}

PERFORMANCE METRICS:
- Total Return: {performance.total_return:.2%}
- Annualized Return: {performance.annualized_return:.2%}
- Sharpe Ratio: {performance.sharpe_ratio:.3f}
- Sortino Ratio: {performance.sortino_ratio:.3f}
- Win Rate: {performance.win_rate:.2%}

RISK METRICS:
- 1-Day VaR (5%): {risk.var_1d:.3%}
- 5-Day VaR (5%): {risk.var_5d:.3%}
- Expected Shortfall: {risk.expected_shortfall:.3%}
- Maximum Drawdown: {risk.max_drawdown:.2%}
- Drawdown Duration: {risk.drawdown_duration} days
- Volatility (annualized): {risk.volatility:.2%}
- Downside Deviation: {risk.downside_deviation:.2%}

STRESS TEST RESULTS:
"""
        
        for key, value in stress.items():
            report += f"- {key}: {value:.3%}\n"
            
        # Calculate Calmar ratio
        if risk.max_drawdown > 0:
            calmar_ratio = performance.annualized_return / risk.max_drawdown
            report += f"\nCALMAR RATIO: {calmar_ratio:.3f}"
            
        # Transaction cost analysis
        total_costs = sum(backtest_results['transaction_costs'])
        report += f"\nTRANSACTION COSTS: ${total_costs:,.0f} ({total_costs/config.initial_capital:.2%} of capital)"
        
        return report
    
    def create_summary_dict(self, backtest_results: Dict) -> Dict:
        performance = backtest_results['performance_metrics']
        risk = backtest_results['risk_metrics']
        
        return {
            'total_return': performance.total_return,
            'annualized_return': performance.annualized_return,
            'sharpe_ratio': performance.sharpe_ratio,
            'sortino_ratio': performance.sortino_ratio,
            'max_drawdown': risk.max_drawdown,
            'volatility': risk.volatility,
            'var_1d': risk.var_1d,
            'win_rate': performance.win_rate,
            'transaction_costs': sum(backtest_results['transaction_costs'])
        }
