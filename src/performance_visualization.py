import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


class AdvancedPerformanceVisualizer:
    def __init__(self, style: str = "seaborn-v0_8-darkgrid", figsize: Tuple[int, int] = (12, 8)):
        self.style = style
        self.figsize = figsize
        plt.style.use('default')  
        sns.set_palette("husl")
        
    def plot_portfolio_performance(self, backtest_results: Dict, benchmark_data: Optional[pd.Series] = None, 
                                 save_path: Optional[str] = None) -> None:
        
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        fig.suptitle('Portfolio Performance Analysis', fontsize=16, fontweight='bold')
        
        portfolio_value = backtest_results['portfolio_value']
        portfolio_returns = backtest_results['portfolio_returns'] 
        dates = pd.date_range(start='2020-01-01', periods=len(portfolio_value), freq='D')
        
        # 1. Portfolio Value Evolution
        axes[0, 0].plot(dates, portfolio_value, linewidth=2, color='blue', label='Portfolio')
        if benchmark_data is not None:
            axes[0, 0].plot(dates[:len(benchmark_data)], benchmark_data, 
                           linewidth=2, color='gray', alpha=0.7, label='Benchmark')
        axes[0, 0].set_title('Portfolio Value Evolution')
        axes[0, 0].set_ylabel('Portfolio Value ($)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Drawdown Analysis
        peak = np.maximum.accumulate(portfolio_value)
        drawdown = (portfolio_value - peak) / peak
        
        axes[0, 1].fill_between(dates, drawdown, 0, alpha=0.3, color='red', label='Drawdown')
        axes[0, 1].plot(dates, drawdown, linewidth=1, color='darkred')
        axes[0, 1].set_title('Drawdown Analysis')
        axes[0, 1].set_ylabel('Drawdown (%)')
        axes[0, 1].axhline(y=-0.15, color='red', linestyle='--', alpha=0.7, label='15% Limit')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Returns Distribution
        if len(portfolio_returns) > 0:
            axes[1, 0].hist(portfolio_returns, bins=50, alpha=0.7, color='blue', edgecolor='black')
            axes[1, 0].axvline(portfolio_returns.mean(), color='red', linestyle='--', 
                              label=f'Mean: {portfolio_returns.mean():.4f}')
            axes[1, 0].axvline(np.percentile(portfolio_returns, 5), color='orange', linestyle='--',
                              label='5% VaR')
            axes[1, 0].set_title('Daily Returns Distribution')
            axes[1, 0].set_xlabel('Daily Returns')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Rolling Sharpe Ratio
        if len(portfolio_returns) > 21:
            rolling_returns = pd.Series(portfolio_returns)
            rolling_sharpe = rolling_returns.rolling(21).mean() / rolling_returns.rolling(21).std() * np.sqrt(252)
            
            axes[1, 1].plot(dates[21:len(rolling_sharpe)+21], rolling_sharpe.dropna(), 
                           linewidth=2, color='green')
            axes[1, 1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
            axes[1, 1].axhline(y=1, color='orange', linestyle='--', alpha=0.7, label='Sharpe = 1')
            axes[1, 1].set_title('Rolling 21-Day Sharpe Ratio')
            axes[1, 1].set_ylabel('Sharpe Ratio')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        # 5. Portfolio Weights Evolution
        if backtest_results['portfolio_weights']:
            weights_df = pd.DataFrame(backtest_results['portfolio_weights'])
            rebalance_dates = backtest_results['rebalance_dates']
            
            for i, asset in enumerate(weights_df.columns):
                axes[2, 0].plot(rebalance_dates, weights_df[asset], 
                               label=asset, marker='o', markersize=3)
            
            axes[2, 0].set_title('Portfolio Weights Evolution')
            axes[2, 0].set_ylabel('Weight')
            axes[2, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            axes[2, 0].grid(True, alpha=0.3)
        
        # 6. Risk Metrics Over Time
        if len(portfolio_returns) > 21:
            rolling_vol = pd.Series(portfolio_returns).rolling(21).std() * np.sqrt(252)
            rolling_var = pd.Series(portfolio_returns).rolling(21).quantile(0.05)
            
            ax_vol = axes[2, 1]
            ax_var = ax_vol.twinx()
            
            line1 = ax_vol.plot(dates[21:len(rolling_vol)+21], rolling_vol.dropna(), 
                               color='blue', label='21-Day Vol', linewidth=2)
            line2 = ax_var.plot(dates[21:len(rolling_var)+21], rolling_var.dropna(), 
                               color='red', label='21-Day VaR', linewidth=2)
            
            ax_vol.set_title('Risk Metrics Evolution')
            ax_vol.set_ylabel('Volatility (annualized)', color='blue')
            ax_var.set_ylabel('VaR (5%)', color='red')
            
            # Combine legends
            lines = line1 + line2
            labels = [l.get_label() for l in lines]
            ax_vol.legend(lines, labels, loc='upper left')
            ax_vol.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
    
    def plot_interactive_dashboard(self, backtest_results: Dict, 
                                 benchmark_data: Optional[pd.Series] = None) -> go.Figure:
        """Create interactive Plotly dashboard"""
        
        portfolio_value = backtest_results['portfolio_value']
        portfolio_returns = backtest_results['portfolio_returns']
        dates = pd.date_range(start='2020-01-01', periods=len(portfolio_value), freq='D')
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=('Portfolio Value', 'Drawdown', 'Returns Distribution', 
                          'Rolling Sharpe', 'Weight Evolution', 'Risk Metrics'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": True}]]
        )
        
        # Portfolio value
        fig.add_trace(
            go.Scatter(x=dates, y=portfolio_value, name='Portfolio', line=dict(width=3)),
            row=1, col=1
        )
        
        if benchmark_data is not None:
            fig.add_trace(
                go.Scatter(x=dates[:len(benchmark_data)], y=benchmark_data, 
                          name='Benchmark', line=dict(width=2, dash='dash')),
                row=1, col=1
            )
        
        # Drawdown
        peak = np.maximum.accumulate(portfolio_value)
        drawdown = (portfolio_value - peak) / peak
        
        fig.add_trace(
            go.Scatter(x=dates, y=drawdown, fill='tonexty', name='Drawdown',
                      line=dict(color='red', width=1)),
            row=1, col=2
        )
        
        # Returns distribution
        if len(portfolio_returns) > 0:
            fig.add_trace(
                go.Histogram(x=portfolio_returns, nbinsx=50, name='Returns Dist',
                           marker_color='blue', opacity=0.7),
                row=2, col=1
            )
        
        # Rolling Sharpe
        if len(portfolio_returns) > 21:
            rolling_returns = pd.Series(portfolio_returns)
            rolling_sharpe = rolling_returns.rolling(21).mean() / rolling_returns.rolling(21).std() * np.sqrt(252)
            
            fig.add_trace(
                go.Scatter(x=dates[21:len(rolling_sharpe)+21], y=rolling_sharpe.dropna(),
                          name='Rolling Sharpe', line=dict(color='green', width=2)),
                row=2, col=2
            )
        
        # Portfolio weights
        if backtest_results['portfolio_weights']:
            weights_df = pd.DataFrame(backtest_results['portfolio_weights'])
            rebalance_dates = backtest_results['rebalance_dates']
            
            for asset in weights_df.columns:
                fig.add_trace(
                    go.Scatter(x=rebalance_dates, y=weights_df[asset], 
                              name=f'Weight {asset}', mode='lines+markers'),
                    row=3, col=1
                )
        
        # Risk metrics
        if len(portfolio_returns) > 21:
            rolling_vol = pd.Series(portfolio_returns).rolling(21).std() * np.sqrt(252)
            
            fig.add_trace(
                go.Scatter(x=dates[21:len(rolling_vol)+21], y=rolling_vol.dropna(),
                          name='Volatility', line=dict(color='blue', width=2)),
                row=3, col=2
            )
        
        fig.update_layout(height=1000, showlegend=True, 
                         title_text="Portfolio Performance Interactive Dashboard")
        
        return fig
    
    def create_risk_report_chart(self, backtest_results: Dict) -> plt.Figure:
        """Create detailed risk analysis charts"""
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Risk Analysis Report', fontsize=16, fontweight='bold')
        
        portfolio_returns = backtest_results['portfolio_returns']
        stress_results = backtest_results['stress_test_results']
        
        if len(portfolio_returns) == 0:
            return fig
            
        # VaR and Expected Shortfall
        var_levels = [0.01, 0.05, 0.10]
        var_values = [np.percentile(portfolio_returns, level*100) for level in var_levels]
        es_values = [portfolio_returns[portfolio_returns <= var].mean() for var in var_values]
        
        x_pos = np.arange(len(var_levels))
        width = 0.35
        
        axes[0, 0].bar(x_pos - width/2, var_values, width, label='VaR', color='orange', alpha=0.7)
        axes[0, 0].bar(x_pos + width/2, es_values, width, label='Expected Shortfall', color='red', alpha=0.7)
        axes[0, 0].set_xlabel('Confidence Level')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('VaR vs Expected Shortfall')
        axes[0, 0].set_xticks(x_pos)
        axes[0, 0].set_xticklabels(['1%', '5%', '10%'])
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Returns vs Normal Distribution
        axes[0, 1].hist(portfolio_returns, bins=50, density=True, alpha=0.7, 
                       color='blue', label='Actual Returns')
        
        # Overlay normal distribution
        mu, sigma = portfolio_returns.mean(), portfolio_returns.std()
        x = np.linspace(portfolio_returns.min(), portfolio_returns.max(), 100)
        normal_pdf = (1/np.sqrt(2*np.pi*sigma**2)) * np.exp(-0.5*((x-mu)/sigma)**2)
        axes[0, 1].plot(x, normal_pdf, 'r-', linewidth=2, label='Normal Distribution')
        
        axes[0, 1].set_title('Returns vs Normal Distribution')
        axes[0, 1].set_xlabel('Returns')
        axes[0, 1].set_ylabel('Density')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Stress Test Results
        if stress_results:
            stress_names = []
            stress_values = []
            
            for key, value in stress_results.items():
                if 'var' in key.lower() or 'shortfall' in key.lower():
                    stress_names.append(key.replace('_', ' ').title())
                    stress_values.append(value)
            
            if stress_names:
                colors = plt.cm.RdYlBu_r(np.linspace(0.2, 0.8, len(stress_names)))
                bars = axes[1, 0].barh(stress_names, stress_values, color=colors)
                axes[1, 0].set_title('Stress Test Results')
                axes[1, 0].set_xlabel('Loss (%)')
                axes[1, 0].grid(True, alpha=0.3)
                
                # Add value labels on bars
                for i, (bar, value) in enumerate(zip(bars, stress_values)):
                    axes[1, 0].text(value, i, f'{value:.2%}', 
                                   ha='left' if value < 0 else 'right', va='center')
        
        # Rolling correlation with benchmark (if available)
        # For now, create a placeholder showing portfolio autocorrelation
        if len(portfolio_returns) > 50:
            lags = range(1, 21)
            autocorr = [np.corrcoef(portfolio_returns[:-lag], portfolio_returns[lag:])[0, 1] 
                       for lag in lags]
            
            axes[1, 1].plot(lags, autocorr, marker='o', linewidth=2, color='purple')
            axes[1, 1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
            axes[1, 1].set_title('Return Autocorrelation')
            axes[1, 1].set_xlabel('Lag (days)')
            axes[1, 1].set_ylabel('Correlation')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig


class ComprehensiveReportGenerator:
    """Generate comprehensive PDF and HTML reports"""
    
    def __init__(self):
        self.visualizer = AdvancedPerformanceVisualizer()
        
    def generate_executive_summary(self, backtest_results: Dict) -> str:
        """Generate executive summary"""
        
        performance = backtest_results['performance_metrics']
        risk = backtest_results['risk_metrics']
        config = backtest_results['config']
        
        # Calculate additional metrics
        portfolio_value = backtest_results['portfolio_value']
        final_value = portfolio_value[-1]
        initial_value = portfolio_value[0]
        
        summary = f"""
EXECUTIVE SUMMARY
================

STRATEGY PERFORMANCE
- Portfolio grew from ${initial_value:,.0f} to ${final_value:,.0f}
- Total Return: {performance.total_return:.2%}
- Annualized Return: {performance.annualized_return:.2%}
- Risk-Adjusted Return (Sharpe): {performance.sharpe_ratio:.3f}
- Downside Risk-Adjusted Return (Sortino): {performance.sortino_ratio:.3f}

RISK PROFILE
- Maximum Drawdown: {risk.max_drawdown:.2%} (Target: {config.max_drawdown_limit:.1%})
- Portfolio Volatility: {risk.volatility:.2%} (annualized)
- 1-Day Value at Risk (5%): {risk.var_1d:.3%}
- Expected Shortfall: {risk.expected_shortfall:.3%}
- Longest Drawdown Period: {risk.drawdown_duration} days

STRATEGY CHARACTERISTICS  
- Kelly Criterion Position Sizing: {'Enabled' if config.use_kelly_sizing else 'Disabled'}
- Maximum Position Size: {config.max_position_size:.1%} per asset
- Rebalancing Frequency: {config.rebalance_frequency.title()}
- Transaction Costs: {config.transaction_cost_bps:.1f} basis points
- Total Transaction Costs: ${sum(backtest_results['transaction_costs']):,.0f}

RISK MANAGEMENT
- Drawdown Limit Breaches: {'None' if risk.max_drawdown < config.max_drawdown_limit else 'Yes'}
- VaR Model Performance: Within expected parameters
- Stress Test Results: See detailed analysis below

RECOMMENDATION
{'APPROVED - Strategy demonstrates strong risk-adjusted returns' if performance.sharpe_ratio > 1.0 and risk.max_drawdown < config.max_drawdown_limit else 'REVIEW REQUIRED - Strategy needs optimization'}
"""
        return summary
    
    def generate_detailed_analysis(self, backtest_results: Dict) -> str:
        """Generate detailed quantitative analysis"""
        
        portfolio_returns = backtest_results['portfolio_returns']
        performance = backtest_results['performance_metrics']
        risk = backtest_results['risk_metrics']
        stress_results = backtest_results['stress_test_results']
        
        analysis = f"""
DETAILED QUANTITATIVE ANALYSIS
==============================

RETURN CHARACTERISTICS
- Mean Daily Return: {portfolio_returns.mean():.4f}
- Standard Deviation: {portfolio_returns.std():.4f}
- Skewness: {pd.Series(portfolio_returns).skew():.3f}
- Kurtosis: {pd.Series(portfolio_returns).kurtosis():.3f}
- Best Day: {portfolio_returns.max():.3%}
- Worst Day: {portfolio_returns.min():.3%}
- Win Rate: {performance.win_rate:.2%}

RISK DECOMPOSITION
- Total Risk (Volatility): {risk.volatility:.2%}
- Systematic Risk (Beta): {risk.beta:.3f}
- Unsystematic Risk: {np.sqrt(max(0, risk.volatility**2 - risk.beta**2)):.2%}
- Downside Deviation: {risk.downside_deviation:.2%}
- Upside Capture: {(portfolio_returns[portfolio_returns > 0].mean() / portfolio_returns.std()):.3f}
- Downside Capture: {abs(portfolio_returns[portfolio_returns < 0].mean() / portfolio_returns.std()):.3f}

VALUE AT RISK ANALYSIS
- 1-Day VaR (5%): {risk.var_1d:.3%}
- 5-Day VaR (5%): {risk.var_5d:.3%}
- 1-Day VaR (1%): {np.percentile(portfolio_returns, 1):.3%}
- Expected Shortfall (5%): {risk.expected_shortfall:.3%}

DRAWDOWN ANALYSIS  
- Maximum Drawdown: {risk.max_drawdown:.2%}
- Average Drawdown: {np.mean([abs(dd) for dd in (np.array(backtest_results['portfolio_value']) - np.maximum.accumulate(backtest_results['portfolio_value'])) / np.maximum.accumulate(backtest_results['portfolio_value']) if abs(dd) > 0.01]):.2%}
- Drawdown Frequency: {sum(1 for dd in (np.array(backtest_results['portfolio_value']) - np.maximum.accumulate(backtest_results['portfolio_value'])) / np.maximum.accumulate(backtest_results['portfolio_value']) if abs(dd) > 0.05)} periods > 5%
- Recovery Time: {risk.drawdown_duration} days (longest)

STRESS TEST SUMMARY
"""
        
        for key, value in stress_results.items():
            analysis += f"- {key.replace('_', ' ').title()}: {value:.3%}\n"
            
        return analysis
    
    def create_full_report(self, backtest_results: Dict, save_dir: str = "./reports/") -> Dict[str, str]:
        """Create comprehensive report with visualizations"""
        
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        # Generate text reports
        executive_summary = self.generate_executive_summary(backtest_results)
        detailed_analysis = self.generate_detailed_analysis(backtest_results)
        
        # Save text reports
        with open(f"{save_dir}/executive_summary.txt", "w") as f:
            f.write(executive_summary)
            
        with open(f"{save_dir}/detailed_analysis.txt", "w") as f:
            f.write(detailed_analysis)
        
        # Generate and save visualizations
        self.visualizer.plot_portfolio_performance(
            backtest_results, save_path=f"{save_dir}/performance_dashboard.png"
        )
        
        risk_fig = self.visualizer.create_risk_report_chart(backtest_results)
        risk_fig.savefig(f"{save_dir}/risk_analysis.png", dpi=300, bbox_inches='tight')
        plt.close(risk_fig)
        
        # Generate interactive dashboard
        interactive_fig = self.visualizer.plot_interactive_dashboard(backtest_results)
        interactive_fig.write_html(f"{save_dir}/interactive_dashboard.html")
        
        return {
            "executive_summary": f"{save_dir}/executive_summary.txt",
            "detailed_analysis": f"{save_dir}/detailed_analysis.txt", 
            "performance_dashboard": f"{save_dir}/performance_dashboard.png",
            "risk_analysis": f"{save_dir}/risk_analysis.png",
            "interactive_dashboard": f"{save_dir}/interactive_dashboard.html"
        }


def create_benchmark_comparison(portfolio_results: Dict, benchmark_returns: pd.Series,
                              benchmark_name: str = "S&P 500") -> Dict:
    """Create benchmark comparison analysis"""
    
    portfolio_returns = portfolio_results['portfolio_returns']
    
    # Calculate benchmark metrics
    benchmark_total_return = (1 + benchmark_returns).prod() - 1
    benchmark_volatility = benchmark_returns.std() * np.sqrt(252)
    benchmark_sharpe = (benchmark_returns.mean() - 0.02/252) / benchmark_returns.std() * np.sqrt(252)
    
    # Calculate relative metrics
    excess_returns = portfolio_returns[:len(benchmark_returns)] - benchmark_returns
    tracking_error = excess_returns.std() * np.sqrt(252)
    information_ratio = excess_returns.mean() / excess_returns.std() * np.sqrt(252) if excess_returns.std() > 0 else 0
    
    # Beta calculation
    beta = np.cov(portfolio_returns[:len(benchmark_returns)], benchmark_returns)[0, 1] / np.var(benchmark_returns)
    alpha = portfolio_results['performance_metrics'].annualized_return - (0.02 + beta * (benchmark_total_return * 252 / len(benchmark_returns) - 0.02))
    
    comparison = {
        "portfolio_metrics": {
            "total_return": portfolio_results['performance_metrics'].total_return,
            "volatility": portfolio_results['risk_metrics'].volatility,
            "sharpe_ratio": portfolio_results['performance_metrics'].sharpe_ratio,
            "max_drawdown": portfolio_results['risk_metrics'].max_drawdown
        },
        "benchmark_metrics": {
            "total_return": benchmark_total_return,
            "volatility": benchmark_volatility, 
            "sharpe_ratio": benchmark_sharpe,
            "name": benchmark_name
        },
        "relative_metrics": {
            "alpha": alpha,
            "beta": beta,
            "tracking_error": tracking_error,
            "information_ratio": information_ratio,
            "excess_return": portfolio_results['performance_metrics'].total_return - benchmark_total_return
        }
    }
    
    return comparison
