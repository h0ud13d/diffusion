# visualization.py - Advanced visualization and reporting system
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Try to import plotly, but make it optional
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("Plotly not available - interactive dashboards will be disabled")

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class PerformanceVisualizer:
    """Advanced visualization for backtest results"""
    
    def __init__(self, figsize: Tuple[int, int] = (15, 10)):
        self.figsize = figsize
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                      '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    
    def plot_portfolio_performance(self, results_dict: Dict[str, any], 
                                 benchmark_results: Optional[any] = None,
                                 save_path: Optional[str] = None):
        """Plot comprehensive portfolio performance comparison"""
        
        fig, axes = plt.subplots(2, 2, figsize=(20, 12))
        fig.suptitle('Portfolio Performance Analysis', fontsize=16, fontweight='bold')
        
        # 1. Cumulative Returns
        ax1 = axes[0, 0]
        for i, (strategy, results) in enumerate(results_dict.items()):
            cumret = (1 + results.returns).cumprod()
            ax1.plot(cumret.index, cumret.values, label=strategy, 
                    color=self.colors[i % len(self.colors)], linewidth=2)
        
        if benchmark_results is not None:
            cumret_bench = (1 + benchmark_results.returns).cumprod()
            ax1.plot(cumret_bench.index, cumret_bench.values, 
                    label='Benchmark', color='black', linestyle='--', linewidth=2)
        
        ax1.set_title('Cumulative Returns', fontweight='bold')
        ax1.set_ylabel('Cumulative Return')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Drawdowns
        ax2 = axes[0, 1]
        for i, (strategy, results) in enumerate(results_dict.items()):
            ax2.fill_between(results.drawdowns.index, results.drawdowns.values, 0,
                           alpha=0.6, color=self.colors[i % len(self.colors)], label=strategy)
        
        ax2.set_title('Drawdowns', fontweight='bold')
        ax2.set_ylabel('Drawdown (%)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Rolling Sharpe Ratio
        ax3 = axes[1, 0]
        for i, (strategy, results) in enumerate(results_dict.items()):
            rolling_sharpe = results.returns.rolling(252).apply(
                lambda x: np.sqrt(252) * x.mean() / x.std() if x.std() > 0 else 0
            )
            ax3.plot(rolling_sharpe.index, rolling_sharpe.values, 
                    label=strategy, color=self.colors[i % len(self.colors)], linewidth=2)
        
        ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax3.set_title('Rolling 1-Year Sharpe Ratio', fontweight='bold')
        ax3.set_ylabel('Sharpe Ratio')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Return Distribution
        ax4 = axes[1, 1]
        returns_data = []
        strategy_names = []
        for strategy, results in results_dict.items():
            returns_data.extend(results.returns.values)
            strategy_names.extend([strategy] * len(results.returns))
        
        returns_df = pd.DataFrame({'returns': returns_data, 'strategy': strategy_names})
        sns.boxplot(data=returns_df, x='strategy', y='returns', ax=ax4)
        ax4.set_title('Return Distribution', fontweight='bold')
        ax4.set_ylabel('Daily Returns')
        ax4.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_risk_metrics(self, results_dict: Dict[str, any], save_path: Optional[str] = None):
        """Plot risk metrics comparison"""
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Risk Metrics Analysis', fontsize=16, fontweight='bold')
        
        # Prepare data
        strategies = list(results_dict.keys())
        metrics = {
            'Volatility': [results.volatility for results in results_dict.values()],
            'Max Drawdown': [abs(results.max_drawdown) for results in results_dict.values()],
            'VaR 95%': [abs(results.var_95) for results in results_dict.values()],
            'Expected Shortfall': [abs(results.expected_shortfall_95) for results in results_dict.values()]
        }
        
        # Plot each metric
        for i, (metric, values) in enumerate(metrics.items()):
            ax = axes[i // 2, i % 2]
            bars = ax.bar(strategies, values, color=self.colors[:len(strategies)], alpha=0.7)
            ax.set_title(f'{metric}', fontweight='bold')
            ax.set_ylabel(metric)
            ax.tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_position_heatmap(self, positions_df: pd.DataFrame, 
                            title: str = "Position Weights Over Time",
                            save_path: Optional[str] = None):
        """Plot position weights as a heatmap"""
        
        # Resample to weekly data for better visualization
        weekly_positions = positions_df.resample('W').last()
        
        fig, ax = plt.subplots(figsize=(15, 8))
        
        # Create heatmap
        sns.heatmap(weekly_positions.T, cmap='RdBu_r', center=0, 
                   ax=ax, cbar_kws={'label': 'Position Weight'})
        
        ax.set_title(title, fontweight='bold', fontsize=14)
        ax.set_xlabel('Date')
        ax.set_ylabel('Assets')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_correlation_matrix(self, returns_data: pd.DataFrame, 
                              title: str = "Asset Return Correlations",
                              save_path: Optional[str] = None):
        """Plot correlation matrix of asset returns"""
        
        corr_matrix = returns_data.corr()
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create heatmap
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                   square=True, ax=ax, fmt='.2f')
        
        ax.set_title(title, fontweight='bold', fontsize=14)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

class InteractiveDashboard:
    """Interactive Plotly-based dashboard"""
    
    def create_performance_dashboard(self, results_dict: Dict[str, any], 
                                   benchmark_results: Optional[any] = None):
        """Create interactive performance dashboard"""
        
        if not PLOTLY_AVAILABLE:
            print("Plotly not available - skipping interactive dashboard")
            return None
            
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Cumulative Returns', 'Rolling Sharpe Ratio',
                          'Drawdowns', 'Monthly Returns Heatmap'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        colors = px.colors.qualitative.Set1
        
        # 1. Cumulative Returns
        for i, (strategy, results) in enumerate(results_dict.items()):
            cumret = (1 + results.returns).cumprod()
            fig.add_trace(
                go.Scatter(x=cumret.index, y=cumret.values, 
                          name=strategy, line=dict(color=colors[i % len(colors)])),
                row=1, col=1
            )
        
        if benchmark_results is not None:
            cumret_bench = (1 + benchmark_results.returns).cumprod()
            fig.add_trace(
                go.Scatter(x=cumret_bench.index, y=cumret_bench.values,
                          name='Benchmark', line=dict(color='black', dash='dash')),
                row=1, col=1
            )
        
        # 2. Rolling Sharpe Ratio
        for i, (strategy, results) in enumerate(results_dict.items()):
            rolling_sharpe = results.returns.rolling(252).apply(
                lambda x: np.sqrt(252) * x.mean() / x.std() if x.std() > 0 else 0
            )
            fig.add_trace(
                go.Scatter(x=rolling_sharpe.index, y=rolling_sharpe.values,
                          name=f"{strategy} Sharpe", line=dict(color=colors[i % len(colors)])),
                row=1, col=2
            )
        
        # 3. Drawdowns
        for i, (strategy, results) in enumerate(results_dict.items()):
            fig.add_trace(
                go.Scatter(x=results.drawdowns.index, y=results.drawdowns.values,
                          fill='tonexty' if i > 0 else 'tozeroy',
                          name=f"{strategy} DD", 
                          line=dict(color=colors[i % len(colors)])),
                row=2, col=1
            )
        
        # Update layout
        fig.update_layout(
            title_text="Portfolio Performance Dashboard",
            title_x=0.5,
            height=800,
            showlegend=True,
            template="plotly_white"
        )
        
        return fig
    
    def create_risk_dashboard(self, results_dict: Dict[str, any]):
        """Create interactive risk metrics dashboard"""
        
        if not PLOTLY_AVAILABLE:
            print("Plotly not available - skipping risk dashboard")
            return None
            
        strategies = list(results_dict.keys())
        
        # Prepare risk metrics data
        risk_metrics = pd.DataFrame({
            'Strategy': strategies,
            'Volatility': [results.volatility for results in results_dict.values()],
            'Max Drawdown': [abs(results.max_drawdown) for results in results_dict.values()],
            'VaR 95%': [abs(results.var_95) for results in results_dict.values()],
            'Sharpe Ratio': [results.sharpe_ratio for results in results_dict.values()],
            'Sortino Ratio': [results.sortino_ratio for results in results_dict.values()]
        })
        
        # Create radar chart
        fig = go.Figure()
        
        for i, strategy in enumerate(strategies):
            strategy_data = risk_metrics[risk_metrics['Strategy'] == strategy].iloc[0]
            
            fig.add_trace(go.Scatterpolar(
                r=[strategy_data['Volatility'], 
                   strategy_data['Max Drawdown'],
                   strategy_data['VaR 95%'],
                   strategy_data['Sharpe Ratio'],
                   strategy_data['Sortino Ratio']],
                theta=['Volatility', 'Max Drawdown', 'VaR 95%', 'Sharpe Ratio', 'Sortino Ratio'],
                fill='toself',
                name=strategy
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, max(risk_metrics[['Volatility', 'Max Drawdown', 'VaR 95%']].max())]
                )),
            title="Risk Metrics Comparison",
            showlegend=True
        )
        
        return fig

class ReportGenerator:
    """Generate comprehensive performance reports"""
    
    def __init__(self):
        self.visualizer = PerformanceVisualizer()
        self.dashboard = InteractiveDashboard()
    
    def generate_html_report(self, results_dict: Dict[str, any], 
                           metadata: Dict,
                           output_path: str = "backtest_report.html"):
        """Generate comprehensive HTML report"""
        
        # Calculate summary statistics
        summary_stats = self._calculate_summary_stats(results_dict)
        
        # Create HTML content
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Diffusion Model Backtest Report</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    margin: 40px;
                    background-color: #f5f5f5;
                }}
                .header {{
                    background-color: #2c3e50;
                    color: white;
                    padding: 20px;
                    border-radius: 10px;
                    margin-bottom: 20px;
                }}
                .section {{
                    background-color: white;
                    padding: 20px;
                    margin-bottom: 20px;
                    border-radius: 10px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }}
                table {{
                    width: 100%;
                    border-collapse: collapse;
                    margin: 20px 0;
                }}
                th, td {{
                    border: 1px solid #ddd;
                    padding: 12px;
                    text-align: left;
                }}
                th {{
                    background-color: #f2f2f2;
                    font-weight: bold;
                }}
                .metric-positive {{ color: #27ae60; }}
                .metric-negative {{ color: #e74c3c; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Diffusion Model Backtest Report</h1>
                <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <div class="section">
                <h2>Executive Summary</h2>
                <p>This report presents the performance analysis of diffusion model-based trading strategies
                on a portfolio of {len(metadata.get('assets', []))} assets over the backtesting period.</p>
                {self._create_summary_table(summary_stats)}
            </div>
            
            <div class="section">
                <h2>Model Configuration</h2>
                <ul>
                    <li><strong>Assets:</strong> {', '.join(metadata.get('assets', []))}</li>
                    <li><strong>Sequence Length:</strong> {metadata.get('sequence_length', 'N/A')} days</li>
                    <li><strong>Prediction Horizon:</strong> {metadata.get('prediction_horizon', 'N/A')} days</li>
                    <li><strong>Feature Dimensions:</strong> {metadata.get('feature_dim', 'N/A')}</li>
                </ul>
            </div>
            
            <div class="section">
                <h2>Performance Metrics</h2>
                {self._create_performance_table(results_dict)}
            </div>
            
            <div class="section">
                <h2>Risk Analysis</h2>
                {self._create_risk_table(results_dict)}
            </div>
            
        </body>
        </html>
        """
        
        # Write to file
        with open(output_path, 'w') as f:
            f.write(html_content)
        
        print(f"HTML report generated: {output_path}")
    
    def _calculate_summary_stats(self, results_dict: Dict[str, any]) -> Dict:
        """Calculate summary statistics across all strategies"""
        
        all_returns = [results.total_return for results in results_dict.values()]
        all_sharpe = [results.sharpe_ratio for results in results_dict.values()]
        all_drawdowns = [results.max_drawdown for results in results_dict.values()]
        
        return {
            'best_strategy': max(results_dict.keys(), key=lambda k: results_dict[k].total_return),
            'worst_strategy': min(results_dict.keys(), key=lambda k: results_dict[k].total_return),
            'avg_return': np.mean(all_returns),
            'avg_sharpe': np.mean(all_sharpe),
            'avg_drawdown': np.mean(all_drawdowns)
        }
    
    def _create_summary_table(self, summary_stats: Dict) -> str:
        """Create HTML summary table"""
        
        return f"""
        <table>
            <tr><th>Metric</th><th>Value</th></tr>
            <tr><td>Best Performing Strategy</td><td class="metric-positive">{summary_stats['best_strategy']}</td></tr>
            <tr><td>Worst Performing Strategy</td><td class="metric-negative">{summary_stats['worst_strategy']}</td></tr>
            <tr><td>Average Return</td><td>{summary_stats['avg_return']:.2%}</td></tr>
            <tr><td>Average Sharpe Ratio</td><td>{summary_stats['avg_sharpe']:.3f}</td></tr>
            <tr><td>Average Max Drawdown</td><td>{summary_stats['avg_drawdown']:.2%}</td></tr>
        </table>
        """
    
    def _create_performance_table(self, results_dict: Dict[str, any]) -> str:
        """Create HTML performance metrics table"""
        
        html = """
        <table>
            <tr>
                <th>Strategy</th>
                <th>Total Return</th>
                <th>Annualized Return</th>
                <th>Volatility</th>
                <th>Sharpe Ratio</th>
                <th>Max Drawdown</th>
                <th>Calmar Ratio</th>
            </tr>
        """
        
        for strategy, results in results_dict.items():
            color_class = "metric-positive" if results.total_return > 0 else "metric-negative"
            html += f"""
            <tr>
                <td><strong>{strategy}</strong></td>
                <td class="{color_class}">{results.total_return:.2%}</td>
                <td class="{color_class}">{results.annualized_return:.2%}</td>
                <td>{results.volatility:.2%}</td>
                <td>{results.sharpe_ratio:.3f}</td>
                <td class="metric-negative">{results.max_drawdown:.2%}</td>
                <td>{results.calmar_ratio:.3f}</td>
            </tr>
            """
        
        html += "</table>"
        return html
    
    def _create_risk_table(self, results_dict: Dict[str, any]) -> str:
        """Create HTML risk metrics table"""
        
        html = """
        <table>
            <tr>
                <th>Strategy</th>
                <th>VaR 95%</th>
                <th>Expected Shortfall</th>
                <th>Sortino Ratio</th>
                <th>Downside Deviation</th>
            </tr>
        """
        
        for strategy, results in results_dict.items():
            html += f"""
            <tr>
                <td><strong>{strategy}</strong></td>
                <td class="metric-negative">{results.var_95:.2%}</td>
                <td class="metric-negative">{results.expected_shortfall_95:.2%}</td>
                <td>{results.sortino_ratio:.3f}</td>
                <td>{results.downside_deviation:.2%}</td>
            </tr>
            """
        
        html += "</table>"
        return html
    
    def save_results_to_csv(self, results_dict: Dict[str, any], output_dir: str = "results/"):
        """Save detailed results to CSV files"""
        
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        for strategy, results in results_dict.items():
            # Portfolio values and returns
            portfolio_data = pd.DataFrame({
                'portfolio_value': results.portfolio_value,
                'returns': results.returns,
                'drawdowns': results.drawdowns
            })
            portfolio_data.to_csv(f"{output_dir}/{strategy}_portfolio.csv")
            
            # Positions
            if not results.positions.empty:
                results.positions.to_csv(f"{output_dir}/{strategy}_positions.csv")
            
            # Trades
            if not results.trades.empty:
                results.trades.to_csv(f"{output_dir}/{strategy}_trades.csv")
        
        print(f"Detailed results saved to {output_dir}")

def create_model_comparison_plots(model_predictions: Dict[str, np.ndarray],
                                actual_returns: np.ndarray,
                                dates: np.ndarray,
                                save_path: Optional[str] = None):
    """Create plots comparing model predictions to actual returns"""
    
    fig, axes = plt.subplots(2, 2, figsize=(20, 12))
    fig.suptitle('Model Prediction Analysis', fontsize=16, fontweight='bold')
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    # 1. Time series comparison
    ax1 = axes[0, 0]
    ax1.plot(dates, actual_returns, label='Actual', color='black', linewidth=2)
    for i, (model_name, predictions) in enumerate(model_predictions.items()):
        ax1.plot(dates[:len(predictions)], predictions, 
                label=f'{model_name} Prediction', 
                color=colors[i % len(colors)], alpha=0.7, linewidth=1.5)
    ax1.set_title('Predictions vs Actual Returns')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Correlation analysis
    ax2 = axes[0, 1]
    correlations = []
    model_names = []
    for model_name, predictions in model_predictions.items():
        min_len = min(len(predictions), len(actual_returns))
        corr = np.corrcoef(predictions[:min_len], actual_returns[:min_len])[0, 1]
        correlations.append(corr)
        model_names.append(model_name)
    
    bars = ax2.bar(model_names, correlations, color=colors[:len(model_names)], alpha=0.7)
    ax2.set_title('Prediction Correlation with Actual Returns')
    ax2.set_ylabel('Correlation')
    ax2.tick_params(axis='x', rotation=45)
    
    # Add correlation values on bars
    for bar, corr in zip(bars, correlations):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
               f'{corr:.3f}', ha='center', va='bottom')
    
    # 3. Prediction error distribution
    ax3 = axes[1, 0]
    for i, (model_name, predictions) in enumerate(model_predictions.items()):
        min_len = min(len(predictions), len(actual_returns))
        errors = predictions[:min_len] - actual_returns[:min_len]
        ax3.hist(errors, bins=50, alpha=0.6, label=model_name, color=colors[i % len(colors)])
    ax3.set_title('Prediction Error Distribution')
    ax3.set_xlabel('Prediction Error')
    ax3.set_ylabel('Frequency')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Directional accuracy
    ax4 = axes[1, 1]
    accuracies = []
    for model_name, predictions in model_predictions.items():
        min_len = min(len(predictions), len(actual_returns))
        pred_direction = np.sign(predictions[:min_len])
        actual_direction = np.sign(actual_returns[:min_len])
        accuracy = np.mean(pred_direction == actual_direction)
        accuracies.append(accuracy)
    
    bars = ax4.bar(model_names, accuracies, color=colors[:len(model_names)], alpha=0.7)
    ax4.set_title('Directional Accuracy')
    ax4.set_ylabel('Accuracy')
    ax4.set_ylim([0, 1])
    ax4.tick_params(axis='x', rotation=45)
    
    # Add accuracy values on bars
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
               f'{acc:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()