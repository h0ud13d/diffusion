#!/usr/bin/env python3
import json
import threading
import webbrowser
from datetime import datetime
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler
import pandas as pd
import numpy as np
from typing import Dict, Any
import pickle
import glob
from pathlib import Path
    


class PortfolioDashboardHandler(BaseHTTPRequestHandler):
    def __init__(self, portfolio_results, *args, **kwargs):
        self.portfolio_results = portfolio_results
        super().__init__(*args, **kwargs)
    
    def do_GET(self):
        if self.path == '/' or self.path == '/index.html':
            self.serve_dashboard()
        elif self.path == '/api/data':
            self.serve_data()
        elif self.path.startswith('/static/'):
            self.serve_static()
        else:
            self.send_error(404)
    
    def serve_dashboard(self):
        html_content = self.generate_dashboard_html()
        
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        self.wfile.write(html_content.encode())
    
    def serve_data(self):
        data = self.prepare_dashboard_data()
        
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(data, default=str).encode())
    
    def serve_static(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/plain')
        self.end_headers()
        self.wfile.write(b'/* Static content */')
    
    def prepare_dashboard_data(self):
        results = self.portfolio_results
        
        # Portfolio value series
        portfolio_values = results['portfolio_value'].tolist()
        
        start_date = datetime(2020, 1, 1) 
        dates = [(start_date.timestamp() + i * 86400) * 1000 for i in range(len(portfolio_values))]
        
        perf = results['performance_metrics']
        risk = results['risk_metrics']
        
        returns = results['portfolio_returns']
        returns_hist = np.histogram(returns, bins=50)
        
        peak = np.maximum.accumulate(portfolio_values)
        drawdown = [(val - pk) / pk * 100 for val, pk in zip(portfolio_values, peak)]
        
        data = {
            'portfolio_value': {
                'dates': dates,
                'values': portfolio_values
            },
            'drawdown': {
                'dates': dates,
                'values': drawdown
            },
            'returns_distribution': {
                'bins': returns_hist[1][:-1].tolist(),
                'counts': returns_hist[0].tolist()
            },
            'performance_metrics': {
                'total_return': perf.total_return,
                'annualized_return': perf.annualized_return,
                'sharpe_ratio': perf.sharpe_ratio,
                'sortino_ratio': perf.sortino_ratio,
                'win_rate': perf.win_rate
            },
            'risk_metrics': {
                'max_drawdown': risk.max_drawdown,
                'volatility': risk.volatility,
                'var_1d': risk.var_1d,
                'expected_shortfall': risk.expected_shortfall,
                'drawdown_duration': risk.drawdown_duration
            },
            'stress_test_results': results['stress_test_results'],
            'transaction_costs': {
                'total_costs': sum(results['transaction_costs']),
                'cost_ratio': sum(results['transaction_costs']) / results['config'].initial_capital
            }
        }

        inpaint = results.get('inpainting_overlay')
        if inpaint is not None:
            try:
                inpaint_dates = [datetime.fromisoformat(d).timestamp() * 1000 for d in inpaint['dates']]
            except Exception:
                inpaint_dates = list(range(len(inpaint['nvda_actual'])))
            data['inpainting_overlay'] = {
                'dates': inpaint_dates,
                'nvda_actual': inpaint['nvda_actual'],
                'nvda_pred': inpaint['nvda_pred'],
                'msft_actual': inpaint['msft_actual']
            }

        return data
    
    def generate_dashboard_html(self):
        """Generate the dashboard HTML"""
        return """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Portfolio Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chartjs-adapter-date-fns/dist/chartjs-adapter-date-fns.bundle.min.js"></script>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #fafafa;
            color: #333;
            line-height: 1.6;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }
        
        .header {
            text-align: center;
            margin-bottom: 30px;
            padding: 20px 0;
            border-bottom: 2px solid #e0e0e0;
        }
        
        .header h1 {
            font-size: 2.5em;
            font-weight: 300;
            color: #2c3e50;
            margin-bottom: 10px;
        }
        
        .timestamp {
            color: #7f8c8d;
            font-size: 0.9em;
        }
        
        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        
        .card {
            background: white;
            border-radius: 8px;
            padding: 20px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.05);
            border: 1px solid #e0e0e0;
        }
        
        .card h3 {
            font-size: 1.2em;
            margin-bottom: 15px;
            color: #2c3e50;
            border-bottom: 2px solid #3498db;
            padding-bottom: 8px;
        }
        
        .metric {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 10px;
            padding: 8px 0;
            border-bottom: 1px solid #f0f0f0;
        }
        
        .metric:last-child {
            border-bottom: none;
        }
        
        .metric-label {
            font-weight: 500;
            color: #555;
        }
        
        .metric-value {
            font-weight: 600;
            font-size: 1.1em;
        }
        
        .positive { color: #27ae60; }
        .negative { color: #e74c3c; }
        .neutral { color: #7f8c8d; }
        
        .chart-container {
            position: relative;
            height: 300px;
            margin: 15px 0;
        }
        
        .large-chart {
            height: 400px;
        }
        
        .loading {
            text-align: center;
            padding: 40px;
            color: #7f8c8d;
        }
        
        .status-indicator {
            display: inline-block;
            width: 12px;
            height: 12px;
            border-radius: 50%;
            margin-right: 8px;
        }
        
        .status-good { background: #27ae60; }
        .status-warning { background: #f39c12; }
        .status-danger { background: #e74c3c; }
        
        @media (max-width: 768px) {
            .container { padding: 10px; }
            .header h1 { font-size: 2em; }
            .grid { grid-template-columns: 1fr; }
        }
    </style>
</head>
<body>
    <div class="container">
        <header class="header">
            <h1>Portfolio Performance Dashboard</h1>
            <div class="timestamp">Generated: <span id="timestamp"></span></div>
        </header>
        
        <div id="loading" class="loading">
            <div>Loading portfolio data...</div>
        </div>
        
        <div id="dashboard" style="display: none;">
            <!-- Performance Overview -->
            <div class="grid">
                <div class="card">
                    <h3>Performance Overview</h3>
                    <div class="metric">
                        <span class="metric-label">Total Return</span>
                        <span class="metric-value" id="total-return">--</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Annualized Return</span>
                        <span class="metric-value" id="annual-return">--</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Sharpe Ratio</span>
                        <span class="metric-value" id="sharpe-ratio">--</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Win Rate</span>
                        <span class="metric-value" id="win-rate">--</span>
                    </div>
                </div>
                
                <div class="card">
                    <h3>Risk Metrics</h3>
                    <div class="metric">
                        <span class="metric-label">Maximum Drawdown</span>
                        <span class="metric-value" id="max-drawdown">--</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Volatility</span>
                        <span class="metric-value" id="volatility">--</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Daily VaR (5%)</span>
                        <span class="metric-value" id="var-5">--</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Expected Shortfall</span>
                        <span class="metric-value" id="expected-shortfall">--</span>
                    </div>
                </div>
                
                <div class="card">
                    <h3>Trading Costs</h3>
                    <div class="metric">
                        <span class="metric-label">Total Costs</span>
                        <span class="metric-value" id="total-costs">--</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Cost Ratio</span>
                        <span class="metric-value" id="cost-ratio">--</span>
                    </div>
                </div>
                
                <div class="card">
                    <h3>Risk Assessment</h3>
                    <div class="metric">
                        <span class="metric-label">
                            <span class="status-indicator" id="drawdown-status"></span>
                            Drawdown Risk
                        </span>
                        <span class="metric-value" id="drawdown-assessment">--</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">
                            <span class="status-indicator" id="performance-status"></span>
                            Performance
                        </span>
                        <span class="metric-value" id="performance-assessment">--</span>
                    </div>
                </div>
            </div>
            
            <!-- Charts -->
            <div class="grid">
                <div class="card">
                    <h3>Portfolio Value Evolution</h3>
                    <div class="chart-container large-chart">
                        <canvas id="portfolioChart"></canvas>
                    </div>
                </div>
                
                <div class="card">
                    <h3>Drawdown Analysis</h3>
                    <div class="chart-container">
                        <canvas id="drawdownChart"></canvas>
                    </div>
                </div>
            </div>
            
            <div class="card" id="inpaintingCard" style="display: none;">
                <h3>Inpainting: NVDA vs Inpainted vs MSFT</h3>
                <div class="chart-container large-chart">
                    <canvas id="inpaintingChart"></canvas>
                </div>
            </div>

            <div class="card">
                <h3>Returns Distribution</h3>
                <div class="chart-container">
                    <canvas id="returnsChart"></canvas>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        // Format currency
        function formatCurrency(value) {
            return new Intl.NumberFormat('en-US', {
                style: 'currency',
                currency: 'USD',
                minimumFractionDigits: 0,
                maximumFractionDigits: 0
            }).format(value);
        }
        
        // Format percentage
        function formatPercent(value) {
            return (value * 100).toFixed(2) + '%';
        }
        
        // Format number
        function formatNumber(value, decimals = 3) {
            return value.toFixed(decimals);
        }
        
        // Get color class for value
        function getColorClass(value) {
            if (value > 0) return 'positive';
            if (value < 0) return 'negative';
            return 'neutral';
        }
        
        // Set timestamp
        document.getElementById('timestamp').textContent = new Date().toLocaleString();
        
        // Load and display data
        fetch('/api/data')
            .then(response => response.json())
            .then(data => {
                console.log('Portfolio data loaded:', data);
                // Hide loading
                document.getElementById('loading').style.display = 'none';
                document.getElementById('dashboard').style.display = 'block';
                
                // Update performance metrics
                const perf = data.performance_metrics;
                document.getElementById('total-return').textContent = formatPercent(perf.total_return);
                document.getElementById('total-return').className = 'metric-value ' + getColorClass(perf.total_return);
                
                document.getElementById('annual-return').textContent = formatPercent(perf.annualized_return);
                document.getElementById('annual-return').className = 'metric-value ' + getColorClass(perf.annualized_return);
                
                document.getElementById('sharpe-ratio').textContent = formatNumber(perf.sharpe_ratio);
                document.getElementById('sharpe-ratio').className = 'metric-value ' + getColorClass(perf.sharpe_ratio);
                
                document.getElementById('win-rate').textContent = formatPercent(perf.win_rate);
                
                // Update risk metrics
                const risk = data.risk_metrics;
                document.getElementById('max-drawdown').textContent = formatPercent(risk.max_drawdown);
                document.getElementById('max-drawdown').className = 'metric-value negative';
                
                document.getElementById('volatility').textContent = formatPercent(risk.volatility);
                document.getElementById('var-5').textContent = formatPercent(risk.var_1d);
                document.getElementById('expected-shortfall').textContent = formatPercent(risk.expected_shortfall);
                
                // Update trading costs
                const costs = data.transaction_costs;
                document.getElementById('total-costs').textContent = formatCurrency(costs.total_costs);
                document.getElementById('cost-ratio').textContent = formatPercent(costs.cost_ratio);
                
                // Risk assessment
                const drawdownOk = risk.max_drawdown < 0.15;
                const performanceOk = perf.sharpe_ratio > 0.5;
                
                document.getElementById('drawdown-status').className = 'status-indicator ' + 
                    (drawdownOk ? 'status-good' : 'status-danger');
                document.getElementById('drawdown-assessment').textContent = drawdownOk ? 'Within Limits' : 'Exceeded';
                
                document.getElementById('performance-status').className = 'status-indicator ' + 
                    (performanceOk ? 'status-good' : (perf.sharpe_ratio > 0 ? 'status-warning' : 'status-danger'));
                document.getElementById('performance-assessment').textContent = 
                    performanceOk ? 'Good' : (perf.sharpe_ratio > 0 ? 'Fair' : 'Poor');
                
                // Create charts with error handling
                try {
                    console.log('Creating portfolio chart...');
                    createPortfolioChart(data.portfolio_value);
                    console.log('Portfolio chart created');
                } catch (e) {
                    console.error('Error creating portfolio chart:', e);
                }
                
                try {
                    console.log('Creating drawdown chart...');
                    createDrawdownChart(data.drawdown);
                    console.log('Drawdown chart created');
                } catch (e) {
                    console.error('Error creating drawdown chart:', e);
                }
                
                // Inpainting overlay (if present)
                try {
                    if (data.inpainting_overlay) {
                        document.getElementById('inpaintingCard').style.display = 'block';
                        createInpaintingChart(data.inpainting_overlay);
                        console.log('Inpainting chart created');
                    }
                } catch (e) {
                    console.error('Error creating inpainting chart:', e);
                }

                try {
                    console.log('Creating returns chart...');
                    createReturnsChart(data.returns_distribution);
                    console.log('Returns chart created');
                } catch (e) {
                    console.error('Error creating returns chart:', e);
                }
            })
            .catch(error => {
                console.error('Error loading data:', error);
                document.getElementById('loading').innerHTML = 
                    '<div style="color: #e74c3c;">Error loading portfolio data</div>';
            });
        
        function createPortfolioChart(data) {
            const ctx = document.getElementById('portfolioChart').getContext('2d');
            
            // Create simple labels (Day 1, Day 2, etc.)
            const labels = data.values.map((_, i) => `Day ${i + 1}`);
            
            new Chart(ctx, {
                type: 'line',
                data: {
                    labels: labels,
                    datasets: [{
                        label: 'Portfolio Value',
                        data: data.values,
                        borderColor: '#3498db',
                        backgroundColor: 'rgba(52, 152, 219, 0.1)',
                        borderWidth: 2,
                        fill: true,
                        tension: 0.1
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: { display: false }
                    },
                    scales: {
                        x: {
                            grid: { display: false },
                            ticks: {
                                maxTicksLimit: 10
                            }
                        },
                        y: {
                            beginAtZero: false,
                            ticks: {
                                callback: function(value) {
                                    return formatCurrency(value);
                                }
                            }
                        }
                    },
                    interaction: {
                        intersect: false,
                        mode: 'index'
                    }
                }
            });
        }
        
        function createDrawdownChart(data) {
            const ctx = document.getElementById('drawdownChart').getContext('2d');
            
            const labels = data.values.map((_, i) => `Day ${i + 1}`);
            
            new Chart(ctx, {
                type: 'line',
                data: {
                    labels: labels,
                    datasets: [{
                        label: 'Drawdown',
                        data: data.values,
                        borderColor: '#e74c3c',
                        backgroundColor: 'rgba(231, 76, 60, 0.1)',
                        borderWidth: 1,
                        fill: true
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: { display: false }
                    },
                    scales: {
                        x: {
                            grid: { display: false },
                            ticks: {
                                maxTicksLimit: 8
                            }
                        },
                        y: {
                            ticks: {
                                callback: function(value) {
                                    return value.toFixed(1) + '%';
                                }
                            }
                        }
                    }
                }
            });
        }
        
        function createReturnsChart(data) {
            const ctx = document.getElementById('returnsChart').getContext('2d');
            new Chart(ctx, {
                type: 'bar',
                data: {
                    labels: data.bins.map(b => (b * 100).toFixed(2) + '%'),
                    datasets: [{
                        label: 'Frequency',
                        data: data.counts,
                        backgroundColor: 'rgba(52, 152, 219, 0.7)',
                        borderColor: '#3498db',
                        borderWidth: 1
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: { display: false }
                    },
                    scales: {
                        x: { 
                            title: { display: true, text: 'Daily Returns' },
                            grid: { display: false },
                            ticks: {
                                maxTicksLimit: 10
                            }
                        },
                        y: { 
                            title: { display: true, text: 'Frequency' },
                            beginAtZero: true
                        }
                    }
                }
            });
        }
        
        function createInpaintingChart(data) {
            const ctx = document.getElementById('inpaintingChart').getContext('2d');
            const labels = data.dates; // timestamps (ms) or indices
            new Chart(ctx, {
                type: 'line',
                data: {
                    labels: labels,
                    datasets: [
                        {
                            label: 'NVDA Actual',
                            data: data.nvda_actual,
                            borderColor: '#FFA500',
                            backgroundColor: 'rgba(255,165,0,0.08)',
                            borderWidth: 2,
                            fill: false
                        },
                        {
                            label: 'NVDA Inpainted',
                            data: data.nvda_pred,
                            borderColor: 'rgba(255,165,0,0.6)',
                            backgroundColor: 'rgba(255,165,0,0.06)',
                            borderDash: [6, 4],
                            borderWidth: 2,
                            fill: false
                        },
                        {
                            label: 'MSFT Actual',
                            data: data.msft_actual,
                            borderColor: '#0000FF',
                            backgroundColor: 'rgba(0,0,255,0.06)',
                            borderWidth: 1.5,
                            fill: false,
                            tension: 0.2,
                            pointRadius: 0
                        }
                    ]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: { position: 'top' }
                    },
                    scales: {
                        x: {
                            type: 'time',
                            time: { unit: 'day' },
                            grid: { display: false }
                        },
                        y: {
                            ticks: {
                                callback: function(value) {
                                    return (value * 100).toFixed(2) + '%';
                                }
                            },
                            grid: { color: 'rgba(0,0,0,0.05)' }
                        }
                    }
                }
            });
        }
    </script>
</body>
</html>"""

    def log_message(self, format, *args):
        """Suppress default logging"""
        pass


def create_portfolio_server(portfolio_results, port=8080):
    
    def handler(*args, **kwargs):
        return PortfolioDashboardHandler(portfolio_results, *args, **kwargs)
    
    server = HTTPServer(('localhost', port), handler)
    return server


def start_portfolio_dashboard(portfolio_results, port=8080, open_browser=True):
    print(f"Starting Portfolio Dashboard Server...")
    print(f"URL: http://localhost:{port}")
    print(f"Dashboard ready with portfolio analytics")
    print(f"Press Ctrl+C to stop the server")
    
    server = create_portfolio_server(portfolio_results, port)
    
    if open_browser:
        threading.Timer(1.0, lambda: webbrowser.open(f'http://localhost:{port}')).start()
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Shutting down dashboard server...")
        server.shutdown()


if __name__ == "__main__":
    data_dir = Path(__file__).parent.parent / "data"
    
    result_patterns = [
        "backtest_results_*.pkl",
        "portfolio_results_*.pkl", 
        "latest_results.pkl"
    ]
    
    latest_results = None
    for pattern in result_patterns:
        files = list(data_dir.glob(pattern))
        if files:
            latest_file = max(files, key=lambda x: x.stat().st_mtime)
            try:
                with open(latest_file, 'rb') as f:
                    latest_results = pickle.load(f)
                print(f"Loaded results from: {latest_file}")
                break
            except Exception as e:
                print(f"Error loading {latest_file}: {e}")
                continue
    
    if latest_results is None:
        print("No saved backtest results found in data/ directory.")
        print("Please run backtest.py first to generate results, or save results with:")
        print("  pickle.dump(backtest_results, open('data/latest_results.pkl', 'wb'))")
        import sys
        sys.exit(1)
    
    print("Starting dashboard with most recent backtest results...")
    start_portfolio_dashboard(latest_results)
