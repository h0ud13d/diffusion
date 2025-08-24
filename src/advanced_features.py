# advanced_features.py - Sophisticated financial feature engineering for diffusion models
import pandas as pd
import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

@dataclass
class FeatureConfig:
    """Configuration for advanced feature engineering"""
    # Technical indicators
    use_ta_indicators: bool = True
    sma_periods: List[int] = None
    ema_periods: List[int] = None
    rsi_periods: List[int] = None
    macd_config: Dict = None
    bb_periods: List[int] = None
    
    # Market microstructure
    use_microstructure: bool = True
    vwap_periods: List[int] = None
    volume_profile_bins: int = 20
    
    # Alternative data
    use_alternative_data: bool = True
    sentiment_features: bool = True
    macro_features: bool = True
    
    # Cross-asset features
    use_cross_asset: bool = True
    correlation_periods: List[int] = None
    beta_periods: List[int] = None
    
    # Time-based features
    use_temporal_features: bool = True
    cyclical_encoding: bool = True
    regime_detection: bool = True
    
    # Risk features
    use_risk_features: bool = True
    var_periods: List[int] = None
    volatility_periods: List[int] = None
    
    def __post_init__(self):
        # Set defaults
        if self.sma_periods is None:
            self.sma_periods = [5, 10, 20, 50, 200]
        if self.ema_periods is None:
            self.ema_periods = [12, 26, 50]
        if self.rsi_periods is None:
            self.rsi_periods = [14, 21]
        if self.macd_config is None:
            self.macd_config = {"fast": 12, "slow": 26, "signal": 9}
        if self.bb_periods is None:
            self.bb_periods = [20]
        if self.vwap_periods is None:
            self.vwap_periods = [5, 20]
        if self.correlation_periods is None:
            self.correlation_periods = [20, 60, 252]
        if self.beta_periods is None:
            self.beta_periods = [60, 252]
        if self.var_periods is None:
            self.var_periods = [20, 60]
        if self.volatility_periods is None:
            self.volatility_periods = [5, 10, 20, 60]

class AdvancedTechnicalIndicators:
    """State-of-the-art technical analysis features"""
    
    @staticmethod
    def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
        """Relative Strength Index with improved calculation"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    @staticmethod
    def calculate_stoch_rsi(prices: pd.Series, period: int = 14, k_period: int = 3, d_period: int = 3) -> Tuple[pd.Series, pd.Series]:
        """Stochastic RSI"""
        rsi = AdvancedTechnicalIndicators.calculate_rsi(prices, period)
        stoch_rsi = (rsi - rsi.rolling(period).min()) / (rsi.rolling(period).max() - rsi.rolling(period).min())
        k_percent = stoch_rsi.rolling(k_period).mean() * 100
        d_percent = k_percent.rolling(d_period).mean()
        return k_percent, d_percent
    
    @staticmethod
    def calculate_williams_r(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Williams %R"""
        highest_high = high.rolling(period).max()
        lowest_low = low.rolling(period).min()
        wr = -100 * (highest_high - close) / (highest_high - lowest_low)
        return wr
    
    @staticmethod
    def calculate_cci(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20) -> pd.Series:
        """Commodity Channel Index"""
        tp = (high + low + close) / 3
        sma_tp = tp.rolling(period).mean()
        mad = tp.rolling(period).apply(lambda x: np.mean(np.abs(x - x.mean())))
        cci = (tp - sma_tp) / (0.015 * mad)
        return cci
    
    @staticmethod
    def calculate_adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Average Directional Index"""
        tr1 = high - low
        tr2 = np.abs(high - close.shift())
        tr3 = np.abs(low - close.shift())
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        
        plus_dm = np.where((high.diff() > low.diff().abs()) & (high.diff() > 0), high.diff(), 0)
        minus_dm = np.where((low.diff().abs() > high.diff()) & (low.diff() < 0), low.diff().abs(), 0)
        
        atr = tr.rolling(period).mean()
        plus_di = 100 * (pd.Series(plus_dm).rolling(period).mean() / atr)
        minus_di = 100 * (pd.Series(minus_dm).rolling(period).mean() / atr)
        
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(period).mean()
        
        return adx
    
    @staticmethod
    def calculate_ichimoku(high: pd.Series, low: pd.Series, close: pd.Series) -> Dict[str, pd.Series]:
        """Ichimoku Cloud components"""
        # Tenkan-sen (Conversion Line)
        tenkan_sen = (high.rolling(9).max() + low.rolling(9).min()) / 2
        
        # Kijun-sen (Base Line)
        kijun_sen = (high.rolling(26).max() + low.rolling(26).min()) / 2
        
        # Senkou Span A (Leading Span A)
        senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(26)
        
        # Senkou Span B (Leading Span B)
        senkou_span_b = ((high.rolling(52).max() + low.rolling(52).min()) / 2).shift(26)
        
        # Chikou Span (Lagging Span)
        chikou_span = close.shift(-26)
        
        return {
            'tenkan_sen': tenkan_sen,
            'kijun_sen': kijun_sen,
            'senkou_span_a': senkou_span_a,
            'senkou_span_b': senkou_span_b,
            'chikou_span': chikou_span
        }

class MarketMicrostructureFeatures:
    """Advanced market microstructure and volume analysis"""
    
    @staticmethod
    def calculate_vwap(high: pd.Series, low: pd.Series, close: pd.Series, 
                      volume: pd.Series, period: int = 20) -> pd.Series:
        """Volume Weighted Average Price"""
        typical_price = (high + low + close) / 3
        return (typical_price * volume).rolling(period).sum() / volume.rolling(period).sum()
    
    @staticmethod
    def calculate_twap(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20) -> pd.Series:
        """Time Weighted Average Price"""
        typical_price = (high + low + close) / 3
        return typical_price.rolling(period).mean()
    
    @staticmethod
    def calculate_volume_profile(prices: pd.Series, volume: pd.Series, bins: int = 20) -> Dict:
        """Volume Profile analysis"""
        price_min, price_max = prices.min(), prices.max()
        price_bins = pd.cut(prices, bins=bins, include_lowest=True)
        
        volume_profile = volume.groupby(price_bins).sum()
        poc = volume_profile.idxmax()  # Point of Control
        
        # Value Area (70% of volume)
        total_volume = volume_profile.sum()
        target_volume = total_volume * 0.7
        sorted_profile = volume_profile.sort_values(ascending=False)
        cumulative_volume = 0
        value_area_bins = []
        
        for bin_range, vol in sorted_profile.items():
            cumulative_volume += vol
            value_area_bins.append(bin_range)
            if cumulative_volume >= target_volume:
                break
        
        return {
            'volume_profile': volume_profile,
            'poc': poc,
            'value_area': value_area_bins,
            'va_high': max([bin_range.right for bin_range in value_area_bins]),
            'va_low': min([bin_range.left for bin_range in value_area_bins])
        }
    
    @staticmethod
    def calculate_order_flow_imbalance(high: pd.Series, low: pd.Series, close: pd.Series,
                                     volume: pd.Series) -> pd.Series:
        """Estimated order flow imbalance"""
        # Simplified order flow proxy using price and volume
        price_change = close.diff()
        volume_weighted_price_change = price_change * volume
        return volume_weighted_price_change.rolling(20).sum()
    
    @staticmethod
    def calculate_money_flow_index(high: pd.Series, low: pd.Series, close: pd.Series,
                                 volume: pd.Series, period: int = 14) -> pd.Series:
        """Money Flow Index"""
        typical_price = (high + low + close) / 3
        money_flow = typical_price * volume
        
        positive_flow = money_flow.where(typical_price.diff() > 0, 0).rolling(period).sum()
        negative_flow = money_flow.where(typical_price.diff() < 0, 0).rolling(period).sum()
        
        money_ratio = positive_flow / negative_flow
        mfi = 100 - (100 / (1 + money_ratio))
        
        return mfi

class RegimeDetection:
    """Advanced market regime detection"""
    
    @staticmethod
    def detect_volatility_regime(returns: pd.Series, lookback: int = 252) -> pd.Series:
        """Volatility regime detection using rolling statistics"""
        vol = returns.rolling(20).std() * np.sqrt(252)  # Annualized volatility
        vol_ma = vol.rolling(lookback).mean()
        vol_std = vol.rolling(lookback).std()
        
        # Define regimes: 0 = Low Vol, 1 = Normal Vol, 2 = High Vol
        regime = pd.Series(1, index=returns.index)  # Default to normal
        regime[vol < (vol_ma - 0.5 * vol_std)] = 0  # Low vol
        regime[vol > (vol_ma + 0.5 * vol_std)] = 2  # High vol
        
        return regime
    
    @staticmethod
    def detect_trend_regime(prices: pd.Series, fast_period: int = 50, slow_period: int = 200) -> pd.Series:
        """Trend regime detection using moving averages"""
        fast_ma = prices.rolling(fast_period).mean()
        slow_ma = prices.rolling(slow_period).mean()
        
        # 0 = Downtrend, 1 = Sideways, 2 = Uptrend
        regime = pd.Series(1, index=prices.index)
        regime[fast_ma > slow_ma * 1.02] = 2  # Uptrend
        regime[fast_ma < slow_ma * 0.98] = 0  # Downtrend
        
        return regime
    
    @staticmethod
    def detect_mean_reversion_regime(returns: pd.Series, lookback: int = 60) -> pd.Series:
        """Mean reversion vs momentum regime detection"""
        # Calculate autocorrelation
        autocorr = returns.rolling(lookback).apply(lambda x: x.autocorr(lag=1))
        
        # Negative autocorr = mean reversion, positive = momentum
        regime = pd.Series(1, index=returns.index)  # Default neutral
        regime[autocorr < -0.1] = 0  # Mean reversion
        regime[autocorr > 0.1] = 2   # Momentum
        
        return regime

class AlternativeDataFeatures:
    """Features from alternative data sources"""
    
    @staticmethod
    def create_cyclical_features(dates: pd.DatetimeIndex) -> pd.DataFrame:
        """Cyclical time-based features"""
        features = pd.DataFrame(index=dates)
        
        # Day of week
        features['dow_sin'] = np.sin(2 * np.pi * dates.dayofweek / 7)
        features['dow_cos'] = np.cos(2 * np.pi * dates.dayofweek / 7)
        
        # Month of year
        features['moy_sin'] = np.sin(2 * np.pi * dates.month / 12)
        features['moy_cos'] = np.cos(2 * np.pi * dates.month / 12)
        
        # Day of month
        features['dom_sin'] = np.sin(2 * np.pi * dates.day / 31)
        features['dom_cos'] = np.cos(2 * np.pi * dates.day / 31)
        
        # Quarter
        features['quarter_sin'] = np.sin(2 * np.pi * dates.quarter / 4)
        features['quarter_cos'] = np.cos(2 * np.pi * dates.quarter / 4)
        
        # Market anomalies
        features['is_monday'] = (dates.dayofweek == 0).astype(int)
        features['is_friday'] = (dates.dayofweek == 4).astype(int)
        features['is_month_end'] = dates.is_month_end.astype(int)
        features['is_month_start'] = dates.is_month_start.astype(int)
        features['is_quarter_end'] = dates.is_quarter_end.astype(int)
        
        return features
    
    @staticmethod
    def create_macro_proxies(dates: pd.DatetimeIndex) -> pd.DataFrame:
        """Macro economic proxy features (simplified)"""
        features = pd.DataFrame(index=dates)
        
        # Economic calendar proxies (simplified)
        # In practice, these would come from real economic data APIs
        np.random.seed(42)  # For reproducibility
        features['vix_proxy'] = np.random.normal(20, 5, len(dates))  # VIX proxy
        features['yield_curve_proxy'] = np.random.normal(2.5, 1, len(dates))  # Yield curve proxy
        features['dxy_proxy'] = np.random.normal(100, 10, len(dates))  # Dollar index proxy
        features['oil_proxy'] = np.random.normal(70, 15, len(dates))  # Oil price proxy
        features['gold_proxy'] = np.random.normal(1800, 200, len(dates))  # Gold price proxy
        
        return features

class AdvancedFeatureEngineer:
    """Main feature engineering class combining all techniques"""
    
    def __init__(self, config: FeatureConfig):
        self.config = config
        self.ta_indicators = AdvancedTechnicalIndicators()
        self.microstructure = MarketMicrostructureFeatures()
        self.regime_detector = RegimeDetection()
        self.alt_data = AlternativeDataFeatures()
    
    def engineer_features(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Main feature engineering pipeline"""
        result_df = df.copy()
        
        # Ensure required columns
        required_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Convert Date to datetime
        result_df['Date'] = pd.to_datetime(result_df['Date'])
        result_df.set_index('Date', inplace=True)
        
        # Basic price features
        result_df[f'returns_{symbol}'] = np.log(result_df['Adj Close']).diff().fillna(0)
        result_df[f'log_volume_{symbol}'] = np.log(result_df['Volume'] + 1)
        
        # Technical indicators
        if self.config.use_ta_indicators:
            result_df = self._add_technical_indicators(result_df, symbol)
        
        # Market microstructure
        if self.config.use_microstructure:
            result_df = self._add_microstructure_features(result_df, symbol)
        
        # Temporal features
        if self.config.use_temporal_features:
            result_df = self._add_temporal_features(result_df, symbol)
        
        # Risk features
        if self.config.use_risk_features:
            result_df = self._add_risk_features(result_df, symbol)
        
        # Alternative data
        if self.config.use_alternative_data:
            result_df = self._add_alternative_features(result_df, symbol)
        
        # Clean up
        result_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        result_df.fillna(method='ffill', inplace=True)
        result_df.fillna(0, inplace=True)
        
        # Reset index
        result_df.reset_index(inplace=True)
        
        return result_df
    
    def _add_technical_indicators(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Add technical analysis indicators"""
        high, low, close, volume = df['High'], df['Low'], df['Adj Close'], df['Volume']
        
        # Moving averages
        for period in self.config.sma_periods:
            df[f'sma_{period}_{symbol}'] = close.rolling(period).mean()
            df[f'price_to_sma_{period}_{symbol}'] = close / df[f'sma_{period}_{symbol}'] - 1
        
        for period in self.config.ema_periods:
            df[f'ema_{period}_{symbol}'] = close.ewm(span=period).mean()
            df[f'price_to_ema_{period}_{symbol}'] = close / df[f'ema_{period}_{symbol}'] - 1
        
        # RSI variants
        for period in self.config.rsi_periods:
            df[f'rsi_{period}_{symbol}'] = self.ta_indicators.calculate_rsi(close, period)
        
        # Stochastic RSI
        k, d = self.ta_indicators.calculate_stoch_rsi(close)
        df[f'stoch_rsi_k_{symbol}'] = k
        df[f'stoch_rsi_d_{symbol}'] = d
        
        # Williams %R
        df[f'williams_r_{symbol}'] = self.ta_indicators.calculate_williams_r(high, low, close)
        
        # CCI
        df[f'cci_{symbol}'] = self.ta_indicators.calculate_cci(high, low, close)
        
        # ADX
        df[f'adx_{symbol}'] = self.ta_indicators.calculate_adx(high, low, close)
        
        # MACD
        config = self.config.macd_config
        ema_fast = close.ewm(span=config['fast']).mean()
        ema_slow = close.ewm(span=config['slow']).mean()
        df[f'macd_{symbol}'] = ema_fast - ema_slow
        df[f'macd_signal_{symbol}'] = df[f'macd_{symbol}'].ewm(span=config['signal']).mean()
        df[f'macd_histogram_{symbol}'] = df[f'macd_{symbol}'] - df[f'macd_signal_{symbol}']
        
        # Bollinger Bands
        for period in self.config.bb_periods:
            sma = close.rolling(period).mean()
            std = close.rolling(period).std()
            df[f'bb_upper_{period}_{symbol}'] = sma + 2 * std
            df[f'bb_lower_{period}_{symbol}'] = sma - 2 * std
            df[f'bb_position_{period}_{symbol}'] = (close - sma) / (2 * std)
            df[f'bb_width_{period}_{symbol}'] = (df[f'bb_upper_{period}_{symbol}'] - 
                                                df[f'bb_lower_{period}_{symbol}']) / sma
        
        # Ichimoku
        ichimoku = self.ta_indicators.calculate_ichimoku(high, low, close)
        for name, series in ichimoku.items():
            df[f'ichimoku_{name}_{symbol}'] = series
        
        return df
    
    def _add_microstructure_features(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Add market microstructure features"""
        high, low, close, volume = df['High'], df['Low'], df['Adj Close'], df['Volume']
        
        # VWAP
        for period in self.config.vwap_periods:
            df[f'vwap_{period}_{symbol}'] = self.microstructure.calculate_vwap(high, low, close, volume, period)
            df[f'price_to_vwap_{period}_{symbol}'] = close / df[f'vwap_{period}_{symbol}'] - 1
        
        # TWAP
        for period in self.config.vwap_periods:
            df[f'twap_{period}_{symbol}'] = self.microstructure.calculate_twap(high, low, close, period)
        
        # Money Flow Index
        df[f'mfi_{symbol}'] = self.microstructure.calculate_money_flow_index(high, low, close, volume)
        
        # Order flow proxy
        df[f'order_flow_{symbol}'] = self.microstructure.calculate_order_flow_imbalance(high, low, close, volume)
        
        # Volume features
        df[f'volume_sma_{symbol}'] = volume.rolling(20).mean()
        df[f'volume_ratio_{symbol}'] = volume / df[f'volume_sma_{symbol}']
        
        return df
    
    def _add_temporal_features(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Add time-based features"""
        if self.config.cyclical_encoding:
            cyclical_features = self.alt_data.create_cyclical_features(df.index)
            for col in cyclical_features.columns:
                df[f'{col}_{symbol}'] = cyclical_features[col]
        
        if self.config.regime_detection:
            returns = df[f'returns_{symbol}']
            df[f'vol_regime_{symbol}'] = self.regime_detector.detect_volatility_regime(returns)
            df[f'trend_regime_{symbol}'] = self.regime_detector.detect_trend_regime(df['Adj Close'])
            df[f'mean_reversion_regime_{symbol}'] = self.regime_detector.detect_mean_reversion_regime(returns)
        
        return df
    
    def _add_risk_features(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Add risk-based features"""
        returns = df[f'returns_{symbol}']
        close = df['Adj Close']
        
        # Volatility measures
        for period in self.config.volatility_periods:
            df[f'volatility_{period}d_{symbol}'] = returns.rolling(period).std() * np.sqrt(252)
            df[f'garch_vol_{period}d_{symbol}'] = returns.rolling(period).apply(
                lambda x: np.sqrt(np.mean(x**2))  # Simplified GARCH proxy
            ) * np.sqrt(252)
        
        # VaR estimates
        for period in self.config.var_periods:
            df[f'var_95_{period}d_{symbol}'] = returns.rolling(period).quantile(0.05)
            df[f'var_99_{period}d_{symbol}'] = returns.rolling(period).quantile(0.01)
        
        # Drawdown features
        cumulative = (1 + returns).cumprod()
        peak = cumulative.expanding().max()
        df[f'drawdown_{symbol}'] = (cumulative / peak - 1)
        df[f'time_in_drawdown_{symbol}'] = (df[f'drawdown_{symbol}'] < 0).astype(int)
        
        # Skewness and kurtosis
        for period in [20, 60]:
            df[f'skewness_{period}d_{symbol}'] = returns.rolling(period).skew()
            df[f'kurtosis_{period}d_{symbol}'] = returns.rolling(period).kurt()
        
        return df
    
    def _add_alternative_features(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Add alternative data features"""
        if self.config.macro_features:
            macro_features = self.alt_data.create_macro_proxies(df.index)
            for col in macro_features.columns:
                df[f'{col}_{symbol}'] = macro_features[col]
        
        return df
    
    def add_cross_asset_features(self, merged_df: pd.DataFrame, assets: List[str]) -> pd.DataFrame:
        """Add cross-asset correlation and beta features"""
        if not self.config.use_cross_asset:
            return merged_df
        
        return_cols = [f'returns_{asset}' for asset in assets]
        
        # Rolling correlations
        for i, asset1 in enumerate(assets):
            for j, asset2 in enumerate(assets[i+1:], i+1):
                for period in self.config.correlation_periods:
                    corr_col = f'corr_{period}d_{asset1}_{asset2}'
                    merged_df[corr_col] = merged_df[f'returns_{asset1}'].rolling(period).corr(
                        merged_df[f'returns_{asset2}']
                    ).fillna(0)
        
        # Rolling betas (first asset as market proxy)
        if len(assets) > 1:
            market_returns = merged_df[f'returns_{assets[0]}']
            for asset in assets[1:]:
                for period in self.config.beta_periods:
                    covariance = merged_df[f'returns_{asset}'].rolling(period).cov(market_returns)
                    market_variance = market_returns.rolling(period).var()
                    beta_col = f'beta_{period}d_{asset}_to_{assets[0]}'
                    merged_df[beta_col] = (covariance / market_variance).fillna(0)
        
        return merged_df