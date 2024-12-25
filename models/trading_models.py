# models/trading_models.py
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

class TrendType(Enum):
    STRONG_UPTREND = "STRONG_UPTREND"
    UPTREND = "UPTREND"
    SIDEWAYS = "SIDEWAYS"
    DOWNTREND = "DOWNTREND"
    STRONG_DOWNTREND = "STRONG_DOWNTREND"

@dataclass
class MarketCondition:
    trend: TrendType
    volatility: float
    volume_profile: str
    support_levels: List[float]
    resistance_levels: List[float]
    rsi: float
    macd: Tuple[float, float]
    key_levels: List[float]

class TradingPatterns:
    """Identifies common trading patterns in price data."""
    
    @staticmethod
    def identify_double_bottom(prices: pd.Series, threshold: float = 0.02) -> bool:
        """Identify double bottom pattern."""
        lows = prices[prices == prices.rolling(10).min()]
        if len(lows) < 2:
            return False
            
        last_two_lows = lows.tail(2)
        if len(last_two_lows) == 2:
            price_diff = abs(last_two_lows.iloc[0] - last_two_lows.iloc[1])
            avg_price = last_two_lows.mean()
            return price_diff / avg_price < threshold
        return False
    
    @staticmethod
    def identify_double_top(prices: pd.Series, threshold: float = 0.02) -> bool:
        """Identify double top pattern."""
        highs = prices[prices == prices.rolling(10).max()]
        if len(highs) < 2:
            return False
            
        last_two_highs = highs.tail(2)
        if len(last_two_highs) == 2:
            price_diff = abs(last_two_highs.iloc[0] - last_two_highs.iloc[1])
            avg_price = last_two_highs.mean()
            return price_diff / avg_price < threshold
        return False

class TechnicalIndicators:
    """Calculate various technical indicators."""
    
    @staticmethod
    def calculate_rsi(prices: pd.Series, periods: int = 14) -> float:
        """Calculate Relative Strength Index."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.iloc[-1]
    
    @staticmethod
    def calculate_macd(prices: pd.Series) -> Tuple[float, float]:
        """Calculate MACD and Signal line."""
        exp1 = prices.ewm(span=12, adjust=False).mean()
        exp2 = prices.ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=9, adjust=False).mean()
        return macd.iloc[-1], signal.iloc[-1]
    
    @staticmethod
    def calculate_bollinger_bands(prices: pd.Series, window: int = 20) -> Tuple[float, float, float]:
        """Calculate Bollinger Bands."""
        sma = prices.rolling(window=window).mean()
        std = prices.rolling(window=window).std()
        upper_band = sma + (std * 2)
        lower_band = sma - (std * 2)
        return upper_band.iloc[-1], sma.iloc[-1], lower_band.iloc[-1]

class VolumeAnalysis:
    """Analyze volume patterns and profiles."""
    
    @staticmethod
    def analyze_volume_profile(volume: pd.Series, price: pd.Series) -> Dict[str, float]:
        """Analyze volume distribution across price levels."""
        # Create price bins
        price_bins = pd.qcut(price, q=10)
        volume_profile = volume.groupby(price_bins).sum()
        
        return {
            "high_volume_price": price_bins[volume == volume.max()].iloc[0],
            "volume_concentration": volume_profile.max() / volume_profile.sum(),
            "volume_trend": volume.iloc[-1] / volume.rolling(20).mean().iloc[-1]
        }

class TradingStrategy:
    """Implements trading strategies combining multiple indicators."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.patterns = TradingPatterns()
        self.indicators = TechnicalIndicators()
        self.volume_analyzer = VolumeAnalysis()

    def analyze_market_condition(self, candles: pd.DataFrame) -> MarketCondition:
        """Analyze current market conditions using multiple indicators."""
        close_prices = candles['close']
        volume = candles['volume']
        
        # Calculate indicators
        rsi = self.indicators.calculate_rsi(close_prices)
        macd_values = self.indicators.calculate_macd(close_prices)
        bb_bands = self.indicators.calculate_bollinger_bands(close_prices)
        volume_profile = self.volume_analyzer.analyze_volume_profile(volume, close_prices)
        
        # Identify support and resistance
        support_levels = self._identify_support_levels(candles)
        resistance_levels = self._identify_resistance_levels(candles)
        
        # Determine trend
        trend = self._determine_trend(candles)
        
        # Calculate volatility
        volatility = close_prices.pct_change().std() * np.sqrt(252)
        
        return MarketCondition(
            trend=trend,
            volatility=volatility,
            volume_profile=self._interpret_volume_profile(volume_profile),
            support_levels=support_levels,
            resistance_levels=resistance_levels,
            rsi=rsi,
            macd=macd_values,
            key_levels=sorted(set(support_levels + resistance_levels))
        )

    def generate_signals(self, candles: pd.DataFrame) -> Dict[str, float]:
        """Generate trading signals based on multiple indicators."""
        market_condition = self.analyze_market_condition(candles)
        
        # Calculate individual signals
        trend_signal = self._calculate_trend_signal(market_condition.trend)
        momentum_signal = self._calculate_momentum_signal(market_condition.rsi, market_condition.macd)
        volume_signal = self._calculate_volume_signal(market_condition.volume_profile)
        
        # Combine signals
        combined_signal = (
            trend_signal * 0.4 +
            momentum_signal * 0.4 +
            volume_signal * 0.2
        )
        
        return {
            "combined_signal": combined_signal,
            "trend_signal": trend_signal,
            "momentum_signal": momentum_signal,
            "volume_signal": volume_signal
        }

    def _identify_support_levels(self, candles: pd.DataFrame, window: int = 20) -> List[float]:
        """Identify key support levels."""
        lows = candles['low']
        support_levels = []
        
        for i in range(window, len(lows)):
            if self._is_support(lows, i):
                support_levels.append(lows[i])
        
        return sorted(set([round(level, 8) for level in support_levels[-5:]]))

    def _identify_resistance_levels(self, candles: pd.DataFrame, window: int = 20) -> List[float]:
        """Identify key resistance levels."""
        highs = candles['high']
        resistance_levels = []
        
        for i in range(window, len(highs)):
            if self._is_resistance(highs, i):
                resistance_levels.append(highs[i])
        
        return sorted(set([round(level, 8) for level in resistance_levels[-5:]]))

    @staticmethod
    def _is_support(prices: pd.Series, index: int, window: int = 3) -> bool:
        """Check if a price point is a support level."""
        for i in range(index - window, index + window + 1):
            if i < 0 or i >= len(prices):
                continue
            if prices[i] < prices[index]:
                return False
        return True

    @staticmethod
    def _is_resistance(prices: pd.Series, index: int, window: int = 3) -> bool:
        """Check if a price point is a resistance level."""
        for i in range(index - window, index + window + 1):
            if i < 0 or i >= len(prices):
                continue
            if prices[i] > prices[index]:
                return False
        return True

    @staticmethod
    def _interpret_volume_profile(volume_profile: Dict[str, float]) -> str:
        """Interpret volume profile analysis."""
        if volume_profile['volume_trend'] > 1.5:
            return "INCREASING_SIGNIFICANTLY"
        elif volume_profile['volume_trend'] > 1.1:
            return "INCREASING"
        elif volume_profile['volume_trend'] < 0.5:
            return "DECREASING_SIGNIFICANTLY"
        elif volume_profile['volume_trend'] < 0.9:
            return "DECREASING"
        return "STABLE"

    @staticmethod
    def _calculate_trend_signal(trend: TrendType) -> float:
        """Convert trend type to numerical signal."""
        trend_signals = {
            TrendType.STRONG_UPTREND: 1.0,
            TrendType.UPTREND: 0.5,
            TrendType.SIDEWAYS: 0.0,
            TrendType.DOWNTREND: -0.5,
            TrendType.STRONG_DOWNTREND: -1.0
        }
        return trend_signals[trend]

    @staticmethod
    def _calculate_momentum_signal(rsi: float, macd: Tuple[float, float]) -> float:
        """Calculate momentum signal from RSI and MACD."""
        rsi_signal = (rsi - 50) / 50  # Normalize RSI to [-1, 1]
        macd_signal = 1 if macd[0] > macd[1] else -1
        return (rsi_signal + macd_signal) / 2

    @staticmethod
    def _calculate_volume_signal(volume_profile: str) -> float:
        """Convert volume profile to numerical signal."""
        volume_signals = {
            "INCREASING_SIGNIFICANTLY": 1.0,
            "INCREASING": 0.5,
            "STABLE": 0.0,
            "DECREASING": -0.5,
            "DECREASING_SIGNIFICANTLY": -1.0
        }
        return volume_signals[volume_profile]