# utils/data_handlers.py
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import logging
from enum import Enum

class TimeFrame(Enum):
    M1 = "1m"
    M5 = "5m"
    M15 = "15m"
    M30 = "30m"
    H1 = "1h"
    H4 = "4h"
    D1 = "1d"
    W1 = "1w"

class DataHandler:
    def __init__(self, logger=None):
        self.logger = logger or logging.getLogger(__name__)
        self.cached_data = {}
        self.last_update = {}

    def process_raw_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Process raw market data and add basic technical indicators."""
        df = data.copy()
        
        # Ensure required columns exist
        required_columns = ['open', 'high', 'low', 'close', 'volume', 'timestamp']
        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"Missing required columns. Required: {required_columns}")

        # Convert timestamp if needed
        if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            df['timestamp'] = pd.to_datetime(df['timestamp'])

        # Sort by timestamp
        df = df.sort_values('timestamp')

        # Add basic technical indicators
        df = self._add_moving_averages(df)
        df = self._add_volatility_indicators(df)
        df = self._add_momentum_indicators(df)
        df = self._add_volume_indicators(df)

        return df

    def resample_timeframe(self, data: pd.DataFrame, target_timeframe: TimeFrame) -> pd.DataFrame:
        """Resample data to different timeframe."""
        df = data.copy()
        
        # Convert timeframe enum to pandas offset string
        offset_map = {
            TimeFrame.M1: '1T',
            TimeFrame.M5: '5T',
            TimeFrame.M15: '15T',
            TimeFrame.M30: '30T',
            TimeFrame.H1: '1H',
            TimeFrame.H4: '4H',
            TimeFrame.D1: '1D',
            TimeFrame.W1: '1W'
        }
        
        offset = offset_map[target_timeframe]
        
        # Resample OHLCV data
        resampled = df.resample(offset, on='timestamp').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()
        
        return self.process_raw_data(resampled)

    def calculate_market_profile(self, data: pd.DataFrame, value_area_pct: float = 0.70) -> Dict:
        """Calculate market profile and value area."""
        df = data.copy()
        
        # Calculate price levels
        price_increment = (df['high'].max() - df['low'].min()) / 100
        price_levels = np.arange(df['low'].min(), df['high'].max(), price_increment)
        
        # Build volume profile
        volume_profile = []
        for price in price_levels:
            volume_at_price = df[
                (df['low'] <= price) & 
                (df['high'] >= price)
            ]['volume'].sum()
            volume_profile.append({
                'price': price,
                'volume': volume_at_price
            })
        
        # Calculate value area
        total_volume = sum(v['volume'] for v in volume_profile)
        target_volume = total_volume * value_area_pct
        
        sorted_profile = sorted(volume_profile, key=lambda x: x['volume'], reverse=True)
        cumulative_volume = 0
        value_area_levels = []
        
        for level in sorted_profile:
            cumulative_volume += level['volume']
            value_area_levels.append(level['price'])
            if cumulative_volume >= target_volume:
                break
                
        return {
            'profile': volume_profile,
            'value_area_high': max(value_area_levels),
            'value_area_low': min(value_area_levels),
            'poc': sorted_profile[0]['price']  # Point of Control
        }

    def detect_orderblocks(self, data: pd.DataFrame, lookback: int = 10) -> List[Dict]:
        """Detect potential orderblock zones."""
        df = data.copy()
        orderblocks = []
        
        for i in range(lookback, len(df)):
            section = df.iloc[i-lookback:i+1]
            
            # Bullish orderblock
            if self._is_bullish_orderblock(section):
                orderblocks.append({
                    'type': 'bullish',
                    'high': section.iloc[-2]['high'],
                    'low': section.iloc[-2]['low'],
                    'timestamp': section.iloc[-2]['timestamp']
                })
                
            # Bearish orderblock
            if self._is_bearish_orderblock(section):
                orderblocks.append({
                    'type': 'bearish',
                    'high': section.iloc[-2]['high'],
                    'low': section.iloc[-2]['low'],
                    'timestamp': section.iloc[-2]['timestamp']
                })
                
        return orderblocks

    def identify_key_levels(self, data: pd.DataFrame) -> Dict[str, List[float]]:
        """Identify key price levels including support, resistance, and liquidity zones."""
        df = data.copy()
        
        return {
            'support': self._find_support_levels(df),
            'resistance': self._find_resistance_levels(df),
            'liquidity_above': self._find_liquidity_zones(df, 'above'),
            'liquidity_below': self._find_liquidity_zones(df, 'below')
        }

    def calculate_session_statistics(self, data: pd.DataFrame) -> Dict:
        """Calculate statistics for different trading sessions."""
        df = data.copy()
        
        # Add session information
        df['session'] = df['timestamp'].apply(self._determine_forex_session)
        
        stats = {}
        for session in ['asian', 'london', 'new_york']:
            session_data = df[df['session'] == session]
            stats[session] = {
                'average_range': session_data['high'].max() - session_data['low'].min(),
                'average_volume': session_data['volume'].mean(),
                'most_active_hours': self._find_most_active_hours(session_data)
            }
            
        return stats

    def _add_moving_averages(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add various moving averages to the dataframe."""
        for period in [8, 20, 50, 200]:
            df[f'sma_{period}'] = df['close'].rolling(window=period).mean()
            df[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()
        return df

    def _add_volatility_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volatility indicators."""
        # ATR
        df['atr'] = self._calculate_atr(df)
        
        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(window=20).mean()
        df['bb_std'] = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + (df['bb_std'] * 2)
        df['bb_lower'] = df['bb_middle'] - (df['bb_std'] * 2)
        
        return df

    def _add_momentum_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add momentum indicators."""
        # RSI
        df['rsi'] = self._calculate_rsi(df['close'])
        
        # MACD
        df['macd'], df['macd_signal'], df['macd_hist'] = self._calculate_macd(df['close'])
        
        return df

    def _add_volume_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volume-based indicators."""
        # Volume MA
        df['volume_sma'] = df['volume'].rolling(window=20).mean()
        
        # OBV
        df['obv'] = self._calculate_obv(df)
        
        return df

    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range."""
        high = df['high']
        low = df['low']
        close = df['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(window=period).mean()

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def _calculate_macd(self, prices: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate MACD, Signal line, and Histogram."""
        exp1 = prices.ewm(span=12, adjust=False).mean()
        exp2 = prices.ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=9, adjust=False).mean()
        hist = macd - signal
        return macd, signal, hist

    def _calculate_obv(self, df: pd.DataFrame) -> pd.Series:
        """Calculate On-Balance Volume."""
        obv = pd.Series(index=df.index, dtype=float)
        obv.iloc[0] = df['volume'].iloc[0]
        
        for i in range(1, len(df)):
            if df['close'].iloc[i] > df['close'].iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] + df['volume'].iloc[i]
            elif df['close'].iloc[i] < df['close'].iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] - df['volume'].iloc[i]
            else:
                obv.iloc[i] = obv.iloc[i-1]
        
        return obv

    def _determine_forex_session(self, timestamp: datetime) -> str:
        """Determine which forex session a timestamp belongs to."""
        hour = timestamp.hour
        
        if 0 <= hour < 8:
            return 'asian'
        elif 8 <= hour < 16:
            return 'london'
        else:
            return 'new_york'

    # Additional helper methods would be implemented here...