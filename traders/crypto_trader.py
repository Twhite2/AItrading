# traders/crypto_trader.py
from lumibot.brokers import Ccxt
from lumibot.traders import Trader
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging

from models.ai_model import TradingAIModel
from models.trading_models import TradingStrategy
from utils.risk_manager import RiskManager
from utils.data_handlers import DataHandler
from utils.trade_logger import TradeLogger

class CryptoTrader(Trader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Initialize components
        self.ai_model = kwargs.get('ai_model')
        self.trading_strategy = kwargs.get('trading_strategy')
        self.risk_manager = kwargs.get('risk_manager')
        self.data_handler = DataHandler()
        self.trade_logger = TradeLogger()
        
        # Trading parameters
        self.symbols = kwargs.get('symbols', ['BTC/USDT'])
        self.timeframes = kwargs.get('timeframes', {
            'analysis': '1d',
            'entry': '1h',
            'execution': '5m'
        })
        
        # Initialize logging
        self.logger = logging.getLogger(__name__)
        
        # Trading state
        self.active_trades = {}
        self.pending_orders = {}
        self.market_state = {}

    def initialize(self):
        """Initialize the trading bot."""
        self.sleeptime = "5m"  # Base monitoring interval
        
        # Set up broker connection
        if isinstance(self.broker, Ccxt):
            for symbol in self.symbols:
                self.broker.subscribe_to_feed(symbol)
                
        # Initialize market state
        self._initialize_market_state()

    def on_trading_iteration(self):
        """Main trading logic executed on each iteration."""
        try:
            current_time = self.get_datetime()
            
            # Update market state
            self._update_market_state()
            
            # Process each trading pair
            for symbol in self.symbols:
                self._process_trading_pair(symbol)
                
            # Monitor active positions
            self._monitor_positions()
                
        except Exception as e:
            self.logger.error(f"Error in trading iteration: {str(e)}")

    def _process_trading_pair(self, symbol: str):
        """Process individual trading pair."""
        try:
            # Get current market data
            market_data = self._get_market_data(symbol)
            if market_data is None:
                return
                
            # Get AI prediction
            prediction = self._get_ai_prediction(market_data)
            
            # Get strategy signals
            strategy_signals = self._get_strategy_signals(market_data)
            
            # Combine signals
            trading_decision = self._make_trading_decision(
                symbol, prediction, strategy_signals
            )
            
            # Execute trading decision
            if trading_decision['action'] != 'HOLD':
                self._execute_trading_decision(symbol, trading_decision)
                
        except Exception as e:
            self.logger.error(f"Error processing {symbol}: {str(e)}")

    def _get_market_data(self, symbol: str) -> Optional[Dict]:
        """Get and process market data for analysis."""
        try:
            # Get data for different timeframes
            data = {}
            for purpose, timeframe in self.timeframes.items():
                candles = self.get_historical_data(
                    symbol,
                    timeframe,
                    length=100  # Adjust based on needs
                )
                if candles is not None:
                    data[timeframe] = self.data_handler.process_raw_data(candles)
                    
            if not data:
                return None
                
            # Process market state
            market_state = {
                'key_levels': self.data_handler.identify_key_levels(data[self.timeframes['analysis']]),
                'session_stats': self.data_handler.calculate_session_statistics(data[self.timeframes['entry']]),
                'orderblocks': self.data_handler.detect_orderblocks(data[self.timeframes['entry']])
            }
            
            return {
                'data': data,
                'market_state': market_state
            }
            
        except Exception as e:
            self.logger.error(f"Error getting market data for {symbol}: {str(e)}")
            return None

    def _get_ai_prediction(self, market_data: Dict) -> Dict:
        """Get prediction from AI model."""
        try:
            # Use entry timeframe data for prediction
            prediction, confidence = self.ai_model.predict(
                market_data['data'][self.timeframes['entry']]
            )
            
            return {
                'prediction': prediction,
                'confidence': confidence,
                'analysis': self.ai_model.analyze_market_condition(
                    market_data['data'][self.timeframes['analysis']]
                )
            }
            
        except Exception as e:
            self.logger.error(f"Error getting AI prediction: {str(e)}")
            return {'prediction': 'HOLD', 'confidence': 0.0}

    def _get_strategy_signals(self, market_data: Dict) -> Dict:
        """Get signals from trading strategy."""
        try:
            return self.trading_strategy.generate_signals(market_data['data'])
        except Exception as e:
            self.logger.error(f"Error getting strategy signals: {str(e)}")
            return {}

    def _make_trading_decision(self, 
                             symbol: str, 
                             prediction: Dict, 
                             strategy_signals: Dict) -> Dict:
        """Combine AI and strategy signals to make trading decision."""
        try:
            # Get current position
            position = self.get_position(symbol)
            
            # Combine signals
            if prediction['confidence'] >= self.config['min_confidence']:
                action = prediction['prediction']
                confidence = prediction['confidence']
            else:
                action = 'HOLD'
                confidence = 0.0
                
            # Validate with strategy signals
            if action != 'HOLD' and strategy_signals:
                if not self._validate_signals(action, strategy_signals):
                    action = 'HOLD'
                    
            return {
                'action': action,
                'confidence': confidence,
                'signals': strategy_signals,
                'analysis': prediction.get('analysis', {})
            }
            
        except Exception as e:
            self.logger.error(f"Error making trading decision: {str(e)}")
            return {'action': 'HOLD', 'confidence': 0.0}

    def _execute_trading_decision(self, symbol: str, decision: Dict):
        """Execute trading decision."""
        try:
            position = self.get_position(symbol)
            current_price = self.get_last_price(symbol)
            
            if decision['action'] == 'BUY' and not position:
                self._execute_entry(symbol, 'BUY', current_price, decision)
                
            elif decision['action'] == 'SELL':
                if position and position.direction == "long":
                    self._execute_exit(symbol, current_price, decision)
                elif not position:
                    self._execute_entry(symbol, 'SELL', current_price, decision)
                    
        except Exception as e:
            self.logger.error(f"Error executing trading decision: {str(e)}")

    def _execute_entry(self, symbol: str, direction: str, price: float, decision: Dict):
        """Execute entry order."""
        try:
            # Calculate position size
            account_value = self.get_cash() + self.get_portfolio_value()
            
            # Get stop loss and take profit levels
            levels = self._calculate_trade_levels(symbol, direction, price)
            
            # Calculate position size
            position_size, risk_metrics = self.risk_manager.calculate_position_size(
                account_value,
                price,
                levels['stop_loss'],
                symbol
            )
            
            # Validate trade
            is_valid, message = self.risk_manager.validate_trade(
                symbol,
                direction,
                position_size,
                price,
                levels['stop_loss'],
                levels['take_profit'],
                account_value
            )
            
            if not is_valid:
                self.logger.warning(f"Trade validation failed: {message}")
                return
                
            # Execute order
            order = self.create_order(
                symbol,
                direction.lower(),
                position_size,
                type="market"
            )
            
            if order.status == "filled":
                # Log trade
                self._log_trade_entry(
                    symbol,
                    direction,
                    order,
                    levels,
                    decision,
                    risk_metrics
                )
                
                # Set stop loss and take profit orders
                self._set_trade_orders(symbol, order, levels)
                
        except Exception as e:
            self.logger.error(f"Error executing entry for {symbol}: {str(e)}")

    def _execute_exit(self, symbol: str, price: float, decision: Dict):
        """Execute exit order."""
        try:
            position = self.get_position(symbol)
            if not position:
                return
                
            # Create exit order
            order = self.create_order(
                symbol,
                "sell" if position.direction == "long" else "buy",
                position.quantity,
                type="market"
            )
            
            if order.status == "filled":
                # Log trade exit
                self._log_trade_exit(
                    symbol,
                    order,
                    decision
                )
                
                # Clean up pending orders
                self._cleanup_trade_orders(symbol)
                
        except Exception as e:
            self.logger.error(f"Error executing exit for {symbol}: {str(e)}")

    def _calculate_trade_levels(self, symbol: str, direction: str, entry_price: float) -> Dict:
        """Calculate stop loss and take profit levels."""
        try:
            # Get ATR for dynamic levels
            market_data = self._get_market_data(symbol)
            atr = market_data['data'][self.timeframes['entry']]['atr'].iloc[-1]
            
            # Calculate levels based on ATR
            sl_multiple = self.config.get('sl_atr_multiple', 1.5)
            tp_multiple = self.config.get('tp_atr_multiple', 2.5)
            
            if direction == 'BUY':
                stop_loss = entry_price - (atr * sl_multiple)
                take_profit = entry_price + (atr * tp_multiple)
            else:
                stop_loss = entry_price + (atr * sl_multiple)
                take_profit = entry_price - (atr * tp_multiple)
                
            return {
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'atr': atr
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating trade levels: {str(e)}")
            return None

    def _monitor_positions(self):
        """Monitor and manage open positions."""
        try:
            for symbol in self.symbols:
                position = self.get_position(symbol)
                if not position:
                    continue
                    
                current_price = self.get_last_price(symbol)
                
                # Update position metrics
                metrics = self.risk_manager.update_position(
                    symbol,
                    current_price,
                    self.get_datetime()
                )
                
                # Check risk levels
                if metrics.get('risk_level') == 'CRITICAL':
                    self._execute_emergency_exit(symbol, current_price)
                    
                # Trail stop loss if applicable
                self._update_trailing_stop(symbol, position, current_price)
                
        except Exception as e:
            self.logger.error(f"Error monitoring positions: {str(e)}")

    def _log_trade_entry(self, symbol: str, direction: str, order: object, 
                        levels: Dict, decision: Dict, risk_metrics: Dict):
        """Log trade entry details."""
        try:
            trade_data = {
                'symbol': symbol,
                'direction': direction,
                'entry_price': float(order.executed_price),
                'position_size': float(order.quantity),
                'stop_loss': levels['stop_loss'],
                'take_profit': levels['take_profit'],
                'risk_metrics': risk_metrics,
                'ai_prediction': {
                    'confidence': decision['confidence'],
                    'analysis': decision['analysis']
                },
                'strategy_signals': decision['signals'],
                'timestamp': self.get_datetime().isoformat()
            }
            
            self.trade_logger.log_trade(trade_data)
            
        except Exception as e:
            self.logger.error(f"Error logging trade entry: {str(e)}")

    def _log_trade_exit(self, symbol: str, order: object, decision: Dict):
        """Log trade exit details."""
        try:
            exit_data = {
                'exit_price': float(order.executed_price),
                'exit_time': self.get_datetime().isoformat(),
                'exit_reason': decision.get('reason', 'signal'),
                'ai_analysis': decision.get('analysis', {}),
                'strategy_signals': decision.get('signals', {})
            }
            
            self.trade_logger.update_trade(
                f"{symbol}_{order.timestamp}", exit_data
            )
            
        except Exception as e:
            self.logger.error(f"Error logging trade exit: {str(e)}")

    def _validate_signals(self, action: str, signals: Dict) -> bool:
        """Validate trading signals."""
        try:
            # Combined signal threshold
            threshold = self.config.get('signal_threshold', 0.5)
            
            if action == 'BUY':
                return signals.get('combined_signal', 0) >= threshold
            elif action == 'SELL':
                return signals.get('combined_signal', 0) <= -threshold
                
            return False
            
        except Exception as e:
            self.logger.error(f"Error validating signals: {str(e)}")
            return False

    # Additional helper methods would be implemented here...