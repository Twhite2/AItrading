# utils/risk_manager.py
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import logging
from datetime import datetime, timedelta

class RiskLevel(Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"

@dataclass
class PositionInfo:
    symbol: str
    direction: str  # 'long' or 'short'
    size: float
    entry_price: float
    stop_loss: float
    take_profit: Optional[float]
    risk_amount: float
    risk_ratio: float

class RiskManager:
    def __init__(self, config: Dict):
        """Initialize risk manager with configuration settings."""
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Risk parameters
        self.max_account_risk = config.get('max_account_risk', 0.02)  # 2% max account risk
        self.max_position_size = config.get('max_position_size', 0.05)  # 5% max position size
        self.max_pair_exposure = config.get('max_pair_exposure', 0.10)  # 10% max exposure per pair
        self.max_correlation_exposure = config.get('max_correlation_exposure', 0.20)  # 20% max correlated exposure
        self.max_daily_drawdown = config.get('max_daily_drawdown', 0.05)  # 5% max daily drawdown
        
        # Track positions and exposure
        self.open_positions: Dict[str, PositionInfo] = {}
        self.daily_pnl = 0.0
        self.peak_balance = 0.0

    def calculate_position_size(self,
                              account_balance: float,
                              entry_price: float,
                              stop_loss: float,
                              symbol: str) -> Tuple[float, Dict]:
        """Calculate safe position size based on risk parameters."""
        try:
            # Calculate risk amount in account currency
            risk_amount = account_balance * self.max_account_risk
            
            # Calculate pip value and position size
            pip_risk = abs(entry_price - stop_loss)
            if pip_risk == 0:
                raise ValueError("Invalid stop loss - same as entry price")
                
            position_size = risk_amount / pip_risk
            
            # Apply position size limits
            max_allowed_size = account_balance * self.max_position_size
            if position_size > max_allowed_size:
                position_size = max_allowed_size
                
            # Check symbol exposure limits
            if not self._check_symbol_exposure(symbol, position_size, account_balance):
                position_size = self._adjust_for_symbol_exposure(symbol, account_balance)
                
            return position_size, {
                "risk_amount": risk_amount,
                "risk_per_pip": position_size * 0.0001,  # For forex
                "max_loss": risk_amount,
                "exposure_percentage": (position_size * entry_price) / account_balance
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating position size: {str(e)}")
            raise

    def validate_trade(self,
                      symbol: str,
                      direction: str,
                      size: float,
                      entry_price: float,
                      stop_loss: float,
                      take_profit: Optional[float],
                      account_balance: float) -> Tuple[bool, str]:
        """Validate if a trade meets all risk management criteria."""
        try:
            # Calculate risk metrics
            risk_amount = size * abs(entry_price - stop_loss)
            risk_ratio = risk_amount / account_balance
            
            # Check risk percentage
            if risk_ratio > self.max_account_risk:
                return False, f"Trade risk {risk_ratio:.2%} exceeds maximum allowed {self.max_account_risk:.2%}"
            
            # Check position size
            position_value = size * entry_price
            if position_value / account_balance > self.max_position_size:
                return False, "Position size exceeds maximum allowed"
            
            # Check symbol exposure
            if not self._check_symbol_exposure(symbol, size, account_balance):
                return False, "Symbol exposure exceeds maximum allowed"
            
            # Check correlation exposure
            if not self._check_correlation_exposure(symbol, size, account_balance):
                return False, "Correlated exposure exceeds maximum allowed"
            
            # Validate risk/reward if take_profit is specified
            if take_profit and not self._validate_risk_reward(entry_price, stop_loss, take_profit):
                return False, "Risk/Reward ratio below minimum threshold"
            
            return True, "Trade validated successfully"
            
        except Exception as e:
            self.logger.error(f"Error validating trade: {str(e)}")
            return False, str(e)

    def update_position(self,
                       symbol: str,
                       current_price: float,
                       timestamp: datetime) -> Dict:
        """Update position metrics and check for risk threshold breaches."""
        if symbol not in self.open_positions:
            return {}
            
        position = self.open_positions[symbol]
        
        # Calculate current P&L
        if position.direction == 'long':
            unrealized_pnl = (current_price - position.entry_price) * position.size
        else:
            unrealized_pnl = (position.entry_price - current_price) * position.size
            
        # Update daily P&L
        self.daily_pnl = self._calculate_daily_pnl()
        
        # Check risk thresholds
        risk_level = self._assess_risk_level(position, current_price, unrealized_pnl)
        
        return {
            "unrealized_pnl": unrealized_pnl,
            "risk_level": risk_level,
            "daily_drawdown": self.daily_pnl / self.peak_balance if self.peak_balance > 0 else 0,
            "position_risk": self._calculate_position_risk(position, current_price)
        }

    def adjust_for_drawdown(self,
                           account_balance: float,
                           drawdown_percentage: float) -> Dict[str, float]:
        """Adjust risk parameters based on drawdown."""
        # Calculate adjustment factors
        drawdown_factor = max(0, 1 - (drawdown_percentage / self.max_daily_drawdown))
        
        # Adjust risk parameters
        adjusted_params = {
            "max_account_risk": self.max_account_risk * drawdown_factor,
            "max_position_size": self.max_position_size * drawdown_factor,
            "max_pair_exposure": self.max_pair_exposure * drawdown_factor
        }
        
        return adjusted_params

    def generate_risk_report(self) -> Dict:
        """Generate comprehensive risk report."""
        return {
            "open_positions": len(self.open_positions),
            "total_exposure": self._calculate_total_exposure(),
            "daily_pnl": self.daily_pnl,
            "risk_distribution": self._calculate_risk_distribution(),
            "exposure_distribution": self._calculate_exposure_distribution(),
            "risk_metrics": self._calculate_risk_metrics()
        }

    def _check_symbol_exposure(self,
                             symbol: str,
                             size: float,
                             account_balance: float) -> bool:
        """Check if symbol exposure is within limits."""
        current_exposure = sum(
            pos.size * pos.entry_price
            for pos in self.open_positions.values()
            if pos.symbol == symbol
        )
        
        new_exposure = current_exposure + (size * self._get_current_price(symbol))
        return (new_exposure / account_balance) <= self.max_pair_exposure

    def _check_correlation_exposure(self,
                                  symbol: str,
                                  size: float,
                                  account_balance: float) -> bool:
        """Check exposure to correlated instruments."""
        correlated_pairs = self._get_correlated_pairs(symbol)
        
        correlated_exposure = sum(
            pos.size * pos.entry_price
            for pos in self.open_positions.values()
            if pos.symbol in correlated_pairs
        )
        
        new_exposure = correlated_exposure + (size * self._get_current_price(symbol))
        return (new_exposure / account_balance) <= self.max_correlation_exposure

    def _validate_risk_reward(self,
                            entry_price: float,
                            stop_loss: float,
                            take_profit: float) -> bool:
        """Validate risk/reward ratio."""
        risk = abs(entry_price - stop_loss)
        reward = abs(take_profit - entry_price)
        min_rr_ratio = self.config.get('min_risk_reward_ratio', 1.5)
        
        return (reward / risk) >= min_rr_ratio

    def _assess_risk_level(self,
                          position: PositionInfo,
                          current_price: float,
                          unrealized_pnl: float) -> RiskLevel:
        """Assess current risk level of a position."""
        # Calculate distance to stop loss as percentage
        distance_to_sl = abs(current_price - position.stop_loss) / current_price
        
        if distance_to_sl <= 0.001:  # Very close to stop loss
            return RiskLevel.CRITICAL
        elif unrealized_pnl < -position.risk_amount * 0.8:
            return RiskLevel.HIGH
        elif unrealized_pnl < -position.risk_amount * 0.5:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW

    def _calculate_daily_pnl(self) -> float:
        """Calculate total daily P&L including closed positions."""
        # Implementation would include closed positions P&L tracking
        pass

    def _calculate_position_risk(self,
                               position: PositionInfo,
                               current_price: float) -> float:
        """Calculate current risk exposure of a position."""
        return position.size * abs(current_price - position.stop_loss)

    def _get_current_price(self, symbol: str) -> float:
        """Get current price for a symbol."""
        # Implementation would interface with market data provider
        pass

    def _get_correlated_pairs(self, symbol: str) -> List[str]:
        """Get list of correlated currency pairs."""
        # Implementation would include correlation analysis
        pass

    # Additional helper methods would be implemented here...