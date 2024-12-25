# utils/trade_logger.py
import json
import os
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Union
import logging
from pathlib import Path

class TradeLogger:
    def __init__(self, base_path: str = "docs/trades"):
        """Initialize trade logger with base path for storing logs."""
        self.base_path = Path(base_path)
        self.trades_path = self.base_path / "trades"
        self.analytics_path = self.base_path / "analytics"
        self.daily_summaries_path = self.base_path / "summaries"
        
        # Create necessary directories
        for path in [self.trades_path, self.analytics_path, self.daily_summaries_path]:
            path.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        logging.basicConfig(
            filename=self.base_path / "trading.log",
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger("TradeLogger")

    def log_trade(self, trade_data: Dict) -> str:
        """
        Log a trade with all relevant information.
        Returns the path to the created log file.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        symbol = trade_data.get("symbol", "UNKNOWN")
        
        # Create trade ID
        trade_id = f"{symbol}_{timestamp}"
        
        # Add metadata
        trade_data.update({
            "trade_id": trade_id,
            "logged_at": datetime.now().isoformat(),
            "trade_status": "OPEN"
        })
        
        # Save JSON log
        json_path = self.trades_path / f"{trade_id}.json"
        with open(json_path, "w") as f:
            json.dump(trade_data, f, indent=4)
        
        # Create human-readable log
        self._create_readable_log(trade_data)
        
        # Log to main log file
        self.logger.info(f"New trade logged: {trade_id}")
        
        return str(json_path)

    def update_trade(self, trade_id: str, update_data: Dict) -> None:
        """Update an existing trade log with new information."""
        json_path = self.trades_path / f"{trade_id}.json"
        
        if not json_path.exists():
            self.logger.error(f"Trade log not found: {trade_id}")
            return
        
        # Read existing data
        with open(json_path, "r") as f:
            trade_data = json.load(f)
        
        # Update data
        trade_data.update(update_data)
        trade_data["last_updated"] = datetime.now().isoformat()
        
        # Save updated data
        with open(json_path, "w") as f:
            json.dump(trade_data, f, indent=4)
        
        # Update readable log
        self._create_readable_log(trade_data)
        
        self.logger.info(f"Trade updated: {trade_id}")

    def close_trade(self, trade_id: str, close_data: Dict) -> None:
        """Mark a trade as closed and log final results."""
        json_path = self.trades_path / f"{trade_id}.json"
        
        if not json_path.exists():
            self.logger.error(f"Trade log not found: {trade_id}")
            return
        
        # Read existing data
        with open(json_path, "r") as f:
            trade_data = json.load(f)
        
        # Update with closing data
        trade_data.update(close_data)
        trade_data.update({
            "trade_status": "CLOSED",
            "closed_at": datetime.now().isoformat(),
            "duration": self._calculate_duration(trade_data["logged_at"])
        })
        
        # Calculate final P&L
        trade_data["final_pnl"] = self._calculate_pnl(trade_data)
        
        # Save updated data
        with open(json_path, "w") as f:
            json.dump(trade_data, f, indent=4)
        
        # Update readable log
        self._create_readable_log(trade_data)
        
        # Generate trade analysis
        self._analyze_closed_trade(trade_data)
        
        self.logger.info(f"Trade closed: {trade_id}")

    def generate_daily_summary(self, date: Optional[str] = None) -> Dict:
        """Generate summary of trading activity for a specific date."""
        if date is None:
            date = datetime.now().strftime("%Y%m%d")
        
        trades = self._get_trades_for_date(date)
        
        summary = {
            "date": date,
            "total_trades": len(trades),
            "winning_trades": len([t for t in trades if t.get("final_pnl", 0) > 0]),
            "losing_trades": len([t for t in trades if t.get("final_pnl", 0) < 0]),
            "total_pnl": sum(t.get("final_pnl", 0) for t in trades),
            "win_rate": 0,
            "average_win": 0,
            "average_loss": 0,
            "largest_win": max((t.get("final_pnl", 0) for t in trades), default=0),
            "largest_loss": min((t.get("final_pnl", 0) for t in trades), default=0),
            "trades_by_symbol": self._group_trades_by_symbol(trades)
        }
        
        # Calculate win rate and averages
        if summary["total_trades"] > 0:
            summary["win_rate"] = (summary["winning_trades"] / summary["total_trades"]) * 100
            
        winning_trades = [t for t in trades if t.get("final_pnl", 0) > 0]
        losing_trades = [t for t in trades if t.get("final_pnl", 0) < 0]
        
        if winning_trades:
            summary["average_win"] = sum(t.get("final_pnl", 0) for t in winning_trades) / len(winning_trades)
        if losing_trades:
            summary["average_loss"] = sum(t.get("final_pnl", 0) for t in losing_trades) / len(losing_trades)
        
        # Save summary
        summary_path = self.daily_summaries_path / f"summary_{date}.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=4)
        
        # Create readable summary
        self._create_readable_summary(summary)
        
        return summary

    def _create_readable_log(self, trade_data: Dict) -> None:
        """Create human-readable version of trade log."""
        trade_id = trade_data["trade_id"]
        readable_path = self.trades_path / f"{trade_id}_readable.txt"
        
        with open(readable_path, "w") as f:
            f.write(f"Trade Report - {trade_data['symbol']}\n")
            f.write("=" * 50 + "\n\n")
            
            # Trade Details
            f.write("Trade Details:\n")
            f.write("-" * 20 + "\n")
            f.write(f"Trade ID: {trade_id}\n")
            f.write(f"Symbol: {trade_data['symbol']}\n")
            f.write(f"Direction: {trade_data['direction']}\n")
            f.write(f"Entry Price: {trade_data['entry_price']}\n")
            f.write(f"Position Size: {trade_data['position_size']}\n")
            
            # Strategy Analysis
            if "strategy_analysis" in trade_data:
                f.write("\nStrategy Analysis:\n")
                f.write("-" * 20 + "\n")
                f.write(trade_data['strategy_analysis'] + "\n")
            
            # AI Model Analysis
            if "ai_analysis" in trade_data:
                f.write("\nAI Model Analysis:\n")
                f.write("-" * 20 + "\n")
                f.write(f"Confidence: {trade_data['ai_analysis'].get('confidence', 'N/A')}%\n")
                f.write(f"Prediction: {trade_data['ai_analysis'].get('prediction', 'N/A')}\n")
            
            # Results (if closed)
            if trade_data["trade_status"] == "CLOSED":
                f.write("\nTrade Results:\n")
                f.write("-" * 20 + "\n")
                f.write(f"Exit Price: {trade_data.get('exit_price', 'N/A')}\n")
                f.write(f"P&L: {trade_data.get('final_pnl', 'N/A')}\n")
                f.write(f"Duration: {trade_data.get('duration', 'N/A')}\n")

    def _create_readable_summary(self, summary: Dict) -> None:
        """Create human-readable version of daily summary."""
        summary_path = self.daily_summaries_path / f"summary_{summary['date']}_readable.txt"
        
        with open(summary_path, "w") as f:
            f.write(f"Daily Trading Summary - {summary['date']}\n")
            f.write("=" * 50 + "\n\n")
            
            f.write("Performance Metrics:\n")
            f.write("-" * 20 + "\n")
            f.write(f"Total Trades: {summary['total_trades']}\n")
            f.write(f"Win Rate: {summary['win_rate']:.2f}%\n")
            f.write(f"Total P&L: {summary['total_pnl']:.2f}\n")
            f.write(f"Average Win: {summary['average_win']:.2f}\n")
            f.write(f"Average Loss: {summary['average_loss']:.2f}\n")
            f.write(f"Largest Win: {summary['largest_win']:.2f}\n")
            f.write(f"Largest Loss: {summary['largest_loss']:.2f}\n")
            
            f.write("\nTrades by Symbol:\n")
            f.write("-" * 20 + "\n")
            for symbol, data in summary['trades_by_symbol'].items():
                f.write(f"\n{symbol}:\n")
                f.write(f"  Total Trades: {data['count']}\n")
                f.write(f"  P&L: {data['pnl']:.2f}\n")
                f.write(f"  Win Rate: {data['win_rate']:.2f}%\n")

    def _calculate_duration(self, start_time: str) -> str:
        """Calculate trade duration."""
        start = datetime.fromisoformat(start_time)
        duration = datetime.now() - start
        return str(duration)

    def _calculate_pnl(self, trade_data: Dict) -> float:
        """Calculate trade P&L."""
        direction = trade_data["direction"]
        entry_price = trade_data["entry_price"]
        exit_price = trade_data["exit_price"]
        position_size = trade_data["position_size"]
        
        if direction == "BUY":
            return (exit_price - entry_price) * position_size
        else:
            return (entry_price - exit_price) * position_size

    def _get_trades_for_date(self, date: str) -> List[Dict]:
        """Get all trades for a specific date."""
        trades = []
        for file in self.trades_path.glob("*.json"):
            if not file.stem.endswith("_readable"):
                with open(file, "r") as f:
                    trade = json.load(f)
                    if trade["logged_at"].startswith(date):
                        trades.append(trade)
        return trades

    def _group_trades_by_symbol(self, trades: List[Dict]) -> Dict:
        """Group trades by symbol and calculate metrics."""
        grouped = {}
        for trade in trades:
            symbol = trade["symbol"]
            if symbol not in grouped:
                grouped[symbol] = {
                    "count": 0,
                    "wins": 0,
                    "pnl": 0,
                    "win_rate": 0
                }
            
            grouped[symbol]["count"] += 1
            if trade.get("final_pnl", 0) > 0:
                grouped[symbol]["wins"] += 1
            grouped[symbol]["pnl"] += trade.get("final_pnl", 0)
            
            # Calculate win rate
            grouped[symbol]["win_rate"] = (grouped[symbol]["wins"] / grouped[symbol]["count"]) * 100
            
        return grouped

    def _analyze_closed_trade(self, trade_data: Dict) -> None:
        """Analyze a closed trade and save insights."""
        analysis = {
            "trade_id": trade_data["trade_id"],
            "symbol": trade_data["symbol"],
            "success": trade_data.get("final_pnl", 0) > 0,
            "pnl": trade_data.get("final_pnl", 0),
            "duration": trade_data.get("duration"),
            "strategy_effectiveness": self._analyze_strategy_effectiveness(trade_data),
            "market_conditions": trade_data.get("market_conditions", {}),
            "key_lessons": self._extract_key_lessons(trade_data)
        }
        
        # Save analysis
        analysis_path = self.analytics_path / f"{trade_data['trade_id']}_analysis.json"
        with open(analysis_path, "w") as f:
            json.dump(analysis, f, indent=4)

    def _analyze_strategy_effectiveness(self, trade_data: Dict) -> Dict:
        """Analyze how well the strategy performed."""
        return {
            "entry_accuracy": self._rate_entry_accuracy(trade_data),
            "exit_accuracy": self._rate_exit_accuracy(trade_data),
            "risk_reward_achieved": self._calculate_risk_reward(trade_data),
            "ai_prediction_accuracy": self._verify_ai_prediction(trade_data)
        }

    def _extract_key_lessons(self, trade_data: Dict) -> List[str]:
        """Extract key lessons from the trade."""
        lessons = []
        
        # Extract lessons based on trade performance
        if trade_data.get("final_pnl", 0) > 0:
            if "strategy_analysis" in trade_data:
                lessons.append("Successful strategy implementation")
        else:
            if "stop_loss_hit" in trade_data:
                lessons.append("Review stop loss placement")
        
        return lessons

    # Additional helper methods would be implemented here...