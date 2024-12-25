#!/usr/bin/env python3
import os
import yaml
import logging.config
import argparse
from datetime import datetime
import pytz
from typing import Dict
from pathlib import Path

from models.ai_model import TradingAIModel
from models.trading_models import TradingStrategy
from utils.risk_manager import RiskManager
from utils.data_handlers import DataHandler
from utils.trade_logger import TradeLogger
from traders.crypto_trader import CryptoTrader
from traders.forex_trader import ForexTrader

def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def setup_logging(config_path: str):
    """Setup logging configuration."""
    with open(config_path, 'r') as file:
        logging_config = yaml.safe_load(file)
        logging.config.dictConfig(logging_config)

def initialize_components(config: Dict):
    """Initialize trading components."""
    # Initialize AI model
    ai_model = TradingAIModel(
        model_path=config['model_settings']['model_path'],
        confidence_threshold=config['model_settings']['confidence_threshold']
    )

    # Initialize trading strategy
    trading_strategy = TradingStrategy(config['trading_settings'])

    # Initialize risk manager
    risk_manager = RiskManager(config['risk_settings'])

    return ai_model, trading_strategy, risk_manager

def setup_broker(broker_type: str, config: Dict):
    """Setup broker connection."""
    if broker_type.lower() == 'crypto':
        from lumibot.brokers import Ccxt
        ccxt_config = {
            "exchange_id": config['broker_settings']['exchange'],
            "api_key": os.getenv('EXCHANGE_API_KEY'),
            "secret": os.getenv('EXCHANGE_SECRET_KEY'),
            "exchange_config": {
                'enableRateLimit': config['broker_settings'].get('rate_limit', True),
                'test': config['broker_settings'].get('practice', True)
            }
        }
        
        broker = Ccxt(ccxt_config)
        
    elif broker_type.lower() == 'forex':
        from brokers.deriv_broker import DerivBroker
        # Get symbols and convert to Deriv format
        symbols = config['trading_settings'].get('pairs', [])
        deriv_symbols = [f"frx{pair.replace('_', '')}" for pair in symbols]
        
        # Create data source configuration
        data_source = {
            "provider": "deriv",
            "host": "wss://ws.binaryws.com/websockets/v3",
            "symbols": deriv_symbols
        }
        
        broker = DerivBroker(
            app_id=os.getenv('DERIV_APP_ID'),
            api_token=os.getenv('DERIV_API_TOKEN'),
            demo=config['broker_settings'].get('demo_account', True),
            symbols=deriv_symbols,
            leverage=config['broker_settings'].get('leverage', 100),
            data_source=data_source  # Added data source configuration
        )
    else:
        raise ValueError(f"Unsupported broker type: {broker_type}")
    
    return broker

def create_trader(broker_type: str, config: Dict, components: tuple):
    """Create appropriate trader instance."""
    ai_model, trading_strategy, risk_manager = components
    
    if broker_type.lower() == 'crypto':
        return CryptoTrader(
            broker=setup_broker(broker_type, config),
            ai_model=ai_model,
            trading_strategy=trading_strategy,
            risk_manager=risk_manager,
            symbols=config['trading_settings']['symbols'],
            timeframes=config['trading_settings']['timeframes'],
            budget=config['trading_settings'].get('budget', 0)
        )
    elif broker_type.lower() == 'forex':
        return ForexTrader(
            broker=setup_broker(broker_type, config),
            ai_model=ai_model,
            trading_strategy=trading_strategy,
            risk_manager=risk_manager,
            pairs=config['trading_settings']['pairs'],
            timeframes=config['trading_settings']['timeframes'],
            budget=config['trading_settings'].get('budget', 0)
        )
    else:
        raise ValueError(f"Unsupported trader type: {broker_type}")

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='AI Trading Bot')
    parser.add_argument(
        '--type',
        choices=['crypto', 'forex'],
        required=True,
        help='Type of trading (crypto or forex)'
    )
    parser.add_argument(
        '--config',
        default='config/config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--logging-config',
        default='config/logging_config.yaml',
        help='Path to logging configuration file'
    )
    parser.add_argument(
        '--backtest',
        action='store_true',
        help='Run in backtest mode'
    )
    parser.add_argument(
        '--start-date',
        help='Start date for backtest (YYYY-MM-DD)'
    )
    parser.add_argument(
        '--end-date',
        help='End date for backtest (YYYY-MM-DD)'
    )
    parser.add_argument(
        '--demo',
        action='store_true',
        default=True,
        help='Use demo account (default: True)'
    )
    
    return parser.parse_args()

def run_backtest(trader, start_date: str, end_date: str):
    """Run trader in backtest mode."""
    start = datetime.strptime(start_date, '%Y-%m-%d').replace(tzinfo=pytz.UTC)
    end = datetime.strptime(end_date, '%Y-%m-%d').replace(tzinfo=pytz.UTC)
    
    trader.backtest(
        start,
        end,
        save_positions=True,
        show_plot=True
    )

def main():
    """Main entry point for the trading bot."""
    try:
        # Parse command line arguments
        args = parse_arguments()
        
        # Load configurations
        config = load_config(args.config)
        setup_logging(args.logging_config)
        
        logger = logging.getLogger(__name__)
        logger.info(f"Starting {args.type} trading bot...")
        
        # Update config with command line arguments
        config['broker_settings']['demo_account'] = args.demo
        
        # Initialize components
        components = initialize_components(config)
        
        # Create trader instance
        trader = create_trader(args.type, config, components)
        
        if args.backtest:
            if not (args.start_date and args.end_date):
                raise ValueError("Start and end dates required for backtest mode")
            logger.info("Running in backtest mode...")
            run_backtest(trader, args.start_date, args.end_date)
        else:
            # Run live trading
            logger.info("Starting live trading...")
            trader.run()
            
    except Exception as e:
        logger.error(f"Error running trading bot: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()