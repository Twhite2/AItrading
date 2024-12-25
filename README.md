# AI Trading Bot

An AI-powered trading bot for cryptocurrency and forex markets, integrating the LoRA-Alpaca model for market analysis and decision making.

## Features

- Supports both cryptocurrency and forex trading
- AI-powered trading decisions using LoRA-Alpaca model
- Multi-timeframe analysis
- Advanced risk management
- Detailed trade logging and analysis
- Backtesting capabilities
- Session-aware trading for forex
- Comprehensive position monitoring

## Requirements

- Python 3.8 or higher
- CUDA-capable GPU (recommended)
- 16GB RAM minimum
- Internet connection
- API access to supported exchanges/brokers

## Installation

1. Clone the repository:
```bash
git clone [repository-url]
cd AITradingbot
```

2. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Configure environment variables:
Create a `.env` file with your API keys:
```
EXCHANGE_API_KEY=your_exchange_api_key
EXCHANGE_SECRET_KEY=your_exchange_secret_key
DERIV_API_KEY=your_DERIV_api_key
DERIV_ACCOUNT_ID=your_DERIV_account_id
```

## Usage

### Live Trading

1. Crypto markets:
```bash
python main.py --type crypto
```

2. Forex markets:
```bash
python main.py --type forex
```

### Backtesting

```bash
python main.py --type crypto --backtest --start-date 2023-01-01 --end-date 2023-12-31
```

## Configuration

- Edit `config/config.yaml` for trading parameters
- Edit `config/logging_config.yaml` for logging settings

## Structure

```
AITradingbot/
├── config/            # Configuration files
├── models/           # AI and trading models
├── utils/            # Utility functions
├── traders/          # Trading implementations
├── docs/             # Documentation and logs
└── main.py          # Entry point
```

## Risk Warning

This software is for educational purposes only. Trading carries significant financial risk, and you should only trade with money you can afford to lose.

## License

MIT License - see LICENSE file for details

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create new Pull Request