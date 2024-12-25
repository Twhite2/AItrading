# brokers/deriv_broker.py
from lumibot.brokers import Broker
from lumibot.entities import Order, Position, Asset
from datetime import datetime, timedelta
import asyncio
from deriv_api import DerivAPI
import websockets
import json
import logging
import pandas as pd
from typing import Dict, List, Optional, Union
import time
from concurrent.futures import ThreadPoolExecutor

class DerivBroker(Broker):
    def __init__(self, 
                 app_id: str,
                 api_token: str,
                 demo: bool = True,
                 symbols: List[str] = None,
                 leverage: int = 100,
                 data_source: Dict = None):
        
        # Set up logger first
        self.logger = logging.getLogger(__name__)
        
        # Initialize connection variables
        self.app_id = app_id
        self.api_token = api_token
        self.demo = demo
        self.symbols = symbols or []
        self.leverage = leverage
        
        # Initialize state
        self.api = None
        self.ws = None
        self.active_contracts = {}
        self.price_subscriptions = {}
        
        # Initialize WebSocket connection first
        self._initialize_connection()
        
        # Create default data source if none provided
        if data_source is None:
            data_source = {
                "provider": "deriv",
                "host": f"wss://ws.deriv.com/websockets/v3?app_id={self.app_id}",
                "symbols": self.symbols
            }
        
        # Initialize broker with data source after WebSocket is ready
        super().__init__(name="deriv", data_source=data_source)

    def _initialize_connection(self):
        """Initialize connection to Deriv API."""
        try:
            # Create event loop for WebSocket connection
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            # Initialize WebSocket connection with app_id in URL
            url = f"wss://ws.deriv.com/websockets/v3?app_id={self.app_id}"
            self.ws = loop.run_until_complete(
                websockets.connect(url)
            )
            
            # After connection, send authorization
            auth_request = {
                "authorize": self.api_token,
                "app_id": int(self.app_id)  # Ensure app_id is integer
            }
            
            # Send auth request and wait for response
            auth_response = loop.run_until_complete(self._async_send_request(auth_request))
            
            if auth_response.get('error'):
                raise Exception(f"Authentication failed: {auth_response['error']['message']}")
                
            # Initialize API after successful authentication
            self.api = DerivAPI(
                app_id=self.app_id,
                api_token=self.api_token
            )
            
            # Start message handling in background
            self.executor = ThreadPoolExecutor(max_workers=1)
            self.executor.submit(self._handle_messages)
            
        except Exception as e:
            self.logger.error(f"Error initializing connection: {str(e)}")
            raise

    async def _async_send_request(self, request: Dict) -> Dict:
        """Send request to Deriv API asynchronously."""
        try:
            await self.ws.send(json.dumps(request))
            response = await self.ws.recv()
            return json.loads(response)
        except Exception as e:
            self.logger.error(f"Error in async request: {str(e)}")
            return {"error": {"message": str(e)}}

    def _get_balances_at_broker(self) -> Dict:
        """Get account balances."""
        try:
            request = {
                "balance": 1
            }
            response = self._send_request(request)
            
            if response.get('error'):
                self.logger.error(f"Error getting balances: {response['error']['message']}")
                return {}
                
            return {
                'total': float(response['balance']['balance']),
                'cash': float(response['balance']['balance']),
                'position_value': 0.0
            }
        except Exception as e:
            self.logger.error(f"Error in _get_balances_at_broker: {str(e)}")
            return {}

    def _get_stream_object(self):
        """Get stream object for real-time data."""
        if not self.ws:
            raise ValueError("WebSocket connection not initialized")
        return self.ws

    def _parse_broker_order(self, response: Dict) -> Order:
        """Parse broker order response into Order object."""
        try:
            order_type = "market"  # Deriv uses market orders
            side = "buy" if response['contract_type'].startswith('CALL') else "sell"
            
            return Order(
                identifier=response['contract_id'],
                symbol=response['symbol'],
                side=side,
                quantity=float(response['buy_price']),
                order_type=order_type,
                status="filled",
                timestamp=datetime.fromtimestamp(response['start_time'])
            )
        except Exception as e:
            self.logger.error(f"Error parsing broker order: {str(e)}")
            return None

    def _pull_broker_all_orders(self) -> List[Order]:
        """Get all orders."""
        try:
            orders = []
            for contract in self.active_contracts.values():
                order = self._parse_broker_order(contract)
                if order:
                    orders.append(order)
            return orders
        except Exception as e:
            self.logger.error(f"Error pulling all orders: {str(e)}")
            return []

    def _pull_broker_order(self, order_id: str) -> Optional[Order]:
        """Get specific order by ID."""
        try:
            contract = self.active_contracts.get(order_id)
            if contract:
                return self._parse_broker_order(contract)
            return None
        except Exception as e:
            self.logger.error(f"Error pulling order: {str(e)}")
            return None

    def _pull_position(self, symbol: str) -> Optional[Position]:
        """Get position for specific symbol."""
        try:
            contract = next(
                (c for c in self.active_contracts.values() if c['symbol'] == symbol),
                None
            )
            
            if contract:
                return Position(
                    symbol=contract['symbol'],
                    side="long" if contract['contract_type'].startswith('CALL') else "short",
                    quantity=float(contract['buy_price']),
                    entry_price=float(contract['entry_spot']),
                    current_price=float(contract['current_spot']),
                    timestamp=datetime.fromtimestamp(contract['start_time'])
                )
            return None
        except Exception as e:
            self.logger.error(f"Error pulling position: {str(e)}")
            return None

    def _pull_positions(self) -> List[Position]:
        """Get all positions."""
        try:
            positions = []
            for contract in self.active_contracts.values():
                position = self._pull_position(contract['symbol'])
                if position:
                    positions.append(position)
            return positions
        except Exception as e:
            self.logger.error(f"Error pulling positions: {str(e)}")
            return []

    def _register_stream_events(self):
        """Register stream events."""
        pass  # Using WebSocket message handler instead

    async def _run_stream(self):
        """Run the data stream."""
        while True:
            try:
                message = await self.ws.recv()
                data = json.loads(message)
                self._process_message(data)
            except Exception as e:
                self.logger.error(f"Stream error: {str(e)}")
                await asyncio.sleep(1)

    def _submit_order(self, order: Order):
        """Submit order to broker."""
        try:
            contract_request = self._create_contract_request(order)
            response = self._send_request(contract_request)
            
            if response.get('error'):
                order.status = "rejected"
                self.logger.error(f"Order rejected: {response['error']['message']}")
            else:
                order.status = "filled"
                order.identifier = response['buy']['contract_id']
                self.active_contracts[order.identifier] = response['buy']
                
            return order
        except Exception as e:
            self.logger.error(f"Error submitting order: {str(e)}")
            order.status = "rejected"
            return order

    def cancel_order(self, order: Order):
        """Cancel an order."""
        try:
            if order.identifier in self.active_contracts:
                request = {
                    "sell": order.identifier
                }
                response = self._send_request(request)
                
                if response.get('error'):
                    self.logger.error(f"Error cancelling order: {response['error']['message']}")
                    return False
                    
                self.active_contracts.pop(order.identifier, None)
                return True
            return False
        except Exception as e:
            self.logger.error(f"Error in cancel_order: {str(e)}")
            return False

    def get_historical_account_value(self, start_dt: datetime, end_dt: datetime) -> pd.DataFrame:
        """Get historical account value."""
        try:
            request = {
                "statement": 1,
                "start": int(start_dt.timestamp()),
                "end": int(end_dt.timestamp())
            }
            response = self._send_request(request)
            
            if response.get('error'):
                self.logger.error(f"Error getting statement: {response['error']['message']}")
                return pd.DataFrame()
                
            statements = response.get('statement', {}).get('transactions', [])
            
            # Convert to DataFrame
            df = pd.DataFrame(statements)
            if not df.empty:
                df['timestamp'] = pd.to_datetime(df['transaction_time'], unit='s')
                df['value'] = df['balance_after']
                return df[['timestamp', 'value']]
                
            return pd.DataFrame(columns=['timestamp', 'value'])
            
        except Exception as e:
            self.logger.error(f"Error getting historical account value: {str(e)}")
            return pd.DataFrame()

    def _create_contract_request(self, order: Order) -> Dict:
        """Create contract request from order."""
        return {
            "proposal": 1,
            "amount": order.quantity,
            "basis": "stake",
            "contract_type": "CALL" if order.side == "buy" else "PUT",
            "currency": "USD",
            "duration": 5,  # Default duration in minutes
            "duration_unit": "m",
            "symbol": order.symbol
        }

    def _process_message(self, data: Dict):
        """Process incoming WebSocket message."""
        msg_type = data.get('msg_type')
        
        if msg_type == 'tick':
            self._handle_tick(data)
        elif msg_type == 'proposal':
            self._handle_proposal(data)
        elif msg_type == 'buy':
            self._handle_buy(data)
        elif msg_type == 'sell':
            self._handle_sell(data)
        elif msg_type == 'error':
            self.logger.error(f"Deriv API error: {data.get('error', {}).get('message')}")

    def _handle_messages(self):
        """Handle incoming WebSocket messages."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        while True:
            try:
                message = loop.run_until_complete(self.ws.recv())
                data = json.loads(message)
                self._process_message(data)
            except Exception as e:
                self.logger.error(f"Error handling message: {str(e)}")
                time.sleep(1)

    def _send_request(self, request: Dict) -> Dict:
        """Send request to Deriv API."""
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            async def send():
                await self.ws.send(json.dumps(request))
                response = await self.ws.recv()
                return json.loads(response)
                
            return loop.run_until_complete(send())
            
        except Exception as e:
            self.logger.error(f"Error sending request: {str(e)}")
            return {"error": {"message": str(e)}}

    def _handle_tick(self, data: Dict):
        """Handle tick data updates."""
        tick = data.get('tick', {})
        if tick:
            symbol = tick.get('symbol')
            if symbol in self.active_contracts:
                for contract in self.active_contracts.values():
                    if contract['symbol'] == symbol:
                        contract['current_spot'] = float(tick['quote'])

    def _handle_buy(self, data: Dict):
        """Handle buy confirmation."""
        if data.get('error'):
            self.logger.error(f"Buy error: {data['error']['message']}")
        else:
            contract = data.get('buy', {})
            if contract:
                self.active_contracts[contract['contract_id']] = contract

    def _handle_sell(self, data: Dict):
        """Handle sell confirmation."""
        if data.get('error'):
            self.logger.error(f"Sell error: {data['error']['message']}")
        else:
            contract_id = data.get('sell', {}).get('contract_id')
            if contract_id:
                self.active_contracts.pop(contract_id, None)

    def _handle_proposal(self, data: Dict):
        """Handle proposal response."""
        pass  # Implement if needed for specific functionality

    def __del__(self):
        """Cleanup on deletion."""
        try:
            if self.ws:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(self.ws.close())
            
            if hasattr(self, 'executor'):
                self.executor.shutdown(wait=True)
                
        except Exception as e:
            if hasattr(self, 'logger'):
                self.logger.error(f"Error during cleanup: {str(e)}")