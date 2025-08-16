"""
Real-time Data Feeds Manager for FinGPT Trading System
Handles multiple data sources: MEXC, Binance, Yahoo Finance
"""

import asyncio
import aiohttp
import websockets
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any
import pandas as pd
import numpy as np
import yfinance as yf
import ccxt
import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class MarketTick:
    symbol: str
    price: float
    volume: float
    timestamp: datetime
    bid: float = None
    ask: float = None
    source: str = "unknown"

class DataFeedManager:
    """Manages multiple real-time data feeds"""
    
    def __init__(self, config):
        self.config = config
        self.active_feeds = {}
        self.subscribers = {}
        self.is_running = False
        self.executor = ThreadPoolExecutor(max_workers=20)
        
    async def start(self):
        """Start all data feeds"""
        self.is_running = True
        logger.info("Starting FinGPT Data Feed Manager...")
        
        # Start Yahoo Finance data feed for stocks/forex
        asyncio.create_task(self._yahoo_finance_feed())
        
        # Start cryptocurrency feeds if API keys available
        if self.config.is_api_configured("binance"):
            asyncio.create_task(self._binance_websocket_feed())
        
        logger.info("All data feeds started successfully")
    
    async def stop(self):
        """Stop all data feeds"""
        self.is_running = False
        logger.info("Stopping all data feeds...")
    
    def subscribe(self, symbol: str, callback: Callable[[MarketTick], None]):
        """Subscribe to real-time data for a symbol"""
        if symbol not in self.subscribers:
            self.subscribers[symbol] = []
        self.subscribers[symbol].append(callback)
    
    def unsubscribe(self, symbol: str, callback: Callable[[MarketTick], None]):
        """Unsubscribe from symbol updates"""
        if symbol in self.subscribers:
            self.subscribers[symbol].remove(callback)
    
    async def _notify_subscribers(self, tick: MarketTick):
        """Notify all subscribers of new market data"""
        if tick.symbol in self.subscribers:
            for callback in self.subscribers[tick.symbol]:
                try:
                    await callback(tick) if asyncio.iscoroutinefunction(callback) else callback(tick)
                except Exception as e:
                    logger.error(f"Error in subscriber callback: {e}")
    
    async def _yahoo_finance_feed(self):
        """Yahoo Finance data feed for stocks and forex"""
        stocks = ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN", "NVDA", "META"]
        forex = ["EURUSD=X", "GBPUSD=X", "USDJPY=X"]
        symbols = stocks + forex
        
        while self.is_running:
            try:
                await asyncio.sleep(5)  # Update every 5 seconds
                
                # Fetch data in thread pool to avoid blocking
                data = await asyncio.get_event_loop().run_in_executor(
                    self.executor, self._fetch_yahoo_data, symbols
                )
                
                for symbol, info in data.items():
                    if info:
                        tick = MarketTick(
                            symbol=symbol,
                            price=info.get('regularMarketPrice', 0),
                            volume=info.get('regularMarketVolume', 0),
                            timestamp=datetime.now(),
                            bid=info.get('bid', 0),
                            ask=info.get('ask', 0),
                            source="yahoo_finance"
                        )
                        await self._notify_subscribers(tick)
                        
            except Exception as e:
                logger.error(f"Error in Yahoo Finance feed: {e}")
                await asyncio.sleep(10)
    
    def _fetch_yahoo_data(self, symbols: List[str]) -> Dict[str, Any]:
        """Fetch Yahoo Finance data (runs in thread pool)"""
        data = {}
        try:
            for symbol in symbols:
                ticker = yf.Ticker(symbol)
                info = ticker.info
                data[symbol] = info
        except Exception as e:
            logger.error(f"Error fetching Yahoo data: {e}")
        return data
    
    async def _binance_websocket_feed(self):
        """Binance WebSocket feed for crypto data"""
        symbols = [s.lower() + "@ticker" for s in self.config.trading_pairs]
        stream = "/".join(symbols)
        
        uri = f"wss://stream.binance.com:9443/ws/{stream}"
        
        while self.is_running:
            try:
                async with websockets.connect(uri) as websocket:
                    logger.info("Connected to Binance WebSocket")
                    
                    async for message in websocket:
                        if not self.is_running:
                            break
                        
                        data = json.loads(message)
                        
                        # Parse Binance ticker data
                        if 'e' in data and data['e'] == '24hrTicker':
                            tick = MarketTick(
                                symbol=data['s'],  # Symbol
                                price=float(data['c']),  # Current price
                                volume=float(data['v']),  # Volume
                                timestamp=datetime.now(),
                                bid=float(data['b']),  # Best bid
                                ask=float(data['a']),  # Best ask
                                source="binance"
                            )
                            await self._notify_subscribers(tick)
                            
            except Exception as e:
                logger.error(f"Error in Binance WebSocket: {e}")
                await asyncio.sleep(10)
    
    async def get_historical_data(
        self, 
        symbol: str, 
        interval: str = "1h", 
        limit: int = 100,
        source: str = "auto"
    ) -> pd.DataFrame:
        """Get historical market data"""
        
        try:
            if source == "auto":
                # Auto-detect best source based on symbol
                if symbol.endswith("USDT") or symbol in self.config.trading_pairs:
                    source = "binance"
                else:
                    source = "yahoo"
            
            if source == "yahoo":
                return await self._get_yahoo_historical(symbol, interval, limit)
            elif source == "binance":
                return await self._get_binance_historical(symbol, interval, limit)
            else:
                logger.error(f"Unknown data source: {source}")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Error getting historical data for {symbol}: {e}")
            return pd.DataFrame()
    
    async def _get_yahoo_historical(self, symbol: str, interval: str, limit: int) -> pd.DataFrame:
        """Get historical data from Yahoo Finance"""
        try:
            # Map intervals
            yahoo_intervals = {
                "1m": "1m", "5m": "5m", "15m": "15m", 
                "1h": "1h", "4h": "4h", "1d": "1d"
            }
            
            yf_interval = yahoo_intervals.get(interval, "1h")
            
            # Calculate period based on limit and interval
            if interval in ["1m", "5m"]:
                period = "7d"
            elif interval in ["15m", "1h"]:
                period = "60d"
            else:
                period = "1y"
            
            # Fetch data in thread pool
            ticker = yf.Ticker(symbol)
            data = await asyncio.get_event_loop().run_in_executor(
                self.executor,
                lambda: ticker.history(period=period, interval=yf_interval)
            )
            
            if not data.empty:
                # Standardize column names
                data = data.rename(columns={
                    'Open': 'open', 'High': 'high', 'Low': 'low', 
                    'Close': 'close', 'Volume': 'volume'
                })
                data['timestamp'] = data.index
                return data.tail(limit)
            
            return pd.DataFrame()
            
        except Exception as e:
            logger.error(f"Error fetching Yahoo historical data: {e}")
            return pd.DataFrame()
    
    async def _get_binance_historical(self, symbol: str, interval: str, limit: int) -> pd.DataFrame:
        """Get historical data from Binance"""
        try:
            # Use ccxt for Binance API
            exchange = ccxt.binance({'enableRateLimit': True})
            
            # Map intervals to Binance format
            binance_intervals = {
                "1m": "1m", "5m": "5m", "15m": "15m",
                "1h": "1h", "4h": "4h", "1d": "1d"
            }
            
            timeframe = binance_intervals.get(interval, "1h")
            
            # Fetch OHLCV data
            ohlcv = await asyncio.get_event_loop().run_in_executor(
                self.executor,
                lambda: exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            )
            
            if ohlcv:
                df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                return df
            
            return pd.DataFrame()
            
        except Exception as e:
            logger.error(f"Error fetching Binance historical data: {e}")
            return pd.DataFrame()
    
    async def get_market_summary(self) -> Dict[str, Any]:
        """Get overall market summary"""
        summary = {
            "active_feeds": len(self.active_feeds),
            "subscribed_symbols": len(self.subscribers),
            "status": "running" if self.is_running else "stopped",
            "data_sources": ["yahoo_finance", "binance"],
            "last_update": datetime.now().isoformat()
        }
        
        return summary