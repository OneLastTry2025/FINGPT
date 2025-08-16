"""
ML Enhanced Trading Engine for FinGPT
Core trading engine with machine learning and reinforcement learning capabilities
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
import json

# ML/AI imports
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import warnings
warnings.filterwarnings('ignore')

# Trading system imports
from trading_system.storage import MongoDBManager
from trading_data.data_feeds import DataFeedManager, MarketTick
from config.settings import TradingSystemConfig

logger = logging.getLogger(__name__)

@dataclass
class TradingSignal:
    symbol: str
    action: str  # 'buy', 'sell', 'hold'
    confidence: float
    price: float
    strategy: str
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Position:
    symbol: str
    quantity: float
    entry_price: float
    current_price: float
    unrealized_pnl: float
    timestamp: datetime

class MLEnhancedTradingEngine:
    """Main trading engine with ML capabilities"""
    
    def __init__(self, storage_manager: MongoDBManager):
        self.storage = storage_manager
        self.config = TradingSystemConfig()
        self.data_feed = DataFeedManager(self.config)
        
        # Trading state
        self.is_running = False
        self.positions: Dict[str, Position] = {}
        self.portfolio_value = 100000.0  # Starting with $100k paper money
        self.daily_pnl = 0.0
        
        # ML models (will be loaded/trained)
        self.ml_models = {}
        self.feature_scalers = {}
        
        # Technical analysis indicators
        self.indicators = {}
        
        # Performance metrics
        self.trades_today = 0
        self.win_rate = 0.0
        self.total_trades = 0
        
    async def start_enhanced_engine(
        self, 
        symbols: List[str], 
        enable_ml: bool = True,
        strategies: List[str] = None
    ):
        """Start the enhanced trading engine"""
        try:
            logger.info("Starting FinGPT ML Enhanced Trading Engine...")
            
            self.is_running = True
            
            # Start data feeds
            await self.data_feed.start()
            
            # Subscribe to market data for all symbols
            for symbol in symbols:
                self.data_feed.subscribe(symbol, self._on_market_tick)
            
            # Initialize ML models if enabled
            if enable_ml:
                await self._initialize_ml_models()
            
            # Start trading strategies
            if not strategies:
                strategies = ["momentum", "mean_reversion", "breakout"]
            
            for strategy in strategies:
                asyncio.create_task(self._run_strategy(strategy, symbols))
            
            # Start performance monitoring
            asyncio.create_task(self._performance_monitor())
            
            # Start risk management
            asyncio.create_task(self._risk_management())
            
            logger.info(f"Trading engine started successfully for {len(symbols)} symbols")
            
        except Exception as e:
            logger.error(f"Error starting trading engine: {e}")
            self.is_running = False
            raise
    
    async def stop_engine(self):
        """Stop the trading engine"""
        logger.info("Stopping trading engine...")
        self.is_running = False
        await self.data_feed.stop()
        
        # Close all positions in paper trading
        for symbol in list(self.positions.keys()):
            await self._close_position(symbol, "engine_shutdown")
        
        logger.info("Trading engine stopped")
    
    async def _on_market_tick(self, tick: MarketTick):
        """Handle incoming market data"""
        try:
            # Update position values
            if tick.symbol in self.positions:
                position = self.positions[tick.symbol]
                position.current_price = tick.price
                position.unrealized_pnl = (tick.price - position.entry_price) * position.quantity
            
            # Store the market tick (optional, for analysis)
            await self.storage.log_system_event(
                "info", 
                f"Market tick received",
                {"symbol": tick.symbol, "price": tick.price, "source": tick.source}
            )
            
        except Exception as e:
            logger.error(f"Error processing market tick: {e}")
    
    async def _initialize_ml_models(self):
        """Initialize and train ML models for predictions"""
        logger.info("Initializing ML models...")
        
        try:
            for symbol in self.config.trading_pairs:
                # Create a simple ML model for price direction prediction
                model = RandomForestClassifier(n_estimators=100, random_state=42)
                scaler = StandardScaler()
                
                # Try to load existing model or train new one
                historical_data = await self.data_feed.get_historical_data(symbol, "1h", 1000)
                
                if not historical_data.empty:
                    features, targets = self._prepare_ml_features(historical_data)
                    
                    if len(features) > 50:  # Minimum data for training
                        # Scale features
                        features_scaled = scaler.fit_transform(features)
                        
                        # Train model
                        model.fit(features_scaled, targets)
                        
                        self.ml_models[symbol] = model
                        self.feature_scalers[symbol] = scaler
                        
                        logger.info(f"ML model trained for {symbol}")
                    else:
                        logger.warning(f"Insufficient data to train ML model for {symbol}")
                else:
                    logger.warning(f"No historical data available for {symbol}")
            
            logger.info(f"ML models initialized for {len(self.ml_models)} symbols")
            
        except Exception as e:
            logger.error(f"Error initializing ML models: {e}")
    
    def _prepare_ml_features(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare features and targets for ML training"""
        try:
            # Calculate technical indicators
            data = self._calculate_technical_indicators(data.copy())
            
            # Prepare features
            feature_columns = [
                'rsi', 'macd', 'bb_upper', 'bb_lower', 'sma_20', 'ema_12', 
                'atr', 'volume_sma', 'price_change', 'volume_change'
            ]
            
            # Fill missing values
            data = data.dropna()
            
            features = data[feature_columns].values
            
            # Create targets (1 for price up, 0 for price down)
            data['future_return'] = data['close'].shift(-1) / data['close'] - 1
            targets = (data['future_return'] > 0.001).astype(int).values[:-1]  # Remove last NaN
            features = features[:-1]  # Align with targets
            
            return features, targets
            
        except Exception as e:
            logger.error(f"Error preparing ML features: {e}")
            return np.array([]), np.array([])
    
    def _calculate_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators for the data"""
        try:
            # Simple Moving Average
            data['sma_20'] = data['close'].rolling(window=20).mean()
            data['sma_50'] = data['close'].rolling(window=50).mean()
            
            # Exponential Moving Average
            data['ema_12'] = data['close'].ewm(span=12).mean()
            data['ema_26'] = data['close'].ewm(span=26).mean()
            
            # MACD
            data['macd'] = data['ema_12'] - data['ema_26']
            data['macd_signal'] = data['macd'].ewm(span=9).mean()
            
            # RSI
            delta = data['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            data['rsi'] = 100 - (100 / (1 + rs))
            
            # Bollinger Bands
            bb_period = 20
            bb_std = 2
            data['bb_middle'] = data['close'].rolling(window=bb_period).mean()
            bb_std_dev = data['close'].rolling(window=bb_period).std()
            data['bb_upper'] = data['bb_middle'] + (bb_std_dev * bb_std)
            data['bb_lower'] = data['bb_middle'] - (bb_std_dev * bb_std)
            
            # ATR (Average True Range)
            high_low = data['high'] - data['low']
            high_close = np.abs(data['high'] - data['close'].shift())
            low_close = np.abs(data['low'] - data['close'].shift())
            true_range = np.maximum(high_low, np.maximum(high_close, low_close))
            data['atr'] = true_range.rolling(window=14).mean()
            
            # Volume indicators
            data['volume_sma'] = data['volume'].rolling(window=20).mean()
            
            # Price and volume changes
            data['price_change'] = data['close'].pct_change()
            data['volume_change'] = data['volume'].pct_change()
            
            return data
            
        except Exception as e:
            logger.error(f"Error calculating technical indicators: {e}")
            return data
    
    async def _run_strategy(self, strategy_name: str, symbols: List[str]):
        """Run a specific trading strategy"""
        logger.info(f"Starting {strategy_name} strategy for {len(symbols)} symbols")
        
        while self.is_running:
            try:
                for symbol in symbols:
                    # Get recent market data
                    data = await self.data_feed.get_historical_data(symbol, "1h", 100)
                    
                    if not data.empty:
                        # Apply strategy
                        signal = await self._apply_strategy(strategy_name, symbol, data)
                        
                        if signal and signal.action != "hold":
                            # Execute trading signal
                            await self._execute_signal(signal)
                
                # Wait before next strategy iteration
                await asyncio.sleep(60)  # Run every minute
                
            except Exception as e:
                logger.error(f"Error in {strategy_name} strategy: {e}")
                await asyncio.sleep(10)
    
    async def _apply_strategy(self, strategy_name: str, symbol: str, data: pd.DataFrame) -> Optional[TradingSignal]:
        """Apply a specific trading strategy"""
        try:
            # Calculate indicators
            data = self._calculate_technical_indicators(data)
            
            current = data.iloc[-1]
            
            signal = None
            
            if strategy_name == "momentum":
                signal = self._momentum_strategy(symbol, data, current)
            elif strategy_name == "mean_reversion":
                signal = self._mean_reversion_strategy(symbol, data, current)
            elif strategy_name == "breakout":
                signal = self._breakout_strategy(symbol, data, current)
            
            # Enhance with ML prediction if available
            if signal and symbol in self.ml_models:
                ml_confidence = await self._get_ml_prediction(symbol, data)
                signal.confidence = (signal.confidence + ml_confidence) / 2
            
            return signal
            
        except Exception as e:
            logger.error(f"Error applying {strategy_name} strategy: {e}")
            return None
    
    def _momentum_strategy(self, symbol: str, data: pd.DataFrame, current: pd.Series) -> Optional[TradingSignal]:
        """Momentum trading strategy"""
        try:
            # Simple momentum based on moving averages and RSI
            if pd.isna(current['sma_20']) or pd.isna(current['rsi']):
                return None
            
            confidence = 0.5
            
            # Bullish momentum
            if (current['close'] > current['sma_20'] and 
                current['sma_20'] > current['sma_50'] and
                current['rsi'] > 50 and current['rsi'] < 70):
                
                confidence = min(0.8, 0.5 + (current['rsi'] - 50) / 100)
                
                return TradingSignal(
                    symbol=symbol,
                    action="buy",
                    confidence=confidence,
                    price=current['close'],
                    strategy="momentum",
                    timestamp=datetime.now()
                )
            
            # Bearish momentum
            elif (current['close'] < current['sma_20'] and 
                  current['sma_20'] < current['sma_50'] and
                  current['rsi'] < 50 and current['rsi'] > 30):
                
                confidence = min(0.8, 0.5 + (50 - current['rsi']) / 100)
                
                return TradingSignal(
                    symbol=symbol,
                    action="sell",
                    confidence=confidence,
                    price=current['close'],
                    strategy="momentum",
                    timestamp=datetime.now()
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Error in momentum strategy: {e}")
            return None
    
    def _mean_reversion_strategy(self, symbol: str, data: pd.DataFrame, current: pd.Series) -> Optional[TradingSignal]:
        """Mean reversion trading strategy"""
        try:
            if pd.isna(current['bb_upper']) or pd.isna(current['bb_lower']):
                return None
            
            # Price near lower Bollinger Band - potential buy
            if current['close'] <= current['bb_lower'] * 1.01 and current['rsi'] < 30:
                return TradingSignal(
                    symbol=symbol,
                    action="buy",
                    confidence=0.7,
                    price=current['close'],
                    strategy="mean_reversion",
                    timestamp=datetime.now()
                )
            
            # Price near upper Bollinger Band - potential sell
            elif current['close'] >= current['bb_upper'] * 0.99 and current['rsi'] > 70:
                return TradingSignal(
                    symbol=symbol,
                    action="sell",
                    confidence=0.7,
                    price=current['close'],
                    strategy="mean_reversion",
                    timestamp=datetime.now()
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Error in mean reversion strategy: {e}")
            return None
    
    def _breakout_strategy(self, symbol: str, data: pd.DataFrame, current: pd.Series) -> Optional[TradingSignal]:
        """Breakout trading strategy"""
        try:
            # Look for breakouts above recent high with volume confirmation
            recent_data = data.tail(20)
            
            if len(recent_data) < 20:
                return None
            
            recent_high = recent_data['high'].max()
            recent_low = recent_data['low'].min()
            avg_volume = recent_data['volume'].mean()
            
            # Bullish breakout
            if (current['close'] > recent_high * 1.005 and  # Break above recent high
                current['volume'] > avg_volume * 1.5 and    # Volume confirmation
                current['rsi'] < 80):                       # Not overbought
                
                return TradingSignal(
                    symbol=symbol,
                    action="buy",
                    confidence=0.75,
                    price=current['close'],
                    strategy="breakout",
                    timestamp=datetime.now()
                )
            
            # Bearish breakout
            elif (current['close'] < recent_low * 0.995 and  # Break below recent low
                  current['volume'] > avg_volume * 1.5 and   # Volume confirmation
                  current['rsi'] > 20):                      # Not oversold
                
                return TradingSignal(
                    symbol=symbol,
                    action="sell",
                    confidence=0.75,
                    price=current['close'],
                    strategy="breakout",
                    timestamp=datetime.now()
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Error in breakout strategy: {e}")
            return None
    
    async def _get_ml_prediction(self, symbol: str, data: pd.DataFrame) -> float:
        """Get ML model prediction confidence"""
        try:
            if symbol not in self.ml_models:
                return 0.5
            
            model = self.ml_models[symbol]
            scaler = self.feature_scalers[symbol]
            
            # Prepare current features
            data_with_indicators = self._calculate_technical_indicators(data.copy())
            current = data_with_indicators.iloc[-1]
            
            feature_columns = [
                'rsi', 'macd', 'bb_upper', 'bb_lower', 'sma_20', 'ema_12', 
                'atr', 'volume_sma', 'price_change', 'volume_change'
            ]
            
            features = []
            for col in feature_columns:
                if col in current and not pd.isna(current[col]):
                    features.append(current[col])
                else:
                    features.append(0.0)
            
            features_scaled = scaler.transform([features])
            
            # Get prediction probability
            prob = model.predict_proba(features_scaled)[0]
            
            # Return confidence (max probability)
            return max(prob)
            
        except Exception as e:
            logger.error(f"Error getting ML prediction: {e}")
            return 0.5
    
    async def _execute_signal(self, signal: TradingSignal):
        """Execute a trading signal"""
        try:
            # Check if we should trade this signal
            if signal.confidence < 0.6:  # Minimum confidence threshold
                return
            
            # Risk management checks
            if not await self._risk_check(signal):
                return
            
            # Execute the trade (paper trading)
            if signal.action == "buy":
                await self._open_position(signal)
            elif signal.action == "sell":
                if signal.symbol in self.positions:
                    await self._close_position(signal.symbol, signal.strategy)
                else:
                    # Short selling in paper trading
                    await self._open_position(signal)
            
            # Store the signal
            await self.storage.store_trading_signal(
                signal.symbol, 
                signal.strategy, 
                {
                    "action": signal.action,
                    "confidence": signal.confidence,
                    "price": signal.price,
                    "timestamp": signal.timestamp.isoformat()
                }
            )
            
        except Exception as e:
            logger.error(f"Error executing signal: {e}")
    
    async def _open_position(self, signal: TradingSignal):
        """Open a new trading position"""
        try:
            # Calculate position size based on risk management
            position_size = self._calculate_position_size(signal)
            
            if position_size == 0:
                return
            
            # Create position
            position = Position(
                symbol=signal.symbol,
                quantity=position_size,
                entry_price=signal.price,
                current_price=signal.price,
                unrealized_pnl=0.0,
                timestamp=datetime.now()
            )
            
            self.positions[signal.symbol] = position
            self.trades_today += 1
            self.total_trades += 1
            
            logger.info(f"Opened {signal.action} position: {signal.symbol} @ {signal.price:.4f}, size: {position_size:.4f}")
            
        except Exception as e:
            logger.error(f"Error opening position: {e}")
    
    async def _close_position(self, symbol: str, reason: str):
        """Close an existing position"""
        try:
            if symbol not in self.positions:
                return
            
            position = self.positions[symbol]
            
            # Calculate P&L
            pnl = position.unrealized_pnl
            self.portfolio_value += pnl
            self.daily_pnl += pnl
            
            # Remove position
            del self.positions[symbol]
            
            # Update win rate
            if pnl > 0:
                self.win_rate = (self.win_rate * (self.total_trades - 1) + 1) / self.total_trades
            else:
                self.win_rate = (self.win_rate * (self.total_trades - 1)) / self.total_trades
            
            logger.info(f"Closed position: {symbol}, P&L: {pnl:.2f}, reason: {reason}")
            
        except Exception as e:
            logger.error(f"Error closing position: {e}")
    
    def _calculate_position_size(self, signal: TradingSignal) -> float:
        """Calculate position size based on risk management"""
        try:
            # Kelly Criterion simplified version
            win_rate = max(0.5, self.win_rate) if self.total_trades > 10 else 0.55
            
            # Max position size as % of portfolio
            max_position_value = self.portfolio_value * self.config.risk_settings.max_position_size
            
            # Adjust by confidence
            adjusted_position_value = max_position_value * signal.confidence
            
            # Convert to quantity
            position_size = adjusted_position_value / signal.price
            
            return position_size
            
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return 0.0
    
    async def _risk_check(self, signal: TradingSignal) -> bool:
        """Perform risk management checks"""
        try:
            # Check daily loss limit
            daily_loss_limit = self.portfolio_value * self.config.risk_settings.daily_loss_limit
            if self.daily_pnl < -daily_loss_limit:
                logger.warning("Daily loss limit reached, rejecting signal")
                return False
            
            # Check maximum drawdown
            # (This would need more sophisticated calculation with high-water mark)
            
            # Check if we're already holding this symbol
            if signal.symbol in self.positions and signal.action == "buy":
                logger.info(f"Already holding {signal.symbol}, rejecting buy signal")
                return False
            
            # Check trading hours (if needed)
            # Check market conditions, etc.
            
            return True
            
        except Exception as e:
            logger.error(f"Error in risk check: {e}")
            return False
    
    async def _risk_management(self):
        """Continuous risk management monitoring"""
        while self.is_running:
            try:
                # Check all positions for stop losses
                for symbol, position in list(self.positions.items()):
                    # Simple stop loss at 2% loss
                    if position.unrealized_pnl < -0.02 * abs(position.quantity * position.entry_price):
                        logger.warning(f"Stop loss triggered for {symbol}")
                        await self._close_position(symbol, "stop_loss")
                
                await asyncio.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                logger.error(f"Error in risk management: {e}")
                await asyncio.sleep(30)
    
    async def _performance_monitor(self):
        """Monitor system performance and trading metrics"""
        while self.is_running:
            try:
                # Calculate portfolio metrics
                total_unrealized_pnl = sum(pos.unrealized_pnl for pos in self.positions.values())
                current_portfolio_value = self.portfolio_value + total_unrealized_pnl
                
                metrics = {
                    "portfolio_value": current_portfolio_value,
                    "daily_pnl": self.daily_pnl,
                    "unrealized_pnl": total_unrealized_pnl,
                    "open_positions": len(self.positions),
                    "trades_today": self.trades_today,
                    "win_rate": self.win_rate,
                    "total_trades": self.total_trades
                }
                
                # Store performance metrics
                await self.storage.store_performance_metrics(metrics)
                
                # Store portfolio state
                portfolio_state = {
                    "value": current_portfolio_value,
                    "positions": {
                        symbol: {
                            "quantity": pos.quantity,
                            "entry_price": pos.entry_price,
                            "current_price": pos.current_price,
                            "unrealized_pnl": pos.unrealized_pnl
                        } for symbol, pos in self.positions.items()
                    }
                }
                await self.storage.store_portfolio_state(portfolio_state)
                
                await asyncio.sleep(60)  # Update every minute
                
            except Exception as e:
                logger.error(f"Error in performance monitor: {e}")
                await asyncio.sleep(60)
    
    async def get_engine_status(self) -> Dict[str, Any]:
        """Get current engine status"""
        total_unrealized_pnl = sum(pos.unrealized_pnl for pos in self.positions.values())
        
        return {
            "is_running": self.is_running,
            "portfolio_value": self.portfolio_value + total_unrealized_pnl,
            "daily_pnl": self.daily_pnl,
            "open_positions": len(self.positions),
            "trades_today": self.trades_today,
            "win_rate": self.win_rate,
            "total_trades": self.total_trades,
            "active_symbols": list(self.positions.keys()),
            "ml_models_loaded": len(self.ml_models)
        }