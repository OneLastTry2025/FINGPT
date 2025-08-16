"""
Advanced ML/RL Engine for FinGPT
Utilizes full hardware potential with multiple ML/RL models
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import joblib
import os
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp

# Advanced ML imports
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier

# RL imports
import gymnasium as gym
from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv

# NLP and Sentiment Analysis
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

# Technical Analysis (fallback implementation)
try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    # Logger will be defined below, so we'll log the warning later

logger = logging.getLogger(__name__)

# Log TA-Lib availability warning if needed
if not TALIB_AVAILABLE:
    logger.warning("TA-Lib not available, using manual technical indicators")

class TradingEnvironment(gym.Env):
    """Custom Gym environment for RL trading"""
    
    def __init__(self, data: pd.DataFrame, initial_balance: float = 10000):
        super(TradingEnvironment, self).__init__()
        
        self.data = data.reset_index(drop=True)
        self.initial_balance = initial_balance
        self.current_step = 0
        self.balance = initial_balance
        self.position = 0  # 0: no position, 1: long, -1: short
        self.entry_price = 0
        self.max_steps = len(data) - 1
        
        # Action space: 0 = hold, 1 = buy, 2 = sell
        self.action_space = gym.spaces.Discrete(3)
        
        # Observation space: OHLCV + technical indicators
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32
        )
        
    def reset(self, seed=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.balance = self.initial_balance
        self.position = 0
        self.entry_price = 0
        return self._get_observation(), {}
    
    def step(self, action):
        self.current_step += 1
        reward = 0
        
        if self.current_step >= self.max_steps:
            return self._get_observation(), reward, True, False, {}
        
        current_price = self.data.iloc[self.current_step]['close']
        
        # Execute action
        if action == 1 and self.position != 1:  # Buy
            if self.position == -1:  # Close short position
                reward += (self.entry_price - current_price) / self.entry_price * 100
            self.position = 1
            self.entry_price = current_price
            
        elif action == 2 and self.position != -1:  # Sell
            if self.position == 1:  # Close long position
                reward += (current_price - self.entry_price) / self.entry_price * 100
            self.position = -1
            self.entry_price = current_price
        
        # Hold position reward/penalty
        if self.position == 1:
            reward += (current_price - self.entry_price) / self.entry_price * 0.1
        elif self.position == -1:
            reward += (self.entry_price - current_price) / self.entry_price * 0.1
        
        done = self.current_step >= self.max_steps
        return self._get_observation(), reward, done, False, {}
    
    def _get_observation(self):
        if self.current_step >= len(self.data):
            return np.zeros(10, dtype=np.float32)
        
        row = self.data.iloc[self.current_step]
        
        # Normalize prices relative to first price
        base_price = self.data.iloc[0]['close']
        
        obs = np.array([
            row['open'] / base_price,
            row['high'] / base_price,
            row['low'] / base_price,
            row['close'] / base_price,
            row['volume'] / 1000000,  # Normalize volume
            row.get('rsi', 50) / 100,
            row.get('macd', 0) / 10,
            row.get('bb_upper', row['close']) / base_price,
            row.get('bb_lower', row['close']) / base_price,
            float(self.position)
        ], dtype=np.float32)
        
        return obs

class AdvancedMLEngine:
    """Advanced ML/RL Engine utilizing full system capabilities"""
    
    def __init__(self, hardware_config: Dict[str, Any]):
        self.hardware_config = hardware_config
        
        # Utilize full hardware potential - Optimized for 48-core, 188GB system
        self.cpu_cores = hardware_config.get('parallel_analysis_workers', 36)  # Updated for 48-core
        self.memory_limit_gb = 150  # Increased from 100GB to 150GB for 188GB system
        
        # Initialize model storage
        self.ml_models = {}
        self.rl_agents = {}
        self.feature_scalers = {}
        self.model_ensembles = {}
        
        # Initialize sentiment analyzers
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        self.nlp_pipeline = None
        
        # Optimized thread pools for maximum performance system
        self.thread_executor = ThreadPoolExecutor(max_workers=24)  # Doubled for 48-core system
        self.process_executor = ProcessPoolExecutor(max_workers=24)  # Increased for maximum parallel processing
        
        # Performance tracking
        self.model_performance = {}
        self.training_history = {}
        
        logger.info(f"Advanced ML Engine initialized with {self.cpu_cores} workers, {self.memory_limit_gb}GB memory, 48-core optimization")
    
    async def initialize_advanced_models(self, symbols: List[str]):
        """Initialize advanced ML/RL models for all symbols"""
        logger.info("Initializing advanced ML/RL models...")
        
        try:
            # Initialize NLP pipeline
            await self._initialize_nlp_models()
            
            # Initialize models for each symbol in parallel
            tasks = []
            for symbol in symbols:
                task = asyncio.create_task(self._initialize_symbol_models(symbol))
                tasks.append(task)
            
            await asyncio.gather(*tasks)
            
            logger.info(f"Successfully initialized advanced models for {len(symbols)} symbols")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing advanced models: {e}")
            return False
    
    async def _initialize_nlp_models(self):
        """Initialize NLP models for sentiment analysis"""
        try:
            # Initialize lightweight NLP pipeline
            logger.info("Loading NLP models for sentiment analysis...")
            
            # Use a lightweight model that can run on CPU efficiently
            model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
            
            # Load in thread to avoid blocking
            self.nlp_pipeline = await asyncio.get_event_loop().run_in_executor(
                self.thread_executor,
                lambda: pipeline("sentiment-analysis", model=model_name, return_all_scores=True)
            )
            
            logger.info("NLP sentiment analysis models loaded successfully")
            
        except Exception as e:
            logger.warning(f"Could not load advanced NLP models, using basic sentiment: {e}")
            self.nlp_pipeline = None
    
    async def _initialize_symbol_models(self, symbol: str):
        """Initialize all ML/RL models for a specific symbol"""
        try:
            # Generate synthetic data for initial training
            synthetic_data = self._generate_synthetic_market_data(symbol)
            
            if len(synthetic_data) < 100:
                logger.warning(f"Insufficient data for {symbol}, skipping model training")
                return
            
            # Prepare features and targets
            features, targets = await self._prepare_advanced_features(synthetic_data)
            
            if len(features) < 50:
                logger.warning(f"Insufficient features for {symbol}, skipping")
                return
            
            # Initialize multiple ML models
            await self._train_ensemble_models(symbol, features, targets)
            
            # Initialize RL agent
            await self._initialize_rl_agent(symbol, synthetic_data)
            
            logger.info(f"Advanced models initialized for {symbol}")
            
        except Exception as e:
            logger.error(f"Error initializing models for {symbol}: {e}")
    
    def _generate_synthetic_market_data(self, symbol: str, days: int = 365) -> pd.DataFrame:
        """Generate realistic synthetic market data for training"""
        np.random.seed(hash(symbol) % 2**32)  # Consistent seed per symbol
        
        # Generate realistic price movements
        dates = pd.date_range(
            start=datetime.now() - timedelta(days=days),
            end=datetime.now(),
            freq='H'
        )
        
        # Geometric Brownian Motion with realistic parameters
        n_periods = len(dates)
        dt = 1/24/365  # Hourly data
        mu = 0.1  # Annual return
        sigma = 0.3  # Annual volatility
        
        # Generate price series
        returns = np.random.normal(
            (mu - 0.5 * sigma**2) * dt,
            sigma * np.sqrt(dt),
            n_periods
        )
        
        # Starting prices based on symbol
        start_prices = {
            'BTCUSDT': 45000, 'ETHUSDT': 3000, 'BNBUSDT': 400,
            'ADAUSDT': 0.5, 'DOTUSDT': 8, 'LINKUSDT': 15
        }
        
        start_price = start_prices.get(symbol, 100)
        prices = start_price * np.exp(np.cumsum(returns))
        
        # Generate OHLCV data
        data = []
        for i in range(len(prices)):
            if i == 0:
                open_price = start_price
            else:
                open_price = prices[i-1]
            
            close_price = prices[i]
            
            # Generate high/low with realistic spreads
            spread = abs(close_price - open_price) * 1.5 + close_price * 0.001
            high_price = max(open_price, close_price) + np.random.uniform(0, spread)
            low_price = min(open_price, close_price) - np.random.uniform(0, spread)
            
            # Generate volume
            volume = np.random.lognormal(10, 1) * 1000
            
            data.append({
                'timestamp': dates[i],
                'open': open_price,
                'high': high_price,
                'low': low_price,
                'close': close_price,
                'volume': volume
            })
        
        df = pd.DataFrame(data)
        
        # Add technical indicators
        df = self._add_advanced_technical_indicators(df)
        
        return df
    
    def _add_advanced_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add comprehensive technical indicators using TA-Lib or fallback implementations"""
        try:
            # Price arrays
            open_prices = df['open'].values
            high_prices = df['high'].values
            low_prices = df['low'].values
            close_prices = df['close'].values
            volume = df['volume'].values
            
            if TALIB_AVAILABLE:
                # Use TA-Lib for optimal performance
                # Trend indicators
                df['sma_10'] = talib.SMA(close_prices, timeperiod=10)
                df['sma_20'] = talib.SMA(close_prices, timeperiod=20)
                df['sma_50'] = talib.SMA(close_prices, timeperiod=50)
                df['ema_12'] = talib.EMA(close_prices, timeperiod=12)
                df['ema_26'] = talib.EMA(close_prices, timeperiod=26)
                
                # MACD
                df['macd'], df['macd_signal'], df['macd_hist'] = talib.MACD(close_prices)
                
                # RSI
                df['rsi'] = talib.RSI(close_prices, timeperiod=14)
                
                # Bollinger Bands
                df['bb_upper'], df['bb_middle'], df['bb_lower'] = talib.BBANDS(close_prices)
                
                # Momentum indicators
                df['momentum'] = talib.MOM(close_prices, timeperiod=10)
                df['williams_r'] = talib.WILLR(high_prices, low_prices, close_prices)
                df['cci'] = talib.CCI(high_prices, low_prices, close_prices)
                
                # Volume indicators
                df['ad_line'] = talib.AD(high_prices, low_prices, close_prices, volume)
                df['obv'] = talib.OBV(close_prices, volume)
                
                # Volatility indicators
                df['atr'] = talib.ATR(high_prices, low_prices, close_prices)
                df['natr'] = talib.NATR(high_prices, low_prices, close_prices)
                
                # Pattern recognition
                df['doji'] = talib.CDLDOJI(open_prices, high_prices, low_prices, close_prices)
                df['hammer'] = talib.CDLHAMMER(open_prices, high_prices, low_prices, close_prices)
                df['engulfing'] = talib.CDLENGULFING(open_prices, high_prices, low_prices, close_prices)
            else:
                # Fallback implementations using pandas
                # Simple Moving Averages
                df['sma_10'] = df['close'].rolling(window=10).mean()
                df['sma_20'] = df['close'].rolling(window=20).mean()
                df['sma_50'] = df['close'].rolling(window=50).mean()
                
                # Exponential Moving Averages
                df['ema_12'] = df['close'].ewm(span=12).mean()
                df['ema_26'] = df['close'].ewm(span=26).mean()
                
                # MACD
                df['macd'] = df['ema_12'] - df['ema_26']
                df['macd_signal'] = df['macd'].ewm(span=9).mean()
                df['macd_hist'] = df['macd'] - df['macd_signal']
                
                # RSI
                delta = df['close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss
                df['rsi'] = 100 - (100 / (1 + rs))
                
                # Bollinger Bands
                df['bb_middle'] = df['close'].rolling(window=20).mean()
                bb_std = df['close'].rolling(window=20).std()
                df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
                df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
                
                # Momentum
                df['momentum'] = df['close'] - df['close'].shift(10)
                
                # Williams %R
                high_14 = df['high'].rolling(window=14).max()
                low_14 = df['low'].rolling(window=14).min()
                df['williams_r'] = -100 * ((high_14 - df['close']) / (high_14 - low_14))
                
                # CCI (Commodity Channel Index)
                tp = (df['high'] + df['low'] + df['close']) / 3
                sma_tp = tp.rolling(window=20).mean()
                mad = tp.rolling(window=20).apply(lambda x: np.abs(x - x.mean()).mean())
                df['cci'] = (tp - sma_tp) / (0.015 * mad)
                
                # Accumulation/Distribution Line
                clv = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'])
                clv = clv.fillna(0)
                df['ad_line'] = (clv * df['volume']).cumsum()
                
                # On Balance Volume
                obv = np.where(df['close'] > df['close'].shift(1), df['volume'],
                              np.where(df['close'] < df['close'].shift(1), -df['volume'], 0))
                df['obv'] = pd.Series(obv).cumsum()
                
                # Average True Range
                high_low = df['high'] - df['low']
                high_close = np.abs(df['high'] - df['close'].shift())
                low_close = np.abs(df['low'] - df['close'].shift())
                true_range = np.maximum(high_low, np.maximum(high_close, low_close))
                df['atr'] = pd.Series(true_range).rolling(window=14).mean()
                df['natr'] = (df['atr'] / df['close']) * 100
                
                # Simple pattern recognition (basic implementations)
                # Doji: open and close are very close
                body_size = np.abs(df['close'] - df['open'])
                range_size = df['high'] - df['low']
                df['doji'] = np.where((body_size / range_size) < 0.1, 100, 0)
                
                # Hammer: small body at top of range with long lower shadow
                lower_shadow = df['open'].combine(df['close'], min) - df['low']
                upper_shadow = df['high'] - df['open'].combine(df['close'], max)
                df['hammer'] = np.where((lower_shadow > 2 * body_size) & (upper_shadow < body_size), 100, 0)
                
                # Engulfing: simplified version
                df['engulfing'] = np.where(
                    (df['close'] > df['open']) & 
                    (df['close'].shift(1) < df['open'].shift(1)) &
                    (df['close'] > df['open'].shift(1)) &
                    (df['open'] < df['close'].shift(1)), 100, 0)
            
            # Price action features (same for both)
            df['price_change'] = df['close'].pct_change()
            df['volume_change'] = df['volume'].pct_change()
            df['high_low_ratio'] = df['high'] / df['low']
            df['close_open_ratio'] = df['close'] / df['open']
            
            return df.fillna(method='bfill').fillna(0)
            
        except Exception as e:
            logger.warning(f"Error calculating advanced indicators: {e}")
            return df
    
    async def _prepare_advanced_features(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare advanced feature sets for ML training"""
        try:
            # Technical indicator features
            feature_columns = [
                'sma_10', 'sma_20', 'sma_50', 'ema_12', 'ema_26',
                'macd', 'macd_signal', 'macd_hist', 'rsi',
                'bb_upper', 'bb_middle', 'bb_lower',
                'momentum', 'williams_r', 'cci',
                'atr', 'natr', 'ad_line', 'obv',
                'doji', 'hammer', 'engulfing',
                'price_change', 'volume_change', 'high_low_ratio', 'close_open_ratio'
            ]
            
            # Add rolling statistical features
            for window in [5, 10, 20]:
                data[f'volatility_{window}'] = data['close'].rolling(window).std()
                data[f'volume_ma_{window}'] = data['volume'].rolling(window).mean()
                data[f'price_ma_{window}'] = data['close'].rolling(window).mean()
            
            # Add lagged features
            for lag in [1, 2, 3, 5]:
                data[f'close_lag_{lag}'] = data['close'].shift(lag)
                data[f'volume_lag_{lag}'] = data['volume'].shift(lag)
            
            # Update feature columns
            feature_columns.extend([
                'volatility_5', 'volatility_10', 'volatility_20',
                'volume_ma_5', 'volume_ma_10', 'volume_ma_20',
                'price_ma_5', 'price_ma_10', 'price_ma_20',
                'close_lag_1', 'close_lag_2', 'close_lag_3', 'close_lag_5',
                'volume_lag_1', 'volume_lag_2', 'volume_lag_3', 'volume_lag_5'
            ])
            
            # Filter available columns
            available_columns = [col for col in feature_columns if col in data.columns]
            
            # Clean data
            data = data.dropna()
            
            if len(data) < 50:
                return np.array([]), np.array([])
            
            # Extract features
            features = data[available_columns].values
            
            # Create targets (multi-class: down, neutral, up)
            future_returns = data['close'].shift(-5) / data['close'] - 1
            targets = np.where(future_returns > 0.01, 2,  # Up
                      np.where(future_returns < -0.01, 0, 1))  # Down, Neutral
            
            # Remove last 5 rows (no future data)
            features = features[:-5]
            targets = targets[:-5]
            
            return features, targets
            
        except Exception as e:
            logger.error(f"Error preparing advanced features: {e}")
            return np.array([]), np.array([])
    
    async def _train_ensemble_models(self, symbol: str, features: np.ndarray, targets: np.ndarray):
        """Train ensemble of ML models with advanced algorithms"""
        try:
            if len(features) < 50:
                return
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                features, targets, test_size=0.2, random_state=42, stratify=targets
            )
            
            # Initialize scalers
            scaler = RobustScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            self.feature_scalers[symbol] = scaler
            
            # Train multiple models in parallel (but limit to prevent blocking)
            models_config = {
                'random_forest': RandomForestClassifier(
                    n_estimators=50,  # Reduced from 200
                    max_depth=8,      # Reduced from 10
                    random_state=42,
                    n_jobs=2          # Reduced from -1
                ),
                'xgboost': xgb.XGBClassifier(
                    n_estimators=50,  # Reduced from 200
                    max_depth=4,      # Reduced from 6
                    learning_rate=0.1,
                    random_state=42,
                    n_jobs=2          # Reduced from -1
                )
            }
            
            trained_models = {}
            model_scores = {}
            
            # Train models
            for name, model in models_config.items():
                try:
                    logger.info(f"Training {name} for {symbol}")
                    
                    if name == 'neural_network':
                        model.fit(X_train_scaled, y_train)
                        score = model.score(X_test_scaled, y_test)
                    else:
                        model.fit(X_train, y_train)
                        score = model.score(X_test, y_test)
                    
                    trained_models[name] = model
                    model_scores[name] = score
                    
                    logger.info(f"{name} for {symbol} - Accuracy: {score:.3f}")
                    
                except Exception as e:
                    logger.warning(f"Failed to train {name} for {symbol}: {e}")
            
            # Store trained models
            if trained_models:
                self.ml_models[symbol] = trained_models
                self.model_performance[symbol] = model_scores
                
                logger.info(f"Successfully trained {len(trained_models)} models for {symbol}")
            
        except Exception as e:
            logger.error(f"Error training ensemble models for {symbol}: {e}")
    
    async def _initialize_rl_agent(self, symbol: str, data: pd.DataFrame):
        """Initialize RL agent using stable-baselines3"""
        try:
            logger.info(f"Initializing RL agent for {symbol}")
            
            # Create trading environment
            env = TradingEnvironment(data)
            
            # Wrap environment
            env = DummyVecEnv([lambda: env])
            
            # Initialize PPO agent
            rl_agent = PPO(
                "MlpPolicy",
                env,
                learning_rate=3e-4,
                n_steps=2048,
                batch_size=64,
                n_epochs=10,
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.2,
                verbose=0,
                device='cpu'  # Use CPU for compatibility
            )
            
            # Train agent with reduced timesteps to prevent blocking
            logger.info(f"Training RL agent for {symbol}...")
            rl_agent.learn(total_timesteps=1000)  # Reduced from 10000
            
            # Store trained agent
            self.rl_agents[symbol] = rl_agent
            
            logger.info(f"RL agent trained successfully for {symbol}")
            
        except Exception as e:
            logger.error(f"Error initializing RL agent for {symbol}: {e}")
    
    async def get_ml_prediction(self, symbol: str, features: np.ndarray) -> Dict[str, Any]:
        """Get ensemble ML prediction for a symbol"""
        try:
            if symbol not in self.ml_models:
                return {"confidence": 0.5, "prediction": "neutral", "method": "fallback"}
            
            models = self.ml_models[symbol]
            scaler = self.feature_scalers.get(symbol)
            
            if not models:
                return {"confidence": 0.5, "prediction": "neutral", "method": "fallback"}
            
            # Get predictions from all models
            predictions = []
            probabilities = []
            
            for name, model in models.items():
                try:
                    if name == 'neural_network' and scaler:
                        features_scaled = scaler.transform(features.reshape(1, -1))
                        prob = model.predict_proba(features_scaled)[0]
                    else:
                        prob = model.predict_proba(features.reshape(1, -1))[0]
                    
                    pred = np.argmax(prob)
                    predictions.append(pred)
                    probabilities.append(prob)
                    
                except Exception as e:
                    logger.warning(f"Error getting prediction from {name}: {e}")
            
            if not predictions:
                return {"confidence": 0.5, "prediction": "neutral", "method": "fallback"}
            
            # Ensemble prediction
            ensemble_prob = np.mean(probabilities, axis=0)
            ensemble_pred = np.argmax(ensemble_prob)
            confidence = float(np.max(ensemble_prob))
            
            prediction_map = {0: "bearish", 1: "neutral", 2: "bullish"}
            prediction = prediction_map.get(ensemble_pred, "neutral")
            
            return {
                "confidence": confidence,
                "prediction": prediction,
                "method": "ensemble_ml",
                "models_used": len(models),
                "individual_predictions": predictions
            }
            
        except Exception as e:
            logger.error(f"Error getting ML prediction for {symbol}: {e}")
            return {"confidence": 0.5, "prediction": "neutral", "method": "error_fallback"}
    
    async def get_rl_action(self, symbol: str, observation: np.ndarray) -> Dict[str, Any]:
        """Get RL agent action recommendation"""
        try:
            if symbol not in self.rl_agents:
                return {"action": "hold", "confidence": 0.5, "method": "fallback"}
            
            rl_agent = self.rl_agents[symbol]
            
            # Get action from RL agent
            action, _ = rl_agent.predict(observation, deterministic=True)
            
            action_map = {0: "hold", 1: "buy", 2: "sell"}
            action_name = action_map.get(action[0], "hold")
            
            return {
                "action": action_name,
                "confidence": 0.8,  # RL agents are generally confident
                "method": "reinforcement_learning",
                "raw_action": int(action[0])
            }
            
        except Exception as e:
            logger.error(f"Error getting RL action for {symbol}: {e}")
            return {"action": "hold", "confidence": 0.5, "method": "error_fallback"}
    
    async def analyze_market_sentiment(self, texts: List[str]) -> Dict[str, float]:
        """Analyze market sentiment from news/social media"""
        try:
            if not texts:
                return {"sentiment_score": 0.0, "confidence": 0.5}
            
            # Use VADER for quick sentiment analysis
            scores = []
            for text in texts:
                # Clean text
                clean_text = text.lower().strip()
                
                # Get VADER sentiment
                vader_score = self.sentiment_analyzer.polarity_scores(clean_text)
                compound_score = vader_score['compound']
                
                # Use NLP pipeline if available
                if self.nlp_pipeline:
                    try:
                        nlp_result = self.nlp_pipeline(clean_text[:512])  # Truncate for model limits
                        if nlp_result and len(nlp_result[0]) > 0:
                            # Convert to compound score format
                            for item in nlp_result[0]:
                                if item['label'] == 'LABEL_2':  # Positive
                                    compound_score = (compound_score + item['score']) / 2
                                elif item['label'] == 'LABEL_0':  # Negative
                                    compound_score = (compound_score - item['score']) / 2
                    except Exception as e:
                        logger.warning(f"NLP pipeline error: {e}")
                
                scores.append(compound_score)
            
            # Average sentiment
            avg_sentiment = np.mean(scores)
            confidence = min(0.9, len(texts) / 10.0)  # More texts = higher confidence
            
            return {
                "sentiment_score": float(avg_sentiment),
                "confidence": confidence,
                "texts_analyzed": len(texts)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing sentiment: {e}")
            return {"sentiment_score": 0.0, "confidence": 0.5}
    
    async def get_model_status(self) -> Dict[str, Any]:
        """Get comprehensive model status"""
        total_ml_models = sum(len(models) for models in self.ml_models.values())
        total_rl_agents = len(self.rl_agents)
        
        return {
            "ml_models_loaded": total_ml_models,
            "rl_agents_loaded": total_rl_agents,
            "symbols_with_models": list(self.ml_models.keys()),
            "symbols_with_rl": list(self.rl_agents.keys()),
            "nlp_available": self.nlp_pipeline is not None,
            "hardware_utilization": {
                "cpu_workers": self.cpu_cores,
                "memory_limit_gb": self.memory_limit_gb,
                "parallel_processing": True
            },
            "model_performance": self.model_performance
        }
    
    async def cleanup(self):
        """Cleanup resources"""
        try:
            self.thread_executor.shutdown(wait=True)
            self.process_executor.shutdown(wait=True)
            logger.info("Advanced ML Engine cleanup completed")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")