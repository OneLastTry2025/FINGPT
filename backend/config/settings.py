"""
FinGPT Trading System Configuration
Centralized configuration management for the trading system
"""

import os
from typing import Dict, List, Any
from pydantic import BaseModel, Field
from enum import Enum

class TradingMode(str, Enum):
    PAPER = "paper"
    LIVE = "live"
    BACKTEST = "backtest"

class RiskSettings(BaseModel):
    """Risk management configuration"""
    max_position_size: float = Field(default=0.05, description="Maximum position size as % of portfolio")
    daily_loss_limit: float = Field(default=0.02, description="Maximum daily loss as % of portfolio")
    max_drawdown: float = Field(default=0.10, description="Maximum portfolio drawdown")
    stop_loss_multiplier: float = Field(default=2.0, description="Stop loss ATR multiplier")
    take_profit_multiplier: float = Field(default=3.0, description="Take profit ATR multiplier")

class SystemSettings(BaseModel):
    """System performance configuration - Optimized for 48-core, 188GB RAM"""
    enable_parallel_processing: bool = Field(default=True)
    max_concurrent_symbols: int = Field(default=200)
    memory_cache_size_mb: int = Field(default=16384)  # 16GB cache
    data_compression: bool = Field(default=True)
    parallel_analysis_workers: int = Field(default=36)  # 75% of 48 cores
    data_processing_threads: int = Field(default=24)    # 50% of 48 cores
    strategy_evaluation_workers: int = Field(default=16) # Increased for 99% accuracy
    risk_calculation_threads: int = Field(default=8)     # Doubled
    concurrent_data_feeds: int = Field(default=50)       # Increased for multiple exchanges
    ml_model_workers: int = Field(default=12)             # Dedicated ML workers
    rl_training_processes: int = Field(default=8)         # Dedicated RL processes

class TradingSystemConfig:
    """Main configuration class for the FinGPT Trading System"""
    
    def __init__(self):
        self.trading_mode = TradingMode(os.getenv("TRADING_MODE", "paper"))
        self.risk_settings = RiskSettings()
        self.system_settings = SystemSettings()
        
        # Database configuration
        self.mongo_url = os.getenv("MONGO_URL", "mongodb://localhost:27017")
        self.db_name = os.getenv("DB_NAME", "fingpt_trading_system")
        
        # API Keys (to be provided by user)
        self.api_keys = {
            "binance_api_key": os.getenv("BINANCE_API_KEY"),
            "binance_secret_key": os.getenv("BINANCE_SECRET_KEY"),
            "mexc_api_key": os.getenv("MEXC_API_KEY"),
            "mexc_secret_key": os.getenv("MEXC_SECRET_KEY"),
            "alpha_vantage_api_key": os.getenv("ALPHA_VANTAGE_API_KEY"),
            "news_api_key": os.getenv("NEWS_API_KEY")
        }
        
        # Trading strategies configuration
        self.strategies = {
            "momentum": {
                "enabled": True,
                "lookback_period": 14,
                "threshold": 0.02
            },
            "mean_reversion": {
                "enabled": True,
                "lookback_period": 20,
                "std_dev_threshold": 2.0
            },
            "breakout": {
                "enabled": True,
                "lookback_period": 20,
                "volume_threshold": 1.5
            }
        }
        
        # Supported trading pairs
        self.trading_pairs = [
            "BTCUSDT", "ETHUSDT", "BNBUSDT", "ADAUSDT", "DOTUSDT",
            "LINKUSDT", "LTCUSDT", "BCHUSDT", "XLMUSDT", "EOSUSDT"
        ]
        
        # Data feed intervals
        self.intervals = ["1m", "5m", "15m", "1h", "4h", "1d"]
        
        # Technical indicators configuration
        self.indicators = {
            "sma": [10, 20, 50, 100, 200],
            "ema": [12, 26, 50],
            "rsi": {"period": 14},
            "macd": {"fast": 12, "slow": 26, "signal": 9},
            "bb": {"period": 20, "std": 2},
            "atr": {"period": 14},
            "stoch": {"k_period": 14, "d_period": 3},
            "williams_r": {"period": 14},
            "cci": {"period": 20},
            "momentum": {"period": 10}
        }
    
    def get_active_api_keys(self) -> Dict[str, str]:
        """Get only the API keys that are set"""
        return {k: v for k, v in self.api_keys.items() if v is not None}
    
    def is_api_configured(self, exchange: str) -> bool:
        """Check if API keys for specific exchange are configured"""
        if exchange.lower() == "binance":
            return bool(self.api_keys.get("binance_api_key") and 
                       self.api_keys.get("binance_secret_key"))
        elif exchange.lower() == "mexc":
            return bool(self.api_keys.get("mexc_api_key") and 
                       self.api_keys.get("mexc_secret_key"))
        return False
    
    def get_hardware_optimization_config(self) -> Dict[str, Any]:
        """Get hardware optimization configuration"""
        return {
            "target_cpu": "16-core ARM Neoverse-N1",
            "memory": "62GB RAM available",
            "storage_limit": "116GB high-speed storage",
            "architecture": "ARM64 cloud-optimized",
            "performance_mode": "High-throughput parallel processing",
            "max_concurrent_symbols": self.system_settings.max_concurrent_symbols,
            "parallel_analysis_workers": self.system_settings.parallel_analysis_workers,
            "data_processing_threads": self.system_settings.data_processing_threads,
            "strategy_evaluation_workers": self.system_settings.strategy_evaluation_workers,
            "risk_calculation_threads": self.system_settings.risk_calculation_threads,
            "memory_cache_size_mb": self.system_settings.memory_cache_size_mb,
            "concurrent_data_feeds": self.system_settings.concurrent_data_feeds
        }

# Global configuration instance
config = TradingSystemConfig()