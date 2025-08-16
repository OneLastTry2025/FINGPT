"""
MongoDB Storage Manager for FinGPT Trading System
Handles data persistence, compression, and retrieval
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
import logging
from pymongo import ASCENDING, DESCENDING
import json
import gzip
import base64

logger = logging.getLogger(__name__)

class MongoDBManager:
    """Advanced MongoDB manager with compression and optimization"""
    
    def __init__(self, client: AsyncIOMotorClient, db_name: str):
        self.client = client
        self.db: AsyncIOMotorDatabase = client[db_name]
        self.setup_complete = False
    
    async def setup_collections(self):
        """Setup MongoDB collections with proper indexing"""
        if self.setup_complete:
            return
        
        try:
            # Market data collection
            market_data = self.db.market_data
            await market_data.create_index([
                ("symbol", ASCENDING),
                ("timestamp", DESCENDING),
                ("interval", ASCENDING)
            ])
            
            # Trading signals collection
            signals = self.db.trading_signals
            await signals.create_index([
                ("symbol", ASCENDING),
                ("timestamp", DESCENDING),
                ("strategy", ASCENDING)
            ])
            
            # Portfolio collection
            portfolio = self.db.portfolio
            await portfolio.create_index([
                ("symbol", ASCENDING),
                ("timestamp", DESCENDING)
            ])
            
            # Performance metrics collection
            performance = self.db.performance_metrics
            await performance.create_index([
                ("timestamp", DESCENDING),
                ("metric_type", ASCENDING)
            ])
            
            # System logs collection
            system_logs = self.db.system_logs
            await system_logs.create_index([
                ("timestamp", DESCENDING),
                ("level", ASCENDING)
            ])
            
            self.setup_complete = True
            logger.info("MongoDB collections setup complete")
            
        except Exception as e:
            logger.error(f"Error setting up MongoDB collections: {e}")
            raise
    
    def compress_data(self, data: Dict[str, Any]) -> str:
        """Compress data using gzip for storage efficiency"""
        json_str = json.dumps(data, default=str)
        compressed = gzip.compress(json_str.encode('utf-8'))
        return base64.b64encode(compressed).decode('utf-8')
    
    def decompress_data(self, compressed_data: str) -> Dict[str, Any]:
        """Decompress data from storage"""
        compressed_bytes = base64.b64decode(compressed_data.encode('utf-8'))
        decompressed = gzip.decompress(compressed_bytes)
        return json.loads(decompressed.decode('utf-8'))
    
    async def store_market_data(self, symbol: str, interval: str, data: pd.DataFrame):
        """Store market data with compression"""
        await self.setup_collections()
        
        try:
            # Convert DataFrame to dict for storage
            data_dict = data.to_dict('records')
            
            # Compress the data
            compressed_data = self.compress_data(data_dict)
            
            document = {
                "symbol": symbol,
                "interval": interval,
                "timestamp": datetime.utcnow(),
                "data": compressed_data,
                "record_count": len(data_dict)
            }
            
            await self.db.market_data.insert_one(document)
            logger.debug(f"Stored {len(data_dict)} records for {symbol} {interval}")
            
        except Exception as e:
            logger.error(f"Error storing market data: {e}")
            raise
    
    async def get_market_data(
        self, 
        symbol: str, 
        interval: str, 
        limit: int = 100
    ) -> Optional[pd.DataFrame]:
        """Retrieve market data with decompression"""
        await self.setup_collections()
        
        try:
            document = await self.db.market_data.find_one(
                {"symbol": symbol, "interval": interval},
                sort=[("timestamp", DESCENDING)]
            )
            
            if document and "data" in document:
                data_dict = self.decompress_data(document["data"])
                df = pd.DataFrame(data_dict)
                return df.tail(limit)
            
            return None
            
        except Exception as e:
            logger.error(f"Error retrieving market data: {e}")
            return None
    
    async def store_trading_signal(
        self, 
        symbol: str, 
        strategy: str, 
        signal: Dict[str, Any]
    ):
        """Store trading signals"""
        await self.setup_collections()
        
        try:
            document = {
                "symbol": symbol,
                "strategy": strategy,
                "timestamp": datetime.utcnow(),
                "signal": signal
            }
            
            await self.db.trading_signals.insert_one(document)
            
        except Exception as e:
            logger.error(f"Error storing trading signal: {e}")
            raise
    
    async def get_recent_signals(
        self, 
        symbol: str, 
        strategy: str = None, 
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Get recent trading signals"""
        await self.setup_collections()
        
        try:
            query = {"symbol": symbol}
            if strategy:
                query["strategy"] = strategy
            
            cursor = self.db.trading_signals.find(query).sort("timestamp", DESCENDING).limit(limit)
            signals = await cursor.to_list(limit)
            
            return signals
            
        except Exception as e:
            logger.error(f"Error retrieving signals: {e}")
            return []
    
    async def store_performance_metrics(self, metrics: Dict[str, Any]):
        """Store system performance metrics"""
        await self.setup_collections()
        
        try:
            document = {
                "timestamp": datetime.utcnow(),
                "metric_type": "system_performance",
                "metrics": metrics
            }
            
            await self.db.performance_metrics.insert_one(document)
            
        except Exception as e:
            logger.error(f"Error storing performance metrics: {e}")
            raise
    
    async def get_performance_history(
        self, 
        hours: int = 24
    ) -> List[Dict[str, Any]]:
        """Get performance metrics history"""
        await self.setup_collections()
        
        try:
            since = datetime.utcnow() - timedelta(hours=hours)
            cursor = self.db.performance_metrics.find(
                {"timestamp": {"$gte": since}}
            ).sort("timestamp", DESCENDING)
            
            metrics = await cursor.to_list(1000)
            return metrics
            
        except Exception as e:
            logger.error(f"Error retrieving performance history: {e}")
            return []
    
    async def store_portfolio_state(self, portfolio: Dict[str, Any]):
        """Store current portfolio state"""
        await self.setup_collections()
        
        try:
            document = {
                "timestamp": datetime.utcnow(),
                "portfolio": portfolio
            }
            
            await self.db.portfolio.insert_one(document)
            
        except Exception as e:
            logger.error(f"Error storing portfolio state: {e}")
            raise
    
    async def get_latest_portfolio(self) -> Optional[Dict[str, Any]]:
        """Get the latest portfolio state"""
        await self.setup_collections()
        
        try:
            document = await self.db.portfolio.find_one(
                sort=[("timestamp", DESCENDING)]
            )
            
            return document["portfolio"] if document else None
            
        except Exception as e:
            logger.error(f"Error retrieving portfolio: {e}")
            return None
    
    async def log_system_event(self, level: str, message: str, details: Dict[str, Any] = None):
        """Log system events to MongoDB"""
        await self.setup_collections()
        
        try:
            document = {
                "timestamp": datetime.utcnow(),
                "level": level,
                "message": message,
                "details": details or {}
            }
            
            await self.db.system_logs.insert_one(document)
            
        except Exception as e:
            logger.error(f"Error storing system log: {e}")
    
    async def cleanup_old_data(self, days_to_keep: int = 30):
        """Cleanup old data to maintain performance"""
        await self.setup_collections()
        
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days_to_keep)
            
            # Clean old market data
            result1 = await self.db.market_data.delete_many(
                {"timestamp": {"$lt": cutoff_date}}
            )
            
            # Clean old logs
            result2 = await self.db.system_logs.delete_many(
                {"timestamp": {"$lt": cutoff_date}}
            )
            
            logger.info(f"Cleaned up {result1.deleted_count} market data records and {result2.deleted_count} log records")
            
        except Exception as e:
            logger.error(f"Error cleaning up old data: {e}")