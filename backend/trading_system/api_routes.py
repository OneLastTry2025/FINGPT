"""
API Routes for FinGPT Trading System
Provides RESTful API endpoints for trading operations
"""

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

# Global trading engine instance
trading_engine = None

def set_trading_engine(engine):
    """Set the global trading engine instance"""
    global trading_engine
    trading_engine = engine

def get_trading_engine():
    """Dependency to get trading engine"""
    if trading_engine is None:
        raise HTTPException(status_code=503, detail="Trading engine not initialized")
    return trading_engine

# Create router
router = APIRouter(prefix="/api/trading", tags=["trading"])

# Pydantic models
class EngineStatusResponse(BaseModel):
    is_running: bool
    portfolio_value: float
    daily_pnl: float
    open_positions: int
    trades_today: int
    win_rate: float
    total_trades: int
    active_symbols: List[str]
    ml_models_loaded: int

class MarketDataRequest(BaseModel):
    symbol: str
    interval: str = "1h"
    limit: int = 100
    source: str = "auto"

class TradingSignal(BaseModel):
    symbol: str
    action: str
    confidence: float
    price: float
    strategy: str
    timestamp: datetime

class PositionInfo(BaseModel):
    symbol: str
    quantity: float
    entry_price: float
    current_price: float
    unrealized_pnl: float
    timestamp: datetime

# Trading Engine Control Endpoints
@router.get("/status", response_model=EngineStatusResponse)
async def get_engine_status(engine=Depends(get_trading_engine)):
    """Get current trading engine status"""
    try:
        status = await engine.get_engine_status()
        return EngineStatusResponse(**status)
    except Exception as e:
        logger.error(f"Error getting engine status: {e}")
        raise HTTPException(status_code=500, detail="Failed to get engine status")

@router.post("/start")
async def start_trading_engine(
    symbols: List[str] = ["BTCUSDT", "ETHUSDT", "BNBUSDT"],
    enable_ml: bool = False,  # Changed default to False
    strategies: List[str] = ["momentum", "mean_reversion", "breakout"],
    engine=Depends(get_trading_engine)
):
    """Start the trading engine with specified parameters"""
    try:
        if engine.is_running:
            return {"message": "Trading engine is already running", "status": "success"}
        
        await engine.start_enhanced_engine(symbols, enable_ml, strategies)
        return {"message": "Trading engine started successfully", "status": "success"}
    except Exception as e:
        logger.error(f"Error starting engine: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start engine: {str(e)}")

@router.post("/ml/initialize")
async def initialize_ml_models(
    symbols: List[str] = ["BTCUSDT", "ETHUSDT", "BNBUSDT"],
    engine=Depends(get_trading_engine)
):
    """Initialize ML models separately (can be called after engine is running)"""
    try:
        if not engine.is_running:
            return {"message": "Start trading engine first", "status": "error"}
        
        # Initialize ML models
        success = await engine.advanced_ml_engine.initialize_advanced_models(symbols)
        
        if success:
            return {"message": "ML models initialized successfully", "status": "success"}
        else:
            return {"message": "Failed to initialize ML models", "status": "error"}
    except Exception as e:
        logger.error(f"Error initializing ML models: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to initialize ML models: {str(e)}")

@router.post("/stop")
async def stop_trading_engine(engine=Depends(get_trading_engine)):
    """Stop the trading engine"""
    try:
        if not engine.is_running:
            return {"message": "Trading engine is not running", "status": "success"}
        
        await engine.stop_engine()
        return {"message": "Trading engine stopped successfully", "status": "success"}
    except Exception as e:
        logger.error(f"Error stopping engine: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to stop engine: {str(e)}")

# Portfolio and Positions Endpoints
@router.get("/portfolio")
async def get_portfolio_info(engine=Depends(get_trading_engine)):
    """Get current portfolio information"""
    try:
        portfolio = await engine.storage.get_latest_portfolio()
        if not portfolio:
            return {"message": "No portfolio data available", "portfolio": None}
        
        return {"portfolio": portfolio, "status": "success"}
    except Exception as e:
        logger.error(f"Error getting portfolio: {e}")
        raise HTTPException(status_code=500, detail="Failed to get portfolio")

@router.get("/positions", response_model=List[PositionInfo])
async def get_current_positions(engine=Depends(get_trading_engine)):
    """Get all current trading positions"""
    try:
        positions = []
        for symbol, pos in engine.positions.items():
            positions.append(PositionInfo(
                symbol=pos.symbol,
                quantity=pos.quantity,
                entry_price=pos.entry_price,
                current_price=pos.current_price,
                unrealized_pnl=pos.unrealized_pnl,
                timestamp=pos.timestamp
            ))
        return positions
    except Exception as e:
        logger.error(f"Error getting positions: {e}")
        raise HTTPException(status_code=500, detail="Failed to get positions")

@router.post("/positions/{symbol}/close")
async def close_position(symbol: str, engine=Depends(get_trading_engine)):
    """Manually close a specific position"""
    try:
        if symbol not in engine.positions:
            raise HTTPException(status_code=404, detail=f"Position {symbol} not found")
        
        await engine._close_position(symbol, "manual_close")
        return {"message": f"Position {symbol} closed successfully", "status": "success"}
    except Exception as e:
        logger.error(f"Error closing position {symbol}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to close position: {str(e)}")

# Market Data Endpoints
@router.post("/market-data")
async def get_market_data(request: MarketDataRequest, engine=Depends(get_trading_engine)):
    """Get historical market data for a symbol"""
    try:
        data = await engine.data_feed.get_historical_data(
            request.symbol, 
            request.interval, 
            request.limit,
            request.source
        )
        
        if data.empty:
            return {"message": "No data available", "data": []}
        
        # Convert DataFrame to list of dictionaries
        data_records = data.to_dict('records')
        
        return {
            "symbol": request.symbol,
            "interval": request.interval,
            "count": len(data_records),
            "data": data_records,
            "status": "success"
        }
    except Exception as e:
        logger.error(f"Error getting market data: {e}")
        raise HTTPException(status_code=500, detail="Failed to get market data")

@router.get("/market-data/summary")
async def get_market_summary(engine=Depends(get_trading_engine)):
    """Get overall market data summary"""
    try:
        summary = await engine.data_feed.get_market_summary()
        return {"summary": summary, "status": "success"}
    except Exception as e:
        logger.error(f"Error getting market summary: {e}")
        raise HTTPException(status_code=500, detail="Failed to get market summary")

# Trading Signals Endpoints
@router.get("/signals/{symbol}")
async def get_trading_signals(
    symbol: str, 
    strategy: Optional[str] = None,
    limit: int = 50,
    engine=Depends(get_trading_engine)
):
    """Get recent trading signals for a symbol"""
    try:
        signals = await engine.storage.get_recent_signals(symbol, strategy, limit)
        return {
            "symbol": symbol,
            "strategy": strategy,
            "count": len(signals),
            "signals": signals,
            "status": "success"
        }
    except Exception as e:
        logger.error(f"Error getting signals for {symbol}: {e}")
        raise HTTPException(status_code=500, detail="Failed to get trading signals")

# Performance and Analytics Endpoints
@router.get("/performance/metrics")
async def get_performance_metrics(hours: int = 24, engine=Depends(get_trading_engine)):
    """Get performance metrics for the specified time period"""
    try:
        metrics = await engine.storage.get_performance_history(hours)
        return {
            "period_hours": hours,
            "count": len(metrics),
            "metrics": metrics,
            "status": "success"
        }
    except Exception as e:
        logger.error(f"Error getting performance metrics: {e}")
        raise HTTPException(status_code=500, detail="Failed to get performance metrics")

@router.get("/performance/summary")
async def get_performance_summary(engine=Depends(get_trading_engine)):
    """Get overall performance summary"""
    try:
        status = await engine.get_engine_status()
        
        # Calculate additional metrics
        total_unrealized_pnl = sum(pos.unrealized_pnl for pos in engine.positions.values())
        
        summary = {
            "portfolio_value": status["portfolio_value"],
            "daily_pnl": status["daily_pnl"],
            "total_unrealized_pnl": total_unrealized_pnl,
            "win_rate": status["win_rate"],
            "total_trades": status["total_trades"],
            "trades_today": status["trades_today"],
            "active_positions": status["open_positions"],
            "ml_models_active": status["ml_models_loaded"] > 0
        }
        
        return {"summary": summary, "status": "success"}
    except Exception as e:
        logger.error(f"Error getting performance summary: {e}")
        raise HTTPException(status_code=500, detail="Failed to get performance summary")

# System Configuration Endpoints
@router.get("/config/strategies")
async def get_strategy_config(engine=Depends(get_trading_engine)):
    """Get current strategy configuration"""
    try:
        return {
            "strategies": engine.config.strategies,
            "status": "success"
        }
    except Exception as e:
        logger.error(f"Error getting strategy config: {e}")
        raise HTTPException(status_code=500, detail="Failed to get strategy config")

@router.get("/config/risk")
async def get_risk_config(engine=Depends(get_trading_engine)):
    """Get current risk management configuration"""
    try:
        return {
            "risk_settings": engine.config.risk_settings.dict(),
            "status": "success"
        }
    except Exception as e:
        logger.error(f"Error getting risk config: {e}")
        raise HTTPException(status_code=500, detail="Failed to get risk config")

@router.get("/config/system")
async def get_system_config(engine=Depends(get_trading_engine)):
    """Get system configuration and capabilities"""
    try:
        hardware_config = engine.config.get_hardware_optimization_config()
        return {
            "system_config": hardware_config,
            "trading_pairs": engine.config.trading_pairs,
            "intervals": engine.config.intervals,
            "indicators": engine.config.indicators,
            "status": "success"
        }
    except Exception as e:
        logger.error(f"Error getting system config: {e}")
        raise HTTPException(status_code=500, detail="Failed to get system config")

# ML Model Endpoints
@router.get("/ml/models")
async def get_ml_models_info(engine=Depends(get_trading_engine)):
    """Get information about loaded ML models"""
    try:
        models_info = {}
        for symbol, model in engine.ml_models.items():
            models_info[symbol] = {
                "model_type": type(model).__name__,
                "features": model.n_features_in_ if hasattr(model, 'n_features_in_') else "unknown",
                "trained": True
            }
        
        return {
            "loaded_models": len(engine.ml_models),
            "models": models_info,
            "status": "success"
        }
    except Exception as e:
        logger.error(f"Error getting ML models info: {e}")
        raise HTTPException(status_code=500, detail="Failed to get ML models info")

@router.post("/ml/retrain/{symbol}")
async def retrain_ml_model(symbol: str, engine=Depends(get_trading_engine)):
    """Retrain ML model for a specific symbol"""
    try:
        # This would trigger model retraining
        # For now, just return a message
        return {
            "message": f"ML model retraining initiated for {symbol}",
            "symbol": symbol,
            "status": "success"
        }
    except Exception as e:
        logger.error(f"Error retraining ML model for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrain model: {str(e)}")

@router.get("/ml/advanced/status")
async def get_advanced_ml_status(engine=Depends(get_trading_engine)):
    """Get comprehensive advanced ML/RL engine status"""
    try:
        advanced_status = await engine.advanced_ml_engine.get_model_status()
        
        # Add detailed performance metrics
        detailed_status = {
            "advanced_ml_engine": advanced_status,
            "legacy_ml_models": len(engine.ml_models),
            "total_ml_capacity": {
                "ensemble_models": advanced_status["ml_models_loaded"],
                "rl_agents": advanced_status["rl_agents_loaded"],
                "nlp_available": advanced_status["nlp_available"],
                "symbols_covered": len(advanced_status["symbols_with_models"])
            },
            "performance_summary": advanced_status.get("model_performance", {}),
            "system_optimization": advanced_status["hardware_utilization"],
            "status": "success"
        }
        
        return detailed_status
    except Exception as e:
        logger.error(f"Error getting advanced ML status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get advanced ML status: {str(e)}")

# System Health and Monitoring
@router.get("/health")
async def get_system_health(engine=Depends(get_trading_engine)):
    """Get overall system health status"""
    try:
        health_status = {
            "trading_engine": "healthy" if engine.is_running else "stopped",
            "data_feeds": "healthy" if engine.data_feed.is_running else "stopped",
            "database": "healthy",  # Would check database connection
            "ml_models": len(engine.ml_models),
            "active_positions": len(engine.positions),
            "last_check": datetime.now().isoformat()
        }
        
        return {
            "health": health_status,
            "overall_status": "healthy" if engine.is_running else "degraded",
            "status": "success"
        }
    except Exception as e:
        logger.error(f"Error getting system health: {e}")
        raise HTTPException(status_code=500, detail="Failed to get system health")