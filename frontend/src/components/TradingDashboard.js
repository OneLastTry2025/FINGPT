import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Progress } from '@/components/ui/progress';
import { 
  Activity, 
  TrendingUp, 
  TrendingDown, 
  DollarSign, 
  Target,
  Brain,
  Zap,
  BarChart3,
  Settings,
  Play,
  Pause,
  RefreshCw,
  Cpu,
  AlertCircle,
  CheckCircle,
  TrendingUp as TrendUp,
  TrendingDown as TrendDown,
  Gauge,
  Layers,
  Network,
  Bot,
  PieChart
} from 'lucide-react';
import axios from 'axios';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

const TradingDashboard = () => {
  const [engineStatus, setEngineStatus] = useState(null);
  const [systemInfo, setSystemInfo] = useState(null);
  const [systemPerformance, setSystemPerformance] = useState(null);
  const [positions, setPositions] = useState([]);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  
  // Enhanced ML state
  const [mlStatus, setMlStatus] = useState(null);
  const [mlActivity, setMlActivity] = useState(null);
  const [mlPerformance, setMlPerformance] = useState(null);
  const [finetuningJobs, setFinetuningJobs] = useState([]);
  const [activeTab, setActiveTab] = useState('overview');

  const fetchData = async () => {
    try {
      setRefreshing(true);
      
      // Fetch engine status
      const engineResponse = await axios.get(`${API}/trading/status`);
      setEngineStatus(engineResponse.data);
      
      // Fetch system info
      const infoResponse = await axios.get(`${API}/system/info`);
      setSystemInfo(infoResponse.data);
      
      // Fetch system performance
      const perfResponse = await axios.get(`${API}/system/performance`);
      setSystemPerformance(perfResponse.data);
      
      // Fetch positions
      const positionsResponse = await axios.get(`${API}/trading/positions`);
      setPositions(positionsResponse.data);
      
      // Fetch enhanced ML data
      try {
        const mlStatusResponse = await axios.get(`${API}/trading/ml/advanced/status`);
        setMlStatus(mlStatusResponse.data);
        
        const mlActivityResponse = await axios.get(`${API}/trading/ml/activity/live`);
        setMlActivity(mlActivityResponse.data);
        
        const mlPerfResponse = await axios.get(`${API}/trading/ml/performance/detailed`);
        setMlPerformance(mlPerfResponse.data);
      } catch (mlError) {
        console.warn('ML data unavailable:', mlError);
      }
      
      setLoading(false);
    } catch (error) {
      console.error('Error fetching data:', error);
      setLoading(false);
    } finally {
      setRefreshing(false);
    }
  };

  useEffect(() => {
    fetchData();
    const interval = setInterval(fetchData, 5000); // Refresh every 5 seconds
    return () => clearInterval(interval);
  }, []);

  const handleEngineControl = async (action) => {
    try {
      await axios.post(`${API}/trading/${action}`);
      fetchData();
    } catch (error) {
      console.error(`Error ${action} engine:`, error);
    }
  };

  const startFineTuning = async (symbol, params = {}) => {
    try {
      const response = await axios.post(`${API}/trading/ml/finetune/start/${symbol}`, params);
      if (response.data.status === 'success') {
        setFinetuningJobs(prev => [...prev, response.data.finetuning_job]);
      }
    } catch (error) {
      console.error(`Error starting fine-tuning for ${symbol}:`, error);
    }
  };

  const initializeML = async () => {
    try {
      await axios.post(`${API}/trading/ml/initialize`);
      fetchData();
    } catch (error) {
      console.error('Error initializing ML:', error);
    }
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-center">
          <RefreshCw className="h-12 w-12 animate-spin text-blue-500 mx-auto mb-4" />
          <h2 className="text-xl font-semibold text-gray-900">Loading FinGPT System...</h2>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <div className="bg-white shadow-sm border-b">
        <div className="max-w-7xl mx-auto px-4 py-6">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-4">
              <div className="p-2 bg-blue-100 rounded-lg">
                <Brain className="h-8 w-8 text-blue-600" />
              </div>
              <div>
                <h1 className="text-3xl font-bold text-gray-900">FinGPT Trading System</h1>
                <p className="text-gray-600">AI-Powered Trading Platform v{systemInfo?.version}</p>
              </div>
            </div>
            
            <div className="flex items-center space-x-4">
              <Badge 
                variant={engineStatus?.is_running ? "default" : "secondary"}
                className={engineStatus?.is_running ? "bg-green-500" : ""}
              >
                {engineStatus?.is_running ? "RUNNING" : "STOPPED"}
              </Badge>
              
              <Button 
                onClick={() => handleEngineControl(engineStatus?.is_running ? 'stop' : 'start')}
                variant={engineStatus?.is_running ? "destructive" : "default"}
                size="sm"
              >
                {engineStatus?.is_running ? (
                  <>
                    <Pause className="h-4 w-4 mr-2" />
                    Stop Engine
                  </>
                ) : (
                  <>
                    <Play className="h-4 w-4 mr-2" />
                    Start Engine
                  </>
                )}
              </Button>
              
              <Button 
                onClick={fetchData} 
                variant="outline" 
                size="sm"
                disabled={refreshing}
              >
                <RefreshCw className={`h-4 w-4 mr-2 ${refreshing ? 'animate-spin' : ''}`} />
                Refresh
              </Button>
            </div>
          </div>
        </div>
      </div>

      <div className="max-w-7xl mx-auto px-4 py-6">
        <Tabs defaultValue="overview" className="space-y-6">
          <TabsList className="grid w-full grid-cols-5">
            <TabsTrigger value="overview">Overview</TabsTrigger>
            <TabsTrigger value="positions">Positions</TabsTrigger>
            <TabsTrigger value="performance">Performance</TabsTrigger>
            <TabsTrigger value="system">System</TabsTrigger>
            <TabsTrigger value="ml">AI/ML</TabsTrigger>
          </TabsList>

          {/* Overview Tab */}
          <TabsContent value="overview" className="space-y-6">
            {/* Key Metrics */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
              <Card>
                <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                  <CardTitle className="text-sm font-medium">Portfolio Value</CardTitle>
                  <DollarSign className="h-4 w-4 text-muted-foreground" />
                </CardHeader>
                <CardContent>
                  <div className="text-2xl font-bold text-green-600">
                    ${engineStatus?.portfolio_value?.toLocaleString() || '0'}
                  </div>
                  <p className="text-xs text-muted-foreground">
                    Daily P&L: ${engineStatus?.daily_pnl?.toFixed(2) || '0.00'}
                  </p>
                </CardContent>
              </Card>

              <Card>
                <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                  <CardTitle className="text-sm font-medium">Active Positions</CardTitle>
                  <Target className="h-4 w-4 text-muted-foreground" />
                </CardHeader>
                <CardContent>
                  <div className="text-2xl font-bold">
                    {engineStatus?.open_positions || 0}
                  </div>
                  <p className="text-xs text-muted-foreground">
                    Symbols: {engineStatus?.active_symbols?.length || 0}
                  </p>
                </CardContent>
              </Card>

              <Card>
                <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                  <CardTitle className="text-sm font-medium">Win Rate</CardTitle>
                  <TrendingUp className="h-4 w-4 text-muted-foreground" />
                </CardHeader>
                <CardContent>
                  <div className="text-2xl font-bold text-blue-600">
                    {(engineStatus?.win_rate * 100)?.toFixed(1) || '0.0'}%
                  </div>
                  <p className="text-xs text-muted-foreground">
                    Total Trades: {engineStatus?.total_trades || 0}
                  </p>
                </CardContent>
              </Card>

              <Card>
                <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                  <CardTitle className="text-sm font-medium">Advanced ML Models</CardTitle>
                  <Brain className="h-4 w-4 text-muted-foreground" />
                </CardHeader>
                <CardContent>
                  <div className="text-2xl font-bold text-purple-600">
                    {mlStatus?.total_ml_capacity?.ensemble_models || engineStatus?.ml_models_loaded || 0}
                  </div>
                  <p className="text-xs text-muted-foreground">
                    {mlStatus ? `+${mlStatus.total_ml_capacity.rl_agents || 0} RL Agents` : 'Active Models'}
                  </p>
                  {mlStatus?.total_ml_capacity?.nlp_available && (
                    <Badge variant="outline" className="mt-1 text-xs">NLP Enabled</Badge>
                  )}
                </CardContent>
              </Card>
            </div>

            {/* System Status */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center">
                    <Activity className="h-5 w-5 mr-2" />
                    System Performance
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div>
                    <div className="flex justify-between text-sm">
                      <span>CPU Usage</span>
                      <span>{systemPerformance?.cpu?.usage_percent?.toFixed(1)}%</span>
                    </div>
                    <Progress 
                      value={systemPerformance?.cpu?.usage_percent || 0} 
                      className="h-2"
                    />
                  </div>
                  
                  <div>
                    <div className="flex justify-between text-sm">
                      <span>Memory Usage</span>
                      <span>{systemPerformance?.memory?.usage_percent?.toFixed(1)}%</span>
                    </div>
                    <Progress 
                      value={systemPerformance?.memory?.usage_percent || 0} 
                      className="h-2"
                    />
                  </div>
                  
                  <div>
                    <div className="flex justify-between text-sm">
                      <span>Disk Usage</span>
                      <span>{systemPerformance?.disk?.usage_percent?.toFixed(1)}%</span>
                    </div>
                    <Progress 
                      value={systemPerformance?.disk?.usage_percent || 0} 
                      className="h-2"
                    />
                  </div>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center">
                    <Zap className="h-5 w-5 mr-2" />
                    Hardware Specifications
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-3">
                  <div className="flex justify-between">
                    <span className="text-sm font-medium">CPU Cores:</span>
                    <span className="text-sm">{systemPerformance?.cpu?.core_count}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-sm font-medium">Total Memory:</span>
                    <span className="text-sm">{systemPerformance?.memory?.total_gb} GB</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-sm font-medium">Available Memory:</span>
                    <span className="text-sm">{systemPerformance?.memory?.available_gb} GB</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-sm font-medium">Total Storage:</span>
                    <span className="text-sm">{systemPerformance?.disk?.total_gb} GB</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-sm font-medium">Architecture:</span>
                    <span className="text-sm">ARM64 Cloud-Optimized</span>
                  </div>
                </CardContent>
              </Card>
            </div>

            {/* Trading Features */}
            <Card>
              <CardHeader>
                <CardTitle>System Capabilities</CardTitle>
                <CardDescription>Advanced trading system features and capabilities</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                  {systemInfo?.features?.map((feature, index) => (
                    <div key={index} className="flex items-center space-x-2">
                      <div className="h-2 w-2 bg-green-500 rounded-full"></div>
                      <span className="text-sm">{feature}</span>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          {/* Positions Tab */}
          <TabsContent value="positions" className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center">
                  <Target className="h-5 w-5 mr-2" />
                  Active Trading Positions
                </CardTitle>
                <CardDescription>
                  Current open positions and their performance
                </CardDescription>
              </CardHeader>
              <CardContent>
                {positions?.length > 0 ? (
                  <div className="overflow-x-auto">
                    <table className="w-full">
                      <thead>
                        <tr className="border-b">
                          <th className="text-left py-2">Symbol</th>
                          <th className="text-right py-2">Quantity</th>
                          <th className="text-right py-2">Entry Price</th>
                          <th className="text-right py-2">Current Price</th>
                          <th className="text-right py-2">P&L</th>
                          <th className="text-right py-2">Actions</th>
                        </tr>
                      </thead>
                      <tbody>
                        {positions.map((position, index) => (
                          <tr key={index} className="border-b">
                            <td className="py-3 font-medium">{position.symbol}</td>
                            <td className="py-3 text-right">{position.quantity.toFixed(4)}</td>
                            <td className="py-3 text-right">${position.entry_price.toFixed(4)}</td>
                            <td className="py-3 text-right">${position.current_price.toFixed(4)}</td>
                            <td className={`py-3 text-right font-medium ${
                              position.unrealized_pnl >= 0 ? 'text-green-600' : 'text-red-600'
                            }`}>
                              ${position.unrealized_pnl.toFixed(2)}
                            </td>
                            <td className="py-3 text-right">
                              <Button 
                                size="sm" 
                                variant="outline"
                                onClick={() => {/* Handle close position */}}
                              >
                                Close
                              </Button>
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                ) : (
                  <div className="text-center py-8 text-gray-500">
                    No active positions
                  </div>
                )}
              </CardContent>
            </Card>
          </TabsContent>

          {/* Performance Tab */}
          <TabsContent value="performance" className="space-y-6">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center">
                    <BarChart3 className="h-5 w-5 mr-2" />
                    Trading Performance
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="flex justify-between">
                    <span>Total Trades:</span>
                    <span className="font-medium">{engineStatus?.total_trades || 0}</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Trades Today:</span>
                    <span className="font-medium">{engineStatus?.trades_today || 0}</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Win Rate:</span>
                    <span className="font-medium text-blue-600">
                      {(engineStatus?.win_rate * 100)?.toFixed(1) || '0.0'}%
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span>Daily P&L:</span>
                    <span className={`font-medium ${
                      (engineStatus?.daily_pnl || 0) >= 0 ? 'text-green-600' : 'text-red-600'
                    }`}>
                      ${engineStatus?.daily_pnl?.toFixed(2) || '0.00'}
                    </span>
                  </div>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle>Risk Management</CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="flex justify-between">
                    <span>Max Position Size:</span>
                    <span className="font-medium">5%</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Daily Loss Limit:</span>
                    <span className="font-medium">2%</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Max Drawdown:</span>
                    <span className="font-medium">10%</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Risk Status:</span>
                    <Badge variant="default" className="bg-green-500">
                      Normal
                    </Badge>
                  </div>
                </CardContent>
              </Card>
            </div>
          </TabsContent>

          {/* System Tab */}
          <TabsContent value="system" className="space-y-6">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <Card>
                <CardHeader>
                  <CardTitle>Data Sources</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="space-y-3">
                    <div>
                      <h4 className="font-medium mb-2">Cryptocurrency</h4>
                      <div className="space-y-1">
                        {systemInfo?.data_sources?.crypto?.map((source, index) => (
                          <div key={index} className="flex items-center space-x-2">
                            <div className="h-2 w-2 bg-blue-500 rounded-full"></div>
                            <span className="text-sm">{source}</span>
                          </div>
                        ))}
                      </div>
                    </div>
                    
                    <div>
                      <h4 className="font-medium mb-2">Stocks</h4>
                      <div className="space-y-1">
                        {systemInfo?.data_sources?.stocks?.map((source, index) => (
                          <div key={index} className="flex items-center space-x-2">
                            <div className="h-2 w-2 bg-green-500 rounded-full"></div>
                            <span className="text-sm">{source}</span>
                          </div>
                        ))}
                      </div>
                    </div>
                    
                    <div>
                      <h4 className="font-medium mb-2">Forex</h4>
                      <div className="space-y-1">
                        {systemInfo?.data_sources?.forex?.map((source, index) => (
                          <div key={index} className="flex items-center space-x-2">
                            <div className="h-2 w-2 bg-orange-500 rounded-full"></div>
                            <span className="text-sm">{source}</span>
                          </div>
                        ))}
                      </div>
                    </div>
                  </div>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle>Performance Optimization</CardTitle>
                </CardHeader>
                <CardContent className="space-y-3">
                  <div className="flex justify-between">
                    <span>Parallel Processing:</span>
                    <Badge variant="default" className="bg-green-500">Enabled</Badge>
                  </div>
                  <div className="flex justify-between">
                    <span>Memory Optimization:</span>
                    <Badge variant="default" className="bg-green-500">High</Badge>
                  </div>
                  <div className="flex justify-between">
                    <span>Cache Efficiency:</span>
                    <Badge variant="default" className="bg-green-500">High</Badge>
                  </div>
                  <div className="flex justify-between">
                    <span>Max Concurrent Symbols:</span>
                    <span className="font-medium">100</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Analysis Workers:</span>
                    <span className="font-medium">12</span>
                  </div>
                </CardContent>
              </Card>
            </div>
          </TabsContent>

          {/* Enhanced ML Tab */}
          <TabsContent value="ml" className="space-y-6">
            {/* ML Status Overview */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
              <Card>
                <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                  <CardTitle className="text-sm font-medium">Ensemble Models</CardTitle>
                  <Layers className="h-4 w-4 text-muted-foreground" />
                </CardHeader>
                <CardContent>
                  <div className="text-2xl font-bold text-blue-600">
                    {mlStatus?.total_ml_capacity?.ensemble_models || 0}
                  </div>
                  <p className="text-xs text-muted-foreground">
                    RF + XGBoost per symbol
                  </p>
                </CardContent>
              </Card>

              <Card>
                <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                  <CardTitle className="text-sm font-medium">RL Agents</CardTitle>
                  <Bot className="h-4 w-4 text-muted-foreground" />
                </CardHeader>
                <CardContent>
                  <div className="text-2xl font-bold text-green-600">
                    {mlStatus?.total_ml_capacity?.rl_agents || 0}
                  </div>
                  <p className="text-xs text-muted-foreground">
                    Reinforcement Learning
                  </p>
                </CardContent>
              </Card>

              <Card>
                <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                  <CardTitle className="text-sm font-medium">Avg Accuracy</CardTitle>
                  <Target className="h-4 w-4 text-muted-foreground" />
                </CardHeader>
                <CardContent>
                  <div className="text-2xl font-bold text-emerald-600">
                    {mlPerformance?.overall_system_performance?.average_accuracy ? 
                      `${(mlPerformance.overall_system_performance.average_accuracy * 100).toFixed(1)}%` : 'N/A'}
                  </div>
                  <p className="text-xs text-muted-foreground">
                    System Performance
                  </p>
                </CardContent>
              </Card>

              <Card>
                <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                  <CardTitle className="text-sm font-medium">System Status</CardTitle>
                  <Activity className="h-4 w-4 text-muted-foreground" />
                </CardHeader>
                <CardContent>
                  <div className="flex items-center space-x-2">
                    <CheckCircle className="h-5 w-5 text-green-500" />
                    <span className="text-sm font-medium">
                      {mlActivity?.system_status === 'active' ? 'Active' : 'Inactive'}
                    </span>
                  </div>
                  <p className="text-xs text-muted-foreground">
                    {mlActivity?.total_active_models || 0} models running
                  </p>
                </CardContent>
              </Card>
            </div>

            {/* Live ML Activity */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center">
                  <Zap className="h-5 w-5 mr-2" />
                  Real-time ML Predictions
                </CardTitle>
                <CardDescription>
                  Live predictions from ensemble models and RL agents
                </CardDescription>
              </CardHeader>
              <CardContent>
                {mlActivity?.live_activity ? (
                  <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
                    {Object.entries(mlActivity.live_activity).map(([symbol, data]) => (
                      <Card key={symbol} className="border-l-4 border-l-blue-500">
                        <CardHeader className="pb-2">
                          <div className="flex items-center justify-between">
                            <CardTitle className="text-lg">{symbol}</CardTitle>
                            <Badge variant={data.consensus === 'BUY' ? 'default' : 'destructive'}>
                              {data.consensus}
                            </Badge>
                          </div>
                          <p className="text-sm text-muted-foreground">
                            ${data.current_price} ({data.price_change_24h > 0 ? '+' : ''}{data.price_change_24h}%)
                          </p>
                        </CardHeader>
                        <CardContent className="space-y-3">
                          {/* ML Model Predictions */}
                          <div>
                            <h4 className="text-sm font-medium mb-2">Ensemble Models</h4>
                            {Object.entries(data.ml_predictions || {}).map(([model, pred]) => (
                              <div key={model} className="flex items-center justify-between text-sm">
                                <span className="capitalize">{model.replace('_', ' ')}</span>
                                <div className="flex items-center space-x-2">
                                  <Badge 
                                    variant={pred.prediction === 'BUY' ? 'default' : 'destructive'} 
                                    className="text-xs"
                                  >
                                    {pred.prediction}
                                  </Badge>
                                  <span className="text-muted-foreground">
                                    {(pred.confidence * 100).toFixed(0)}%
                                  </span>
                                </div>
                              </div>
                            ))}
                          </div>
                          
                          {/* RL Predictions */}
                          {data.rl_predictions && (
                            <div>
                              <h4 className="text-sm font-medium mb-2">RL Agent</h4>
                              <div className="flex items-center justify-between text-sm">
                                <span>Action</span>
                                <div className="flex items-center space-x-2">
                                  <Badge 
                                    variant={
                                      data.rl_predictions.action === 'BUY' ? 'default' : 
                                      data.rl_predictions.action === 'SELL' ? 'destructive' : 'secondary'
                                    }
                                    className="text-xs"
                                  >
                                    {data.rl_predictions.action}
                                  </Badge>
                                  <span className="text-muted-foreground">
                                    {(data.rl_predictions.confidence * 100).toFixed(0)}%
                                  </span>
                                </div>
                              </div>
                              <div className="text-xs text-muted-foreground mt-1">
                                Expected Reward: {data.rl_predictions.expected_reward}
                              </div>
                            </div>
                          )}
                        </CardContent>
                      </Card>
                    ))}
                  </div>
                ) : (
                  <div className="text-center py-8">
                    <Brain className="h-12 w-12 mx-auto text-muted-foreground mb-4" />
                    <p className="text-muted-foreground">Initialize ML models to view predictions</p>
                    <Button onClick={initializeML} className="mt-4">
                      Initialize ML Models
                    </Button>
                  </div>
                )}
              </CardContent>
            </Card>

            {/* Model Performance Analytics */}
            {mlPerformance && (
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center">
                    <BarChart3 className="h-5 w-5 mr-2" />
                    Performance Analytics
                  </CardTitle>
                  <CardDescription>
                    Detailed metrics for all ML models
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="space-y-6">
                    {Object.entries(mlPerformance.performance_analytics).map(([symbol, data]) => (
                      <div key={symbol} className="border rounded-lg p-4">
                        <h3 className="text-lg font-semibold mb-4">{symbol}</h3>
                        
                        {/* Ensemble Performance */}
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
                          <div>
                            <h4 className="text-sm font-medium text-muted-foreground mb-2">ENSEMBLE PERFORMANCE</h4>
                            <div className="space-y-2">
                              <div className="flex justify-between">
                                <span className="text-sm">Combined Accuracy</span>
                                <span className="text-sm font-medium">
                                  {(data.ensemble_performance.combined_accuracy * 100).toFixed(2)}%
                                </span>
                              </div>
                              <div className="flex justify-between">
                                <span className="text-sm">Consensus</span>
                                <span className="text-sm font-medium">
                                  {(data.ensemble_performance.prediction_consensus * 100).toFixed(1)}%
                                </span>
                              </div>
                              <div className="flex justify-between">
                                <span className="text-sm">Stability Score</span>
                                <span className="text-sm font-medium">
                                  {(data.ensemble_performance.stability_score * 100).toFixed(1)}%
                                </span>
                              </div>
                            </div>
                          </div>
                          
                          {/* RL Performance */}
                          {data.rl_performance && (
                            <div>
                              <h4 className="text-sm font-medium text-muted-foreground mb-2">RL AGENT PERFORMANCE</h4>
                              <div className="space-y-2">
                                <div className="flex justify-between">
                                  <span className="text-sm">Win Rate</span>
                                  <span className="text-sm font-medium">
                                    {(data.rl_performance.win_rate * 100).toFixed(1)}%
                                  </span>
                                </div>
                                <div className="flex justify-between">
                                  <span className="text-sm">Avg Reward</span>
                                  <span className="text-sm font-medium">
                                    {data.rl_performance.average_reward.toFixed(4)}
                                  </span>
                                </div>
                                <div className="flex justify-between">
                                  <span className="text-sm">Sharpe Ratio</span>
                                  <span className="text-sm font-medium">
                                    {data.rl_performance.sharpe_ratio.toFixed(2)}
                                  </span>
                                </div>
                              </div>
                            </div>
                          )}
                        </div>

                        {/* Individual Model Performance */}
                        <div>
                          <h4 className="text-sm font-medium text-muted-foreground mb-3">INDIVIDUAL MODEL METRICS</h4>
                          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                            {Object.entries(data.models).map(([modelName, metrics]) => (
                              <div key={modelName} className="border rounded p-3">
                                <h5 className="font-medium capitalize mb-2">
                                  {modelName.replace('_', ' ')}
                                </h5>
                                <div className="grid grid-cols-2 gap-2 text-xs">
                                  <div>Accuracy: {(metrics.accuracy * 100).toFixed(1)}%</div>
                                  <div>Precision: {(metrics.precision * 100).toFixed(1)}%</div>
                                  <div>Recall: {(metrics.recall * 100).toFixed(1)}%</div>
                                  <div>F1 Score: {(metrics.f1_score * 100).toFixed(1)}%</div>
                                </div>
                                <Button 
                                  size="sm" 
                                  variant="outline" 
                                  className="w-full mt-2"
                                  onClick={() => startFineTuning(symbol, {
                                    model: modelName,
                                    epochs: 10,
                                    learning_rate: 0.001
                                  })}
                                >
                                  Fine-tune Model
                                </Button>
                              </div>
                            ))}
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                </CardContent>
              </Card>
            )}

            {/* System Hardware Utilization */}
            {mlStatus?.system_optimization && (
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center">
                    <Cpu className="h-5 w-5 mr-2" />
                    Hardware Utilization
                  </CardTitle>
                  <CardDescription>
                    ML engine hardware optimization status
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                    <div className="text-center">
                      <div className="text-2xl font-bold text-blue-600">
                        {mlStatus.system_optimization.cpu_workers}
                      </div>
                      <p className="text-sm text-muted-foreground">CPU Workers</p>
                    </div>
                    <div className="text-center">
                      <div className="text-2xl font-bold text-green-600">
                        {mlStatus.system_optimization.memory_limit_gb}GB
                      </div>
                      <p className="text-sm text-muted-foreground">Memory Allocated</p>
                    </div>
                    <div className="text-center">
                      <div className="text-2xl font-bold text-purple-600">
                        {mlStatus.system_optimization.parallel_processing ? 'ON' : 'OFF'}
                      </div>
                      <p className="text-sm text-muted-foreground">Parallel Processing</p>
                    </div>
                  </div>
                  
                  {mlPerformance?.overall_system_performance && (
                    <div className="mt-4 pt-4 border-t">
                      <div className="flex items-center justify-between text-sm">
                        <span>Prediction Latency</span>
                        <span className="font-medium">
                          {mlPerformance.overall_system_performance.prediction_latency_ms}ms
                        </span>
                      </div>
                      <div className="flex items-center justify-between text-sm mt-2">
                        <span>System Stability</span>
                        <span className="font-medium">
                          {(mlPerformance.overall_system_performance.system_stability * 100).toFixed(1)}%
                        </span>
                      </div>
                    </div>
                  )}
                </CardContent>
              </Card>
            )}

            {/* Fine-tuning Jobs */}
            {finetuningJobs.length > 0 && (
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center">
                    <Settings className="h-5 w-5 mr-2" />
                    Active Fine-tuning Jobs
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="space-y-3">
                    {finetuningJobs.map((job, index) => (
                      <div key={index} className="flex items-center justify-between p-3 border rounded">
                        <div>
                          <div className="font-medium">{job.symbol}</div>
                          <div className="text-sm text-muted-foreground">{job.status}</div>
                        </div>
                        <Badge variant="outline">{job.status}</Badge>
                      </div>
                    ))}
                  </div>
                </CardContent>
              </Card>
            )}
          </TabsContent>
        </Tabs>
      </div>
    </div>
  );
};

export default TradingDashboard;