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
                  <CardTitle className="text-sm font-medium">ML Models</CardTitle>
                  <Brain className="h-4 w-4 text-muted-foreground" />
                </CardHeader>
                <CardContent>
                  <div className="text-2xl font-bold text-purple-600">
                    {engineStatus?.ml_models_loaded || 0}
                  </div>
                  <p className="text-xs text-muted-foreground">
                    Active Models
                  </p>
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

          {/* ML Tab */}
          <TabsContent value="ml" className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center">
                  <Brain className="h-5 w-5 mr-2" />
                  Machine Learning Models
                </CardTitle>
                <CardDescription>
                  AI models powering the trading decisions
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                  <div className="p-4 border rounded-lg">
                    <h4 className="font-medium mb-2">Random Forest</h4>
                    <p className="text-sm text-gray-600 mb-2">Price direction prediction</p>
                    <Badge variant="default" className="bg-green-500">Active</Badge>
                  </div>
                  
                  <div className="p-4 border rounded-lg">
                    <h4 className="font-medium mb-2">Technical Analysis</h4>
                    <p className="text-sm text-gray-600 mb-2">25+ indicators</p>
                    <Badge variant="default" className="bg-green-500">Active</Badge>
                  </div>
                  
                  <div className="p-4 border rounded-lg">
                    <h4 className="font-medium mb-2">Risk Management</h4>
                    <p className="text-sm text-gray-600 mb-2">Kelly Criterion + ATR</p>
                    <Badge variant="default" className="bg-green-500">Active</Badge>
                  </div>
                </div>

                <div className="mt-6">
                  <h4 className="font-medium mb-3">Trading Strategies</h4>
                  <div className="space-y-2">
                    <div className="flex items-center justify-between p-3 border rounded">
                      <span>Momentum Strategy</span>
                      <Badge variant="default" className="bg-blue-500">Running</Badge>
                    </div>
                    <div className="flex items-center justify-between p-3 border rounded">
                      <span>Mean Reversion Strategy</span>
                      <Badge variant="default" className="bg-blue-500">Running</Badge>
                    </div>
                    <div className="flex items-center justify-between p-3 border rounded">
                      <span>Breakout Strategy</span>
                      <Badge variant="default" className="bg-blue-500">Running</Badge>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      </div>
    </div>
  );
};

export default TradingDashboard;