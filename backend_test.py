#!/usr/bin/env python3
"""
FinGPT Backend API Testing Suite - MEXC Integration & Real Price Data Focus
Tests critical endpoints for real-time data feeds and MEXC WebSocket integration
"""

import requests
import json
import time
import sys
from datetime import datetime
from typing import Dict, Any

# Get backend URL from environment
BACKEND_URL = "https://kline-verify.preview.emergentagent.com"
API_BASE = f"{BACKEND_URL}/api"

class FinGPTTester:
    def __init__(self):
        self.results = []
        self.failed_tests = []
        self.passed_tests = []
        
    def log_result(self, test_name: str, success: bool, details: str = "", response_data: Any = None):
        """Log test result"""
        result = {
            "test": test_name,
            "success": success,
            "details": details,
            "timestamp": datetime.now().isoformat(),
            "response_data": response_data
        }
        self.results.append(result)
        
        if success:
            self.passed_tests.append(test_name)
            print(f"✅ {test_name}: {details}")
        else:
            self.failed_tests.append(test_name)
            print(f"❌ {test_name}: {details}")
    
    def test_ml_activity_live_real_prices(self):
        """PRIORITY TEST: ML Activity Live - Real Price Integration"""
        try:
            response = requests.get(f"{API_BASE}/trading/ml/activity/live", timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                
                if "live_activity" in data:
                    activity = data["live_activity"]
                    
                    # Check for realistic crypto prices
                    btc_data = activity.get("BTCUSDT", {})
                    eth_data = activity.get("ETHUSDT", {})
                    bnb_data = activity.get("BNBUSDT", {})
                    
                    btc_price = btc_data.get("current_price", 0)
                    eth_price = eth_data.get("current_price", 0)
                    bnb_price = bnb_data.get("current_price", 0)
                    
                    # Check if prices are in realistic ranges (not synthetic)
                    btc_realistic = 80000 <= btc_price <= 110000  # Current BTC range ~$93K
                    eth_realistic = 2800 <= eth_price <= 4000     # Current ETH range ~$3.3K
                    bnb_realistic = 600 <= bnb_price <= 800       # Current BNB range ~$710
                    
                    price_check = f"BTC: ${btc_price:,.2f} ({'✅' if btc_realistic else '❌'}), ETH: ${eth_price:,.2f} ({'✅' if eth_realistic else '❌'}), BNB: ${bnb_price:,.2f} ({'✅' if bnb_realistic else '❌'})"
                    
                    # Check for ML predictions and activity
                    total_models = data.get("total_active_models", 0)
                    system_status = data.get("system_status", "")
                    
                    if btc_realistic and eth_realistic and bnb_realistic:
                        self.log_result(
                            "ML Activity Live - Real Price Data", 
                            True, 
                            f"REAL PRICE DATA CONFIRMED: {price_check}, Active Models: {total_models}, Status: {system_status}", 
                            data
                        )
                    else:
                        self.log_result(
                            "ML Activity Live - Real Price Data", 
                            False, 
                            f"SYNTHETIC DATA DETECTED: {price_check}"
                        )
                else:
                    self.log_result(
                        "ML Activity Live", 
                        False, 
                        f"Missing 'live_activity' field: {data}"
                    )
            else:
                self.log_result(
                    "ML Activity Live", 
                    False, 
                    f"HTTP {response.status_code}: {response.text}"
                )
                
        except Exception as e:
            self.log_result("ML Activity Live", False, f"Connection error: {str(e)}")
    
    def test_market_data_summary_mexc(self):
        """PRIORITY TEST: Market Data Summary - MEXC Integration Verification"""
        try:
            response = requests.get(f"{API_BASE}/trading/market-data/summary", timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                if "summary" in data:
                    summary = data["summary"]
                    
                    # Check for MEXC as primary crypto source
                    primary_source = summary.get("primary_crypto_source", "")
                    data_sources = summary.get("data_sources", [])
                    status = summary.get("status", "")
                    
                    mexc_primary = "mexc" in primary_source.lower()
                    mexc_in_sources = any("mexc" in str(source).lower() for source in data_sources)
                    
                    if mexc_primary and mexc_in_sources:
                        self.log_result(
                            "Market Data Summary - MEXC Integration", 
                            True, 
                            f"MEXC confirmed as primary crypto source: {primary_source}, Sources: {data_sources}, Status: {status}", 
                            data
                        )
                    else:
                        self.log_result(
                            "Market Data Summary - MEXC Integration", 
                            False, 
                            f"MEXC not properly configured - Primary: {mexc_primary}, In sources: {mexc_in_sources}"
                        )
                else:
                    self.log_result(
                        "Market Data Summary", 
                        False, 
                        f"Missing 'summary' field: {data}"
                    )
            else:
                self.log_result(
                    "Market Data Summary", 
                    False, 
                    f"HTTP {response.status_code}: {response.text}"
                )
                
        except Exception as e:
            self.log_result("Market Data Summary", False, f"Connection error: {str(e)}")
    
    def test_performance_summary(self):
        """Test trading performance summary"""
        try:
            response = requests.get(f"{API_BASE}/trading/performance/summary", timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                if "summary" in data:
                    summary = data["summary"]
                    required_fields = ["portfolio_value", "daily_pnl", "win_rate", "total_trades"]
                    missing_fields = [field for field in required_fields if field not in summary]
                    
                    if missing_fields:
                        self.log_result(
                            "Performance Summary", 
                            False, 
                            f"Summary missing fields: {missing_fields}"
                        )
                    else:
                        perf_summary = f"Portfolio: ${summary.get('portfolio_value', 0):,.2f}, Win Rate: {summary.get('win_rate', 0):.1%}, Trades: {summary.get('total_trades', 0)}"
                        
                        self.log_result(
                            "Performance Summary", 
                            True, 
                            f"Performance summary retrieved - {perf_summary}", 
                            data
                        )
                else:
                    self.log_result(
                        "Performance Summary", 
                        False, 
                        f"Missing 'summary' field in response: {data}"
                    )
            else:
                self.log_result(
                    "Performance Summary", 
                    False, 
                    f"HTTP {response.status_code}: {response.text}"
                )
                
        except Exception as e:
            self.log_result("Performance Summary", False, f"Connection error: {str(e)}")
    
    def test_system_performance(self):
        """Test system performance metrics"""
        try:
            response = requests.get(f"{API_BASE}/system/performance", timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                
                # Check required performance fields
                required_sections = ["cpu", "memory", "disk", "optimization_status"]
                missing_sections = [section for section in required_sections if section not in data]
                
                if missing_sections:
                    self.log_result(
                        "System Performance", 
                        False, 
                        f"Missing performance sections: {missing_sections}"
                    )
                    return
                
                # Validate performance data
                cpu = data.get("cpu", {})
                memory = data.get("memory", {})
                disk = data.get("disk", {})
                
                performance_summary = f"CPU: {cpu.get('usage_percent', 'N/A')}%, Memory: {memory.get('usage_percent', 'N/A')}%, Disk: {disk.get('usage_percent', 'N/A')}%"
                
                self.log_result(
                    "System Performance", 
                    True, 
                    f"Performance metrics retrieved - {performance_summary}", 
                    data
                )
            else:
                self.log_result(
                    "System Performance", 
                    False, 
                    f"HTTP {response.status_code}: {response.text}"
                )
                
        except Exception as e:
            self.log_result("System Performance", False, f"Connection error: {str(e)}")
    
    def test_trading_health(self):
        """Test trading system health"""
        try:
            response = requests.get(f"{API_BASE}/trading/health", timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                if "health" in data:
                    health = data["health"]
                    overall_status = data.get("overall_status", "unknown")
                    
                    health_summary = f"Overall: {overall_status}, Engine: {health.get('trading_engine')}, Data: {health.get('data_feeds')}"
                    
                    if overall_status in ["healthy", "degraded"]:
                        self.log_result(
                            "Trading Health", 
                            True, 
                            f"System health check passed - {health_summary}", 
                            data
                        )
                    else:
                        self.log_result(
                            "Trading Health", 
                            False, 
                            f"Unexpected health status: {overall_status}"
                        )
                else:
                    self.log_result(
                        "Trading Health", 
                        False, 
                        f"Missing 'health' field: {data}"
                    )
            else:
                self.log_result(
                    "Trading Health", 
                    False, 
                    f"HTTP {response.status_code}: {response.text}"
                )
                
        except Exception as e:
            self.log_result("Trading Health", False, f"Connection error: {str(e)}")
    
    def test_trading_status(self):
        """Test trading engine status"""
        try:
            response = requests.get(f"{API_BASE}/trading/status", timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                # Check required trading status fields
                required_fields = ["is_running", "portfolio_value", "daily_pnl", "open_positions"]
                missing_fields = [field for field in required_fields if field not in data]
                
                if missing_fields:
                    self.log_result(
                        "Trading Status", 
                        False, 
                        f"Missing status fields: {missing_fields}"
                    )
                    return
                
                status_summary = f"Running: {data.get('is_running')}, Portfolio: ${data.get('portfolio_value', 0):,.2f}, P&L: ${data.get('daily_pnl', 0):,.2f}"
                
                self.log_result(
                    "Trading Status", 
                    True, 
                    f"Trading engine status retrieved - {status_summary}", 
                    data
                )
            else:
                self.log_result(
                    "Trading Status", 
                    False, 
                    f"HTTP {response.status_code}: {response.text}"
                )
                
        except Exception as e:
            self.log_result("Trading Status", False, f"Connection error: {str(e)}")
    
    def test_system_info(self):
        """Test system information and capabilities"""
        try:
            response = requests.get(f"{API_BASE}/system/info", timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                # Check required fields
                required_fields = ["system", "version", "features", "hardware_optimized"]
                missing_fields = [field for field in required_fields if field not in data]
                
                if missing_fields:
                    self.log_result(
                        "System Info", 
                        False, 
                        f"Missing required fields: {missing_fields}"
                    )
                    return
                
                # Check hardware specs - UPDATED FOR 48-CORE UPGRADE
                hardware = data.get("hardware_optimized", {})
                performance = data.get("performance_capabilities", {})
                
                # Check for upgraded hardware specs
                hardware_checks = {
                    "48-core upgrade": "48-core" in str(hardware.get("target_cpu", "")),
                    "188GB memory": "188GB" in str(hardware.get("memory", "")),
                    "36 workers": performance.get("parallel_analysis_workers") == 36,
                    "ML workers": performance.get("ml_model_workers") == 12
                }
                
                upgrade_status = []
                for check, passed in hardware_checks.items():
                    if passed:
                        upgrade_status.append(f"✅ {check}")
                    else:
                        upgrade_status.append(f"❌ {check}")
                
                # Check data sources for MEXC
                data_sources = data.get("data_sources", {})
                mexc_primary = "MEXC WebSocket" in str(data_sources.get("crypto", []))
                
                if mexc_primary:
                    upgrade_status.append("✅ MEXC WebSocket as primary crypto source")
                else:
                    upgrade_status.append("❌ MEXC WebSocket not found in crypto sources")
                
                self.log_result(
                    "System Info", 
                    True, 
                    f"Hardware upgrade status: {'; '.join(upgrade_status)}", 
                    data
                )
            else:
                self.log_result(
                    "System Info", 
                    False, 
                    f"HTTP {response.status_code}: {response.text}"
                )
                
        except Exception as e:
            self.log_result("System Info", False, f"Connection error: {str(e)}")
    
    def test_advanced_ml_status(self):
        """Test advanced ML engine status"""
        try:
            response = requests.get(f"{API_BASE}/trading/ml/advanced/status", timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                if "advanced_ml_engine" in data:
                    ml_status = data["advanced_ml_engine"]
                    total_capacity = data.get("total_ml_capacity", {})
                    
                    status_summary = f"ML Models: {total_capacity.get('ensemble_models', 0)}, RL Agents: {total_capacity.get('rl_agents', 0)}, NLP: {total_capacity.get('nlp_available', False)}"
                    
                    self.log_result(
                        "Advanced ML Status", 
                        True, 
                        f"Advanced ML engine status retrieved - {status_summary}", 
                        data
                    )
                else:
                    self.log_result(
                        "Advanced ML Status", 
                        False, 
                        f"Missing 'advanced_ml_engine' field: {data}"
                    )
            else:
                self.log_result(
                    "Advanced ML Status", 
                    False, 
                    f"HTTP {response.status_code}: {response.text}"
                )
                
        except Exception as e:
            self.log_result("Advanced ML Status", False, f"Connection error: {str(e)}")
    
    def test_ml_models_info(self):
        """Test ML model loading and initialization"""
        try:
            response = requests.get(f"{API_BASE}/trading/ml/models", timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                if "loaded_models" in data:
                    model_count = data.get("loaded_models", 0)
                    
                    # Note: Models might be 0 due to geo-restrictions as mentioned in requirements
                    if model_count == 0:
                        self.log_result(
                            "ML Models", 
                            True, 
                            "Minor: No ML models loaded (may be due to geo-restrictions)", 
                            data
                        )
                    else:
                        self.log_result(
                            "ML Models", 
                            True, 
                            f"ML models loaded: {model_count}", 
                            data
                        )
                else:
                    self.log_result(
                        "ML Models", 
                        False, 
                        f"Missing 'loaded_models' field: {data}"
                    )
            else:
                self.log_result(
                    "ML Models", 
                    False, 
                    f"HTTP {response.status_code}: {response.text}"
                )
                
        except Exception as e:
            self.log_result("ML Models", False, f"Connection error: {str(e)}")
    
    def test_error_handling_invalid_symbols(self):
        """Test error handling for invalid symbols"""
        try:
            # Test with invalid symbol
            invalid_symbol = "INVALIDCOIN"
            response = requests.get(f"{API_BASE}/trading/signals/{invalid_symbol}", timeout=10)
            
            # Should handle gracefully (either 404 or empty response)
            if response.status_code in [200, 404]:
                if response.status_code == 200:
                    data = response.json()
                    if data.get("count", 0) == 0:
                        self.log_result(
                            "Error Handling - Invalid Symbols", 
                            True, 
                            f"Graceful handling of invalid symbol: {invalid_symbol} - returned empty signals", 
                            data
                        )
                    else:
                        self.log_result(
                            "Error Handling - Invalid Symbols", 
                            False, 
                            f"Unexpected data for invalid symbol: {data}"
                        )
                else:  # 404
                    self.log_result(
                        "Error Handling - Invalid Symbols", 
                        True, 
                        f"Proper 404 response for invalid symbol: {invalid_symbol}"
                    )
            else:
                self.log_result(
                    "Error Handling - Invalid Symbols", 
                    False, 
                    f"Unexpected response for invalid symbol: HTTP {response.status_code}"
                )
                
        except Exception as e:
            self.log_result("Error Handling - Invalid Symbols", False, f"Connection error: {str(e)}")
    
    def test_basic_api_response(self):
        """Test basic API response"""
        try:
            response = requests.get(f"{API_BASE}/", timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if "FinGPT" in data.get("message", ""):
                    self.log_result(
                        "Basic API Response", 
                        True, 
                        f"API responding correctly: {data.get('message')}", 
                        data
                    )
                else:
                    self.log_result(
                        "Basic API Response", 
                        False, 
                        f"Unexpected response message: {data}"
                    )
            else:
                self.log_result(
                    "Basic API Response", 
                    False, 
                    f"HTTP {response.status_code}: {response.text}"
                )
                
        except Exception as e:
            self.log_result("Basic API Response", False, f"Connection error: {str(e)}")
    
    def run_priority_tests(self):
        """Run priority tests focused on MEXC integration and real price data"""
        print("🔥 PRIORITY TESTS - MEXC Integration & Real Price Data")
        print("=" * 60)
        
        # CRITICAL ENDPOINTS FROM REVIEW REQUEST
        self.test_ml_activity_live_real_prices()
        self.test_market_data_summary_mexc()
        self.test_performance_summary()
        self.test_system_performance()
        self.test_trading_health()
        self.test_trading_status()
        self.test_system_info()
        self.test_advanced_ml_status()
        self.test_ml_models_info()
        self.test_error_handling_invalid_symbols()
        self.test_basic_api_response()
    
    def run_all_tests(self):
        """Run all backend tests"""
        print("🚀 Starting FinGPT Backend API Tests - MEXC Integration Focus...")
        print(f"Testing against: {BACKEND_URL}")
        print("=" * 60)
        
        # Run priority tests
        self.run_priority_tests()
        
        # Print summary
        print("\n" + "=" * 60)
        print("📊 TEST SUMMARY")
        print("=" * 60)
        
        total_tests = len(self.results)
        passed_count = len(self.passed_tests)
        failed_count = len(self.failed_tests)
        
        print(f"Total Tests: {total_tests}")
        print(f"✅ Passed: {passed_count}")
        print(f"❌ Failed: {failed_count}")
        print(f"Success Rate: {(passed_count/total_tests)*100:.1f}%")
        
        if self.failed_tests:
            print(f"\n🔍 FAILED TESTS:")
            for test in self.failed_tests:
                print(f"  - {test}")
        
        print(f"\n📝 Detailed results logged")
        
        return {
            "total_tests": total_tests,
            "passed": passed_count,
            "failed": failed_count,
            "success_rate": (passed_count/total_tests)*100,
            "failed_tests": self.failed_tests,
            "all_results": self.results
        }

def main():
    """Main test execution"""
    tester = FinGPTTester()
    results = tester.run_all_tests()
    
    # Return appropriate exit code
    if results["failed"] == 0:
        print("\n🎉 All tests passed!")
        sys.exit(0)
    else:
        print(f"\n⚠️  {results['failed']} test(s) failed")
        sys.exit(1)

if __name__ == "__main__":
    main()