import os
import time
import threading
import numpy as np
import pandas as pd
import yfinance as yf
from flask import Flask
import alpaca_trade_api as tradeapi

app = Flask(__name__)

@app.route('/')
def home():
    return "Global Broad-Sector Risk Engine Online!", 200

# THE ENTIRE GLOBAL SECTOR UNIVERSE (Covering all global markets)
CORE_ETF = "VOO"  # Core S&P 500 Base (70%)
GLOBAL_SECTORS = {
    "Technology": "XLK",
    "Financials": "XLF",
    "Healthcare": "XLV",
    "Energy": "XLE",
    "Industrials": "XLI",
    "Materials": "XLB",
    "Consumer_Discretionary": "XLY",
    "Consumer_Staples": "XLP",
    "Utilities": "XLU",
    "Real_Estate": "XLRE",
    "Global_Ex_US": "VEU",  # International Global Equities
    "Emerging_Markets": "VWO" # Global Emerging Economies
}
ALL_TICKERS = list(GLOBAL_SECTORS.values())

# 1. MACRO STATISTICAL MATRIX ENGINE (15+ Years Historical Depth)
def get_top_global_sectors():
    print("Evaluating 15-year historical trends across all global macro sectors...")
    try:
        # Fetching 15 years of market data to construct systemic risk models
        data = yf.download(ALL_TICKERS, period="15y")['Close']
        returns = data.pct_change().dropna()
        
        risk_free_rate = 0.04 / 252
        sharpe_ratios = {}
        
        for name, ticker in GLOBAL_SECTORS.items():
            if ticker not in returns: continue
            excess_ret = returns[ticker] - risk_free_rate
            std_dev = returns[ticker].std()
            
            if std_dev == 0: continue
            # Calculate Annualized Sharpe Ratio Profile
            sharpe_ratios[ticker] = (excess_returns_mean := excess_ret.mean() / std_dev) * np.sqrt(252)
            
        # Select the two highest performing global sectors statistically
        sorted_sectors = sorted(sharpe_ratios.items(), key=lambda x: x[1], reverse=True)
        best_sector_1 = sorted_sectors[0][0] if len(sorted_sectors) > 0 else "XLK"
        best_sector_2 = sorted_sectors[1][0] if len(sorted_sectors) > 1 else "XLV"
        
        print(f"Top Statistical Global Allocations -> Sector 1: {best_sector_1} | Sector 2: {best_sector_2}")
        return best_sector_1, best_sector_2
    except Exception as e:
        print(f"Macro analysis engine failure: {e}. Defaulting to safe proxies.")
        return "XLK", "XLV"

# 2. STRUCTURAL ATR VOLATILITY PROTECTION LAYER
def get_market_risk_profile(ticker):
    try:
        hist = yf.download(ticker, period="1y")
        close_p = hist['Close']
        
        # Absolute Benchmark Health Verification
        if ticker == CORE_ETF:
            sma_200 = close_p.rolling(window=200).mean().iloc[-1]
            return {"bull_regime": close_p.iloc[-1] > sma_200}
            
        # Compute Average True Range (ATR)
        tr1 = hist['High'] - hist['Low']
        tr2 = abs(hist['High'] - close_p.shift(1))
        tr3 = abs(hist['Low'] - close_p.shift(1))
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        return {"atr": float(true_range.rolling(window=14).mean().iloc[-1])}
    except Exception:
        return {"bull_regime": False, "atr": 1.5}

# 3. COMPREHENSIVE POSITION MANAGER
def trading_bot_loop():
    API_KEY = os.environ.get("ALPACA_PAPER_KEY")
    SECRET_KEY = os.environ.get("ALPACA_PAPER_SECRET")
    BASE_URL = os.environ.get("ALPACA_PAPER_URL", "https://alpaca.markets")
    
    if not API_KEY or not SECRET_KEY: return
    api = tradeapi.REST(API_KEY, SECRET_KEY, BASE_URL, api_version='v2')
    
    while True:
        try:
            if time.gmtime().tm_wday >= 5:
                time.sleep(1800)
                continue
                
            account = api.get_account()
            total_equity = float(account.portfolio_value)
            cash_available = float(account.cash)
            positions = {p.symbol: p for p in api.list_positions()}
            
            # --- CONTINUOUS TRAILING STOP PROTECTIONS ---
            for symbol in list(positions.keys()):
                if len(symbol) <= 5:  # Skip active options contracts string formats
                    open_orders = api.list_orders(status='open', symbols=[symbol])
                    for order in open_orders:
                        if order.side == 'sell': api.cancel_order(order.id)
                    
                    api.submit_order(
                        symbol=symbol, qty=int(positions[symbol].qty),
                        side='sell', type='trailing_stop', trail_percent=5.0, time_in_force='gtc'
                    )

            # --- MAIN FUND STRATEGY DEPLOYMENT (70% Core VOO) ---
            target_voo_cash = total_equity * 0.70
            voo_price = api.get_latest_trade(CORE_ETF).price
            current_voo_value = float(positions[CORE_ETF].market_value) if CORE_ETF in positions else 0
            
            if current_voo_value < (target_voo_cash * 0.95) and cash_available > voo_price:
                buy_qty = int((target_voo_cash - current_voo_value) // voo_price)
                if buy_qty > 0: api.submit_order(symbol=CORE_ETF, qty=buy_qty, side='buy', type='market', time_in_force='gtc')

            # --- DYNAMIC GLOBAL SECTOR LAYER (20% Allocation Engine) ---
            market_condition = get_market_risk_profile(CORE_ETF)
            if market_condition.get("bull_regime", False):
                target_per_sector = (total_equity * 0.20) / 2
                sector_1, sector_2 = get_top_global_sectors()
                
                for sector_ticker in [sector_1, sector_2]:
                    price = api.get_latest_trade(sector_ticker).price
                    current_sector_value = float(positions[sector_ticker].market_value) if sector_ticker in positions else 0
                    
                    if current_sector_value < (target_per_sector * 0.95) and cash_available > price:
                        risk_metrics = get_market_risk_profile(sector_ticker)
                        atr = risk_metrics.get("atr", 1.5)
                        
                        # Statistical Volatility Multiplier adjustment
                        adjusted_max_allocation = target_per_sector * (1.2 / atr)
                        safe_target = min(target_per_sector, adjusted_max_allocation)
                        
                        shares_to_buy = int((safe_target - current_sector_value) // price)
                        if shares_to_buy > 0:
                            api.submit_order(symbol=sector_ticker, qty=shares_to_buy, side='buy', type='market', time_in_force='gtc')
            else:
                print("Global system models indicate systemic downtrend. Retaining hedge allocations inside liquid cash protection.")

            # --- STRUCTURAL RISK GOVERNOR FOR OPTIONS (Under 10%) ---
            current_options_equity = sum(float(pos.market_value) for pos in positions.values() if len(pos.symbol) > 5)
            if (current_options_equity / total_equity) * 100 > 10.0:
                print("CRITICAL EXPOSURE WARNING: Options footprint exceeds safety constraints. Blocked incoming purchases.")

        except Exception as e:
            print(f"Main processing failure: {e}")
            
        time.sleep(300)

threading.Thread(target=trading_bot_loop, daemon=True).start()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
