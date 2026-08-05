import os
import time
import logging
import threading
from math import sqrt, log, exp
from datetime import datetime, timedelta, date
from zoneinfo import ZoneInfo

import yfinance as yf
from flask import Flask

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import (
    MarketOrderRequest,
    LimitOrderRequest,
)
from alpaca.trading.enums import (
    OrderSide,
    TimeInForce,
    OrderType,
    AssetClass,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("lean_risk_engine")

app = Flask(__name__)

@app.route("/")
def home():
    return "Lean Global Multi-Factor Risk Engine Online", 200

@app.route("/health")
def health():
    return "OK", 200

# ============================================================
# CONFIG
# ============================================================

EASTERN_TZ = ZoneInfo("America/New_York")

RISK_FREE_ANNUAL = 0.04
TRADING_DAYS = 252

CORE_ETFS = {
    "VOO": 0.45,
    "VXUS": 0.15,
    "BND": 0.10,
}
CORE_TOTAL_WEIGHT = 0.70

SECTOR_TICKERS = [
    "XLK","XLF","XLV","XLE","XLI","XLB","XLY","XLP","XLU","XLRE","VEU","VWO"
]

SATELLITE_SLEEVE_WEIGHT = 0.20
HEDGE_SLEEVE_WEIGHT = 0.10

HEDGE_INSTRUMENTS = {
    "inverse_equity": "SH",
    "long_duration_bonds": "TLT",
    "gold": "GLD",
}

SHORT_SLEEVE_WEIGHT = 0.05
TOP_N_SHORTS = 3

OPTIONS_ENABLED = True
CALL_OTM_PCT = 0.03
PUT_OTM_PCT = 0.07
OPTIONS_MIN_DTE = 25
OPTIONS_MAX_DTE = 45
COVERED_CALL_MIN_SHARES = 100

SAFE_ASSET_CLASSES = {AssetClass.US_EQUITY, AssetClass.US_OPTION}

MAX_TRADES_PER_DAY = 500
TRADES_TODAY = 0
LAST_TRADE_DAY = None

NO_TRADE_BEFORE = (9, 45)
NO_TRADE_AFTER = (15, 55)

MIN_OPTION_VOLUME = 500
MAX_OPTION_SPREAD_PCT = 0.15

REGIME_BENCHMARK = "VOO"

# ============================================================
# TIME / TRADE WINDOW
# ============================================================

def now_et():
    return datetime.now(EASTERN_TZ)

def in_no_trade_window():
    t = now_et()
    h, m = t.hour, t.minute
    if (h < NO_TRADE_BEFORE[0]) or (h == NO_TRADE_BEFORE[0] and m < NO_TRADE_BEFORE[1]):
        return True
    if (h > NO_TRADE_AFTER[0]) or (h == NO_TRADE_AFTER[0] and m >= NO_TRADE_AFTER[1]):
        return True
    return False

def reset_trade_counter_if_new_day():
    global TRADES_TODAY, LAST_TRADE_DAY
    today = now_et().date()
    if LAST_TRADE_DAY != today:
        LAST_TRADE_DAY = today
        TRADES_TODAY = 0
        log.info("New trading day: trade counter reset.")

def can_trade_today():
    reset_trade_counter_if_new_day()
    if in_no_trade_window():
        log.info("In no-trade window, skipping trades.")
        return False
    return TRADES_TODAY < MAX_TRADES_PER_DAY

def submit_order_safe(trading_client, order_data, label=""):
    global TRADES_TODAY
    reset_trade_counter_if_new_day()
    if TRADES_TODAY >= MAX_TRADES_PER_DAY:
        log.warning("Trade limit reached, skipping order: %s", label)
        return
    if in_no_trade_window():
        log.info("No-trade window active, skipping order: %s", label)
        return
    try:
        trading_client.submit_order(order_data=order_data)
        TRADES_TODAY += 1
        log.info("Order submitted (%d/%d): %s", TRADES_TODAY, MAX_TRADES_PER_DAY, label)
    except Exception as e:
        log.exception("Order failed (%s): %s", label, e)

def assert_safe_order(symbol: str, asset_class):
    if asset_class not in SAFE_ASSET_CLASSES:
        raise ValueError(f"Blocked order for {symbol}: asset class {asset_class} not permitted.")

# ============================================================
# YAHOO FINANCE — LIGHTWEIGHT HELPERS
# ============================================================

def yf_bars(symbol, days=60):
    try:
        df = yf.download(symbol, period=f"{days}d", interval="1d", progress=False)
        return df["Close"].dropna().tolist()
    except Exception as e:
        log.warning("YF failed for %s: %s", symbol, e)
        return []

def latest_price(symbol):
    bars = yf_bars(symbol, days=5)
    return bars[-1] if bars else 0.0

def momentum(symbol, lookback=20):
    bars = yf_bars(symbol, days=lookback+5)
    if len(bars) < lookback+1:
        return 0.0
    return (bars[-1] / bars[-lookback]) - 1.0

def volatility(symbol, window=14):
    bars = yf_bars(symbol, days=window+10)
    if len(bars) < window+1:
        return 0.0
    rets = []
    for i in range(1, len(bars)):
        if bars[i-1] > 0:
            rets.append((bars[i] / bars[i-1]) - 1.0)
    if len(rets) < window:
        return 0.0
    mean = sum(rets[-window:]) / window
    var = sum((r - mean)**2 for r in rets[-window:]) / window
    return sqrt(var) * sqrt(TRADING_DAYS)

# ============================================================
# REGIME / RISK MANAGER (LEAN)
# ============================================================

class LeanRiskManager:
    def regime_risk_score(self):
        bars = yf_bars(REGIME_BENCHMARK, days=220)
        if len(bars) < 200:
            return 0.3
        sma200 = sum(bars[-200:]) / 200
        last = bars[-1]
        below_sma = 0.0 if last > sma200 else 1.0
        last_year = bars[-252:]
        dd = 1.0 - (last / max(last_year))
        dd_score = max(0.0, min(dd / 0.20, 1.0))
        return (below_sma + dd_score) / 2.0

    def hedge_weights(self):
        return {
            "SH": HEDGE_SLEEVE_WEIGHT * 0.4,
            "TLT": HEDGE_SLEEVE_WEIGHT * 0.35,
            "GLD": HEDGE_SLEEVE_WEIGHT * 0.25,
        }

# ============================================================
# BLACK–SCHOLES (LEAN)
# ============================================================

def bs_price(S, K, T, r, sigma, opt_type):
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return 0.0
    d1 = (log(S/K) + (r + 0.5*sigma*sigma)*T) / (sigma*sqrt(T))
    d2 = d1 - sigma*sqrt(T)
    from math import erf
    def cdf(x): return 0.5*(1+erf(x/sqrt(2)))
    if opt_type == "call":
        return S*cdf(d1) - K*exp(-r*T)*cdf(d2)
    else:
        return K*exp(-r*T)*cdf(-d2) - S*cdf(-d1)

# ============================================================
# OPTIONS ENGINE (LEAN)
# ============================================================

class LeanOptionsEngine:
    def __init__(self, trading_client, option_client):
        self.trading = trading_client
        self.opt = option_client

    def find_contracts(self, underlying, contract_type):
        today = now_et().date()
        req = {
            "underlying_symbols": [underlying],
            "status": "active",
            "type": contract_type,
            "expiration_date_gte": today + timedelta(days=OPTIONS_MIN_DTE),
            "expiration_date_lte": today + timedelta(days=OPTIONS_MAX_DTE),
        }
        try:
            resp = self.trading.get_option_contracts(req)
            return list(resp.option_contracts)
        except Exception as e:
            log.warning("Option contract lookup failed for %s: %s", underlying, e)
            return []

    def quote(self, symbol):
        try:
            q = self.opt.get_option_latest_quote({"symbol_or_symbols": symbol})[symbol]
            return float(q.bid_price), float(q.ask_price)
        except Exception as e:
            log.warning("Option quote failed for %s: %s", symbol, e)
            return None, None

    def closest_strike(self, contracts, target):
        if not contracts:
            return None
        return sorted(
            contracts,
            key=lambda c: (abs(float(c.strike_price) - target), c.expiration_date)
        )[0]

    def liquidity_ok(self, contract, bid, ask):
        vol = getattr(contract, "volume", None)
        if vol is None or vol < MIN_OPTION_VOLUME:
            return False
        if not bid or not ask or bid <= 0 or ask <= 0:
            return False
        mid = (bid+ask)/2
        if mid <= 0:
            return False
        return (ask-bid)/mid <= MAX_OPTION_SPREAD_PCT

    def sell_covered_calls(self, underlying, shares, spot):
        if shares < COVERED_CALL_MIN_SHARES or spot <= 0:
            return
        contracts = self.find_contracts(underlying, "call")
        target = spot*(1+CALL_OTM_PCT)
        c = self.closest_strike(contracts, target)
        if not c:
            return
        assert_safe_order(c.symbol, AssetClass.US_OPTION)
        T = (c.expiration_date - now_et().date()).days / TRADING_DAYS
        sigma = max(volatility(underlying), 0.15)
        fair = bs_price(spot, float(c.strike_price), T, RISK_FREE_ANNUAL, sigma, "call")
        bid, ask = self.quote(c.symbol)
        if not self.liquidity_ok(c, bid, ask):
            return
        use = bid if bid and bid > 0 else fair
        qty = int(shares // 100)
        if qty <= 0 or use <= 0:
            return
        order = LimitOrderRequest(
            symbol=c.symbol,
            qty=qty,
            side=OrderSide.SELL,
            type=OrderType.LIMIT,
            time_in_force=TimeInForce.DAY,
            limit_price=round(use,2),
        )
        submit_order_safe(self.trading, order, f"SELL covered calls {c.symbol}")

    def buy_protective_puts(self, underlying, shares, spot, budget):
        if shares < 100 or budget <= 0 or spot <= 0:
            return
        contracts = self.find_contracts(underlying, "put")
        target = spot*(1-PUT_OTM_PCT)
        c = self.closest_strike(contracts, target)
        if not c:
            return
        assert_safe_order(c.symbol, AssetClass.US_OPTION)
        T = (c.expiration_date - now_et().date()).days / TRADING_DAYS
        sigma = max(volatility(underlying), 0.15)
        fair = bs_price(spot, float(c.strike_price), T, RISK_FREE_ANNUAL, sigma, "put")
        bid, ask = self.quote(c.symbol)
        if not self.liquidity_ok(c, bid, ask):
            return
        use = ask if ask and ask > 0 else fair
        max_aff = int(budget // (use*100))
        qty = min(int(shares//100), max_aff)
        if qty <= 0 or use <= 0:
            return
        order = LimitOrderRequest(
            symbol=c.symbol,
            qty=qty,
            side=OrderSide.BUY,
            type=OrderType.LIMIT,
            time_in_force=TimeInForce.DAY,
            limit_price=round(use,2),
        )
        submit_order_safe(self.trading, order, f"BUY protective puts {c.symbol}")

# ============================================================
# MAIN TRADING ENGINE
# ============================================================

class LeanTradingEngine:
    def __init__(self, trading_client, option_client):
        self.trading = trading_client
        self.options = LeanOptionsEngine(trading_client, option_client)
        self.risk = LeanRiskManager()

    def equity(self):
        try:
            return float(self.trading.get_account().equity)
        except Exception as e:
            log.warning("Failed to fetch equity: %s", e)
            return 0.0

    def positions(self):
        try:
            return {p.symbol: p for p in self.trading.get_all_positions()}
        except Exception as e:
            log.warning("Failed to fetch positions: %s", e)
            return {}

    # ---- core sleeve ----
    def run_core(self):
        eq = self.equity()
        if eq <= 0:
            return
        pos = self.positions()
        for sym, w in CORE_ETFS.items():
            target = eq*w
            price = latest_price(sym)
            if price <= 0:
                continue
            cur = float(pos.get(sym).market_value) if sym in pos else 0.0
            diff = target - cur
            if abs(diff)/eq < 0.01:
                continue
            side = OrderSide.BUY if diff>0 else OrderSide.SELL
            qty = int(abs(diff)/price)
            if qty <= 0:
                continue
            assert_safe_order(sym, AssetClass.US_EQUITY)
            order = MarketOrderRequest(symbol=sym, qty=qty, side=side, time_in_force=TimeInForce.DAY)
            submit_order_safe(self.trading, order, f"CORE {sym}")

    # ---- satellite sectors ----
    def run_satellite(self):
        eq = self.equity()
        if eq <= 0:
            return
        pos = self.positions()
        scores = {t: momentum(t,60) for t in SECTOR_TICKERS}
        top = sorted(SECTOR_TICKERS, key=lambda x: scores.get(x, -1e9), reverse=True)[:3]
        per = SATELLITE_SLEEVE_WEIGHT / max(len(top),1)
        for sym in top:
            target = eq*per
            price = latest_price(sym)
            if price <= 0:
                continue
            cur = float(pos.get(sym).market_value) if sym in pos else 0.0
            diff = target - cur
            if abs(diff)/eq < 0.01:
                continue
            side = OrderSide.BUY if diff>0 else OrderSide.SELL
            qty = int(abs(diff)/price)
            if qty <= 0:
                continue
            assert_safe_order(sym, AssetClass.US_EQUITY)
            order = MarketOrderRequest(symbol=sym, qty=qty, side=side, time_in_force=TimeInForce.DAY)
            submit_order_safe(self.trading, order, f"SAT {sym}")

    # ---- shorts ----
    def run_shorts(self):
        eq = self.equity()
        if eq <= 0:
            return
        pos = self.positions()
        scores = {t: momentum(t,60) for t in SECTOR_TICKERS}
        worst = sorted(SECTOR_TICKERS, key=lambda x: scores.get(x, 1e9))[:TOP_N_SHORTS]
        per = SHORT_SLEEVE_WEIGHT / max(len(worst),1)
        for sym in worst:
            price = latest_price(sym)
            if price <= 0:
                continue
            target = eq*per
            qty = int(target/price)
            if qty <= 0:
                continue
            if sym in pos and float(pos[sym].qty) > 0:
                continue
            assert_safe_order(sym, AssetClass.US_EQUITY)
            order = MarketOrderRequest(symbol=sym, qty=qty, side=OrderSide.SELL, time_in_force=TimeInForce.DAY)
            submit_order_safe(self.trading, order, f"SHORT {sym}")

    # ---- hedges ----
    def run_hedges(self):
        eq = self.equity()
        if eq <= 0:
            return
        pos = self.positions()
        weights = self.risk.hedge_weights()
        for sym, w in weights.items():
            target = eq*w
            price = latest_price(sym)
            if price <= 0:
                continue
            cur = float(pos.get(sym).market_value) if sym in pos else 0.0
            diff = target - cur
            if abs(diff)/eq < 0.01:
                continue
            side = OrderSide.BUY if diff>0 else OrderSide.SELL
            qty = int(abs(diff)/price)
            if qty <= 0:
                continue
            assert_safe_order(sym, AssetClass.US_EQUITY)
            order = MarketOrderRequest(symbol=sym, qty=qty, side=side, time_in_force=TimeInForce.DAY)
            submit_order_safe(self.trading, order, f"HEDGE {sym}")

    # ---- options ----
    def run_options(self):
        if not OPTIONS_ENABLED:
            return
        eq = self.equity()
        if eq <= 0:
            return
        pos = self.positions()
        for sym in CORE_ETFS.keys():
            if sym not in pos:
                continue
            shares = float(pos[sym].qty)
            price = latest_price(sym)
            if price <= 0:
                continue
            if shares >= COVERED_CALL_MIN_SHARES:
                self.options.sell_covered_calls(sym, shares, price)
            put_budget = eq*0.01
            self.options.buy_protective_puts(sym, shares, price, put_budget)

    # ---- main ----
    def run_once(self):
        if not can_trade_today():
            return
        regime = self.risk.regime_risk_score()
        log.info("Regime risk score: %.2f", regime)
        self.run_core()
        self.run_satellite()
        self.run_hedges()
        if regime > 0.4:
            self.run_shorts()
        self.run_options()

# ============================================================
# BACKGROUND LOOP FOR RENDER
# ============================================================

def trading_loop():
    api_key = os.getenv("ALPACA_API_KEY", "YOUR_KEY")
    api_secret = os.getenv("ALPACA_API_SECRET", "YOUR_SECRET")
    paper = True

    trading_client = TradingClient(api_key, api_secret, paper=paper)
    option_client = trading_client  # same client for options

    engine = LeanTradingEngine(trading_client, option_client)

    while True:
        try:
            engine.run_once()
        except Exception as e:
            log.exception("Engine error: %s", e)
        time.sleep(300)

threading.Thread(target=trading_loop, daemon=True).start()
