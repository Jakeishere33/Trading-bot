import os
import gc
import time
import logging
import threading
from math import sqrt, log, exp, erf
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import yfinance as yf
from flask import Flask

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import (
    MarketOrderRequest,
    LimitOrderRequest,
    GetOptionContractsRequest,
)
from alpaca.trading.enums import (
    OrderSide,
    TimeInForce,
    OrderType,
    AssetClass,
    AssetStatus,
    ContractType,
)
from alpaca.data.historical.option import OptionHistoricalDataClient
from alpaca.data.requests import OptionLatestQuoteRequest
from alpaca.data.enums import OptionsFeed

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

# ---- Core long-only sleeve (broad ETFs) ----
CORE_ETFS = {
    "VOO": 0.40,
    "VXUS": 0.15,
    "BND": 0.10,
}
CORE_TOTAL_WEIGHT = 0.65

# ---- Satellite / short universe: sector ETFs + individual equities ----
SECTOR_ETFS = [
    "XLK", "XLF", "XLV", "XLE", "XLI", "XLB", "XLY", "XLP", "XLU", "XLRE",
]
EQUITY_UNIVERSE = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "JPM", "XOM",
    "UNH", "PG", "HD", "COST",
]
SATELLITE_UNIVERSE = SECTOR_ETFS + EQUITY_UNIVERSE  # keep this list bounded for memory

SATELLITE_SLEEVE_WEIGHT = 0.20
TOP_N_LONGS = 4

SHORT_SLEEVE_WEIGHT = 0.05
TOP_N_SHORTS = 3

HEDGE_SLEEVE_WEIGHT = 0.10
HEDGE_INSTRUMENTS = {
    "SH": 0.40,   # inverse equity
    "TLT": 0.35,  # long duration bonds
    "GLD": 0.25,  # gold
}

OPTIONS_ENABLED = True
CALL_OTM_PCT = 0.03
PUT_OTM_PCT = 0.07
OPTIONS_MIN_DTE = 25
OPTIONS_MAX_DTE = 45
COVERED_CALL_MIN_SHARES = 100
OPTIONABLE_SYMBOLS = list(CORE_ETFS.keys())  # keep options sleeve small on purpose

SAFE_ASSET_CLASSES = {AssetClass.US_EQUITY, AssetClass.US_OPTION}

MAX_TRADES_PER_DAY = 300
TRADES_TODAY = 0
LAST_TRADE_DAY = None

NO_TRADE_BEFORE = (9, 45)
NO_TRADE_AFTER = (15, 55)

MIN_OPTION_VOLUME = 500
MAX_OPTION_SPREAD_PCT = 0.15

REGIME_BENCHMARK = "VOO"

# yfinance memory hygiene
MAX_BARS_DAYS = 220  # never request more history than this for any calc
LOOP_SLEEP_SECONDS = 300

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
# YAHOO FINANCE — LIGHTWEIGHT, MEMORY-BOUNDED HELPERS
# ============================================================
# Every call downloads only what it needs, extracts a plain list of floats,
# then explicitly drops the DataFrame and forces a gc pass. This keeps peak
# resident memory low enough to run comfortably inside a 512 MB Render plan.

def yf_bars(symbol, days=60):
    days = min(days, MAX_BARS_DAYS)
    try:
        df = yf.download(
            symbol,
            period=f"{days}d",
            interval="1d",
            progress=False,
            threads=False,
            auto_adjust=True,
        )
        if df is None or df.empty:
            return []
        closes = df["Close"].dropna().astype(float).tolist()
        return closes
    except Exception as e:
        log.warning("YF failed for %s: %s", symbol, e)
        return []
    finally:
        try:
            del df
        except Exception:
            pass
        gc.collect()


def latest_price(symbol):
    bars = yf_bars(symbol, days=5)
    return bars[-1] if bars else 0.0


def momentum(symbol, lookback=20):
    bars = yf_bars(symbol, days=lookback + 5)
    if len(bars) < lookback + 1:
        return 0.0
    return (bars[-1] / bars[-lookback]) - 1.0


def volatility(symbol, window=14):
    bars = yf_bars(symbol, days=window + 10)
    if len(bars) < window + 1:
        return 0.0
    rets = []
    for i in range(1, len(bars)):
        if bars[i - 1] > 0:
            rets.append((bars[i] / bars[i - 1]) - 1.0)
    if len(rets) < window:
        return 0.0
    mean = sum(rets[-window:]) / window
    var = sum((r - mean) ** 2 for r in rets[-window:]) / window
    return sqrt(var) * sqrt(TRADING_DAYS)


# ============================================================
# REGIME / RISK MANAGER (LEAN)
# ============================================================

class LeanRiskManager:
    def regime_risk_score(self):
        bars = yf_bars(REGIME_BENCHMARK, days=MAX_BARS_DAYS)
        if len(bars) < 200:
            return 0.3
        sma200 = sum(bars[-200:]) / 200
        last = bars[-1]
        below_sma = 0.0 if last > sma200 else 1.0
        last_year = bars[-252:] if len(bars) >= 252 else bars
        dd = 1.0 - (last / max(last_year))
        dd_score = max(0.0, min(dd / 0.20, 1.0))
        return (below_sma + dd_score) / 2.0

    def hedge_weights(self):
        return {sym: HEDGE_SLEEVE_WEIGHT * pct for sym, pct in HEDGE_INSTRUMENTS.items()}


# ============================================================
# BLACK–SCHOLES (LEAN)
# ============================================================

def _cdf(x):
    return 0.5 * (1 + erf(x / sqrt(2)))


def bs_price(S, K, T, r, sigma, opt_type):
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return 0.0
    d1 = (log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    if opt_type == "call":
        return S * _cdf(d1) - K * exp(-r * T) * _cdf(d2)
    else:
        return K * exp(-r * T) * _cdf(-d2) - S * _cdf(-d1)


# ============================================================
# OPTIONS ENGINE (LEAN) — covered calls + protective puts only
# ============================================================

class LeanOptionsEngine:
    def __init__(self, trading_client, option_client):
        self.trading = trading_client
        self.opt = option_client

    def find_contracts(self, underlying, contract_type):
        today = now_et().date()
        ctype = ContractType.CALL if contract_type == "call" else ContractType.PUT
        req = GetOptionContractsRequest(
            underlying_symbols=[underlying],
            status=AssetStatus.ACTIVE,
            type=ctype,
            expiration_date_gte=today + timedelta(days=OPTIONS_MIN_DTE),
            expiration_date_lte=today + timedelta(days=OPTIONS_MAX_DTE),
        )
        try:
            resp = self.trading.get_option_contracts(req)
            return list(resp.option_contracts)
        except Exception as e:
            log.warning("Option contract lookup failed for %s: %s", underlying, e)
            return []

    def quote(self, symbol):
        try:
            req = OptionLatestQuoteRequest(symbol_or_symbols=symbol, feed=OptionsFeed.INDICATIVE)
            q = self.opt.get_option_latest_quote(req)[symbol]
            return float(q.bid_price), float(q.ask_price)
        except Exception as e:
            log.warning("Option quote failed for %s: %s", symbol, e)
            return None, None

    def closest_strike(self, contracts, target):
        if not contracts:
            return None
        return sorted(
            contracts,
            key=lambda c: (abs(float(c.strike_price) - target), c.expiration_date),
        )[0]

    def liquidity_ok(self, contract, bid, ask):
        vol = getattr(contract, "volume", None)
        if vol is None or vol < MIN_OPTION_VOLUME:
            return False
        if not bid or not ask or bid <= 0 or ask <= 0:
            return False
        mid = (bid + ask) / 2
        if mid <= 0:
            return False
        return (ask - bid) / mid <= MAX_OPTION_SPREAD_PCT

    def sell_covered_calls(self, underlying, shares, spot):
        if shares < COVERED_CALL_MIN_SHARES or spot <= 0:
            return
        contracts = self.find_contracts(underlying, "call")
        target = spot * (1 + CALL_OTM_PCT)
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
            limit_price=round(use, 2),
        )
        submit_order_safe(self.trading, order, f"SELL covered calls {c.symbol}")

    def buy_protective_puts(self, underlying, shares, spot, budget):
        if shares < 100 or budget <= 0 or spot <= 0:
            return
        contracts = self.find_contracts(underlying, "put")
        target = spot * (1 - PUT_OTM_PCT)
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
        max_aff = int(budget // (use * 100))
        qty = min(int(shares // 100), max_aff)
        if qty <= 0 or use <= 0:
            return
        order = LimitOrderRequest(
            symbol=c.symbol,
            qty=qty,
            side=OrderSide.BUY,
            type=OrderType.LIMIT,
            time_in_force=TimeInForce.DAY,
            limit_price=round(use, 2),
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

    def _rebalance_symbol(self, sym, target_value, price, pos, label):
        cur = float(pos.get(sym).market_value) if sym in pos else 0.0
        diff = target_value - cur
        if target_value == 0 and cur == 0:
            return
        eq = max(self.equity(), 1.0)
        if abs(diff) / eq < 0.01:
            return
        side = OrderSide.BUY if diff > 0 else OrderSide.SELL
        qty = int(abs(diff) / price)
        if qty <= 0:
            return
        assert_safe_order(sym, AssetClass.US_EQUITY)
        order = MarketOrderRequest(symbol=sym, qty=qty, side=side, time_in_force=TimeInForce.DAY)
        submit_order_safe(self.trading, order, f"{label} {sym}")

    # ---- core sleeve: broad-market ETFs, long only ----
    def run_core(self):
        eq = self.equity()
        if eq <= 0:
            return
        pos = self.positions()
        for sym, w in CORE_ETFS.items():
            price = latest_price(sym)
            if price <= 0:
                continue
            self._rebalance_symbol(sym, eq * w, price, pos, "CORE")

    # ---- satellite sleeve: top-momentum longs from ETFs + equities ----
    def run_satellite(self, scores=None):
        eq = self.equity()
        if eq <= 0:
            return
        pos = self.positions()
        if scores is None:
            scores = {t: momentum(t, 60) for t in SATELLITE_UNIVERSE}
        top = sorted(SATELLITE_UNIVERSE, key=lambda x: scores.get(x, -1e9), reverse=True)[:TOP_N_LONGS]
        per = SATELLITE_SLEEVE_WEIGHT / max(len(top), 1)
        for sym in top:
            price = latest_price(sym)
            if price <= 0:
                continue
            self._rebalance_symbol(sym, eq * per, price, pos, "SAT")

    # ---- short sleeve: worst-momentum names from ETFs + equities ----
    def run_shorts(self, scores=None):
        eq = self.equity()
        if eq <= 0:
            return
        pos = self.positions()
        if scores is None:
            scores = {t: momentum(t, 60) for t in SATELLITE_UNIVERSE}
        worst = sorted(SATELLITE_UNIVERSE, key=lambda x: scores.get(x, 1e9))[:TOP_N_SHORTS]
        per = SHORT_SLEEVE_WEIGHT / max(len(worst), 1)
        for sym in worst:
            price = latest_price(sym)
            if price <= 0:
                continue
            target = eq * per
            qty = int(target / price)
            if qty <= 0:
                continue
            if sym in pos and float(pos[sym].qty) > 0:
                continue  # don't short a name we're already long
            assert_safe_order(sym, AssetClass.US_EQUITY)
            order = MarketOrderRequest(symbol=sym, qty=qty, side=OrderSide.SELL, time_in_force=TimeInForce.DAY)
            submit_order_safe(self.trading, order, f"SHORT {sym}")

    # ---- hedge sleeve ----
    def run_hedges(self):
        eq = self.equity()
        if eq <= 0:
            return
        pos = self.positions()
        for sym, w in self.risk.hedge_weights().items():
            price = latest_price(sym)
            if price <= 0:
                continue
            self._rebalance_symbol(sym, eq * w, price, pos, "HEDGE")

    # ---- options sleeve: covered calls + protective puts on core ETF holdings ----
    def run_options(self):
        if not OPTIONS_ENABLED:
            return
        eq = self.equity()
        if eq <= 0:
            return
        pos = self.positions()
        for sym in OPTIONABLE_SYMBOLS:
            if sym not in pos:
                continue
            shares = float(pos[sym].qty)
            if shares <= 0:
                continue  # only write/hedge against long shares
            price = latest_price(sym)
            if price <= 0:
                continue
            if shares >= COVERED_CALL_MIN_SHARES:
                self.options.sell_covered_calls(sym, shares, price)
            put_budget = eq * 0.01
            self.options.buy_protective_puts(sym, shares, price, put_budget)

    # ---- main cycle ----
    def run_once(self):
        if not can_trade_today():
            return
        regime = self.risk.regime_risk_score()
        log.info("Regime risk score: %.2f", regime)

        # Momentum scores are shared by the long and short sleeves so we hit
        # Yahoo Finance once per symbol per cycle instead of twice.
        momentum_scores = {t: momentum(t, 60) for t in SATELLITE_UNIVERSE}

        for label, fn in [
            ("core", self.run_core),
            ("satellite", lambda: self.run_satellite(momentum_scores)),
            ("hedges", self.run_hedges),
        ]:
            try:
                fn()
            except Exception as e:
                log.exception("Sleeve '%s' failed, continuing: %s", label, e)

        if regime > 0.4:
            try:
                self.run_shorts(momentum_scores)
            except Exception as e:
                log.exception("Sleeve 'shorts' failed, continuing: %s", e)

        try:
            self.run_options()
        except Exception as e:
            log.exception("Sleeve 'options' failed, continuing: %s", e)

        del momentum_scores
        gc.collect()


# ============================================================
# BACKGROUND LOOP FOR RENDER
# ============================================================

def _env_bool(name, default=True):
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")


def trading_loop():
    api_key = os.getenv("ALPACA_API_KEY", "")
    api_secret = os.getenv("ALPACA_API_SECRET", "")
    paper = _env_bool("ALPACA_PAPER", True)

    if not api_key or not api_secret:
        log.error("ALPACA_API_KEY / ALPACA_API_SECRET not set — trading loop will not start.")
        return

    trading_client = TradingClient(api_key, api_secret, paper=paper)
    # Contract lookup goes through TradingClient; live quotes need the
    # dedicated options market-data client — they are NOT the same object.
    option_data_client = OptionHistoricalDataClient(api_key, api_secret)

    engine = LeanTradingEngine(trading_client, option_data_client)

    while True:
        try:
            engine.run_once()
        except Exception as e:
            log.exception("Engine error: %s", e)
        gc.collect()
        time.sleep(LOOP_SLEEP_SECONDS)


threading.Thread(target=trading_loop, daemon=True).start()

if __name__ == "__main__":
    port = int(os.getenv("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
