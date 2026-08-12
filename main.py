import os
import sys
import gc
import time
import logging
import threading
from math import sqrt, log as ln, exp, erf
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from flask import Flask
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce, OrderType, AssetClass

import yfinance as yf

# ------------------------------------------------------------
# Logging / Flask / Engine state
# ------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger("lean_risk_engine")

app = Flask(__name__)

ENGINE_STATE = {
    "thread_started": False,
    "thread_alive": False,
    "last_cycle_started": None,
    "last_cycle_finished": None,
    "last_error": None,
    "trades_today": 0,
}

@app.route("/")
def home():
    return "Lean Short + Options Risk Engine Online", 200

@app.route("/health")
def health():
    return ENGINE_STATE, 200

# ------------------------------------------------------------
# Config
# ------------------------------------------------------------

EASTERN_TZ = ZoneInfo("America/New_York")

RISK_FREE_ANNUAL = 0.04
TRADING_DAYS = 252

SECTOR_ETFS = [
    "XLK", "XLF", "XLV", "XLE", "XLI", "XLB", "XLY", "XLP", "XLU", "XLRE",
]
EQUITY_UNIVERSE = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "JPM", "XOM",
    "UNH", "PG", "HD", "COST",
]

SEMICONDUCTOR_TICKERS = [
    "NVDA", "AMD", "INTC", "TSM", "AVGO", "QCOM", "TXN", "MU", "LRCX", "AMAT",
    "ADI", "KLAC", "MRVL", "ON", "MCHP", "SWKS", "QRVO", "NXPI", "TER", "ENTG",
    "MPWR", "CRUS", "SLAB", "POWI", "DIOD", "RMBS", "ALGM", "WOLF", "ONTO", "COHU",
]
MANUFACTURING_TICKERS = [
    "CAT", "DE", "GE", "HON", "MMM", "ETN", "EMR", "ITW", "PH", "ROK",
    "DOV", "XYL", "IR", "AME", "ROP", "SNA", "SWK", "CMI", "PCAR", "FAST",
    "GWW", "PNR", "FLS", "AOS", "NDSN", "ALLE", "IEX", "HUBB", "GGG", "CR",
]
PHARMA_TICKERS = [
    "PFE", "JNJ", "MRK", "ABBV", "LLY", "BMY", "AMGN", "GILD", "VRTX", "REGN",
    "BIIB", "ZTS", "MRNA", "ALNY", "INCY", "EXEL", "UTHR", "JAZZ", "RPRX", "SUPN",
    "PCVX", "ARGX", "BMRN", "IONS", "NBIX", "HALO", "CRSP", "VTRS", "TEVA", "ELAN",
]
MINING_TICKERS = [
    "FCX", "NEM", "GOLD", "SCCO", "AEM", "TECK", "RIO", "BHP", "VALE", "MOS",
    "AA", "CLF", "X", "NUE", "STLD", "MP", "CDE", "HL", "PAAS", "AG",
    "SSRM", "EGO", "KGC", "AU", "WPM", "FNV", "RGLD", "ALB", "LAC", "SQM",
]

def _dedupe(seq):
    seen = set()
    out = []
    for s in seq:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out

SHORT_UNIVERSE = _dedupe(SECTOR_ETFS + EQUITY_UNIVERSE +
                         SEMICONDUCTOR_TICKERS + MANUFACTURING_TICKERS +
                         PHARMA_TICKERS + MINING_TICKERS)
OPTIONABLE_SYMBOLS = SHORT_UNIVERSE

SHORT_SLEEVE_WEIGHT = 0.30
TOP_N_SHORTS = 8

OPTIONS_ENABLED = True
HEDGE_CALL_OTM_PCT = 0.05
OPTIONS_MIN_DTE = 25
OPTIONS_MAX_DTE = 45
OPTIONS_HEDGE_BUDGET_PCT = 0.05

SAFE_ASSET_CLASSES = {AssetClass.US_EQUITY, AssetClass.US_OPTION}

MAX_TRADES_PER_DAY = 300
MIN_TRADES_PER_DAY = 5
TRADES_TODAY = 0
LAST_TRADE_DAY = None

NO_TRADE_BEFORE = (9, 45)
NO_TRADE_AFTER = (15, 55)

MIN_OPTION_VOLUME = 500
MAX_OPTION_SPREAD_PCT = 0.15

REGIME_BENCHMARK = "VOO"

MAX_BARS_DAYS = 220
LOOP_SLEEP_SECONDS = 60

# ------------------------------------------------------------
# Time / trade window
# ------------------------------------------------------------

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
        ENGINE_STATE["trades_today"] = 0
        log.info("New trading day: trade counter reset.")

def can_trade_today():
    reset_trade_counter_if_new_day()
    if in_no_trade_window():
        log.info("In no-trade window (%s ET), skipping trades.", now_et().strftime("%H:%M"))
        return False
    return TRADES_TODAY < MAX_TRADES_PER_DAY

def submit_order_safe(trading_client, order_data, label=""):
    global TRADES_TODAY
    reset_trade_counter_if_new_day()
    if TRADES_TODAY >= MAX_TRADES_PER_DAY:
        log.warning("Trade limit reached, skipping order: %s", label)
        return False
    if in_no_trade_window():
        log.info("No-trade window active, skipping order: %s", label)
        return False
    try:
        trading_client.submit_order(order_data=order_data)
        TRADES_TODAY += 1
        ENGINE_STATE["trades_today"] = TRADES_TODAY
        log.info("Order submitted (%d, min-per-day=%d): %s", TRADES_TODAY, MIN_TRADES_PER_DAY, label)
        return True
    except Exception as e:
        log.exception("Order failed (%s): %s", label, e)
        return False

def assert_safe_order(symbol: str, asset_class):
    if asset_class not in SAFE_ASSET_CLASSES:
        raise ValueError(f"Blocked order for {symbol}: asset class {asset_class} not permitted.")

# ------------------------------------------------------------
# Market data (Yahoo Finance)
# ------------------------------------------------------------

_YF_CACHE = {}
_YF_CACHE_TTL_SECONDS = 60

def yf_bars(symbol, days=60):
    days = min(days, MAX_BARS_DAYS)
    cache_key = (symbol, days)
    cached = _YF_CACHE.get(cache_key)
    if cached and (time.time() - cached[0]) < _YF_CACHE_TTL_SECONDS:
        return cached[1]
    period_days = int(days * 1.6) + 10
    try:
        df = yf.Ticker(symbol).history(period=f"{period_days}d", interval="1d", auto_adjust=True)
        if df is None or df.empty or "Close" not in df.columns:
            return []
        closes = df["Close"].dropna().astype(float).tolist()
        closes = closes[-days:]
        _YF_CACHE[cache_key] = (time.time(), closes)
        return closes
    except Exception as e:
        log.warning("Yahoo Finance bars failed for %s: %s", symbol, e)
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

# ------------------------------------------------------------
# Regime / risk manager
# ------------------------------------------------------------

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

# ------------------------------------------------------------
# Black–Scholes
# ------------------------------------------------------------

def _cdf(x):
    return 0.5 * (1 + erf(x / sqrt(2)))

def bs_price(S, K, T, r, sigma, opt_type):
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return 0.0
    d1 = (ln(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    if opt_type == "call":
        return S * _cdf(d1) - K * exp(-r * T) * _cdf(d2)
    else:
        return K * exp(-r * T) * _cdf(-d2) - S * _cdf(-d1)

# ------------------------------------------------------------
# Options engine (Yahoo Finance)
# ------------------------------------------------------------

class LeanOptionsEngine:
    def __init__(self, trading_client):
        self.trading = trading_client

    def yf_option_chain(self, underlying):
        try:
            tk = yf.Ticker(underlying)
            expirations = tk.options
            chains = {}
            today = now_et().date()
            for exp in expirations:
                try:
                    exp_date = datetime.strptime(exp, "%Y-%m-%d").date()
                    dte = (exp_date - today).days
                    if dte < OPTIONS_MIN_DTE or dte > OPTIONS_MAX_DTE:
                        continue
                    chain = tk.option_chain(exp)
                    chains[exp_date] = chain.calls
                except Exception:
                    continue
            return chains
        except Exception as e:
            log.warning("Yahoo option chain failed for %s: %s", underlying, e)
            return {}

    def find_contracts(self, underlying):
        chains = self.yf_option_chain(underlying)
        out = []
        for exp_date, df in chains.items():
            for _, row in df.iterrows():
                out.append({
                    "symbol": row["contractSymbol"],
                    "strike": float(row["strike"]),
                    "expiration": exp_date,
                    "bid": float(row.get("bid", 0) or 0),
                    "ask": float(row.get("ask", 0) or 0),
                    "volume": int(row.get("volume", 0) or 0),
                })
        return out

    def closest_strike(self, contracts, target):
        if not contracts:
            return None
        return sorted(
            contracts,
            key=lambda c: (abs(c["strike"] - target), c["expiration"])
        )[0]

    def liquidity_ok(self, c):
        if c["volume"] < MIN_OPTION_VOLUME:
            return False
        bid, ask = c["bid"], c["ask"]
        if bid <= 0 or ask <= 0:
            return False
        mid = (bid + ask) / 2
        if mid <= 0:
            return False
        return (ask - bid) / mid <= MAX_OPTION_SPREAD_PCT

    def buy_protective_calls(self, underlying, shares_short, spot, budget):
        if shares_short < 100 or budget <= 0 or spot <= 0:
            return False

        contracts = self.find_contracts(underlying)
        if not contracts:
            return False

        target = spot * (1 + HEDGE_CALL_OTM_PCT)
        c = self.closest_strike(contracts, target)
        if not c:
            return False

        assert_safe_order(c["symbol"], AssetClass.US_OPTION)

        T = (c["expiration"] - now_et().date()).days / TRADING_DAYS
        sigma = max(volatility(underlying), 0.15)
        fair = bs_price(spot, c["strike"], T, RISK_FREE_ANNUAL, sigma, "call")

        use = c["ask"] if c["ask"] > 0 else fair
        if use <= 0:
            return False

        max_aff = int(budget // (use * 100))
        qty = min(int(shares_short // 100), max_aff)
        if qty <= 0:
            return False

        order = LimitOrderRequest(
            symbol=c["symbol"],
            qty=qty,
            side=OrderSide.BUY,
            type=OrderType.LIMIT,
            time_in_force=TimeInForce.DAY,
            limit_price=round(use, 2),
        )
        return submit_order_safe(self.trading, order, f"BUY protective calls {c['symbol']}")

# ------------------------------------------------------------
# Main trading engine
# ------------------------------------------------------------

class LeanTradingEngine:
    def __init__(self, trading_client):
        self.trading = trading_client
        self.options = LeanOptionsEngine(trading_client)
        self.risk = LeanRiskManager()

    def equity(self):
        try:
            return float(self.trading.get_account().equity)
        except Exception as e:
            log.error("Failed to fetch equity: %s", e)
            return 0.0

    def positions(self):
        try:
            return {p.symbol: p for p in self.trading.get_all_positions()}
        except Exception as e:
            log.warning("Failed to fetch positions: %s", e)
            return {}

    def _open_short(self, sym, price, target_value):
        qty = int(target_value / price)
        if qty <= 0:
            return False
        assert_safe_order(sym, AssetClass.US_EQUITY)
        order = MarketOrderRequest(
            symbol=sym,
            qty=qty,
            side=OrderSide.SELL,
            time_in_force=TimeInForce.DAY,
        )
        return submit_order_safe(self.trading, order, f"SHORT {sym}")

    def run_shorts(self, scores, regime, rank_offset=0, extra=0):
        eq = self.equity()
        if eq <= 0:
            return 0
        pos = self.positions()
        ranked = sorted(SHORT_UNIVERSE, key=lambda x: scores.get(x, 1e9))
        n = TOP_N_SHORTS + extra
        candidates = ranked[rank_offset:rank_offset + n]
        per = (SHORT_SLEEVE_WEIGHT / max(TOP_N_SHORTS, 1)) * (0.5 + regime)
        placed = 0
        for sym in candidates:
            if sym in pos and float(pos[sym].qty) < 0:
                continue
            if sym in pos and float(pos[sym].qty) > 0:
                continue
            price = latest_price(sym)
            if price <= 0:
                continue
            if self._open_short(sym, price, eq * per):
                placed += 1
        return placed

    def run_options(self):
        if not OPTIONS_ENABLED:
            return 0
        eq = self.equity()
        if eq <= 0:
            return 0
        pos = self.positions()
        budget_total = eq * OPTIONS_HEDGE_BUDGET_PCT
        shorts = [(s, p) for s, p in pos.items() if float(p.qty) < 0 and s in OPTIONABLE_SYMBOLS]
        if not shorts:
            return 0
        budget_each = budget_total / len(shorts)
        placed = 0
        for sym, p in shorts:
            shares_short = abs(float(p.qty))
            if shares_short < 100:
                continue
            price = latest_price(sym)
            if price <= 0:
                continue
            if self.options.buy_protective_calls(sym, shares_short, price, budget_each):
                placed += 1
        return placed

    def ensure_minimum_trades(self, momentum_scores, regime):
        global TRADES_TODAY
        offset = TOP_N_SHORTS
        attempts = 0
        max_attempts = len(SHORT_UNIVERSE)
        while TRADES_TODAY < MIN_TRADES_PER_DAY and attempts < max_attempts:
            if not can_trade_today():
                break
            placed = self.run_shorts(momentum_scores, regime, rank_offset=offset, extra=0)
            if placed == 0:
                offset += TOP_N_SHORTS
            attempts += TOP_N_SHORTS
            if offset >= len(SHORT_UNIVERSE):
                break
        if TRADES_TODAY < MIN_TRADES_PER_DAY:
            log.warning(
                "Could not reach MIN_TRADES_PER_DAY (%d/%d) — universe/no-trade window exhausted.",
                TRADES_TODAY, MIN_TRADES_PER_DAY,
            )

    def run_once(self):
        if not can_trade_today():
            return
        regime = self.risk.regime_risk_score()
        log.info("Regime risk score: %.2f", regime)

        momentum_scores = {t: momentum(t, 60) for t in SHORT_UNIVERSE}

        try:
            self.run_shorts(momentum_scores, regime)
        except Exception as e:
            log.exception("Sleeve 'shorts' failed: %s", e)

        try:
            self.run_options()
        except Exception as e:
            log.exception("Sleeve 'options' failed: %s", e)

        try:
            self.ensure_minimum_trades(momentum_scores, regime)
        except Exception as e:
            log.exception("ensure_minimum_trades failed: %s", e)

        del momentum_scores
        gc.collect()

# ------------------------------------------------------------
# Background loop
# ------------------------------------------------------------

def _env_bool(name, default=True):
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")

def trading_loop():
    ENGINE_STATE["thread_started"] = True
    ENGINE_STATE["thread_alive"] = True

    api_key = os.getenv("ALPACA_API_KEY", "")
    api_secret = os.getenv("ALPACA_API_SECRET", "")
    paper = _env_bool("ALPACA_PAPER", True)

    if not api_key or not api_secret:
        msg = "ALPACA_API_KEY / ALPACA_API_SECRET not set — trading loop will NOT start."
        log.error(msg)
        ENGINE_STATE["last_error"] = msg
        ENGINE_STATE["thread_alive"] = False
        return

    log.info("Env vars found. paper=%s. Connecting to Alpaca...", paper)

    try:
        trading_client = TradingClient(api_key, api_secret, paper=paper)
        account = trading_client.get_account()
        log.info(
            "Connected OK. account_status=%s equity=%s paper=%s",
            account.status, account.equity, paper,
        )
    except Exception as e:
        msg = f"Startup connection to Alpaca failed: {e}"
        log.exception(msg)
        ENGINE_STATE["last_error"] = msg
        ENGINE_STATE["thread_alive"] = False
        return

    engine = LeanTradingEngine(trading_client)

    while True:
        ENGINE_STATE["last_cycle_started"] = now_et().isoformat()
        try:
            engine.run_once()
            ENGINE_STATE["last_error"] = None
        except Exception as e:
            log.exception("Engine error: %s", e)
            ENGINE_STATE["last_error"] = str(e)
        ENGINE_STATE["last_cycle_finished"] = now_et().isoformat()
        gc.collect()
        log.info("Cycle complete. Sleeping %ss.", LOOP_SLEEP_SECONDS)
        time.sleep(LOOP_SLEEP_SECONDS)

def _thread_wrapper():
    try:
        trading_loop()
    except Exception as e:
        log.exception("Trading thread crashed: %s", e)
        ENGINE_STATE["last_error"] = f"Thread crashed: {e}"
        ENGINE_STATE["thread_alive"] = False

threading.Thread(target=_thread_wrapper, daemon=True).start()
log.info("Trading loop thread launched.")

if __name__ == "__main__":
    port = int(os.getenv("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
