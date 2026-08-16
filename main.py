import os
import sys
import gc
import time
import random
import logging
import threading
from math import sqrt, log as ln, exp, erf
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import requests
from flask import Flask

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce, OrderType, AssetClass


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    stream=sys.stdout,
)

log = logging.getLogger("lean_risk_engine")


# ============================================================
# FLASK / ENGINE STATE
# ============================================================

app = Flask(__name__)

ENGINE_STATE = {
    "thread_started": False,
    "thread_alive": False,
    "last_cycle_started": None,
    "last_cycle_finished": None,
    "last_error": None,
    "trades_today": 0,
    "hedges_today": 0,
    "market_open": None,
    "data_provider": "Alpaca",
    "data_failures": 0,
    "monte_carlo": None,
    "risk_scalar": 1.0,
}


@app.route("/")
def home():
    return "Lean Long/Short + Options Risk Engine Online - Alpaca Data", 200


@app.route("/health")
def health():
    return ENGINE_STATE, 200


# ============================================================
# CONFIG
# ============================================================

EASTERN_TZ = ZoneInfo("America/New_York")

RISK_FREE_ANNUAL = 0.04
TRADING_DAYS = 252


# ============================================================
# UNIVERSE
# ============================================================

SECTOR_ETFS = [
    "XLK", "XLF", "XLV", "XLE", "XLI",
    "XLB", "XLY", "XLP", "XLU", "XLRE",
]

SEMICONDUCTOR_TICKERS = [
    "NVDA", "AMD", "INTC", "TSM", "AVGO",
    "QCOM", "TXN", "MU", "LRCX", "AMAT",
    "ADI", "KLAC", "MRVL", "ON", "MCHP",
    "SWKS", "QRVO", "NXPI", "TER", "ENTG",
    "MPWR", "CRUS", "SLAB", "POWI", "DIOD",
    "RMBS", "ALGM", "WOLF", "ONTO", "COHU",
]

MANUFACTURING_TICKERS = [
    "CAT", "DE", "GE", "HON", "MMM",
    "ETN", "EMR", "ITW", "PH", "ROK",
    "DOV", "XYL", "IR", "AME", "ROP",
    "SNA", "SWK", "CMI", "PCAR", "FAST",
    "GWW", "PNR", "FLS", "AOS", "NDSN",
    "ALLE", "IEX", "HUBB", "GGG", "CR",
]

MINING_TICKERS = [
    "FCX", "NEM", "GOLD", "SCCO", "AEM",
    "TECK", "RIO", "BHP", "VALE", "MOS",
    "AA", "CLF", "X", "NUE", "STLD",
    "MP", "CDE", "HL", "PAAS", "AG",
    "SSRM", "EGO", "KGC", "AU", "WPM",
    "FNV", "RGLD", "ALB", "LAC", "SQM",
]

# Mid/large-cap E&P, refining, midstream, and oilfield-services names.
# Excludes XOM/CVX (mega-cap) and any ticker that has been delisted
# via merger (e.g. HES -> Chevron, MRO/PXD -> ConocoPhillips/Exxon).
ENERGY_TICKERS = [
    "COP", "EOG", "OXY", "MPC", "PSX",
    "VLO", "WMB", "OKE", "KMI", "TRGP",
    "DVN", "FANG", "HAL", "SLB", "BKR",
    "APA", "CTRA", "EQT", "PBF", "DINO",
    "CVI", "PARR", "DK", "CHRD", "MTDR",
    "AR", "RRC", "SM", "NOG", "TALO",
]

# Transportation/logistics, aerospace parts, waste/environmental
# services, and building-products names -- distinct from the
# heavy-machinery names already covered in MANUFACTURING_TICKERS.
INDUSTRIALS_TICKERS = [
    "WAB", "JBHT", "ODFL", "XPO", "CHRW",
    "LSTR", "EXPD", "WERN", "SAIA", "RXO",
    "TXT", "HEI", "TDG", "CW", "WWD",
    "AXON", "LII", "WSO", "JCI", "CSL",
    "MAS", "VMC", "MLM", "WM", "RSG",
    "CTAS", "ROL", "PWR", "FIX", "EME",
]


def _dedupe(seq):
    seen = set()
    out = []
    for symbol in seq:
        symbol = symbol.upper().strip()
        if symbol and symbol not in seen:
            seen.add(symbol)
            out.append(symbol)
    return out


SHORT_UNIVERSE = _dedupe(
    SECTOR_ETFS
    + SEMICONDUCTOR_TICKERS
    + MANUFACTURING_TICKERS
    + MINING_TICKERS
    + ENERGY_TICKERS
    + INDUSTRIALS_TICKERS
)

# Longs trade the same universe as shorts (symmetric momentum book).
LONG_UNIVERSE = SHORT_UNIVERSE

OPTIONABLE_SYMBOLS = set(SHORT_UNIVERSE)

# ------------------------------------------------------------
# SECTOR MAP -- used purely for diversification caps so that no
# single sector (e.g. mining, during a lithium-miner drawdown)
# can dominate the short book, the long book, or the option-hedge
# budget. This directly fixes the "keeps buying LAC and nothing
# else" behavior: previously PRIORITY_MIN_SLOTS reserved 12/20
# short slots for semis+manufacturing+mining, so a persistently
# weak name like LAC would win the "weakest momentum" ranking on
# nearly every cycle, get shorted, and then get hedged on repeat.
# ------------------------------------------------------------

SECTOR_MAP = {}

for _t in SECTOR_ETFS:
    SECTOR_MAP[_t] = "ETF"

for _t in SEMICONDUCTOR_TICKERS:
    SECTOR_MAP[_t] = "Semiconductors"

for _t in MANUFACTURING_TICKERS:
    SECTOR_MAP[_t] = "Manufacturing"

for _t in MINING_TICKERS:
    SECTOR_MAP[_t] = "Mining"

for _t in ENERGY_TICKERS:
    SECTOR_MAP[_t] = "Energy"

for _t in INDUSTRIALS_TICKERS:
    SECTOR_MAP[_t] = "Industrials"


def pick_diversified(ranked_symbols, top_n, max_per_sector, exclude=None):
    """
    Walk a ranked (best-candidate-first) symbol list and greedily fill
    up to `top_n` slots while capping how many symbols can come from
    any single sector. If the cap makes it impossible to fill every
    slot (not enough diverse candidates), a second pass fills the
    remainder from the leftover pool ignoring the cap, so the sleeve
    still reaches its target size on thin days.
    """

    exclude = exclude or set()

    chosen = []
    sector_counts = {}
    leftover = []

    for symbol in ranked_symbols:

        if symbol in exclude:
            continue

        if len(chosen) >= top_n:
            break

        sector = SECTOR_MAP.get(symbol, "Other")

        if sector_counts.get(sector, 0) < max_per_sector:
            chosen.append(symbol)
            sector_counts[sector] = sector_counts.get(sector, 0) + 1
        else:
            leftover.append(symbol)

    if len(chosen) < top_n:
        chosen_set = set(chosen)
        for symbol in leftover:
            if len(chosen) >= top_n:
                break
            if symbol not in chosen_set:
                chosen.append(symbol)
                chosen_set.add(symbol)

    return chosen


# ============================================================
# STRATEGY CONFIG
# ============================================================

SHORT_SLEEVE_WEIGHT = 0.30
TOP_N_SHORTS = 20
MAX_SHORTS_PER_SECTOR = 4

LONG_SLEEVE_WEIGHT = 0.20
TOP_N_LONGS = 15
MAX_LONGS_PER_SECTOR = 3

OPTIONS_ENABLED = True

HEDGE_CALL_OTM_PCT = 0.05

OPTIONS_MIN_DTE = 25
OPTIONS_MAX_DTE = 45

# Trimmed down: options are a hedge overlay, not the main book.
OPTIONS_HEDGE_BUDGET_PCT = 0.03

# Hard cap on how many option contracts can be *held* (not bought
# per pass) for any single underlying at once.
MAX_OPTION_CONTRACTS_PER_TICKER = 3

# Hard cap on how many *new* underlyings can get their first hedge
# opened in a given day. Topping up an already-partially-hedged
# name up to MAX_OPTION_CONTRACTS_PER_TICKER doesn't count against
# this -- it only throttles how many distinct names get touched.
OPTIONS_MAX_NEW_HEDGES_PER_DAY = 5

# No ceiling on trades per day -- set to None to disable the cap
# entirely (submit_order_safe / can_trade_today treat None as
# "unlimited").
MAX_TRADES_PER_DAY = None

# The engine will keep opening additional longs/shorts (from the
# next-best ranked momentum candidates, never arbitrary picks) after
# the normal sleeve + options passes if the day's trade count hasn't
# reached this floor yet. It stops early if it simply runs out of
# valid, not-already-held candidates -- it will not fabricate trades
# against missing/bad data.
MIN_TRADES_PER_DAY = 20

TRADES_TODAY = 0
LAST_TRADE_DAY = None

HEDGES_TODAY = 0
LAST_HEDGE_DAY = None


# ============================================================
# MARKET WINDOW
# ============================================================

NO_TRADE_BEFORE = (4, 0)
NO_TRADE_AFTER = (20, 0)


# ============================================================
# OPTIONS
# ============================================================

MIN_OPTION_VOLUME = 50
MAX_OPTION_SPREAD_PCT = 0.15

REGIME_BENCHMARK = "VOO"

MAX_BARS_DAYS = 220

LOOP_SLEEP_SECONDS = 60


# ============================================================
# MONTE CARLO RISK CONFIG
# ============================================================

MC_ENABLED = True
MC_SIMULATIONS = 2000
MC_HORIZON_DAYS = 5
MC_BLOCK_SIZE = 5          # resample contiguous blocks to keep cross-asset correlation
MC_VAR_CONFIDENCE = 0.95

# If the simulated 5-day 95% VaR on the *intended* book exceeds this
# fraction of equity, position sizing for that cycle is scaled down
# proportionally (floor at MC_MIN_RISK_SCALAR).
MC_MAX_PORTFOLIO_VAR_PCT = 0.08
MC_MIN_RISK_SCALAR = 0.25


# ============================================================
# ALPACA DATA CONFIG
# ============================================================

ALPACA_DATA_URL = "https://data.alpaca.markets"

# IEX is generally the safest choice for accounts without
# the SIP subscription.
ALPACA_STOCK_FEED = os.getenv("ALPACA_STOCK_FEED", "iex")

# Options:
# "indicative" is usable without OPRA subscription.
# "opra" requires the appropriate market-data subscription.
ALPACA_OPTION_FEED = os.getenv("ALPACA_OPTION_FEED", "indicative")

DATA_BATCH_SIZE = 50
DATA_MAX_RETRIES = 5
DATA_RETRY_BASE_DELAY = 1.5
DATA_CACHE_TTL_SECONDS = 60
HTTP_TIMEOUT = 15


# ============================================================
# SHARED HTTP SESSION
# ============================================================

_HTTP = requests.Session()

_HTTP.headers.update({
    "User-Agent": "LeanRiskEngine/2.1",
    "Accept": "application/json",
    "Connection": "keep-alive",
})


# ============================================================
# ALPACA CREDENTIALS
# ============================================================

ALPACA_API_KEY = ""
ALPACA_API_SECRET = ""


def initialize_credentials():
    global ALPACA_API_KEY
    global ALPACA_API_SECRET

    ALPACA_API_KEY = os.getenv("ALPACA_API_KEY", "").strip()
    ALPACA_API_SECRET = os.getenv("ALPACA_API_SECRET", "").strip()

    if not ALPACA_API_KEY or not ALPACA_API_SECRET:
        raise RuntimeError("ALPACA_API_KEY / ALPACA_API_SECRET are not configured.")


def alpaca_headers():
    return {
        "APCA-API-KEY-ID": ALPACA_API_KEY,
        "APCA-API-SECRET-KEY": ALPACA_API_SECRET,
        "Accept": "application/json",
    }


# ============================================================
# GENERIC ALPACA DATA REQUEST
# ============================================================

def alpaca_data_get(path, params=None, label=""):
    """
    Resilient GET wrapper for Alpaca market data.
    Handles 429 rate limits, 500-series errors, connection failures,
    timeouts, and malformed JSON.
    """

    url = f"{ALPACA_DATA_URL}{path}"
    last_error = None

    for attempt in range(1, DATA_MAX_RETRIES + 1):
        try:
            response = _HTTP.get(
                url,
                params=params or {},
                headers=alpaca_headers(),
                timeout=HTTP_TIMEOUT,
            )

            status = response.status_code

            if 200 <= status < 300:
                try:
                    return response.json()
                except ValueError as exc:
                    raise RuntimeError(f"Invalid JSON returned by Alpaca: {exc}")

            if status == 429:
                retry_after = response.headers.get("Retry-After")
                if retry_after:
                    try:
                        delay = float(retry_after)
                    except Exception:
                        delay = DATA_RETRY_BASE_DELAY * (2 ** (attempt - 1))
                else:
                    delay = DATA_RETRY_BASE_DELAY * (2 ** (attempt - 1))

                delay += random.uniform(0.2, 0.8)
                log.warning(
                    "Alpaca rate limit (%s) %s attempt %d/%d. Sleeping %.1fs.",
                    label, path, attempt, DATA_MAX_RETRIES, delay,
                )
                time.sleep(delay)
                continue

            if status >= 500:
                delay = DATA_RETRY_BASE_DELAY * (2 ** (attempt - 1))
                delay += random.uniform(0.2, 0.8)
                log.warning(
                    "Alpaca server error %s (%s) attempt %d/%d. Retrying in %.1fs.",
                    status, label, attempt, DATA_MAX_RETRIES, delay,
                )
                time.sleep(delay)
                continue

            try:
                body = response.json()
            except Exception:
                body = response.text[:500]

            raise RuntimeError(f"Alpaca HTTP {status}: {body}")

        except Exception as exc:
            last_error = exc

            if attempt < DATA_MAX_RETRIES:
                delay = DATA_RETRY_BASE_DELAY * (2 ** (attempt - 1))
                delay += random.uniform(0.2, 0.8)
                log.warning(
                    "Alpaca request failed (%s) attempt %d/%d: %s. Retrying in %.1fs.",
                    label, attempt, DATA_MAX_RETRIES, exc, delay,
                )
                time.sleep(delay)
            else:
                log.error(
                    "Alpaca request failed (%s) after %d attempts: %s",
                    label, DATA_MAX_RETRIES, exc,
                )

    raise last_error or RuntimeError(f"Unknown Alpaca data failure: {label}")


# ============================================================
# DATA CACHE
# ============================================================

_BARS_CACHE = {}
_PRICE_CACHE = {}
_OPTIONS_CACHE = {}


def cache_get(cache, key):
    item = cache.get(key)
    if not item:
        return None
    timestamp, value = item
    if time.time() - timestamp > DATA_CACHE_TTL_SECONDS:
        return None
    return value


def cache_put(cache, key, value):
    cache[key] = (time.time(), value)


# ============================================================
# DATE HELPERS
# ============================================================

def now_et():
    return datetime.now(EASTERN_TZ)


def utc_now():
    return datetime.now(timezone.utc)


def iso_utc(dt):
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


# ============================================================
# ALPACA HISTORICAL STOCK DATA
# ============================================================

def alpaca_bars_many(symbols, days=60):
    """
    Download historical daily bars for many symbols in batches.
    Returns: {"AAPL": [price, price, ...], "MSFT": [...], ...}
    """

    symbols = _dedupe(symbols)
    if not symbols:
        return {}

    days = min(int(days), MAX_BARS_DAYS)

    cache_key = (tuple(sorted(symbols)), days, ALPACA_STOCK_FEED)
    cached = cache_get(_BARS_CACHE, cache_key)
    if cached is not None:
        return cached

    calendar_days = int(days * 1.55) + 15
    end_dt = utc_now()
    start_dt = end_dt - timedelta(days=calendar_days)

    result = {symbol: [] for symbol in symbols}

    for start_index in range(0, len(symbols), DATA_BATCH_SIZE):
        batch = symbols[start_index:start_index + DATA_BATCH_SIZE]

        params = {
            "symbols": ",".join(batch),
            "timeframe": "1Day",
            "start": iso_utc(start_dt),
            "end": iso_utc(end_dt),
            "feed": ALPACA_STOCK_FEED,
            "adjustment": "split",
            "sort": "asc",
            "limit": 10000,
        }

        try:
            data = alpaca_data_get(
                "/v2/stocks/bars",
                params=params,
                label=f"bars:{','.join(batch[:3])}",
            )

            bars = data.get("bars", {})
            if not isinstance(bars, dict):
                log.warning("Malformed Alpaca bars response for batch.")
                continue

            for symbol in batch:
                symbol_bars = bars.get(symbol, [])
                closes = []
                for bar in symbol_bars:
                    try:
                        close = float(bar["c"])
                        if close > 0:
                            closes.append(close)
                    except Exception:
                        continue
                result[symbol] = closes[-days:]

        except Exception as exc:
            ENGINE_STATE["data_failures"] += 1
            log.error("Historical data batch failed: %s", exc)
            # Do NOT destroy the entire engine because one batch failed.
            continue

    cache_put(_BARS_CACHE, cache_key, result)
    return result


def yf_bars(symbol, days=60):
    """Compatibility wrapper -- no longer uses Yahoo Finance."""
    symbol = symbol.upper()
    data = alpaca_bars_many([symbol], days=days)
    return data.get(symbol, [])


# ============================================================
# BULK DATA PRELOAD
# ============================================================

def preload_market_data():
    log.info("Loading Alpaca market data for %d symbols...", len(SHORT_UNIVERSE))

    data = alpaca_bars_many(SHORT_UNIVERSE, days=MAX_BARS_DAYS)

    successful = sum(1 for symbol in SHORT_UNIVERSE if len(data.get(symbol, [])) >= 30)
    failed = len(SHORT_UNIVERSE) - successful

    log.info(
        "Alpaca market data loaded: %d/%d symbols usable, %d failed.",
        successful, len(SHORT_UNIVERSE), failed,
    )

    return data


# ============================================================
# LATEST PRICES
# ============================================================

def latest_prices_many(symbols):
    symbols = _dedupe(symbols)
    if not symbols:
        return {}

    cache_key = (tuple(sorted(symbols)), ALPACA_STOCK_FEED)
    cached = cache_get(_PRICE_CACHE, cache_key)
    if cached is not None:
        return cached

    result = {}

    for start_index in range(0, len(symbols), DATA_BATCH_SIZE):
        batch = symbols[start_index:start_index + DATA_BATCH_SIZE]

        params = {"symbols": ",".join(batch), "feed": ALPACA_STOCK_FEED}

        try:
            data = alpaca_data_get(
                "/v2/stocks/trades/latest",
                params=params,
                label=f"latest:{','.join(batch[:3])}",
            )

            trades = data.get("trades", {})
            if not isinstance(trades, dict):
                continue

            for symbol in batch:
                trade = trades.get(symbol)
                if not trade:
                    continue
                try:
                    price = float(trade["p"])
                    if price > 0:
                        result[symbol] = price
                except Exception:
                    continue

        except Exception as exc:
            ENGINE_STATE["data_failures"] += 1
            log.warning("Latest-price batch failed: %s", exc)

    cache_put(_PRICE_CACHE, cache_key, result)
    return result


def latest_price(symbol):
    prices = latest_prices_many([symbol])
    price = prices.get(symbol.upper(), 0.0)
    if price > 0:
        return price
    bars = yf_bars(symbol, days=5)
    return bars[-1] if bars else 0.0


# ============================================================
# MOMENTUM
# ============================================================

def momentum_from_bars(bars, lookback=20):
    if not bars:
        return None
    if len(bars) < lookback + 1:
        return None
    try:
        return (bars[-1] / bars[-lookback]) - 1.0
    except Exception:
        return None


def momentum(symbol, lookback=20):
    bars = yf_bars(symbol, days=lookback + 5)
    return momentum_from_bars(bars, lookback)


# ============================================================
# VOLATILITY
# ============================================================

def volatility_from_bars(bars, window=14):
    if not bars:
        return None
    if len(bars) < window + 1:
        return None

    returns = []
    for i in range(1, len(bars)):
        previous = bars[i - 1]
        current = bars[i]
        if previous <= 0:
            continue
        try:
            returns.append((current / previous) - 1.0)
        except Exception:
            continue

    if len(returns) < window:
        return None

    sample = returns[-window:]
    mean = sum(sample) / len(sample)
    variance = sum((r - mean) ** 2 for r in sample) / len(sample)

    return sqrt(variance) * sqrt(TRADING_DAYS)


def volatility(symbol, window=14):
    bars = yf_bars(symbol, days=window + 10)
    value = volatility_from_bars(bars, window)
    return value if value is not None else 0.0


def daily_returns(bars):
    returns = []
    for i in range(1, len(bars)):
        previous = bars[i - 1]
        if previous <= 0:
            continue
        try:
            returns.append((bars[i] / previous) - 1.0)
        except Exception:
            continue
    return returns


# ============================================================
# MONTE CARLO PORTFOLIO RISK
# ============================================================

def monte_carlo_var(
    book,
    market_data,
    n_sims=MC_SIMULATIONS,
    horizon=MC_HORIZON_DAYS,
    block=MC_BLOCK_SIZE,
    confidence=MC_VAR_CONFIDENCE,
):
    """
    Block-bootstrap Monte Carlo simulation of `horizon`-day forward
    portfolio return, given a proposed book.

    book: list of (symbol, side) tuples, side = +1.0 for long,
          -1.0 for short. Equal-weighted across the book.

    Each simulation repeatedly samples a random contiguous block of
    historical daily-return rows (the SAME calendar block for every
    symbol in that step), which preserves realistic cross-asset
    correlation instead of treating each name as independent.

    Returns a dict with n_sims, symbols_used, mean_return_pct,
    var_pct (loss at the given confidence, negative = a loss), and
    cvar_pct (average loss beyond the VaR cutoff). Returns None if
    there isn't enough history to simulate anything meaningful.
    """

    if not book:
        return None

    returns_by_symbol = {}

    for symbol, _side in book:
        bars = market_data.get(symbol, [])
        rets = daily_returns(bars)
        if len(rets) >= block * 2:
            returns_by_symbol[symbol] = rets

    usable_book = [(s, side) for s, side in book if s in returns_by_symbol]

    if not usable_book:
        return None

    weight = 1.0 / len(usable_book)
    min_len = min(len(returns_by_symbol[s]) for s, _ in usable_book)

    if min_len < block:
        return None

    sim_results = []

    for _ in range(n_sims):
        cumulative = {s: 1.0 for s, _ in usable_book}
        days_simulated = 0

        while days_simulated < horizon:
            start = random.randint(0, min_len - block)
            steps = min(block, horizon - days_simulated)

            for step in range(steps):
                idx = start + step
                for symbol, _side in usable_book:
                    r = returns_by_symbol[symbol][idx]
                    cumulative[symbol] *= (1.0 + r)

            days_simulated += steps

        portfolio_return = 0.0
        for symbol, side in usable_book:
            asset_return = cumulative[symbol] - 1.0
            portfolio_return += weight * side * asset_return

        sim_results.append(portfolio_return)

    sim_results.sort()

    tail_idx = int((1.0 - confidence) * len(sim_results))
    tail_idx = max(0, min(tail_idx, len(sim_results) - 1))

    var_return = sim_results[tail_idx]
    tail = sim_results[: tail_idx + 1]
    cvar_return = sum(tail) / len(tail) if tail else var_return
    mean_return = sum(sim_results) / len(sim_results)

    return {
        "n_sims": len(sim_results),
        "symbols_used": len(usable_book),
        "horizon_days": horizon,
        "confidence": confidence,
        "mean_return_pct": mean_return * 100.0,
        "var_pct": var_return * 100.0,
        "cvar_pct": cvar_return * 100.0,
    }


def risk_scalar_from_mc(mc_result):
    """
    Turn a monte_carlo_var() result into a position-size multiplier.
    If simulated tail loss exceeds the configured risk budget, sizing
    is scaled down proportionally (with a floor so the book never
    goes fully to zero from this check alone).
    """

    if not mc_result:
        return 1.0

    var_loss_pct = abs(min(mc_result["var_pct"], 0.0))
    budget_pct = MC_MAX_PORTFOLIO_VAR_PCT * 100.0

    if var_loss_pct <= budget_pct or var_loss_pct <= 0:
        return 1.0

    scalar = budget_pct / var_loss_pct
    return max(MC_MIN_RISK_SCALAR, min(scalar, 1.0))


# ============================================================
# REGIME / RISK MANAGER
# ============================================================

class LeanRiskManager:

    def __init__(self, market_data=None):
        self.market_data = market_data or {}

    def regime_risk_score(self):
        bars = self.market_data.get(REGIME_BENCHMARK, [])

        if len(bars) < 200:
            log.warning(
                "Not enough %s history for regime calculation. Using neutral regime.",
                REGIME_BENCHMARK,
            )
            return 0.3

        sma200 = sum(bars[-200:]) / 200
        last = bars[-1]

        below_sma = 0.0 if last > sma200 else 1.0

        last_year = bars[-252:] if len(bars) >= 252 else bars
        highest = max(last_year)

        dd = 0.0 if highest <= 0 else 1.0 - (last / highest)
        dd_score = max(0.0, min(dd / 0.20, 1.0))

        return (below_sma + dd_score) / 2.0


# ============================================================
# BLACK-SCHOLES
# ============================================================

def _cdf(x):
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def bs_price(S, K, T, r, sigma, opt_type):
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return 0.0

    d1 = (ln(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)

    if opt_type == "call":
        return S * _cdf(d1) - K * exp(-r * T) * _cdf(d2)

    return K * exp(-r * T) * _cdf(-d2) - S * _cdf(-d1)


def bs_delta(S, K, T, r, sigma, opt_type):
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return 0.0

    d1 = (ln(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrt(T))

    if opt_type == "call":
        return _cdf(d1)

    return _cdf(d1) - 1.0


# ============================================================
# OCC OPTION SYMBOL PARSER
# ============================================================

def parse_occ_option_symbol(symbol):
    """
    OCC option format: AAPL250117C00200000
    root = AAPL, expiration = 2025-01-17, type = C, strike = 200.00
    """

    try:
        if len(symbol) < 15:
            return None

        tail = symbol[-15:]
        option_type = tail[6]

        if option_type not in ("C", "P"):
            return None

        date_str = tail[:6]
        strike_str = tail[7:]

        year = int(date_str[0:2])
        month = int(date_str[2:4])
        day = int(date_str[4:6])

        expiration = datetime(2000 + year, month, day).date()
        strike = int(strike_str) / 1000.0
        root = symbol[:-15]

        return (root, expiration, option_type, strike)

    except Exception:
        return None


# ============================================================
# ALPACA OPTIONS DATA
# ============================================================

class LeanOptionsEngine:

    def __init__(self, trading_client):
        self.trading = trading_client

    # --------------------------------------------------------
    # OPTION CHAIN
    # --------------------------------------------------------

    def option_chain(self, underlying):
        underlying = underlying.upper()

        cache_key = (underlying, OPTIONS_MIN_DTE, OPTIONS_MAX_DTE, ALPACA_OPTION_FEED)
        cached = cache_get(_OPTIONS_CACHE, cache_key)
        if cached is not None:
            return cached

        today = now_et().date()
        min_expiration = today + timedelta(days=OPTIONS_MIN_DTE)
        max_expiration = today + timedelta(days=OPTIONS_MAX_DTE)

        params = {
            "feed": ALPACA_OPTION_FEED,
            "type": "call",
            "expiration_date_gte": min_expiration.isoformat(),
            "expiration_date_lte": max_expiration.isoformat(),
            "limit": 1000,
        }

        try:
            data = alpaca_data_get(
                f"/v1beta1/options/snapshots/{underlying}",
                params=params,
                label=f"option-chain:{underlying}",
            )

            snapshots = data.get("snapshots", {})
            contracts = []

            if isinstance(snapshots, dict):
                for contract_symbol, snapshot in snapshots.items():
                    try:
                        parts = parse_occ_option_symbol(contract_symbol)
                        if not parts:
                            continue

                        root, expiration, option_type, strike = parts

                        if option_type != "C":
                            continue

                        if expiration < min_expiration or expiration > max_expiration:
                            continue

                        latest_quote = snapshot.get("latestQuote", {}) or {}
                        latest_trade = snapshot.get("latestTrade", {}) or {}

                        bid = float(latest_quote.get("bp", 0) or 0)
                        ask = float(latest_quote.get("ap", 0) or 0)
                        volume = int(latest_trade.get("s", 0) or 0)

                        greeks = snapshot.get("greeks", {}) or {}

                        contracts.append({
                            "symbol": contract_symbol,
                            "strike": strike,
                            "expiration": expiration,
                            "bid": bid,
                            "ask": ask,
                            "volume": volume,
                            "delta": float(greeks.get("delta", 0) or 0),
                            "iv": float(snapshot.get("impliedVolatility", 0) or 0),
                        })

                    except Exception:
                        continue

            cache_put(_OPTIONS_CACHE, cache_key, contracts)
            return contracts

        except Exception as exc:
            ENGINE_STATE["data_failures"] += 1
            log.warning("Alpaca option chain failed for %s: %s", underlying, exc)
            return []

    # --------------------------------------------------------
    # FIND CONTRACT
    # --------------------------------------------------------

    def find_contract(self, underlying, spot):
        contracts = self.option_chain(underlying)
        if not contracts:
            return None

        target_strike = spot * (1.0 + HEDGE_CALL_OTM_PCT)

        valid = [c for c in contracts if self.liquidity_ok(c)]
        if not valid:
            return None

        valid.sort(key=lambda c: (abs(c["strike"] - target_strike), c["expiration"]))
        return valid[0]

    # --------------------------------------------------------
    # LIQUIDITY
    # --------------------------------------------------------

    def liquidity_ok(self, contract):
        bid = contract["bid"]
        ask = contract["ask"]

        if bid <= 0 or ask <= 0:
            return False
        if ask < bid:
            return False

        mid = (bid + ask) / 2.0
        if mid <= 0:
            return False

        spread = (ask - bid) / mid
        return spread <= MAX_OPTION_SPREAD_PCT

    # --------------------------------------------------------
    # PROTECTIVE CALL
    # --------------------------------------------------------

    def buy_protective_calls(self, underlying, shares_short, spot, budget, already_held=0):
        """
        already_held: number of contracts already open on this
        underlying (from a prior cycle). Sizing is capped by the
        *remaining* room up to MAX_OPTION_CONTRACTS_PER_TICKER, so
        the same underlying can no longer be bought without limit
        every 60-second cycle.
        """

        if shares_short < 100 or budget <= 0 or spot <= 0:
            return False

        remaining_capacity = MAX_OPTION_CONTRACTS_PER_TICKER - already_held

        if remaining_capacity <= 0:
            log.info(
                "%s already at max hedge contracts (%d/%d). Skipping.",
                underlying, already_held, MAX_OPTION_CONTRACTS_PER_TICKER,
            )
            return False

        contract = self.find_contract(underlying, spot)
        if not contract:
            log.info("No liquid protective call found for %s.", underlying)
            return False

        symbol = contract["symbol"]
        assert_safe_order(symbol, AssetClass.US_OPTION)

        expiration = contract["expiration"]
        dte = (expiration - now_et().date()).days
        if dte <= 0:
            return False

        T = dte / TRADING_DAYS

        sigma = volatility(underlying, window=20)
        if sigma <= 0:
            sigma = max(contract.get("iv", 0), 0.20)

        fair_value = bs_price(spot, contract["strike"], T, RISK_FREE_ANNUAL, sigma, "call")
        delta = bs_delta(spot, contract["strike"], T, RISK_FREE_ANNUAL, sigma, "call")

        bid = contract["bid"]
        ask = contract["ask"]
        mid = (bid + ask) / 2.0

        if mid <= 0:
            return False

        if fair_value > 0:
            max_reasonable_price = fair_value * 1.20
            limit_price = min(ask, max_reasonable_price)
        else:
            limit_price = mid

        if limit_price <= 0:
            return False

        max_affordable = int(budget // (limit_price * 100))
        contracts_by_shares = int(shares_short // 100)

        qty = min(contracts_by_shares, max_affordable, remaining_capacity)

        if qty <= 0:
            return False

        order = LimitOrderRequest(
            symbol=symbol,
            qty=qty,
            side=OrderSide.BUY,
            type=OrderType.LIMIT,
            time_in_force=TimeInForce.DAY,
            limit_price=round(limit_price, 2),
        )

        log.info(
            "Protective call candidate %s | spot=%.2f strike=%.2f dte=%d "
            "bid=%.2f ask=%.2f fair=%.2f delta=%.3f already_held=%d qty=%d",
            symbol, spot, contract["strike"], dte, bid, ask, fair_value, delta,
            already_held, qty,
        )

        return submit_order_safe(self.trading, order, f"BUY protective call {symbol}")


# ============================================================
# MARKET / TRADING WINDOW
# ============================================================

_CLOCK_CACHE = {"ts": 0.0, "is_open": None}
_CLOCK_CACHE_TTL_SECONDS = 30


def _fallback_in_no_trade_window():
    t = now_et()
    h = t.hour
    m = t.minute

    if h < NO_TRADE_BEFORE[0] or (h == NO_TRADE_BEFORE[0] and m < NO_TRADE_BEFORE[1]):
        return True

    if h > NO_TRADE_AFTER[0] or (h == NO_TRADE_AFTER[0] and m >= NO_TRADE_AFTER[1]):
        return True

    return False


def market_is_open(trading_client):
    now = time.time()

    if (_CLOCK_CACHE["is_open"] is not None
            and (now - _CLOCK_CACHE["ts"]) < _CLOCK_CACHE_TTL_SECONDS):
        return _CLOCK_CACHE["is_open"]

    try:
        clock = trading_client.get_clock()
        is_open = bool(clock.is_open)

        _CLOCK_CACHE["ts"] = now
        _CLOCK_CACHE["is_open"] = is_open
        ENGINE_STATE["market_open"] = is_open

        return is_open

    except Exception as exc:
        log.warning("Alpaca clock failed: %s. Using ET fallback.", exc)

        fallback = not _fallback_in_no_trade_window()
        ENGINE_STATE["market_open"] = fallback

        return fallback


# ============================================================
# TRADE / HEDGE COUNTERS
# ============================================================

def reset_trade_counter_if_new_day():
    global TRADES_TODAY, LAST_TRADE_DAY

    today = now_et().date()

    if LAST_TRADE_DAY != today:
        LAST_TRADE_DAY = today
        TRADES_TODAY = 0
        ENGINE_STATE["trades_today"] = 0
        log.info("New trading day: trade counter reset.")


def reset_hedge_counter_if_new_day():
    global HEDGES_TODAY, LAST_HEDGE_DAY

    today = now_et().date()

    if LAST_HEDGE_DAY != today:
        LAST_HEDGE_DAY = today
        HEDGES_TODAY = 0
        ENGINE_STATE["hedges_today"] = 0
        log.info("New trading day: new-hedge counter reset.")


def can_trade_today(trading_client):
    reset_trade_counter_if_new_day()

    if not market_is_open(trading_client):
        log.info("Market closed. Skipping trades.")
        return False

    return MAX_TRADES_PER_DAY is None or TRADES_TODAY < MAX_TRADES_PER_DAY


# ============================================================
# SAFE ORDER SUBMISSION
# ============================================================

def submit_order_safe(trading_client, order_data, label=""):
    global TRADES_TODAY

    reset_trade_counter_if_new_day()

    if MAX_TRADES_PER_DAY is not None and TRADES_TODAY >= MAX_TRADES_PER_DAY:
        log.warning("Trade limit reached. Skipping order: %s", label)
        return False

    if not market_is_open(trading_client):
        log.info("Market closed. Skipping order: %s", label)
        return False

    try:
        result = trading_client.submit_order(order_data=order_data)

        TRADES_TODAY += 1
        ENGINE_STATE["trades_today"] = TRADES_TODAY

        log.info(
            "ORDER SUBMITTED | trade #%d | %s | id=%s",
            TRADES_TODAY, label, getattr(result, "id", "unknown"),
        )

        return True

    except Exception as exc:
        log.exception("Order failed (%s): %s", label, exc)
        return False


def assert_safe_order(symbol, asset_class):
    if asset_class not in {AssetClass.US_EQUITY, AssetClass.US_OPTION}:
        raise ValueError(f"Blocked order for {symbol}: asset class {asset_class}")


def _position_qty(position):
    try:
        return float(position.qty)
    except Exception:
        return 0.0


# ============================================================
# MAIN TRADING ENGINE
# ============================================================

class LeanTradingEngine:

    def __init__(self, trading_client, market_data):
        self.trading = trading_client
        self.market_data = market_data
        self.options = LeanOptionsEngine(trading_client)
        self.risk = LeanRiskManager(market_data)

    # --------------------------------------------------------
    # ACCOUNT
    # --------------------------------------------------------

    def equity(self):
        try:
            account = self.trading.get_account()
            return float(account.equity)
        except Exception as exc:
            log.error("Failed to fetch equity: %s", exc)
            return 0.0

    # --------------------------------------------------------
    # POSITIONS
    # --------------------------------------------------------

    def positions(self):
        try:
            positions = self.trading.get_all_positions()
            return {p.symbol: p for p in positions}
        except Exception as exc:
            log.warning("Failed to fetch positions: %s", exc)
            return {}

    def existing_option_contracts(self, positions, underlying):
        """Sum contracts currently held on `underlying` across all its
        open option positions (any strike/expiration)."""

        total = 0.0

        for symbol, position in positions.items():
            if symbol in SHORT_UNIVERSE:
                continue  # equity position, not an option contract

            parsed = parse_occ_option_symbol(symbol)
            if not parsed:
                continue

            if parsed[0] != underlying:
                continue

            total += abs(_position_qty(position))

        return int(total)

    # --------------------------------------------------------
    # SIZING
    # --------------------------------------------------------

    def _target_short_size(self, eq, regime):
        per_position = SHORT_SLEEVE_WEIGHT / max(TOP_N_SHORTS, 1)
        # Slightly increase size in higher-risk regimes.
        regime_multiplier = 0.5 + regime
        return eq * per_position * regime_multiplier

    def _target_long_size(self, eq, regime):
        per_position = LONG_SLEEVE_WEIGHT / max(TOP_N_LONGS, 1)
        # Longs get trimmed, not boosted, in higher-risk regimes.
        regime_multiplier = 1.5 - regime
        return eq * per_position * regime_multiplier

    # --------------------------------------------------------
    # ORDER HELPERS
    # --------------------------------------------------------

    def _open_short(self, symbol, price, target_value):
        if price <= 0:
            return False

        qty = int(target_value / price)
        if qty <= 0:
            return False

        assert_safe_order(symbol, AssetClass.US_EQUITY)

        order = MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=OrderSide.SELL,
            time_in_force=TimeInForce.DAY,
        )

        return submit_order_safe(self.trading, order, f"SHORT {symbol}")

    def _open_long(self, symbol, price, target_value):
        if price <= 0:
            return False

        qty = int(target_value / price)
        if qty <= 0:
            return False

        assert_safe_order(symbol, AssetClass.US_EQUITY)

        order = MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=OrderSide.BUY,
            time_in_force=TimeInForce.DAY,
        )

        return submit_order_safe(self.trading, order, f"LONG {symbol}")

    # --------------------------------------------------------
    # SHORT SLEEVE
    # --------------------------------------------------------

    def run_shorts(self, scores, regime, prices, risk_scalar=1.0):
        eq = self.equity()
        if eq <= 0:
            return 0

        positions = self.positions()

        valid_symbols = [
            s for s in SHORT_UNIVERSE
            if scores.get(s) is not None and prices.get(s, 0) > 0
        ]

        if not valid_symbols:
            log.warning("No symbols have valid data. No shorts will be opened.")
            return 0

        already_short = {
            s for s, p in positions.items()
            if s in SHORT_UNIVERSE and _position_qty(p) < 0
        }

        # Weakest momentum first.
        ranked = sorted(valid_symbols, key=lambda x: scores[x])

        candidates = pick_diversified(
            ranked, TOP_N_SHORTS, MAX_SHORTS_PER_SECTOR, exclude=already_short,
        )

        target_per_position = self._target_short_size(eq, regime) * risk_scalar

        placed = 0

        for symbol in candidates:
            price = prices.get(symbol, 0.0)
            if price <= 0:
                continue

            log.info(
                "SHORT candidate %s (%s) | momentum=%.2f%% price=%.2f",
                symbol, SECTOR_MAP.get(symbol, "Other"), scores[symbol] * 100, price,
            )

            if self._open_short(symbol, price, target_per_position):
                placed += 1

        return placed

    # --------------------------------------------------------
    # LONG SLEEVE
    # --------------------------------------------------------

    def run_longs(self, scores, regime, prices, risk_scalar=1.0):
        eq = self.equity()
        if eq <= 0:
            return 0

        positions = self.positions()

        valid_symbols = [
            s for s in LONG_UNIVERSE
            if scores.get(s) is not None and prices.get(s, 0) > 0
        ]

        if not valid_symbols:
            log.warning("No symbols have valid data. No longs will be opened.")
            return 0

        already_long = {
            s for s, p in positions.items()
            if s in LONG_UNIVERSE and _position_qty(p) > 0
        }

        # Strongest momentum first.
        ranked = sorted(valid_symbols, key=lambda x: scores[x], reverse=True)

        candidates = pick_diversified(
            ranked, TOP_N_LONGS, MAX_LONGS_PER_SECTOR, exclude=already_long,
        )

        target_per_position = self._target_long_size(eq, regime) * risk_scalar

        placed = 0

        for symbol in candidates:
            price = prices.get(symbol, 0.0)
            if price <= 0:
                continue

            log.info(
                "LONG candidate %s (%s) | momentum=%.2f%% price=%.2f",
                symbol, SECTOR_MAP.get(symbol, "Other"), scores[symbol] * 100, price,
            )

            if self._open_long(symbol, price, target_per_position):
                placed += 1

        return placed

    # --------------------------------------------------------
    # TOP-UP TO DAILY MINIMUM (alternates shorts/longs)
    # --------------------------------------------------------

    def top_up_to_minimum(self, scores, regime, prices, risk_scalar=1.0):
        """
        If the day's trade count is still below MIN_TRADES_PER_DAY after
        the normal sleeves + options pass, alternate opening additional
        shorts (next-weakest) and longs (next-strongest) from real,
        ranked, not-already-held candidates until the minimum is met or
        both ranked lists run out.
        """

        reset_trade_counter_if_new_day()

        still_needed = MIN_TRADES_PER_DAY - TRADES_TODAY
        if still_needed <= 0:
            return 0

        eq = self.equity()
        if eq <= 0:
            return 0

        positions = self.positions()

        already_short = {
            s for s, p in positions.items()
            if s in SHORT_UNIVERSE and _position_qty(p) < 0
        }
        already_long = {
            s for s, p in positions.items()
            if s in LONG_UNIVERSE and _position_qty(p) > 0
        }

        valid_symbols = [
            s for s in SHORT_UNIVERSE
            if scores.get(s) is not None and prices.get(s, 0) > 0
        ]

        if not valid_symbols:
            log.info(
                "Below daily trade minimum (%d/%d) but no additional data available.",
                TRADES_TODAY, MIN_TRADES_PER_DAY,
            )
            return 0

        ranked_weak = sorted(
            [s for s in valid_symbols if s not in already_short],
            key=lambda x: scores[x],
        )
        ranked_strong = sorted(
            [s for s in valid_symbols if s not in already_long],
            key=lambda x: scores[x],
            reverse=True,
        )

        short_target = self._target_short_size(eq, regime) * risk_scalar
        long_target = self._target_long_size(eq, regime) * risk_scalar

        placed = 0
        si, li = 0, 0
        want_short = True

        while TRADES_TODAY < MIN_TRADES_PER_DAY and (si < len(ranked_weak) or li < len(ranked_strong)):

            did_something = False

            if want_short and si < len(ranked_weak):
                symbol = ranked_weak[si]
                si += 1
                price = prices.get(symbol, 0.0)
                if price > 0:
                    log.info("TOP-UP short candidate %s (trades today=%d, min=%d)",
                             symbol, TRADES_TODAY, MIN_TRADES_PER_DAY)
                    if self._open_short(symbol, price, short_target):
                        placed += 1
                did_something = True

            elif li < len(ranked_strong):
                symbol = ranked_strong[li]
                li += 1
                price = prices.get(symbol, 0.0)
                if price > 0:
                    log.info("TOP-UP long candidate %s (trades today=%d, min=%d)",
                             symbol, TRADES_TODAY, MIN_TRADES_PER_DAY)
                    if self._open_long(symbol, price, long_target):
                        placed += 1
                did_something = True

            elif si < len(ranked_weak):
                symbol = ranked_weak[si]
                si += 1
                price = prices.get(symbol, 0.0)
                if price > 0:
                    if self._open_short(symbol, price, short_target):
                        placed += 1
                did_something = True

            if not did_something:
                break

            want_short = not want_short

        return placed

    # --------------------------------------------------------
    # OPTIONS (bug fix: caps new hedges/day + existing holdings)
    # --------------------------------------------------------

    def run_options(self):
        global HEDGES_TODAY

        if not OPTIONS_ENABLED:
            return 0

        reset_hedge_counter_if_new_day()

        eq = self.equity()
        if eq <= 0:
            return 0

        positions = self.positions()

        budget_total = eq * OPTIONS_HEDGE_BUDGET_PCT

        shorts = [
            (symbol, position)
            for symbol, position in positions.items()
            if symbol in OPTIONABLE_SYMBOLS and _position_qty(position) < 0
        ]

        if not shorts:
            return 0

        budget_each = budget_total / max(len(shorts), 1)

        prices = latest_prices_many([symbol for symbol, _ in shorts])

        placed = 0

        for symbol, position in shorts:
            shares_short = abs(_position_qty(position))
            if shares_short < 100:
                continue

            spot = prices.get(symbol, 0.0)
            if spot <= 0:
                continue

            already_held = self.existing_option_contracts(positions, symbol)

            # Throttle how many *new* underlyings get hedged per day.
            # Topping up an already-hedged name doesn't count against
            # this cap -- it's already-fine coverage, not new spend.
            if already_held == 0 and HEDGES_TODAY >= OPTIONS_MAX_NEW_HEDGES_PER_DAY:
                log.info(
                    "Daily new-hedge cap (%d) reached. Skipping first hedge on %s.",
                    OPTIONS_MAX_NEW_HEDGES_PER_DAY, symbol,
                )
                continue

            was_new = already_held == 0

            if self.options.buy_protective_calls(
                symbol, shares_short, spot, budget_each, already_held=already_held,
            ):
                placed += 1
                if was_new:
                    HEDGES_TODAY += 1
                    ENGINE_STATE["hedges_today"] = HEDGES_TODAY

        return placed

    # --------------------------------------------------------
    # ONE CYCLE
    # --------------------------------------------------------

    def run_once(self):
        if not can_trade_today(self.trading):
            return

        # ----------------------------------------------------
        # PRELOAD EVERYTHING
        # ----------------------------------------------------

        self.market_data = preload_market_data()
        self.risk.market_data = self.market_data

        # ----------------------------------------------------
        # REGIME
        # ----------------------------------------------------

        regime = self.risk.regime_risk_score()
        log.info("Regime risk score: %.2f", regime)

        # ----------------------------------------------------
        # MOMENTUM
        # ----------------------------------------------------

        momentum_scores = {}

        for symbol in SHORT_UNIVERSE:
            bars = self.market_data.get(symbol, [])
            score = momentum_from_bars(bars, lookback=60)
            if score is None:
                continue
            momentum_scores[symbol] = score

        log.info(
            "Momentum data available for %d/%d symbols.",
            len(momentum_scores), len(SHORT_UNIVERSE),
        )

        # ----------------------------------------------------
        # PRICES
        # ----------------------------------------------------

        prices = latest_prices_many(list(momentum_scores.keys()))

        # ----------------------------------------------------
        # MONTE CARLO RISK CHECK (on the *intended* book, before
        # any orders go out) -- scales sizing down if simulated
        # tail risk is too high for the configured risk budget.
        # ----------------------------------------------------

        risk_scalar = 1.0

        if MC_ENABLED:
            try:
                valid_symbols = [
                    s for s in SHORT_UNIVERSE
                    if momentum_scores.get(s) is not None and prices.get(s, 0) > 0
                ]

                ranked_weak = sorted(valid_symbols, key=lambda x: momentum_scores[x])
                ranked_strong = sorted(
                    valid_symbols, key=lambda x: momentum_scores[x], reverse=True,
                )

                proposed_shorts = pick_diversified(ranked_weak, TOP_N_SHORTS, MAX_SHORTS_PER_SECTOR)
                proposed_longs = pick_diversified(ranked_strong, TOP_N_LONGS, MAX_LONGS_PER_SECTOR)

                book = [(s, -1.0) for s in proposed_shorts] + [(s, 1.0) for s in proposed_longs]

                mc_result = monte_carlo_var(book, self.market_data)
                risk_scalar = risk_scalar_from_mc(mc_result)

                ENGINE_STATE["monte_carlo"] = mc_result
                ENGINE_STATE["risk_scalar"] = risk_scalar

                if mc_result:
                    log.info(
                        "Monte Carlo (%d sims, %dd horizon, %d symbols): "
                        "mean=%.2f%% VaR95=%.2f%% CVaR95=%.2f%% -> risk_scalar=%.2f",
                        mc_result["n_sims"], mc_result["horizon_days"],
                        mc_result["symbols_used"], mc_result["mean_return_pct"],
                        mc_result["var_pct"], mc_result["cvar_pct"], risk_scalar,
                    )
                else:
                    log.info("Monte Carlo: not enough history to simulate this cycle.")

            except Exception as exc:
                log.exception("Monte Carlo risk check failed: %s", exc)
                risk_scalar = 1.0

        # ----------------------------------------------------
        # SHORTS
        # ----------------------------------------------------

        try:
            self.run_shorts(momentum_scores, regime, prices, risk_scalar=risk_scalar)
        except Exception as exc:
            log.exception("Short sleeve failed: %s", exc)

        # ----------------------------------------------------
        # LONGS
        # ----------------------------------------------------

        try:
            self.run_longs(momentum_scores, regime, prices, risk_scalar=risk_scalar)
        except Exception as exc:
            log.exception("Long sleeve failed: %s", exc)

        # ----------------------------------------------------
        # OPTIONS (hedge overlay on the short book only)
        # ----------------------------------------------------

        try:
            self.run_options()
        except Exception as exc:
            log.exception("Options sleeve failed: %s", exc)

        # ----------------------------------------------------
        # TOP UP TO DAILY MINIMUM
        #
        # Only pulls from real, ranked momentum candidates that
        # aren't already held -- never fabricated or random trades.
        # Stops naturally if there simply aren't enough valid
        # candidates left (e.g. widespread data outage).
        # ----------------------------------------------------

        try:
            self.top_up_to_minimum(momentum_scores, regime, prices, risk_scalar=risk_scalar)
        except Exception as exc:
            log.exception("Trade top-up failed: %s", exc)

        # ----------------------------------------------------
        # CYCLE SUMMARY
        # ----------------------------------------------------

        log.info(
            "Cycle complete. Valid momentum=%d. Trades today=%d. Hedges today=%d.",
            len(momentum_scores), TRADES_TODAY, HEDGES_TODAY,
        )

        gc.collect()


# ============================================================
# ENVIRONMENT HELPERS
# ============================================================

def _env_bool(name, default=True):
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")


# ============================================================
# TRADING LOOP
# ============================================================

def trading_loop():
    ENGINE_STATE["thread_started"] = True
    ENGINE_STATE["thread_alive"] = True

    try:
        initialize_credentials()
    except Exception as exc:
        log.error("Credential initialization failed: %s", exc)
        ENGINE_STATE["last_error"] = str(exc)
        ENGINE_STATE["thread_alive"] = False
        return

    paper = _env_bool("ALPACA_PAPER", True)

    log.info("Starting Lean Risk Engine.")
    log.info("Data provider: Alpaca")
    log.info("Stock feed: %s", ALPACA_STOCK_FEED)
    log.info("Options feed: %s", ALPACA_OPTION_FEED)
    log.info("Paper trading: %s", paper)

    # --------------------------------------------------------
    # ALPACA CONNECTION
    # --------------------------------------------------------

    try:
        trading_client = TradingClient(ALPACA_API_KEY, ALPACA_API_SECRET, paper=paper)
        account = trading_client.get_account()

        log.info(
            "Connected to Alpaca. status=%s equity=%s paper=%s",
            account.status, account.equity, paper,
        )

    except Exception as exc:
        msg = f"Startup connection to Alpaca failed: {exc}"
        log.exception(msg)
        ENGINE_STATE["last_error"] = msg
        ENGINE_STATE["thread_alive"] = False
        return

    # --------------------------------------------------------
    # ENGINE
    # --------------------------------------------------------

    engine = LeanTradingEngine(trading_client, {})

    # --------------------------------------------------------
    # MAIN LOOP
    # --------------------------------------------------------

    while True:
        ENGINE_STATE["last_cycle_started"] = now_et().isoformat()

        try:
            engine.run_once()
            ENGINE_STATE["last_error"] = None
        except Exception as exc:
            log.exception("Engine cycle error: %s", exc)
            ENGINE_STATE["last_error"] = str(exc)

        ENGINE_STATE["last_cycle_finished"] = now_et().isoformat()

        gc.collect()

        log.info("Cycle finished. Sleeping %ss.", LOOP_SLEEP_SECONDS)
        time.sleep(LOOP_SLEEP_SECONDS)


# ============================================================
# THREAD WRAPPER
# ============================================================

def _thread_wrapper():
    try:
        trading_loop()
    except Exception as exc:
        log.exception("Trading thread crashed: %s", exc)
        ENGINE_STATE["last_error"] = f"Thread crashed: {exc}"
        ENGINE_STATE["thread_alive"] = False


# ============================================================
# START ENGINE
# ============================================================

threading.Thread(target=_thread_wrapper, daemon=True).start()

log.info("Trading loop thread launched.")


# ============================================================
# FLASK SERVER
# ============================================================

if __name__ == "__main__":
    port = int(os.getenv("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
