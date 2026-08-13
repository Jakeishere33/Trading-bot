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
    "market_open": None,
    "data_provider": "Alpaca",
    "data_failures": 0,
}


@app.route("/")
def home():
    return "Lean Short + Options Risk Engine Online - Alpaca Data", 200


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

EQUITY_UNIVERSE = [
    "AAPL", "MSFT", "NVDA", "AMZN",
    "GOOGL", "META", "JPM", "XOM",
    "UNH", "PG", "HD", "COST",
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

PHARMA_TICKERS = [
    "PFE", "JNJ", "MRK", "ABBV", "LLY",
    "BMY", "AMGN", "GILD", "VRTX", "REGN",
    "BIIB", "ZTS", "MRNA", "ALNY", "INCY",
    "EXEL", "UTHR", "JAZZ", "RPRX", "SUPN",
    "PCVX", "ARGX", "BMRN", "IONS", "NBIX",
    "HALO", "CRSP", "VTRS", "TEVA", "ELAN",
]

MINING_TICKERS = [
    "FCX", "NEM", "GOLD", "SCCO", "AEM",
    "TECK", "RIO", "BHP", "VALE", "MOS",
    "AA", "CLF", "X", "NUE", "STLD",
    "MP", "CDE", "HL", "PAAS", "AG",
    "SSRM", "EGO", "KGC", "AU", "WPM",
    "FNV", "RGLD", "ALB", "LAC", "SQM",
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
    + EQUITY_UNIVERSE
    + SEMICONDUCTOR_TICKERS
    + MANUFACTURING_TICKERS
    + PHARMA_TICKERS
    + MINING_TICKERS
)

OPTIONABLE_SYMBOLS = set(SHORT_UNIVERSE)


# ============================================================
# STRATEGY CONFIG
# ============================================================

SHORT_SLEEVE_WEIGHT = 0.30
TOP_N_SHORTS = 8

OPTIONS_ENABLED = True

HEDGE_CALL_OTM_PCT = 0.05

OPTIONS_MIN_DTE = 25
OPTIONS_MAX_DTE = 45

OPTIONS_HEDGE_BUDGET_PCT = 0.05

MAX_TRADES_PER_DAY = 300

# IMPORTANT:
# Do not force trades just because a minimum was not reached.
# This prevents bad trades during data outages.
MIN_TRADES_PER_DAY = 0

TRADES_TODAY = 0
LAST_TRADE_DAY = None


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
# ALPACA DATA CONFIG
# ============================================================

ALPACA_DATA_URL = "https://data.alpaca.markets"

# IEX is generally the safest choice for accounts without
# the SIP subscription.
ALPACA_STOCK_FEED = os.getenv("ALPACA_STOCK_FEED", "iex")

# Options:
# "indicative" is usable without OPRA subscription.
# "opra" requires the appropriate market-data subscription.
ALPACA_OPTION_FEED = os.getenv(
    "ALPACA_OPTION_FEED",
    "indicative",
)

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
    "User-Agent": "LeanRiskEngine/2.0",
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
        raise RuntimeError(
            "ALPACA_API_KEY / ALPACA_API_SECRET are not configured."
        )


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

    Handles:
      429 rate limits
      500-series server errors
      connection failures
      timeouts
      malformed JSON
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

            # ----------------------------
            # SUCCESS
            # ----------------------------

            if 200 <= status < 300:

                try:
                    return response.json()

                except ValueError as exc:
                    raise RuntimeError(
                        f"Invalid JSON returned by Alpaca: {exc}"
                    )

            # ----------------------------
            # RATE LIMIT
            # ----------------------------

            if status == 429:

                retry_after = response.headers.get("Retry-After")

                if retry_after:
                    try:
                        delay = float(retry_after)
                    except Exception:
                        delay = DATA_RETRY_BASE_DELAY * (2 ** (attempt - 1))
                else:
                    delay = DATA_RETRY_BASE_DELAY * (
                        2 ** (attempt - 1)
                    )

                delay += random.uniform(0.2, 0.8)

                log.warning(
                    "Alpaca rate limit (%s) %s attempt %d/%d. "
                    "Sleeping %.1fs.",
                    label,
                    path,
                    attempt,
                    DATA_MAX_RETRIES,
                    delay,
                )

                time.sleep(delay)
                continue

            # ----------------------------
            # SERVER ERRORS
            # ----------------------------

            if status >= 500:

                delay = DATA_RETRY_BASE_DELAY * (
                    2 ** (attempt - 1)
                )

                delay += random.uniform(0.2, 0.8)

                log.warning(
                    "Alpaca server error %s (%s) attempt %d/%d. "
                    "Retrying in %.1fs.",
                    status,
                    label,
                    attempt,
                    DATA_MAX_RETRIES,
                    delay,
                )

                time.sleep(delay)
                continue

            # ----------------------------
            # CLIENT ERROR
            # ----------------------------

            try:
                body = response.json()
            except Exception:
                body = response.text[:500]

            raise RuntimeError(
                f"Alpaca HTTP {status}: {body}"
            )

        except Exception as exc:

            last_error = exc

            if attempt < DATA_MAX_RETRIES:

                delay = DATA_RETRY_BASE_DELAY * (
                    2 ** (attempt - 1)
                )

                delay += random.uniform(0.2, 0.8)

                log.warning(
                    "Alpaca request failed (%s) attempt %d/%d: %s. "
                    "Retrying in %.1fs.",
                    label,
                    attempt,
                    DATA_MAX_RETRIES,
                    exc,
                    delay,
                )

                time.sleep(delay)

            else:

                log.error(
                    "Alpaca request failed (%s) after %d attempts: %s",
                    label,
                    DATA_MAX_RETRIES,
                    exc,
                )

    raise last_error or RuntimeError(
        f"Unknown Alpaca data failure: {label}"
    )


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
    return dt.astimezone(timezone.utc).isoformat().replace(
        "+00:00",
        "Z",
    )


# ============================================================
# ALPACA HISTORICAL STOCK DATA
# ============================================================

def alpaca_bars_many(symbols, days=60):
    """
    Download historical daily bars for many symbols in batches.

    This replaces yfinance completely.

    Returns:
        {
            "AAPL": [price, price, price, ...],
            "MSFT": [...],
        }
    """

    symbols = _dedupe(symbols)

    if not symbols:
        return {}

    days = min(int(days), MAX_BARS_DAYS)

    cache_key = (
        tuple(sorted(symbols)),
        days,
        ALPACA_STOCK_FEED,
    )

    cached = cache_get(_BARS_CACHE, cache_key)

    if cached is not None:
        return cached

    # We request extra calendar days because trading days
    # are fewer than calendar days.
    calendar_days = int(days * 1.55) + 15

    end_dt = utc_now()
    start_dt = end_dt - timedelta(days=calendar_days)

    result = {
        symbol: []
        for symbol in symbols
    }

    # --------------------------------------------------------
    # BATCH REQUESTS
    # --------------------------------------------------------

    for start_index in range(
        0,
        len(symbols),
        DATA_BATCH_SIZE,
    ):

        batch = symbols[
            start_index:
            start_index + DATA_BATCH_SIZE
        ]

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
                log.warning(
                    "Malformed Alpaca bars response for batch."
                )
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

            log.error(
                "Historical data batch failed: %s",
                exc,
            )

            # IMPORTANT:
            # Do NOT destroy the entire engine because one
            # batch failed. Continue with other batches.

            continue

    cache_put(
        _BARS_CACHE,
        cache_key,
        result,
    )

    return result


def yf_bars(symbol, days=60):
    """
    Compatibility wrapper.

    The rest of the engine can continue calling yf_bars(),
    but it no longer uses Yahoo Finance.
    """

    symbol = symbol.upper()

    data = alpaca_bars_many(
        [symbol],
        days=days,
    )

    return data.get(symbol, [])


# ============================================================
# BULK DATA PRELOAD
# ============================================================

def preload_market_data():
    """
    Load the entire universe once per cycle.

    This is a major improvement over making one request per
    ticker.
    """

    log.info(
        "Loading Alpaca market data for %d symbols...",
        len(SHORT_UNIVERSE),
    )

    data = alpaca_bars_many(
        SHORT_UNIVERSE,
        days=MAX_BARS_DAYS,
    )

    successful = sum(
        1
        for symbol in SHORT_UNIVERSE
        if len(data.get(symbol, [])) >= 30
    )

    failed = len(SHORT_UNIVERSE) - successful

    log.info(
        "Alpaca market data loaded: %d/%d symbols usable, %d failed.",
        successful,
        len(SHORT_UNIVERSE),
        failed,
    )

    return data


# ============================================================
# LATEST PRICES
# ============================================================

def latest_prices_many(symbols):
    """
    Get latest prices in batches.

    Uses Alpaca latest trades.
    """

    symbols = _dedupe(symbols)

    if not symbols:
        return {}

    cache_key = (
        tuple(sorted(symbols)),
        ALPACA_STOCK_FEED,
    )

    cached = cache_get(
        _PRICE_CACHE,
        cache_key,
    )

    if cached is not None:
        return cached

    result = {}

    for start_index in range(
        0,
        len(symbols),
        DATA_BATCH_SIZE,
    ):

        batch = symbols[
            start_index:
            start_index + DATA_BATCH_SIZE
        ]

        params = {
            "symbols": ",".join(batch),
            "feed": ALPACA_STOCK_FEED,
        }

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

            log.warning(
                "Latest-price batch failed: %s",
                exc,
            )

    cache_put(
        _PRICE_CACHE,
        cache_key,
        result,
    )

    return result


def latest_price(symbol):
    prices = latest_prices_many([symbol])

    price = prices.get(
        symbol.upper(),
        0.0,
    )

    if price > 0:
        return price

    # Fallback to most recent daily bar.
    bars = yf_bars(
        symbol,
        days=5,
    )

    return bars[-1] if bars else 0.0


# ============================================================
# MOMENTUM
# ============================================================

def momentum_from_bars(
    bars,
    lookback=20,
):

    if not bars:
        return None

    if len(bars) < lookback + 1:
        return None

    try:
        return (
            bars[-1] / bars[-lookback]
        ) - 1.0

    except Exception:
        return None


def momentum(
    symbol,
    lookback=20,
):

    bars = yf_bars(
        symbol,
        days=lookback + 5,
    )

    return momentum_from_bars(
        bars,
        lookback,
    )


# ============================================================
# VOLATILITY
# ============================================================

def volatility_from_bars(
    bars,
    window=14,
):

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
            returns.append(
                (current / previous) - 1.0
            )

        except Exception:
            continue

    if len(returns) < window:
        return None

    sample = returns[-window:]

    mean = sum(sample) / len(sample)

    variance = sum(
        (r - mean) ** 2
        for r in sample
    ) / len(sample)

    return sqrt(
        variance
    ) * sqrt(TRADING_DAYS)


def volatility(
    symbol,
    window=14,
):

    bars = yf_bars(
        symbol,
        days=window + 10,
    )

    value = volatility_from_bars(
        bars,
        window,
    )

    return value if value is not None else 0.0


# ============================================================
# REGIME / RISK MANAGER
# ============================================================

class LeanRiskManager:

    def __init__(self, market_data=None):
        self.market_data = market_data or {}

    def regime_risk_score(self):

        bars = self.market_data.get(
            REGIME_BENCHMARK,
            [],
        )

        if len(bars) < 200:

            log.warning(
                "Not enough %s history for regime calculation. "
                "Using neutral regime.",
                REGIME_BENCHMARK,
            )

            return 0.3

        sma200 = sum(
            bars[-200:]
        ) / 200

        last = bars[-1]

        below_sma = (
            0.0
            if last > sma200
            else 1.0
        )

        last_year = (
            bars[-252:]
            if len(bars) >= 252
            else bars
        )

        highest = max(last_year)

        if highest <= 0:
            dd = 0.0
        else:
            dd = 1.0 - (
                last / highest
            )

        dd_score = max(
            0.0,
            min(
                dd / 0.20,
                1.0,
            ),
        )

        return (
            below_sma
            + dd_score
        ) / 2.0


# ============================================================
# BLACK-SCHOLES
# ============================================================

def _cdf(x):
    return 0.5 * (
        1.0 + erf(
            x / sqrt(2.0)
        )
    )


def bs_price(
    S,
    K,
    T,
    r,
    sigma,
    opt_type,
):

    if (
        T <= 0
        or sigma <= 0
        or S <= 0
        or K <= 0
    ):
        return 0.0

    d1 = (
        ln(S / K)
        + (
            r
            + 0.5 * sigma * sigma
        ) * T
    ) / (
        sigma * sqrt(T)
    )

    d2 = (
        d1
        - sigma * sqrt(T)
    )

    if opt_type == "call":

        return (
            S * _cdf(d1)
            - K
            * exp(-r * T)
            * _cdf(d2)
        )

    return (
        K
        * exp(-r * T)
        * _cdf(-d2)
        - S
        * _cdf(-d1)
    )


def bs_delta(
    S,
    K,
    T,
    r,
    sigma,
    opt_type,
):

    if (
        T <= 0
        or sigma <= 0
        or S <= 0
        or K <= 0
    ):
        return 0.0

    d1 = (
        ln(S / K)
        + (
            r
            + 0.5 * sigma * sigma
        ) * T
    ) / (
        sigma * sqrt(T)
    )

    if opt_type == "call":
        return _cdf(d1)

    return _cdf(d1) - 1.0


# ============================================================
# ALPACA OPTIONS DATA
# ============================================================

class LeanOptionsEngine:

    def __init__(
        self,
        trading_client,
    ):

        self.trading = trading_client

    # --------------------------------------------------------
    # OPTION CHAIN
    # --------------------------------------------------------

    def option_chain(
        self,
        underlying,
    ):

        underlying = underlying.upper()

        cache_key = (
            underlying,
            OPTIONS_MIN_DTE,
            OPTIONS_MAX_DTE,
            ALPACA_OPTION_FEED,
        )

        cached = cache_get(
            _OPTIONS_CACHE,
            cache_key,
        )

        if cached is not None:
            return cached

        today = now_et().date()

        min_expiration = (
            today
            + timedelta(
                days=OPTIONS_MIN_DTE
            )
        )

        max_expiration = (
            today
            + timedelta(
                days=OPTIONS_MAX_DTE
            )
        )

        params = {
            "feed": ALPACA_OPTION_FEED,
            "type": "call",
            "expiration_date_gte":
                min_expiration.isoformat(),
            "expiration_date_lte":
                max_expiration.isoformat(),
            "limit": 1000,
        }

        try:

            data = alpaca_data_get(
                f"/v1beta1/options/snapshots/{underlying}",
                params=params,
                label=f"option-chain:{underlying}",
            )

            snapshots = data.get(
                "snapshots",
                {},
            )

            contracts = []

            if isinstance(
                snapshots,
                dict,
            ):

                for contract_symbol, snapshot in snapshots.items():

                    try:

                        parts = parse_occ_option_symbol(
                            contract_symbol
                        )

                        if not parts:
                            continue

                        (
                            root,
                            expiration,
                            option_type,
                            strike,
                        ) = parts

                        if option_type != "C":
                            continue

                        if (
                            expiration < min_expiration
                            or expiration > max_expiration
                        ):
                            continue

                        latest_quote = (
                            snapshot.get(
                                "latestQuote",
                                {}
                            )
                            or {}
                        )

                        latest_trade = (
                            snapshot.get(
                                "latestTrade",
                                {}
                            )
                            or {}
                        )

                        bid = float(
                            latest_quote.get(
                                "bp",
                                0
                            )
                            or 0
                        )

                        ask = float(
                            latest_quote.get(
                                "ap",
                                0
                            )
                            or 0
                        )

                        volume = int(
                            latest_trade.get(
                                "s",
                                0
                            )
                            or 0
                        )

                        greeks = (
                            snapshot.get(
                                "greeks",
                                {}
                            )
                            or {}
                        )

                        contracts.append({
                            "symbol":
                                contract_symbol,
                            "strike":
                                strike,
                            "expiration":
                                expiration,
                            "bid":
                                bid,
                            "ask":
                                ask,
                            "volume":
                                volume,
                            "delta":
                                float(
                                    greeks.get(
                                        "delta",
                                        0
                                    )
                                    or 0
                                ),
                            "iv":
                                float(
                                    snapshot.get(
                                        "impliedVolatility",
                                        0
                                    )
                                    or 0
                                ),
                        })

                    except Exception:
                        continue

            cache_put(
                _OPTIONS_CACHE,
                cache_key,
                contracts,
            )

            return contracts

        except Exception as exc:

            ENGINE_STATE["data_failures"] += 1

            log.warning(
                "Alpaca option chain failed for %s: %s",
                underlying,
                exc,
            )

            return []


    # --------------------------------------------------------
    # FIND CONTRACT
    # --------------------------------------------------------

    def find_contract(
        self,
        underlying,
        spot,
    ):

        contracts = self.option_chain(
            underlying
        )

        if not contracts:
            return None

        target_strike = (
            spot
            * (
                1.0
                + HEDGE_CALL_OTM_PCT
            )
        )

        valid = []

        for contract in contracts:

            if not self.liquidity_ok(
                contract
            ):
                continue

            valid.append(
                contract
            )

        if not valid:
            return None

        valid.sort(
            key=lambda c: (
                abs(
                    c["strike"]
                    - target_strike
                ),
                c["expiration"],
            )
        )

        return valid[0]


    # --------------------------------------------------------
    # LIQUIDITY
    # --------------------------------------------------------

    def liquidity_ok(
        self,
        contract,
    ):

        bid = contract["bid"]
        ask = contract["ask"]

        if bid <= 0 or ask <= 0:
            return False

        if ask < bid:
            return False

        mid = (
            bid + ask
        ) / 2.0

        if mid <= 0:
            return False

        spread = (
            ask - bid
        ) / mid

        if spread > MAX_OPTION_SPREAD_PCT:
            return False

        return True


    # --------------------------------------------------------
    # PROTECTIVE CALL
    # --------------------------------------------------------

    def buy_protective_calls(
        self,
        underlying,
        shares_short,
        spot,
        budget,
    ):

        if (
            shares_short < 100
            or budget <= 0
            or spot <= 0
        ):
            return False

        contract = self.find_contract(
            underlying,
            spot,
        )

        if not contract:
            log.info(
                "No liquid protective call found for %s.",
                underlying,
            )
            return False

        symbol = contract["symbol"]

        assert_safe_order(
            symbol,
            AssetClass.US_OPTION,
        )

        expiration = contract[
            "expiration"
        ]

        dte = (
            expiration
            - now_et().date()
        ).days

        if dte <= 0:
            return False

        T = (
            dte
            / TRADING_DAYS
        )

        sigma = volatility(
            underlying,
            window=20,
        )

        if sigma <= 0:
            sigma = max(
                contract.get(
                    "iv",
                    0
                ),
                0.20,
            )

        fair_value = bs_price(
            spot,
            contract["strike"],
            T,
            RISK_FREE_ANNUAL,
            sigma,
            "call",
        )

        delta = bs_delta(
            spot,
            contract["strike"],
            T,
            RISK_FREE_ANNUAL,
            sigma,
            "call",
        )

        bid = contract["bid"]
        ask = contract["ask"]

        # Use the ask only if it is reasonable.
        # Otherwise use a midpoint-based limit.
        mid = (
            bid + ask
        ) / 2.0

        if mid <= 0:
            return False

        # Do not pay an absurd amount over theoretical value.
        if fair_value > 0:

            max_reasonable_price = (
                fair_value * 1.20
            )

            limit_price = min(
                ask,
                max_reasonable_price,
            )

        else:

            limit_price = mid

        if limit_price <= 0:
            return False

        max_affordable = int(
            budget
            // (
                limit_price
                * 100
            )
        )

        contracts_by_shares = int(
            shares_short // 100
        )

        qty = min(
            contracts_by_shares,
            max_affordable,
        )

        if qty <= 0:
            return False

        order = LimitOrderRequest(
            symbol=symbol,
            qty=qty,
            side=OrderSide.BUY,
            type=OrderType.LIMIT,
            time_in_force=TimeInForce.DAY,
            limit_price=round(
                limit_price,
                2,
            ),
        )

        log.info(
            "Protective call candidate %s | "
            "spot=%.2f strike=%.2f dte=%d "
            "bid=%.2f ask=%.2f fair=%.2f delta=%.3f",
            symbol,
            spot,
            contract["strike"],
            dte,
            bid,
            ask,
            fair_value,
            delta,
        )

        return submit_order_safe(
            self.trading,
            order,
            f"BUY protective call {symbol}",
        )


# ============================================================
# OCC OPTION SYMBOL PARSER
# ============================================================

def parse_occ_option_symbol(symbol):

    """
    OCC option format:

    AAPL250117C00200000

    root = AAPL
    expiration = 2025-01-17
    type = C
    strike = 200.00
    """

    try:

        if len(symbol) < 15:
            return None

        # OCC format has 15-character strike/type tail.
        tail = symbol[-15:]

        option_type = tail[6]

        if option_type not in (
            "C",
            "P",
        ):
            return None

        date_str = tail[:6]

        strike_str = tail[7:]

        year = int(
            date_str[0:2]
        )

        month = int(
            date_str[2:4]
        )

        day = int(
            date_str[4:6]
        )

        expiration = datetime(
            2000 + year,
            month,
            day,
        ).date()

        strike = (
            int(strike_str)
            / 1000.0
        )

        root = symbol[:-15]

        return (
            root,
            expiration,
            option_type,
            strike,
        )

    except Exception:
        return None


# ============================================================
# MARKET / TRADING WINDOW
# ============================================================

_CLOCK_CACHE = {
    "ts": 0.0,
    "is_open": None,
}

_CLOCK_CACHE_TTL_SECONDS = 30


def _fallback_in_no_trade_window():

    t = now_et()

    h = t.hour
    m = t.minute

    if (
        h < NO_TRADE_BEFORE[0]
        or (
            h == NO_TRADE_BEFORE[0]
            and m < NO_TRADE_BEFORE[1]
        )
    ):
        return True

    if (
        h > NO_TRADE_AFTER[0]
        or (
            h == NO_TRADE_AFTER[0]
            and m >= NO_TRADE_AFTER[1]
        )
    ):
        return True

    return False


def market_is_open(
    trading_client,
):

    now = time.time()

    if (
        _CLOCK_CACHE["is_open"]
        is not None
        and (
            now
            - _CLOCK_CACHE["ts"]
        ) < _CLOCK_CACHE_TTL_SECONDS
    ):
        return _CLOCK_CACHE["is_open"]

    try:

        clock = (
            trading_client.get_clock()
        )

        is_open = bool(
            clock.is_open
        )

        _CLOCK_CACHE["ts"] = now
        _CLOCK_CACHE["is_open"] = is_open

        ENGINE_STATE[
            "market_open"
        ] = is_open

        return is_open

    except Exception as exc:

        log.warning(
            "Alpaca clock failed: %s. "
            "Using ET fallback.",
            exc,
        )

        fallback = not (
            _fallback_in_no_trade_window()
        )

        ENGINE_STATE[
            "market_open"
        ] = fallback

        return fallback


# ============================================================
# TRADE COUNTER
# ============================================================

def reset_trade_counter_if_new_day():

    global TRADES_TODAY
    global LAST_TRADE_DAY

    today = now_et().date()

    if LAST_TRADE_DAY != today:

        LAST_TRADE_DAY = today

        TRADES_TODAY = 0

        ENGINE_STATE[
            "trades_today"
        ] = 0

        log.info(
            "New trading day: trade counter reset."
        )


def can_trade_today(
    trading_client,
):

    reset_trade_counter_if_new_day()

    if not market_is_open(
        trading_client
    ):
        log.info(
            "Market closed. Skipping trades."
        )
        return False

    return (
        TRADES_TODAY
        < MAX_TRADES_PER_DAY
    )


# ============================================================
# SAFE ORDER SUBMISSION
# ============================================================

def submit_order_safe(
    trading_client,
    order_data,
    label="",
):

    global TRADES_TODAY

    reset_trade_counter_if_new_day()

    if (
        TRADES_TODAY
        >= MAX_TRADES_PER_DAY
    ):
        log.warning(
            "Trade limit reached. "
            "Skipping order: %s",
            label,
        )
        return False

    if not market_is_open(
        trading_client
    ):
        log.info(
            "Market closed. "
            "Skipping order: %s",
            label,
        )
        return False

    try:

        result = (
            trading_client.submit_order(
                order_data=order_data
            )
        )

        TRADES_TODAY += 1

        ENGINE_STATE[
            "trades_today"
        ] = TRADES_TODAY

        log.info(
            "ORDER SUBMITTED | "
            "trade #%d | %s | id=%s",
            TRADES_TODAY,
            label,
            getattr(
                result,
                "id",
                "unknown",
            ),
        )

        return True

    except Exception as exc:

        log.exception(
            "Order failed (%s): %s",
            label,
            exc,
        )

        return False


def assert_safe_order(
    symbol,
    asset_class,
):

    if asset_class not in {
        AssetClass.US_EQUITY,
        AssetClass.US_OPTION,
    }:
        raise ValueError(
            f"Blocked order for {symbol}: "
            f"asset class {asset_class}"
        )


# ============================================================
# MAIN TRADING ENGINE
# ============================================================

class LeanTradingEngine:

    def __init__(
        self,
        trading_client,
        market_data,
    ):

        self.trading = trading_client

        self.market_data = (
            market_data
        )

        self.options = (
            LeanOptionsEngine(
                trading_client
            )
        )

        self.risk = (
            LeanRiskManager(
                market_data
            )
        )

    # --------------------------------------------------------
    # ACCOUNT
    # --------------------------------------------------------

    def equity(self):

        try:

            account = (
                self.trading.get_account()
            )

            return float(
                account.equity
            )

        except Exception as exc:

            log.error(
                "Failed to fetch equity: %s",
                exc,
            )

            return 0.0

    # --------------------------------------------------------
    # POSITIONS
    # --------------------------------------------------------

    def positions(self):

        try:

            positions = (
                self.trading.get_all_positions()
            )

            return {
                p.symbol: p
                for p in positions
            }

        except Exception as exc:

            log.warning(
                "Failed to fetch positions: %s",
                exc,
            )

            return {}

    # --------------------------------------------------------
    # OPEN SHORT
    # --------------------------------------------------------

    def _open_short(
        self,
        symbol,
        price,
        target_value,
    ):

        if price <= 0:
            return False

        qty = int(
            target_value
            / price
        )

        if qty <= 0:
            return False

        assert_safe_order(
            symbol,
            AssetClass.US_EQUITY,
        )

        order = MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=OrderSide.SELL,
            time_in_force=TimeInForce.DAY,
        )

        return submit_order_safe(
            self.trading,
            order,
            f"SHORT {symbol}",
        )

    # --------------------------------------------------------
    # SHORT STRATEGY
    # --------------------------------------------------------

    def run_shorts(
        self,
        scores,
        regime,
        prices,
    ):

        eq = self.equity()

        if eq <= 0:
            return 0

        positions = self.positions()

        valid_symbols = [
            symbol
            for symbol in SHORT_UNIVERSE
            if scores.get(symbol) is not None
            and prices.get(symbol, 0) > 0
        ]

        if not valid_symbols:

            log.warning(
                "No symbols have valid data. "
                "No shorts will be opened."
            )

            return 0

        # Lower momentum = weaker stock.
        ranked = sorted(
            valid_symbols,
            key=lambda x: scores[x],
        )

        candidates = ranked[
            :TOP_N_SHORTS
        ]

        per_position = (
            SHORT_SLEEVE_WEIGHT
            / max(
                TOP_N_SHORTS,
                1,
            )
        )

        # Slightly increase size in higher-risk regimes.
        regime_multiplier = (
            0.5
            + regime
        )

        target_per_position = (
            eq
            * per_position
            * regime_multiplier
        )

        placed = 0

        for symbol in candidates:

            # Never duplicate an existing short.
            if symbol in positions:

                try:

                    existing_qty = float(
                        positions[
                            symbol
                        ].qty
                    )

                except Exception:
                    existing_qty = 0

                if existing_qty != 0:
                    continue

            price = prices.get(
                symbol,
                0.0,
            )

            if price <= 0:
                continue

            log.info(
                "SHORT candidate %s | "
                "momentum=%.2f%% price=%.2f",
                symbol,
                scores[symbol] * 100,
                price,
            )

            if self._open_short(
                symbol,
                price,
                target_per_position,
            ):
                placed += 1

        return placed

    # --------------------------------------------------------
    # OPTIONS
    # --------------------------------------------------------

    def run_options(self):

        if not OPTIONS_ENABLED:
            return 0

        eq = self.equity()

        if eq <= 0:
            return 0

        positions = self.positions()

        budget_total = (
            eq
            * OPTIONS_HEDGE_BUDGET_PCT
        )

        shorts = []

        for symbol, position in positions.items():

            try:

                qty = float(
                    position.qty
                )

            except Exception:
                continue

            if (
                qty < 0
                and symbol in OPTIONABLE_SYMBOLS
            ):
                shorts.append(
                    (
                        symbol,
                        position,
                    )
                )

        if not shorts:
            return 0

        budget_each = (
            budget_total
            / max(
                len(shorts),
                1,
            )
        )

        prices = latest_prices_many(
            [
                symbol
                for symbol, _ in shorts
            ]
        )

        placed = 0

        for symbol, position in shorts:

            try:
                shares_short = abs(
                    float(position.qty)
                )
            except Exception:
                continue

            if shares_short < 100:
                continue

            spot = prices.get(
                symbol,
                0.0,
            )

            if spot <= 0:
                continue

            if self.options.buy_protective_calls(
                symbol,
                shares_short,
                spot,
                budget_each,
            ):
                placed += 1

        return placed

    # --------------------------------------------------------
    # ONE CYCLE
    # --------------------------------------------------------

    def run_once(self):

        if not can_trade_today(
            self.trading
        ):
            return

        # ----------------------------------------------------
        # PRELOAD EVERYTHING
        # ----------------------------------------------------

        self.market_data = (
            preload_market_data()
        )

        self.risk.market_data = (
            self.market_data
        )

        # ----------------------------------------------------
        # REGIME
        # ----------------------------------------------------

        regime = (
            self.risk.regime_risk_score()
        )

        log.info(
            "Regime risk score: %.2f",
            regime,
        )

        # ----------------------------------------------------
        # MOMENTUM
        # ----------------------------------------------------

        momentum_scores = {}

        for symbol in SHORT_UNIVERSE:

            bars = self.market_data.get(
                symbol,
                [],
            )

            score = momentum_from_bars(
                bars,
                lookback=60,
            )

            if score is None:
                continue

            momentum_scores[
                symbol
            ] = score

        log.info(
            "Momentum data available for %d/%d symbols.",
            len(momentum_scores),
            len(SHORT_UNIVERSE),
        )

        # ----------------------------------------------------
        # PRICES
        # ----------------------------------------------------

        prices = latest_prices_many(
            list(
                momentum_scores.keys()
            )
        )

        # ----------------------------------------------------
        # SHORTS
        # ----------------------------------------------------

        try:

            self.run_shorts(
                momentum_scores,
                regime,
                prices,
            )

        except Exception as exc:

            log.exception(
                "Short sleeve failed: %s",
                exc,
            )

        # ----------------------------------------------------
        # OPTIONS
        # ----------------------------------------------------

        try:

            self.run_options()

        except Exception as exc:

            log.exception(
                "Options sleeve failed: %s",
                exc,
            )

        # ----------------------------------------------------
        # NO FORCED TRADES
        # ----------------------------------------------------

        log.info(
            "Cycle complete. "
            "Valid momentum=%d. "
            "Trades today=%d.",
            len(momentum_scores),
            TRADES_TODAY,
        )

        gc.collect()


# ============================================================
# ENVIRONMENT HELPERS
# ============================================================

def _env_bool(
    name,
    default=True,
):

    value = os.getenv(name)

    if value is None:
        return default

    return (
        value.strip().lower()
        in (
            "1",
            "true",
            "yes",
            "on",
        )
    )


# ============================================================
# TRADING LOOP
# ============================================================

def trading_loop():

    ENGINE_STATE[
        "thread_started"
    ] = True

    ENGINE_STATE[
        "thread_alive"
    ] = True

    try:

        initialize_credentials()

    except Exception as exc:

        log.error(
            "Credential initialization failed: %s",
            exc,
        )

        ENGINE_STATE[
            "last_error"
        ] = str(exc)

        ENGINE_STATE[
            "thread_alive"
        ] = False

        return

    paper = _env_bool(
        "ALPACA_PAPER",
        True,
    )

    log.info(
        "Starting Lean Risk Engine."
    )

    log.info(
        "Data provider: Alpaca"
    )

    log.info(
        "Stock feed: %s",
        ALPACA_STOCK_FEED,
    )

    log.info(
        "Options feed: %s",
        ALPACA_OPTION_FEED,
    )

    log.info(
        "Paper trading: %s",
        paper,
    )

    # --------------------------------------------------------
    # ALPACA CONNECTION
    # --------------------------------------------------------

    try:

        trading_client = TradingClient(
            ALPACA_API_KEY,
            ALPACA_API_SECRET,
            paper=paper,
        )

        account = (
            trading_client.get_account()
        )

        log.info(
            "Connected to Alpaca. "
            "status=%s equity=%s paper=%s",
            account.status,
            account.equity,
            paper,
        )

    except Exception as exc:

        msg = (
            f"Startup connection to Alpaca failed: "
            f"{exc}"
        )

        log.exception(msg)

        ENGINE_STATE[
            "last_error"
        ] = msg

        ENGINE_STATE[
            "thread_alive"
        ] = False

        return

    # --------------------------------------------------------
    # ENGINE
    # --------------------------------------------------------

    engine = LeanTradingEngine(
        trading_client,
        {},
    )

    # --------------------------------------------------------
    # MAIN LOOP
    # --------------------------------------------------------

    while True:

        ENGINE_STATE[
            "last_cycle_started"
        ] = now_et().isoformat()

        try:

            engine.run_once()

            ENGINE_STATE[
                "last_error"
            ] = None

        except Exception as exc:

            log.exception(
                "Engine cycle error: %s",
                exc,
            )

            ENGINE_STATE[
                "last_error"
            ] = str(exc)

        ENGINE_STATE[
            "last_cycle_finished"
        ] = now_et().isoformat()

        gc.collect()

        log.info(
            "Cycle finished. "
            "Sleeping %ss.",
            LOOP_SLEEP_SECONDS,
        )

        time.sleep(
            LOOP_SLEEP_SECONDS
        )


# ============================================================
# THREAD WRAPPER
# ============================================================

def _thread_wrapper():

    try:

        trading_loop()

    except Exception as exc:

        log.exception(
            "Trading thread crashed: %s",
            exc,
        )

        ENGINE_STATE[
            "last_error"
        ] = (
            f"Thread crashed: {exc}"
        )

        ENGINE_STATE[
            "thread_alive"
        ] = False


# ============================================================
# START ENGINE
# ============================================================

threading.Thread(
    target=_thread_wrapper,
    daemon=True,
).start()

log.info(
    "Trading loop thread launched."
)


# ============================================================
# FLASK SERVER
# ============================================================

if __name__ == "__main__":

    port = int(
        os.getenv(
            "PORT",
            10000,
        )
    )

    app.run(
        host="0.0.0.0",
        port=port,
    )
