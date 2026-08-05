import os
import re
import time
import random
import threading
import logging
from datetime import date, timedelta, datetime
from zoneinfo import ZoneInfo
from math import log, sqrt, exp

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import minimize
from scipy.stats import norm
from flask import Flask
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import (
    MarketOrderRequest,
    LimitOrderRequest,
    TrailingStopOrderRequest,
    GetOptionContractsRequest,
    GetOrdersRequest,
)
from alpaca.trading.enums import (
    OrderSide,
    TimeInForce,
    OrderType,
    ContractType,
    AssetStatus,
    AssetClass,
    QueryOrderStatus,
)
from alpaca.data.historical.stock import StockHistoricalDataClient
from alpaca.data.historical.option import OptionHistoricalDataClient
from alpaca.data.requests import StockLatestTradeRequest, OptionLatestQuoteRequest

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("risk_engine")

app = Flask(__name__)


@app.route("/")
def home():
    return "Global Multi-Factor Risk Engine Online (ETFs/NYSE/NASDAQ/Options)!", 200


@app.route("/health")
def health():
    # Simple, fast health check for UptimeRobot / Render
    return "OK", 200


# ============================================================
# CONFIG
# ============================================================

HISTORY_YEARS = 12
RISK_FREE_ANNUAL = 0.04
TRADING_DAYS = 252

# Core 70% sleeve — ONLY ETFs
CORE_ETFS = {
    "VOO": 0.45,
    "VXUS": 0.15,
    "BND": 0.10,
}
CORE_TOTAL_WEIGHT = 0.70

SECTOR_UNIVERSE = {
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
    "Global_Ex_US": "VEU",
    "Emerging_Markets": "VWO",
}
SECTOR_TICKERS = list(SECTOR_UNIVERSE.values())
TOP_N_SATELLITE = 3

HEDGE_INSTRUMENTS = {
    "long_duration_bonds": "TLT",
    "gold": "GLD",
    "inverse_equity": "SH",
}
REGIME_BENCHMARK = "VOO"
VIX_TICKER = "^VIX"

SATELLITE_HEDGE_TOTAL = 1.0 - CORE_TOTAL_WEIGHT   # 0.30
MIN_HEDGE_WEIGHT = 0.02
MAX_HEDGE_WEIGHT = 0.07   # hedge < 7% of portfolio (per instrument)

# Split the 30% satellite+hedge sleeve between the two purposes.
# (Previously the code accidentally gave the hedges the entire sleeve,
# leaving nothing for sector satellites — fixed by explicitly splitting it.)
HEDGE_SLEEVE_WEIGHT = 0.10
SATELLITE_SLEEVE_WEIGHT = SATELLITE_HEDGE_TOTAL - HEDGE_SLEEVE_WEIGHT  # 0.20

# Individual-stock satellite sleeve — carved out of SATELLITE_SLEEVE_WEIGHT
# so equities can sit alongside sector ETFs in the satellite book.
STOCK_SLEEVE_SHARE_OF_SATELLITE = 0.40  # 40% of the satellite budget -> single names
TOP_N_STOCKS = 5

# Options overlay config
OPTIONS_ENABLED = True
CALL_OTM_PCT = 0.03
PUT_OTM_PCT = 0.07
OPTIONS_MIN_DTE = 25
OPTIONS_MAX_DTE = 45
PUT_HEDGE_SHARE = 0.25
OPTIONS_RUN_INTERVAL_SECONDS = 24 * 3600
MAX_OPTIONS_PCT_OF_EQUITY = 2.0   # options <= 2% of equity
COVERED_CALL_MIN_SHARES = 100     # 1 option contract = 100 shares

# ---- Short equity sleeve ----
# A small, separately-capped sleeve for outright short equity positions,
# sized off the worst-ranked names in the liquid equity universe. Requires
# a margin account with shorting enabled on Alpaca — orders simply fail
# (and get logged) if the account doesn't support it.
SHORT_SLEEVE_WEIGHT = 0.05         # 5% of equity, short-side risk budget AT NEUTRAL regime
MIN_SHORT_SLEEVE_MULTIPLIER = 0.3  # in strongly bullish regimes, shrink the short sleeve to 30% of base
MAX_SHORT_SLEEVE_MULTIPLIER = 1.6  # in strongly bearish regimes, scale it up to 160% of base
MAX_SHORT_POSITION_WEIGHT = 0.03   # no single short > 3% of equity
MAX_SHORT_EXPOSURE_PCT = 8.0       # hard cap: total short mkt value <= 8% of equity
TOP_N_SHORTS = 3
SHORT_STOP_LOSS_PCT = 0.15         # cover if price rises 15% against the short

# ---- Additional options legs: long calls (speculative) + short puts (cash-secured) ----
CALL_LONG_OTM_PCT = 0.05                 # long calls bought ~5% OTM
PUT_LONG_OTM_PCT = 0.05                  # speculative long puts bought ~5% OTM
PUT_SHORT_OTM_PCT = 0.10                 # short puts sold ~10% OTM (reduce assignment odds)
LONG_CALL_BUDGET_PCT_OF_EQUITY = 0.02    # total speculative long-call budget
LONG_PUT_BUDGET_PCT_OF_EQUITY = 0.02     # total speculative long-put budget
SHORT_PUT_CASH_RESERVE_PCT = 0.05        # cash reserved for potential assignment
TOP_N_LONG_CALLS = 3
TOP_N_LONG_PUTS = 3
TOP_N_SHORT_PUTS = 3

SAFE_ASSET_CLASSES = {AssetClass.US_EQUITY, AssetClass.US_OPTION}

TRAILING_STOP_PCT = 5.0
REBALANCE_BAND = 0.05
LOOP_SLEEP_SECONDS = 300
WEEKEND_SLEEP_SECONDS = 1800

# Position caps
MAX_ETF_WEIGHT = 0.40       # no single ETF > 40%
MAX_SATELLITE_WEIGHT = 0.15 # no single sector/stock > 15%

# Option liquidity filters
MIN_OPTION_VOLUME = 500
MAX_OPTION_SPREAD_PCT = 0.15  # bid/ask spread <= 15% of mid

# Stock liquidity filters
MIN_STOCK_VOLUME = 500000
MIN_STOCK_MARKET_CAP = 2e9
MIN_STOCK_PRICE = 5.0
MAX_STOCK_SPREAD_PCT = 0.005  # 0.5%

# Trade limit
MAX_TRADES_PER_DAY = 30
TRADES_TODAY = 0
LAST_TRADE_DAY = None

# Volatility throttle
VIX_PAUSE_LEVEL = 40
VIX_SLOW_LEVEL = 30

# Drawdown kill-switch
MAX_POSITION_LOSS_PCT = 0.25   # exit if >25% loss
REDUCE_POSITION_LOSS_PCT = 0.15
PORTFOLIO_PAUSE_DD_PCT = 0.10
PORTFOLIO_EXIT_SATELLITE_DD_PCT = 0.20

# Holding period / turnover
MIN_HOLD_DAYS = 5
MAX_TURNOVER_PCT_PER_REBAL = 0.20

# No trade windows (ET time)
NO_TRADE_BEFORE = (9, 45)   # 9:45 AM ET
NO_TRADE_AFTER = (15, 55)   # 3:55 PM ET

# Correlation cap
MAX_CORRELATION = 0.85

# Eastern timezone used for all market-hours logic below
EASTERN_TZ = ZoneInfo("America/New_York")


# ============================================================
# YFINANCE RETRY WRAPPER (handles rate limiting)
# ============================================================

YF_MAX_RETRIES = 2
YF_BASE_DELAY_SECONDS = 2.5


def yf_call_with_retry(func, *args, max_retries=YF_MAX_RETRIES, base_delay=YF_BASE_DELAY_SECONDS, **kwargs):
    """
    Call a yfinance-backed function with exponential backoff + jitter on
    rate-limit errors. Non-rate-limit errors are raised immediately (no
    point retrying a bad ticker or a genuine network failure the same way).
    """
    last_exc = None
    for attempt in range(max_retries):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            last_exc = e
            msg = str(e)
            is_rate_limit = (
                "Rate limit" in msg
                or "Too Many Requests" in msg
                or "YFRateLimitError" in type(e).__name__
            )
            if not is_rate_limit or attempt == max_retries - 1:
                raise
            delay = base_delay * (2 ** attempt) + random.uniform(0, 1.5)
            log.warning(
                f"yfinance rate limited, retrying in {delay:.1f}s "
                f"(attempt {attempt + 1}/{max_retries})..."
            )
            time.sleep(delay)
    raise last_exc


def yf_download(*args, **kwargs):
    return yf_call_with_retry(yf.download, *args, **kwargs)


def yf_ticker_info(ticker):
    return yf_call_with_retry(lambda: yf.Ticker(ticker).info)


def yf_ticker_news(ticker):
    return yf_call_with_retry(lambda: yf.Ticker(ticker).news)


# ============================================================
# SAFETY GUARD
# ============================================================

def assert_safe_order(symbol: str, asset_class):
    if asset_class not in SAFE_ASSET_CLASSES:
        raise ValueError(f"Blocked order for {symbol}: asset class {asset_class} is not permitted (no forex/crypto).")


# ============================================================
# TIME / TRADE WINDOW HELPERS
# ============================================================

def now_et():
    # Real US/Eastern wall-clock time (handles EST/EDT automatically),
    # regardless of what timezone the server itself runs in.
    return datetime.now(EASTERN_TZ)


def in_no_trade_window():
    t = now_et()
    h, m = t.hour, t.minute
    # Before first 15 minutes after the open
    if (h < NO_TRADE_BEFORE[0]) or (h == NO_TRADE_BEFORE[0] and m < NO_TRADE_BEFORE[1]):
        return True
    # After last 5 minutes before the close
    if (h > NO_TRADE_AFTER[0]) or (h == NO_TRADE_AFTER[0] and m >= NO_TRADE_AFTER[1]):
        return True
    return False


# ============================================================
# TRADE LIMIT / SUBMIT WRAPPER
# ============================================================

def reset_trade_counter_if_new_day():
    global TRADES_TODAY, LAST_TRADE_DAY
    today = now_et().date()
    if LAST_TRADE_DAY != today:
        LAST_TRADE_DAY = today
        TRADES_TODAY = 0
        log.info("New trading day: trade counter reset.")


def can_trade_today() -> bool:
    reset_trade_counter_if_new_day()
    if in_no_trade_window():
        log.info("In no-trade window (open/close). Skipping trades.")
        return False
    return TRADES_TODAY < MAX_TRADES_PER_DAY


def submit_order_safe(trading_client: TradingClient, order_data, label: str = ""):
    global TRADES_TODAY
    reset_trade_counter_if_new_day()
    if TRADES_TODAY >= MAX_TRADES_PER_DAY:
        log.warning(f"Trade limit reached ({MAX_TRADES_PER_DAY} per day). Skipping order: {label}")
        return
    if in_no_trade_window():
        log.info(f"No-trade window active. Skipping order: {label}")
        return
    try:
        trading_client.submit_order(order_data=order_data)
        TRADES_TODAY += 1
        log.info(f"Order submitted ({TRADES_TODAY}/{MAX_TRADES_PER_DAY} today): {label}")
    except Exception as e:
        log.exception(f"Order submission failed for {label}: {e}")


# ============================================================
# NYSE/NASDAQ STOCK UNIVERSE
# ============================================================

def get_nyse_nasdaq_universe(trading_client: TradingClient):
    assets = trading_client.get_all_assets()
    tickers = [
        a.symbol
        for a in assets
        if a.asset_class == AssetClass.US_EQUITY
        and a.status == AssetStatus.ACTIVE
        and a.tradable
        and a.exchange in ("NYSE", "NASDAQ")
    ]
    log.info(f"Raw NYSE/NASDAQ universe size: {len(tickers)}")
    return tickers


def filter_liquid_stocks(tickers):
    liquid = []
    for t in tickers:
        try:
            info = yf_ticker_info(t)
            if (
                info.get("averageVolume", 0) >= MIN_STOCK_VOLUME
                and info.get("marketCap", 0) >= MIN_STOCK_MARKET_CAP
                and info.get("regularMarketPrice", 0) >= MIN_STOCK_PRICE
            ):
                liquid.append(t)
        except Exception as e:
            log.warning(f"Liquidity filter failed for {t}: {e}")
            continue
        time.sleep(0.3)
    log.info(f"Filtered liquid NYSE/NASDAQ stocks: {len(liquid)}")
    return liquid


# ============================================================
# BLACK–SCHOLES PRICER
# ============================================================

def black_scholes_price(
    S: float, K: float, T: float, r: float, sigma: float, option_type: str
) -> float:
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return 0.0

    d1 = (log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)

    if option_type == "call":
        return S * norm.cdf(d1) - K * exp(-r * T) * norm.cdf(d2)
    else:
        return K * exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


# ============================================================
# DATA ENGINE
# ============================================================

class DataEngine:
    def __init__(self, tickers, years=HISTORY_YEARS):
        self.tickers = list(dict.fromkeys(tickers))
        self.years = years
        self.close = pd.DataFrame()
        self.returns = pd.DataFrame()

    def fetch(self):
        log.info(f"Downloading {self.years}y of history for {len(self.tickers)} tickers...")
        raw = yf_download(
            self.tickers, period=f"{self.years}y",
            auto_adjust=True, progress=False, group_by="column", threads=False,
        )
        if isinstance(raw.columns, pd.MultiIndex):
            close = raw["Close"]
        else:
            close = raw[["Close"]]
            close.columns = self.tickers
        self.close = close.dropna(how="all")

        # yf.download() swallows per-ticker failures internally (logs them,
        # returns NaN columns) rather than raising — so the retry wrapper
        # above never sees them. Detect and retry those tickers individually,
        # where a single-ticker download does raise on failure and our
        # backoff wrapper can actually kick in.
        missing = [
            t for t in self.tickers
            if t not in self.close.columns or self.close[t].dropna().empty
        ]
        for t in missing:
            try:
                log.info(f"Retrying failed ticker {t} individually...")
                single_raw = yf_call_with_retry(
                    yf.download, t, period=f"{self.years}y",
                    auto_adjust=True, progress=False, threads=False,
                )
                if isinstance(single_raw.columns, pd.MultiIndex):
                    s_close = single_raw["Close"].iloc[:, 0]
                else:
                    s_close = single_raw["Close"]
                self.close[t] = s_close
            except Exception as e:
                log.warning(f"Individual retry failed for {t}, leaving it out of this cycle: {e}")

        self.close = self.close.dropna(how="all")
        self.returns = self.close.pct_change(fill_method=None).dropna(how="all")
        return self.close, self.returns

    def sharpe_ratios(self):
        rf_daily = RISK_FREE_ANNUAL / TRADING_DAYS
        result = {}
        for ticker in self.tickers:
            if ticker not in self.returns:
                continue
            r = self.returns[ticker].dropna()
            if len(r) < 30 or r.std() == 0:
                continue
            excess = r - rf_daily
            result[ticker] = float((excess.mean() / r.std()) * np.sqrt(TRADING_DAYS))
        return result

    def momentum_12_1(self):
        result = {}
        for ticker in self.tickers:
            if ticker not in self.close:
                continue
            s = self.close[ticker].dropna()
            if len(s) < 260:
                continue
            p_now = s.iloc[-21]
            p_then = s.iloc[-260]
            if p_then > 0:
                result[ticker] = float(p_now / p_then - 1.0)
        return result

    def volatility(self, window=14):
        result = {}
        for ticker in self.tickers:
            if ticker not in self.returns:
                continue
            r = self.returns[ticker].dropna()
            if len(r) < window:
                continue
            result[ticker] = float(r.rolling(window).std().iloc[-1] * np.sqrt(TRADING_DAYS))
        return result

    def covariance_matrix(self, tickers, lookback_days=252):
        sub = self.returns[tickers].dropna().tail(lookback_days)
        return sub.cov() * TRADING_DAYS


def get_underlying_vol(data_engine: DataEngine, underlying: str, default_sigma: float = 0.20) -> float:
    vol = data_engine.volatility().get(underlying)
    return vol if vol and vol > 0 else default_sigma


# ============================================================
# FUNDAMENTAL ANALYZER
# ============================================================

class FundamentalAnalyzer:
    FIELDS_LOWER_IS_BETTER = ["debtToEquity", "trailingPE"]
    FIELDS_HIGHER_IS_BETTER = ["returnOnEquity", "profitMargins", "currentRatio", "earningsGrowth"]

    def __init__(self):
        self._cache = {}

    def _get_info(self, ticker):
        if ticker in self._cache:
            return self._cache[ticker]
        try:
            info = yf_ticker_info(ticker) or {}
        except Exception as e:
            log.warning(f"Fundamentals fetch failed for {ticker}: {e}")
            info = {}
        self._cache[ticker] = info
        return info

    def score(self, ticker):
        info = self._get_info(ticker)
        pts, n = 0.0, 0
        for field in self.FIELDS_HIGHER_IS_BETTER:
            val = info.get(field)
            if val is not None:
                pts += np.tanh(val)
                n += 1
        for field in self.FIELDS_LOWER_IS_BETTER:
            val = info.get(field)
            if val is not None and val > 0:
                pts += np.tanh(1.0 / val)
                n += 1
        return pts / n if n > 0 else 0.0


# ============================================================
# NEWS SENTIMENT
# ============================================================

class NewsSentimentAnalyzer:
    def __init__(self):
        self.vader = SentimentIntensityAnalyzer()

    def score(self, ticker, max_headlines=8):
        try:
            news_items = yf_ticker_news(ticker) or []
        except Exception as e:
            log.warning(f"News fetch failed for {ticker}: {e}")
            return 0.0
        if not news_items:
            return 0.0
        scores = []
        for item in news_items[:max_headlines]:
            title = item.get("title") or item.get("content", {}).get("title", "")
            if not title:
                continue
            scores.append(self.vader.polarity_scores(title)["compound"])
        return float(np.mean(scores)) if scores else 0.0


# ============================================================
# COMPOSITE SCORER
# ============================================================

def zscore(d: dict) -> dict:
    if not d:
        return {}
    vals = np.array(list(d.values()), dtype=float)
    std = vals.std()
    if std == 0:
        return {k: 0.0 for k in d}
    mean = vals.mean()
    return {k: (v - mean) / std for k, v in d.items()}


def composite_scores(data_engine, fundamentals, sentiment, tickers, weights=None):
    weights = weights or {"sharpe": 0.35, "momentum": 0.30, "fundamentals": 0.20, "sentiment": 0.15}
    sharpe = zscore({t: v for t, v in data_engine.sharpe_ratios().items() if t in tickers})
    momentum = zscore({t: v for t, v in data_engine.momentum_12_1().items() if t in tickers})

    fund_raw = {}
    sent_raw = {}
    for t in tickers:
        fund_raw[t] = fundamentals.score(t)
        time.sleep(0.2)
        sent_raw[t] = sentiment.score(t)
        time.sleep(0.2)
    fund = zscore(fund_raw)
    sent = zscore(sent_raw)

    scores = {}
    for t in tickers:
        scores[t] = (
            weights["sharpe"] * sharpe.get(t, 0.0)
            + weights["momentum"] * momentum.get(t, 0.0)
            + weights["fundamentals"] * fund.get(t, 0.0)
            + weights["sentiment"] * sent.get(t, 0.0)
        )
    return scores


def select_satellite_weights(data_engine, scores, sector_tickers, top_n=TOP_N_SATELLITE, total_weight=SATELLITE_SLEEVE_WEIGHT):
    """
    Pick the top-N scoring sector ETFs and size them with the mean-variance
    optimizer (correlation-capped) over the satellite sleeve's weight budget.
    Falls back to equal weighting if covariance data isn't usable.
    """
    ranked = sorted(sector_tickers, key=lambda t: scores.get(t, -1e9), reverse=True)
    top = [t for t in ranked[:top_n] if t in scores]
    if not top:
        return {}

    try:
        cov = data_engine.covariance_matrix(top)
        cov_ok = not cov.isnull().values.any() and cov.shape[0] == len(top)
    except Exception as e:
        log.warning(f"Satellite covariance matrix failed, falling back to equal weight: {e}")
        cov_ok = False

    if cov_ok and len(top) > 1:
        expected_returns = {t: scores[t] for t in top}
        weights = optimize_weights(expected_returns, cov, total_weight, max_weight_per_asset=MAX_SATELLITE_WEIGHT)
    else:
        per = total_weight / len(top)
        weights = {t: min(per, MAX_SATELLITE_WEIGHT) for t in top}

    return weights


# ============================================================
# MEAN-VARIANCE OPTIMIZER WITH CORRELATION CAP
# ============================================================

def enforce_correlation_cap(cov_matrix, tickers, max_corr=MAX_CORRELATION):
    # Simple heuristic: drop assets that are too correlated with already-selected ones
    selected = []
    for t in tickers:
        if not selected:
            selected.append(t)
            continue
        ok = True
        for s in selected:
            var_t = cov_matrix.loc[t, t]
            var_s = cov_matrix.loc[s, s]
            cov_ts = cov_matrix.loc[t, s]
            if var_t <= 0 or var_s <= 0:
                continue
            corr = cov_ts / (sqrt(var_t) * sqrt(var_s))
            if corr > max_corr:
                ok = False
                break
        if ok:
            selected.append(t)
    return selected


def optimize_weights(expected_returns, cov_matrix, total_weight, max_weight_per_asset=0.6):
    tickers = list(expected_returns.keys())
    if not tickers:
        return {}

    # Enforce correlation cap before optimization
    tickers = enforce_correlation_cap(cov_matrix, tickers, MAX_CORRELATION)
    n = len(tickers)
    if n == 0:
        return {}
    if n == 1:
        return {tickers[0]: total_weight}

    mu = np.array([expected_returns[t] for t in tickers])
    cov = cov_matrix.loc[tickers, tickers].values
    rf = RISK_FREE_ANNUAL

    def neg_sharpe(w):
        port_ret = np.dot(w, mu)
        port_vol = np.sqrt(np.dot(w.T, np.dot(cov, w)))
        if port_vol == 0:
            return 1e6
        return -(port_ret - rf) / port_vol

    bounds = [(0.0, max_weight_per_asset) for _ in tickers]
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    x0 = np.array([1.0 / n] * n)

    result = minimize(neg_sharpe, x0, method="SLSQP", bounds=bounds, constraints=constraints)
    raw_weights = result.x if result.success else x0
    return {t: float(w) * total_weight for t, w in zip(tickers, raw_weights)}


# ============================================================
# REGIME / RISK MANAGER + VOLATILITY THROTTLE
# ============================================================

class RiskManager:
    def __init__(self, data_engine):
        self.de = data_engine

    def regime_risk_score(self):
        score_components = []
        px = self.de.close.get(REGIME_BENCHMARK)
        if px is not None and len(px.dropna()) > 200:
            sma200 = px.rolling(200).mean().iloc[-1]
            last = px.iloc[-1]
            score_components.append(0.0 if last > sma200 else 1.0)
            last_year = px.tail(252)
            drawdown = 1.0 - (last / last_year.max())
            score_components.append(float(np.clip(drawdown / 0.20, 0, 1)))
        try:
            vix_hist = yf_download(VIX_TICKER, period="6mo", progress=False, auto_adjust=True)["Close"]
            vix_level = float(vix_hist.iloc[-1])
            score_components.append(float(np.clip((vix_level - 15) / 20, 0, 1)))
        except Exception as e:
            log.warning(f"VIX fetch failed: {e}")
        return float(np.mean(score_components)) if score_components else 0.3

    def current_vix_level(self):
        try:
            vix_hist = yf_download(VIX_TICKER, period="5d", progress=False, auto_adjust=True)["Close"]
            return float(vix_hist.iloc[-1])
        except Exception as e:
            log.warning(f"VIX fetch failed for throttle: {e}")
            return 20.0

    def hedge_weight_breakdown(self, hedge_total_weight):
        vol = self.de.volatility()
        equity_vol = vol.get(REGIME_BENCHMARK, 0.15)
        if equity_vol > 0.22:
            split = {"inverse_equity": 0.45, "long_duration_bonds": 0.30, "gold": 0.25}
        else:
            split = {"inverse_equity": 0.25, "long_duration_bonds": 0.40, "gold": 0.35}
        return {name: hedge_total_weight * pct for name, pct in split.items()}


# ============================================================
# OPTIONS ENGINE
# ============================================================

OCC_SYMBOL_RE = re.compile(r"^([A-Z]{1,6})(\d{6})([CP])(\d{8})$")


def parse_occ_symbol(symbol):
    m = OCC_SYMBOL_RE.match(symbol)
    if not m:
        return None
    root, exp, cp, strike = m.groups()
    return {
        "underlying": root,
        "expiration": datetime.strptime(exp, "%y%m%d").date(),
        "type": "call" if cp == "C" else "put",
        "strike": int(strike) / 1000.0,
    }


def options_position_summary(positions):
    summary = {}
    for p in positions:
        parsed = parse_occ_symbol(p.symbol)
        if not parsed:
            continue
        key = (parsed["underlying"], parsed["type"])
        summary[key] = summary.get(key, 0.0) + float(p.qty)
    return summary


class OptionsEngine:
    def __init__(
        self,
        trading_client: TradingClient,
        option_data_client: OptionHistoricalDataClient,
        data_engine: DataEngine,
    ):
        self.trading = trading_client
        self.data = option_data_client
        self.de = data_engine

    def _find_contracts(self, underlying, contract_type, min_dte=OPTIONS_MIN_DTE, max_dte=OPTIONS_MAX_DTE):
        today = now_et().date()
        req = GetOptionContractsRequest(
            underlying_symbols=[underlying],
            status=AssetStatus.ACTIVE,
            type=contract_type,
            expiration_date_gte=today + timedelta(days=min_dte),
            expiration_date_lte=today + timedelta(days=max_dte),
        )
        try:
            resp = self.trading.get_option_contracts(req)
            return list(resp.option_contracts)
        except Exception as e:
            log.warning(f"Option contract lookup failed for {underlying}: {e}")
            return []

    def _quote(self, symbol):
        try:
            req = OptionLatestQuoteRequest(symbol_or_symbols=symbol)
            q = self.data.get_option_latest_quote(req)[symbol]
            return float(q.bid_price), float(q.ask_price)
        except Exception as e:
            log.warning(f"Option quote failed for {symbol}: {e}")
            return None, None

    def _option_liquidity_ok(self, contract, bid, ask):
        vol = getattr(contract, "volume", None)
        if vol is None or vol < MIN_OPTION_VOLUME:
            log.info(f"Skipping illiquid option {contract.symbol}: volume={vol}")
            return False
        if bid is None or ask is None or bid <= 0 or ask <= 0:
            return False
        mid = (bid + ask) / 2.0
        spread = ask - bid
        if mid <= 0:
            return False
        spread_pct = spread / mid
        if spread_pct > MAX_OPTION_SPREAD_PCT:
            log.info(
                f"Skipping wide-spread option {contract.symbol}: "
                f"spread_pct={spread_pct:.2%} > {MAX_OPTION_SPREAD_PCT:.2%}"
            )
            return False
        return True

    def _closest_strike(self, contracts, target_price):
        if not contracts:
            return None
        contracts_sorted = sorted(
            contracts, key=lambda c: (abs(float(c.strike_price) - target_price), c.expiration_date)
        )
        return contracts_sorted[0]

    def sell_covered_calls(self, underlying, shares_held, existing_short_call_qty, spot_price):
        max_contracts = int(shares_held // 100)
        to_sell = max_contracts - existing_short_call_qty
        if to_sell <= 0 or spot_price <= 0:
            return
        candidates = self._find_contracts(underlying, ContractType.CALL)
        contract = self._closest_strike(candidates, spot_price * (1 + CALL_OTM_PCT))
        if not contract:
            return
        assert_safe_order(contract.symbol, AssetClass.US_OPTION)

        T = (contract.expiration_date - now_et().date()).days / TRADING_DAYS
        sigma = get_underlying_vol(self.de, underlying)
        bs_price = black_scholes_price(
            S=spot_price,
            K=float(contract.strike_price),
            T=T,
            r=RISK_FREE_ANNUAL,
            sigma=sigma,
            option_type="call",
        )

        bid, ask = self._quote(contract.symbol)
        use_price = bid if bid and bid > 0 else bs_price
        if not self._option_liquidity_ok(contract, bid, ask):
            return
        if use_price <= 0:
            return

        order = LimitOrderRequest(
            symbol=contract.symbol,
            qty=to_sell,
            side=OrderSide.SELL,
            type=OrderType.LIMIT,
            time_in_force=TimeInForce.DAY,
            limit_price=round(use_price, 2),
        )
        submit_order_safe(self.trading, order, label=f"SELL covered calls {contract.symbol}")
        log.info(
            f"SELL {to_sell} covered call(s) {contract.symbol} @ {use_price:.2f} "
            f"(BS={bs_price:.2f}, income sleeve)"
        )

    def buy_protective_puts(self, underlying, shares_to_hedge, existing_long_put_qty, spot_price, dollar_budget):
        max_contracts = int(shares_to_hedge // 100)
        to_buy = max_contracts - existing_long_put_qty
        if to_buy <= 0 or spot_price <= 0 or dollar_budget <= 0:
            return
        candidates = self._find_contracts(underlying, ContractType.PUT)
        contract = self._closest_strike(candidates, spot_price * (1 - PUT_OTM_PCT))
        if not contract:
            return
        assert_safe_order(contract.symbol, AssetClass.US_OPTION)

        T = (contract.expiration_date - now_et().date()).days / TRADING_DAYS
        sigma = get_underlying_vol(self.de, underlying)
        bs_price = black_scholes_price(
            S=spot_price,
            K=float(contract.strike_price),
            T=T,
            r=RISK_FREE_ANNUAL,
            sigma=sigma,
            option_type="put",
        )

        bid, ask = self._quote(contract.symbol)
        use_price = ask if ask and ask > 0 else bs_price
        if not self._option_liquidity_ok(contract, bid, ask):
            return
        if use_price <= 0:
            return

        affordable = int(dollar_budget // (use_price * 100))
        qty = max(0, min(to_buy, affordable))
        if qty <= 0:
            return

        order = LimitOrderRequest(
            symbol=contract.symbol,
            qty=qty,
            side=OrderSide.BUY,
            type=OrderType.LIMIT,
            time_in_force=TimeInForce.DAY,
            limit_price=round(use_price, 2),
        )
        submit_order_safe(self.trading, order, label=f"BUY protective puts {contract.symbol}")
        log.info(
            f"BUY {qty} protective put(s) {contract.symbol} @ {use_price:.2f} "
            f"(BS={bs_price:.2f}, hedge sleeve)"
        )

    def buy_long_calls(self, underlying, spot_price, dollar_budget, existing_long_call_qty=0):
        """
        Speculative directional long calls. Sized purely off dollar_budget —
        max loss is the premium paid, so unlike covered calls / protective
        puts this doesn't need to be linked to an existing share count.
        """
        if spot_price <= 0 or dollar_budget <= 0:
            return
        candidates = self._find_contracts(underlying, ContractType.CALL)
        contract = self._closest_strike(candidates, spot_price * (1 + CALL_LONG_OTM_PCT))
        if not contract:
            return
        assert_safe_order(contract.symbol, AssetClass.US_OPTION)

        T = (contract.expiration_date - now_et().date()).days / TRADING_DAYS
        sigma = get_underlying_vol(self.de, underlying)
        bs_price = black_scholes_price(
            S=spot_price, K=float(contract.strike_price), T=T,
            r=RISK_FREE_ANNUAL, sigma=sigma, option_type="call",
        )

        bid, ask = self._quote(contract.symbol)
        use_price = ask if ask and ask > 0 else bs_price
        if not self._option_liquidity_ok(contract, bid, ask):
            return
        if use_price <= 0:
            return

        qty = int(dollar_budget // (use_price * 100))
        if qty <= 0:
            return

        order = LimitOrderRequest(
            symbol=contract.symbol, qty=qty, side=OrderSide.BUY, type=OrderType.LIMIT,
            time_in_force=TimeInForce.DAY, limit_price=round(use_price, 2),
        )
        submit_order_safe(self.trading, order, label=f"BUY long calls {contract.symbol}")
        log.info(
            f"BUY {qty} long call(s) {contract.symbol} @ {use_price:.2f} "
            f"(BS={bs_price:.2f}, speculative)"
        )

    def buy_speculative_puts(self, underlying, spot_price, dollar_budget, existing_long_put_qty=0):
        """
        Speculative long puts — a directional bearish bet, distinct from
        buy_protective_puts() which hedges shares actually held. This is
        sized purely off dollar_budget (max loss = premium paid), used on
        names with a negative composite score rather than names in the
        long book.
        """
        if spot_price <= 0 or dollar_budget <= 0:
            return
        candidates = self._find_contracts(underlying, ContractType.PUT)
        contract = self._closest_strike(candidates, spot_price * (1 - PUT_LONG_OTM_PCT))
        if not contract:
            return
        assert_safe_order(contract.symbol, AssetClass.US_OPTION)

        T = (contract.expiration_date - now_et().date()).days / TRADING_DAYS
        sigma = get_underlying_vol(self.de, underlying)
        bs_price = black_scholes_price(
            S=spot_price, K=float(contract.strike_price), T=T,
            r=RISK_FREE_ANNUAL, sigma=sigma, option_type="put",
        )

        bid, ask = self._quote(contract.symbol)
        use_price = ask if ask and ask > 0 else bs_price
        if not self._option_liquidity_ok(contract, bid, ask):
            return
        if use_price <= 0:
            return

        qty = int(dollar_budget // (use_price * 100))
        if qty <= 0:
            return

        order = LimitOrderRequest(
            symbol=contract.symbol, qty=qty, side=OrderSide.BUY, type=OrderType.LIMIT,
            time_in_force=TimeInForce.DAY, limit_price=round(use_price, 2),
        )
        submit_order_safe(self.trading, order, label=f"BUY speculative puts {contract.symbol}")
        log.info(
            f"BUY {qty} speculative put(s) {contract.symbol} @ {use_price:.2f} "
            f"(BS={bs_price:.2f}, bearish alpha)"
        )

    def sell_short_puts(self, underlying, spot_price, cash_reserve_available, existing_short_put_qty=0):
        """
        Cash-secured short puts only. Sized so that qty * strike * 100 never
        exceeds cash_reserve_available — i.e. capped at what could actually
        be paid in cash if assigned, not margin. This deliberately avoids
        naked/undefined downside. Assignment still means being forced to buy
        the stock at the strike, so PUT_SHORT_OTM_PCT is kept wide enough
        that this is a deliberate, acceptable outcome rather than a surprise.
        """
        if spot_price <= 0 or cash_reserve_available <= 0:
            return
        candidates = self._find_contracts(underlying, ContractType.PUT)
        contract = self._closest_strike(candidates, spot_price * (1 - PUT_SHORT_OTM_PCT))
        if not contract:
            return
        assert_safe_order(contract.symbol, AssetClass.US_OPTION)

        strike = float(contract.strike_price)
        T = (contract.expiration_date - now_et().date()).days / TRADING_DAYS
        sigma = get_underlying_vol(self.de, underlying)
        bs_price = black_scholes_price(
            S=spot_price, K=strike, T=T,
            r=RISK_FREE_ANNUAL, sigma=sigma, option_type="put",
        )

        bid, ask = self._quote(contract.symbol)
        use_price = bid if bid and bid > 0 else bs_price
        if not self._option_liquidity_ok(contract, bid, ask):
            return
        if use_price <= 0 or strike <= 0:
            return

        max_assignable_qty = int(cash_reserve_available // (strike * 100))
        qty = max_assignable_qty
        if qty <= 0:
            return

        order = LimitOrderRequest(
            symbol=contract.symbol, qty=qty, side=OrderSide.SELL, type=OrderType.LIMIT,
            time_in_force=TimeInForce.DAY, limit_price=round(use_price, 2),
        )
        submit_order_safe(self.trading, order, label=f"SELL short puts {contract.symbol}")
        log.info(
            f"SELL {qty} short put(s) {contract.symbol} @ {use_price:.2f} "
            f"(BS={bs_price:.2f}, cash-secured, strike={strike})"
        )


# ============================================================
# EXECUTION LAYER
# ============================================================

def get_price(stock_data_client, ticker):
    req = StockLatestTradeRequest(symbol_or_symbols=ticker)
    trade = stock_data_client.get_stock_latest_trade(req)[ticker]
    return float(trade.price)


def stock_liquidity_ok(ticker):
    """
    Returns True if the spread check actually passed, or if liquidity data
    couldn't be fetched at all (e.g. yfinance rate limited) — in which case
    we proceed but log it clearly, rather than silently blocking a trade on
    a data-availability problem rather than a real liquidity problem. Only
    returns False when we actually got quote data and it failed the spread
    test.
    """
    try:
        info = yf_ticker_info(ticker)
    except Exception as e:
        log.warning(
            f"Liquidity data unavailable for {ticker} ({e}); "
            f"proceeding without spread check for this cycle."
        )
        return True

    bid = info.get("bid", None)
    ask = info.get("ask", None)
    if bid is None or ask is None or bid <= 0 or ask <= 0:
        log.warning(
            f"No usable bid/ask for {ticker}; proceeding without spread check for this cycle."
        )
        return True
    mid = (bid + ask) / 2.0
    spread = ask - bid
    if mid <= 0:
        return True
    spread_pct = spread / mid
    if spread_pct > MAX_STOCK_SPREAD_PCT:
        log.info(f"Skipping illiquid/wide-spread stock {ticker}: spread_pct={spread_pct:.2%}")
        return False
    return True


def rebalance_to_target(
    trading_client,
    stock_data_client,
    positions,
    total_equity,
    cash_available,
    ticker,
    target_weight,
    is_core=False,
):
    if is_core:
        target_weight = min(target_weight, MAX_ETF_WEIGHT)
    else:
        target_weight = min(target_weight, MAX_SATELLITE_WEIGHT)

    target_value = total_equity * target_weight
    current_value = float(positions[ticker].market_value) if ticker in positions else 0.0
    if target_value <= 0:
        return
    drift = abs(current_value - target_value) / target_value if target_value else 1.0
    if drift < REBALANCE_BAND or current_value >= target_value:
        return
    try:
        price = get_price(stock_data_client, ticker)
    except Exception as e:
        log.warning(f"Price fetch failed for {ticker}: {e}")
        return
    if price <= 0 or cash_available < price:
        return
    if not is_core and not stock_liquidity_ok(ticker):
        return
    qty = int((target_value - current_value) // price)
    if qty > 0 and can_trade_today():
        assert_safe_order(ticker, AssetClass.US_EQUITY)
        order = MarketOrderRequest(symbol=ticker, qty=qty, side=OrderSide.BUY, time_in_force=TimeInForce.GTC)
        submit_order_safe(trading_client, order, label=f"BUY {ticker} toward {target_weight:.3f}")
        log.info(f"BUY {qty} {ticker} toward target weight {target_weight:.3f}")


def rebalance_short_to_target(
    trading_client,
    stock_data_client,
    positions,
    total_equity,
    ticker,
    target_weight,
):
    """
    Opens/increases a short equity position toward target_weight (a positive
    weight representing dollar short exposure). This only ever *sells to
    open/add to* a short — it never touches a ticker already held long
    elsewhere in the portfolio, so it can't accidentally liquidate a long
    position instead of opening a short.
    """
    target_weight = min(target_weight, MAX_SHORT_POSITION_WEIGHT)
    target_value = total_equity * target_weight
    if target_value <= 0:
        return

    pos = positions.get(ticker)
    if pos is not None and getattr(pos, "side", None) != "short":
        # Already held long here — never short a name we're also long.
        return

    current_short_value = abs(float(pos.market_value)) if pos is not None else 0.0
    drift = abs(current_short_value - target_value) / target_value if target_value else 1.0
    if drift < REBALANCE_BAND or current_short_value >= target_value:
        return

    try:
        price = get_price(stock_data_client, ticker)
    except Exception as e:
        log.warning(f"Price fetch failed for {ticker} (short): {e}")
        return
    if price <= 0:
        return
    if not stock_liquidity_ok(ticker):
        return

    qty = int((target_value - current_short_value) // price)
    if qty > 0 and can_trade_today():
        assert_safe_order(ticker, AssetClass.US_EQUITY)
        order = MarketOrderRequest(symbol=ticker, qty=qty, side=OrderSide.SELL, time_in_force=TimeInForce.GTC)
        submit_order_safe(trading_client, order, label=f"SHORT {ticker} toward {target_weight:.3f}")
        log.info(f"SELL SHORT {qty} {ticker} toward target short weight {target_weight:.3f}")


def apply_trailing_stops(trading_client, positions):
    for symbol, pos in list(positions.items()):
        if pos.asset_class != AssetClass.US_EQUITY:
            continue
        if getattr(pos, "side", None) == "short":
            # A trailing *sell* stop is backwards for a short position (it
            # would add to the short, not protect it). Shorts get their own
            # buy-side stop in apply_short_stop_losses() instead.
            continue
        open_orders = trading_client.get_orders(
            GetOrdersRequest(status=QueryOrderStatus.OPEN, symbols=[symbol])
        )
        for order in open_orders:
            if order.side == OrderSide.SELL:
                try:
                    trading_client.cancel_order_by_id(order.id)
                except Exception as e:
                    log.warning(f"Failed to cancel order {order.id} for {symbol}: {e}")
        if can_trade_today():
            order = TrailingStopOrderRequest(
                symbol=symbol, qty=abs(float(pos.qty)), side=OrderSide.SELL,
                time_in_force=TimeInForce.GTC, trail_percent=TRAILING_STOP_PCT,
            )
            submit_order_safe(trading_client, order, label=f"Trailing stop {symbol}")


def check_options_exposure(positions, total_equity):
    options_equity = sum(
        float(p.market_value) for p in positions.values() if p.asset_class == AssetClass.US_OPTION
    )
    pct = (options_equity / total_equity) * 100 if total_equity else 0
    if pct > MAX_OPTIONS_PCT_OF_EQUITY:
        log.warning(f"Options exposure {pct:.1f}% exceeds {MAX_OPTIONS_PCT_OF_EQUITY}% cap. Blocking new options orders.")
    return pct


def check_short_exposure(positions, total_equity):
    short_equity = sum(
        abs(float(p.market_value))
        for p in positions.values()
        if p.asset_class == AssetClass.US_EQUITY and getattr(p, "side", None) == "short"
    )
    pct = (short_equity / total_equity) * 100 if total_equity else 0
    if pct > MAX_SHORT_EXPOSURE_PCT:
        log.warning(f"Short exposure {pct:.1f}% exceeds {MAX_SHORT_EXPOSURE_PCT}% cap. Blocking new short orders.")
    return pct


# ============================================================
# KILL-SWITCHES: POSITION LOSS + PORTFOLIO DRAWDOWN
# ============================================================

def check_position_losses(trading_client, positions):
    for symbol, p in positions.items():
        if p.asset_class != AssetClass.US_EQUITY:
            continue
        try:
            cost_basis = float(p.avg_entry_price)
            market_price = float(p.market_value) / abs(float(p.qty)) if p.qty != 0 else cost_basis
            if cost_basis <= 0:
                continue
            loss_pct = (cost_basis - market_price) / cost_basis
            if loss_pct >= MAX_POSITION_LOSS_PCT and can_trade_today():
                order = MarketOrderRequest(
                    symbol=symbol,
                    qty=abs(float(p.qty)),
                    side=OrderSide.SELL,
                    time_in_force=TimeInForce.GTC,
                )
                submit_order_safe(trading_client, order, label=f"EXIT {symbol} due to loss {loss_pct:.2%}")
                log.warning(f"EXIT {symbol} due to large loss {loss_pct:.2%}")
            elif loss_pct >= REDUCE_POSITION_LOSS_PCT and can_trade_today():
                reduce_qty = int(abs(float(p.qty)) * 0.5)
                if reduce_qty > 0:
                    order = MarketOrderRequest(
                        symbol=symbol,
                        qty=reduce_qty,
                        side=OrderSide.SELL,
                        time_in_force=TimeInForce.GTC,
                    )
                    submit_order_safe(trading_client, order, label=f"REDUCE {symbol} due to loss {loss_pct:.2%}")
                    log.warning(f"REDUCE {symbol} due to moderate loss {loss_pct:.2%}")
        except Exception as e:
            log.warning(f"Position loss check failed for {symbol}: {e}")


def apply_short_stop_losses(trading_client, positions):
    """
    Mirror of check_position_losses for the short side: a short loses money
    when price rises, so cover (buy to close) once the adverse move exceeds
    SHORT_STOP_LOSS_PCT. Long positions are untouched here — they're already
    handled by check_position_losses.
    """
    for symbol, p in positions.items():
        if p.asset_class != AssetClass.US_EQUITY or getattr(p, "side", None) != "short":
            continue
        try:
            cost_basis = float(p.avg_entry_price)
            qty = abs(float(p.qty))
            if cost_basis <= 0 or qty <= 0:
                continue
            market_price = abs(float(p.market_value)) / qty
            adverse_move_pct = (market_price - cost_basis) / cost_basis
            if adverse_move_pct >= SHORT_STOP_LOSS_PCT and can_trade_today():
                order = MarketOrderRequest(
                    symbol=symbol,
                    qty=qty,
                    side=OrderSide.BUY,
                    time_in_force=TimeInForce.GTC,
                )
                submit_order_safe(
                    trading_client, order,
                    label=f"COVER SHORT {symbol} due to adverse move {adverse_move_pct:.2%}",
                )
                log.warning(f"COVER SHORT {symbol}: price up {adverse_move_pct:.2%} against short position")
        except Exception as e:
            log.warning(f"Short stop-loss check failed for {symbol}: {e}")


def check_portfolio_drawdown(account, initial_equity):
    """
    Simple portfolio drawdown kill-switch.
    initial_equity can be set to the starting equity for the session.
    """
    try:
        current_equity = float(account.equity)
        if initial_equity <= 0:
            return
        dd_pct = (initial_equity - current_equity) / initial_equity
        if dd_pct >= PORTFOLIO_EXIT_SATELLITE_DD_PCT:
            log.warning(f"Portfolio drawdown {dd_pct:.2%} exceeds satellite exit threshold.")
            # Here you could exit satellite positions, etc.
        elif dd_pct >= PORTFOLIO_PAUSE_DD_PCT:
            log.warning(f"Portfolio drawdown {dd_pct:.2%} exceeds pause threshold.")
            # You could set a global pause flag, etc.
    except Exception as e:
        log.warning(f"Portfolio drawdown check failed: {e}")


# ============================================================
# DAILY DATA CACHE (avoids re-hitting yfinance every 5-min loop)
# ============================================================
# Historical prices, fundamentals, and news sentiment don't meaningfully
# change minute to minute. Re-downloading 12y of history + per-ticker
# fundamentals/news on every 5-minute loop hammers Yahoo Finance and
# reliably trips its rate limiter. Instead, fetch this bundle once per
# trading day and reuse it for every loop iteration that day. Only live
# prices (fetched from Alpaca, not Yahoo) need to be fresh every loop.
_DATA_CACHE = {
    "date": None,
    "de": None,
    "fundamentals": None,
    "sentiment": None,
    "scores": None,
    "regime_score": None,
    "hedge_weights": None,
    "liquid_stocks": None,
}


def get_daily_market_data(trading_client, base_tickers):
    today = now_et().date()
    if _DATA_CACHE["date"] == today and _DATA_CACHE["de"] is not None:
        log.info("Using cached market data for today (fetched earlier this trading day).")
        return (
            _DATA_CACHE["de"],
            _DATA_CACHE["fundamentals"],
            _DATA_CACHE["sentiment"],
            _DATA_CACHE["scores"],
            _DATA_CACHE["regime_score"],
            _DATA_CACHE["hedge_weights"],
            _DATA_CACHE["liquid_stocks"],
        )

    log.info("Refreshing daily market data cache (history, fundamentals, sentiment, regime, universe)...")

    # Pull + liquidity-filter the NYSE/NASDAQ stock universe once per day.
    # This is what actually lets individual equities into the satellite
    # sleeve (previously this scan was disabled and the bot only ever
    # considered ETFs). It's a slow, sequential yfinance scan, so it's
    # capped and only run once per trading day via this cache.
    try:
        raw_universe = get_nyse_nasdaq_universe(trading_client)
        liquid_stocks = filter_liquid_stocks(raw_universe)
    except Exception as e:
        log.warning(f"Equity universe scan failed, continuing with ETFs only: {e}")
        liquid_stocks = []

    tickers = list(dict.fromkeys(base_tickers + liquid_stocks))

    de = DataEngine(tickers)
    de.fetch()
    fundamentals = FundamentalAnalyzer()
    sentiment = NewsSentimentAnalyzer()
    scores = composite_scores(de, fundamentals, sentiment, tickers)

    rm = RiskManager(de)
    regime_score = rm.regime_risk_score()
    hedge_weights = rm.hedge_weight_breakdown(HEDGE_SLEEVE_WEIGHT)

    _DATA_CACHE.update({
        "date": today,
        "de": de,
        "fundamentals": fundamentals,
        "sentiment": sentiment,
        "scores": scores,
        "regime_score": regime_score,
        "hedge_weights": hedge_weights,
        "liquid_stocks": liquid_stocks,
    })
    return de, fundamentals, sentiment, scores, regime_score, hedge_weights, liquid_stocks


def select_stock_weights(data_engine, scores, stock_tickers, top_n=TOP_N_STOCKS, total_weight=0.0):
    """
    Same approach as select_satellite_weights but over the individual-equity
    universe, sized within its own carve-out of the satellite sleeve so it
    never crowds out the sector ETFs.
    """
    if total_weight <= 0 or not stock_tickers:
        return {}
    ranked = sorted(stock_tickers, key=lambda t: scores.get(t, -1e9), reverse=True)
    top = [t for t in ranked[:top_n] if t in scores and scores[t] > 0]
    if not top:
        return {}

    try:
        cov = data_engine.covariance_matrix(top)
        cov_ok = not cov.isnull().values.any() and cov.shape[0] == len(top)
    except Exception as e:
        log.warning(f"Stock-sleeve covariance matrix failed, falling back to equal weight: {e}")
        cov_ok = False

    if cov_ok and len(top) > 1:
        expected_returns = {t: scores[t] for t in top}
        weights = optimize_weights(expected_returns, cov, total_weight, max_weight_per_asset=MAX_SATELLITE_WEIGHT)
    else:
        per = total_weight / len(top)
        weights = {t: min(per, MAX_SATELLITE_WEIGHT) for t in top}

    return weights


def select_short_weights(scores, candidate_tickers, exclude_tickers, top_n=TOP_N_SHORTS, total_weight=SHORT_SLEEVE_WEIGHT):
    """
    Picks the worst-scoring names (excluding anything already held long
    elsewhere) to short. Equal-weighted and capped per name rather than
    mean-variance optimized — this is a smaller, higher-risk sleeve that
    doesn't need the same treatment as the long sleeves.
    """
    eligible = [t for t in candidate_tickers if t not in exclude_tickers]
    ranked = sorted(eligible, key=lambda t: scores.get(t, 1e9))  # ascending: worst first
    worst = [t for t in ranked[:top_n] if t in scores and scores[t] < 0]
    if not worst:
        return {}
    per = total_weight / len(worst)
    return {t: min(per, MAX_SHORT_POSITION_WEIGHT) for t in worst}


# ============================================================
# MAIN TRADING LOOP (BACKGROUND THREAD)
# ============================================================

def trading_loop():
    log.info("Trading loop started.")
    backoff_seconds = 30

    # Read Alpaca credentials from environment
    api_key = os.environ.get("ALPACA_API_KEY")
    api_secret = os.environ.get("ALPACA_API_SECRET")
    paper = os.environ.get("ALPACA_PAPER", "true").lower() == "true"

    if not api_key or not api_secret:
        log.error("Missing ALPACA_API_KEY or ALPACA_API_SECRET in environment.")
        return

    trading_client = TradingClient(api_key, api_secret, paper=paper)
    stock_data_client = StockHistoricalDataClient(api_key, api_secret)
    option_data_client = OptionHistoricalDataClient(api_key, api_secret)

    # Capture initial equity for drawdown checks
    try:
        initial_account = trading_client.get_account()
        initial_equity = float(initial_account.equity)
    except Exception as e:
        log.warning(f"Failed to fetch initial account equity: {e}")
        initial_equity = 0.0

    while True:
        start_ts = time.time()
        try:
            # Account + positions
            account = trading_client.get_account()
            positions_list = trading_client.get_all_positions()
            positions = {p.symbol: p for p in positions_list}
            total_equity = float(account.equity)
            cash_available = float(account.cash)

            # Universe: core ETFs + sector ETFs, plus the liquid NYSE/NASDAQ
            # equity universe (fetched/filtered once per day inside
            # get_daily_market_data). This is what lets individual stocks
            # into the satellite sleeve alongside the sector ETFs.
            base_tickers = list(CORE_ETFS.keys()) + SECTOR_TICKERS

            # Data + scores (cached once per trading day to avoid yfinance rate limits)
            de, fundamentals, sentiment, scores, regime_score, hedge_weights, liquid_stocks = (
                get_daily_market_data(trading_client, base_tickers)
            )

            log.info(
                f"Regime score: {regime_score:.3f}, hedge weights: {hedge_weights}, "
                f"liquid equity universe size: {len(liquid_stocks)}"
            )

            # Rebalance core ETFs
            for ticker, w in CORE_ETFS.items():
                rebalance_to_target(
                    trading_client,
                    stock_data_client,
                    positions,
                    total_equity,
                    cash_available,
                    ticker,
                    w * CORE_TOTAL_WEIGHT,
                    is_core=True,
                )

            # Satellite sleeve is split between sector ETFs and individual
            # equities, so single-name stocks can now sit alongside the
            # sector bets and give the hedge sleeve real diversification
            # to work with instead of ETFs-only.
            stock_sleeve_budget = SATELLITE_SLEEVE_WEIGHT * STOCK_SLEEVE_SHARE_OF_SATELLITE
            sector_sleeve_budget = SATELLITE_SLEEVE_WEIGHT - stock_sleeve_budget

            satellite_weights = select_satellite_weights(
                de, scores, SECTOR_TICKERS, total_weight=sector_sleeve_budget
            )
            log.info(f"Satellite (sector ETF) weights: {satellite_weights}")
            for ticker, w in satellite_weights.items():
                rebalance_to_target(
                    trading_client,
                    stock_data_client,
                    positions,
                    total_equity,
                    cash_available,
                    ticker,
                    w,
                    is_core=False,
                )

            stock_weights = select_stock_weights(
                de, scores, liquid_stocks, total_weight=stock_sleeve_budget
            )
            log.info(f"Satellite (individual equity) weights: {stock_weights}")
            for ticker, w in stock_weights.items():
                rebalance_to_target(
                    trading_client,
                    stock_data_client,
                    positions,
                    total_equity,
                    cash_available,
                    ticker,
                    w,
                    is_core=False,
                )

            # ---- Short equity sleeve ----
            # Separately-capped short book, drawn from the worst-scoring
            # names in the liquid equity universe, excluding anything held
            # long elsewhere in the portfolio — this never shorts a name
            # the long sleeves are also buying. Sized by the same regime
            # score that drives TLT/GLD/SH, so it actually functions as a
            # hedge (more short exposure when the regime is bearish, less
            # when it's bullish) instead of sitting at a flat weight.
            long_held_tickers = {s for s, p in positions.items() if getattr(p, "side", None) != "short"}
            short_universe = [t for t in liquid_stocks if t not in long_held_tickers]
            short_exposure_pct = check_short_exposure(positions, total_equity)
            if short_exposure_pct >= MAX_SHORT_EXPOSURE_PCT:
                log.warning("Short exposure at/above cap; skipping new short orders this cycle.")
            else:
                # regime_score is 0 (calm/bullish) .. 1 (stressed/bearish).
                # Map that onto [MIN_SHORT_SLEEVE_MULTIPLIER, MAX_SHORT_SLEEVE_MULTIPLIER]
                # around the base SHORT_SLEEVE_WEIGHT.
                short_regime_multiplier = MIN_SHORT_SLEEVE_MULTIPLIER + regime_score * (
                    MAX_SHORT_SLEEVE_MULTIPLIER - MIN_SHORT_SLEEVE_MULTIPLIER
                )
                short_sleeve_budget = SHORT_SLEEVE_WEIGHT * short_regime_multiplier
                log.info(
                    f"Short sleeve budget: {short_sleeve_budget:.3f} "
                    f"(base {SHORT_SLEEVE_WEIGHT}, regime multiplier {short_regime_multiplier:.2f})"
                )
                short_weights = select_short_weights(
                    scores, short_universe, long_held_tickers, total_weight=short_sleeve_budget
                )
                log.info(f"Short equity weights: {short_weights}")
                for ticker, w in short_weights.items():
                    rebalance_short_to_target(
                        trading_client, stock_data_client, positions, total_equity, ticker, w,
                    )

            # Hedge sleeve — inverse equity / long bonds / gold, sized by regime.
            for name, w in hedge_weights.items():
                symbol = HEDGE_INSTRUMENTS.get(name)
                if not symbol:
                    continue
                w_clamped = float(np.clip(w, MIN_HEDGE_WEIGHT, MAX_HEDGE_WEIGHT))
                rebalance_to_target(
                    trading_client,
                    stock_data_client,
                    positions,
                    total_equity,
                    cash_available,
                    symbol,
                    w_clamped,
                    is_core=False,
                )

            # Options overlay — income (covered calls) + protection (protective
            # puts). Puts are sized off the hedge sleeve's dollar budget so the
            # options book stays inside MAX_OPTIONS_PCT_OF_EQUITY.
            if OPTIONS_ENABLED:
                options_pct = check_options_exposure(positions, total_equity)
                opt_engine = OptionsEngine(trading_client, option_data_client, de)
                opt_summary = options_position_summary(positions_list)

                if options_pct >= MAX_OPTIONS_PCT_OF_EQUITY:
                    log.warning("Options exposure at/above cap; skipping new options orders this cycle.")
                else:
                    # Covered calls on any equity or ETF position (core, sector,
                    # or individual stock) that holds at least one full lot of
                    # 100 shares — previously this was hard-restricted to
                    # CORE_ETFS only, which meant it almost never fired unless
                    # you held >=100 shares of VOO/VXUS/BND specifically.
                    for symbol, pos in positions.items():
                        if pos.asset_class != AssetClass.US_EQUITY:
                            continue
                        if float(pos.qty) < COVERED_CALL_MIN_SHARES:
                            continue
                        try:
                            spot = get_price(stock_data_client, symbol)
                        except Exception as e:
                            log.warning(f"Price fetch failed for {symbol} (covered call): {e}")
                            continue
                        existing_short_calls = int(opt_summary.get((symbol, "call"), 0))
                        opt_engine.sell_covered_calls(
                            underlying=symbol,
                            shares_held=float(pos.qty),
                            existing_short_call_qty=existing_short_calls,
                            spot_price=spot,
                        )

                    # Protective puts — hedge the equity/ETF book using the
                    # hedge sleeve's dollar budget. Previously implemented on
                    # OptionsEngine but never called from the loop, so no
                    # downside protection was ever actually bought.
                    hedge_dollar_budget = total_equity * HEDGE_SLEEVE_WEIGHT * PUT_HEDGE_SHARE
                    equity_positions = [
                        (s, p) for s, p in positions.items()
                        if p.asset_class == AssetClass.US_EQUITY and float(p.qty) >= COVERED_CALL_MIN_SHARES
                    ]
                    if equity_positions and hedge_dollar_budget > 0:
                        # Split the put budget evenly across eligible names so
                        # no single underlying eats the whole hedge budget.
                        per_name_budget = hedge_dollar_budget / len(equity_positions)
                        for symbol, pos in equity_positions:
                            try:
                                spot = get_price(stock_data_client, symbol)
                            except Exception as e:
                                log.warning(f"Price fetch failed for {symbol} (protective put): {e}")
                                continue
                            existing_long_puts = int(opt_summary.get((symbol, "put"), 0))
                            opt_engine.buy_protective_puts(
                                underlying=symbol,
                                shares_to_hedge=float(pos.qty),
                                existing_long_put_qty=existing_long_puts,
                                spot_price=spot,
                                dollar_budget=per_name_budget,
                            )

                    # Speculative long calls — best-scoring names not
                    # already core holdings, sized off a small dedicated
                    # dollar budget (max loss = premium paid).
                    long_call_budget_total = total_equity * LONG_CALL_BUDGET_PCT_OF_EQUITY
                    call_candidates = [
                        t for t in (list(CORE_ETFS.keys()) + liquid_stocks)
                        if scores.get(t, -1e9) > 0
                    ]
                    ranked_calls = sorted(
                        call_candidates, key=lambda t: scores.get(t, -1e9), reverse=True
                    )[:TOP_N_LONG_CALLS]
                    if ranked_calls and long_call_budget_total > 0:
                        per_name_call_budget = long_call_budget_total / len(ranked_calls)
                        for symbol in ranked_calls:
                            try:
                                spot = get_price(stock_data_client, symbol)
                            except Exception as e:
                                log.warning(f"Price fetch failed for {symbol} (long call): {e}")
                                continue
                            existing_long_calls = int(opt_summary.get((symbol, "call"), 0))
                            opt_engine.buy_long_calls(
                                underlying=symbol,
                                spot_price=spot,
                                dollar_budget=per_name_call_budget,
                                existing_long_call_qty=existing_long_calls,
                            )

                    # Speculative long puts — mirror of the long-calls block
                    # above, but expressing bearish conviction on the
                    # worst-scoring names (the same candidate list driving
                    # the short-equity sleeve). Distinct from
                    # buy_protective_puts(), which only hedges shares
                    # actually held — this is a standalone directional bet.
                    long_put_budget_total = total_equity * LONG_PUT_BUDGET_PCT_OF_EQUITY
                    put_speculative_candidates = [
                        t for t in short_universe if scores.get(t, 1e9) < 0
                    ]
                    ranked_puts = sorted(
                        put_speculative_candidates, key=lambda t: scores.get(t, 1e9)
                    )[:TOP_N_LONG_PUTS]
                    if ranked_puts and long_put_budget_total > 0:
                        per_name_put_budget = long_put_budget_total / len(ranked_puts)
                        for symbol in ranked_puts:
                            try:
                                spot = get_price(stock_data_client, symbol)
                            except Exception as e:
                                log.warning(f"Price fetch failed for {symbol} (speculative put): {e}")
                                continue
                            existing_spec_puts = int(opt_summary.get((symbol, "put"), 0))
                            opt_engine.buy_speculative_puts(
                                underlying=symbol,
                                spot_price=spot,
                                dollar_budget=per_name_put_budget,
                                existing_long_put_qty=existing_spec_puts,
                            )

                    # Cash-secured short puts — worst-case outcome is being
                    # assigned the stock at the strike, so this is sized
                    # only up to cash actually available to cover that (not
                    # margin), and only on names we'd be OK owning (the same
                    # well-scored names as the long calls above, not the
                    # short-sleeve's worst-scoring names).
                    put_cash_budget_total = min(cash_available, total_equity * SHORT_PUT_CASH_RESERVE_PCT)
                    put_candidates = ranked_calls[:TOP_N_SHORT_PUTS]
                    if put_candidates and put_cash_budget_total > 0:
                        per_name_put_cash = put_cash_budget_total / len(put_candidates)
                        for symbol in put_candidates:
                            try:
                                spot = get_price(stock_data_client, symbol)
                            except Exception as e:
                                log.warning(f"Price fetch failed for {symbol} (short put): {e}")
                                continue
                            existing_short_puts = int(opt_summary.get((symbol, "put"), 0))
                            opt_engine.sell_short_puts(
                                underlying=symbol,
                                spot_price=spot,
                                cash_reserve_available=per_name_put_cash,
                                existing_short_put_qty=existing_short_puts,
                            )

            # Kill-switches
            check_position_losses(trading_client, positions)
            apply_short_stop_losses(trading_client, positions)
            apply_trailing_stops(trading_client, positions)
            check_options_exposure(positions, total_equity)
            check_short_exposure(positions, total_equity)
            check_portfolio_drawdown(account, initial_equity)

            loop_duration = time.time() - start_ts
            log.info(f"Trading loop completed in {loop_duration:.1f}s.")
            time.sleep(LOOP_SLEEP_SECONDS)

        except Exception as e:
            log.exception(f"Trading loop error: {e}")
            log.info(f"Sleeping {backoff_seconds}s before retry.")
            time.sleep(backoff_seconds)


_trading_loop_started = False
_trading_loop_lock = threading.Lock()


def start_background_trading_loop():
    global _trading_loop_started

    def _runner():
        # Delay so Flask/Gunicorn is fully up before heavy work
        time.sleep(5)
        trading_loop()

    with _trading_loop_lock:
        if _trading_loop_started:
            log.info("Trading loop already started in this process; skipping duplicate start.")
            return
        _trading_loop_started = True

    t = threading.Thread(target=_runner, daemon=True)
    t.start()
    log.info("Background trading loop thread started.")


# ============================================================
# ENTRYPOINT
# ============================================================

# Start the trading loop at import time. This runs both when the file is
# executed directly (`python trading_bot.py`) AND when a WSGI server like
# Gunicorn imports this module and grabs `app` — Gunicorn never sets
# __name__ == "__main__", so anything inside that guard would silently
# never run under Gunicorn. The lock above prevents a double-start if
# multiple Gunicorn workers import this module (keep --workers 1 regardless,
# so you don't get duplicate order submissions from separate processes).
start_background_trading_loop()

if __name__ == "__main__":
    # Local/dev run: use Flask's built-in server directly.
    port = int(os.environ.get("PORT", 10000))
    log.info(f"Starting Flask app on port {port}...")
    app.run(host="0.0.0.0", port=port, threaded=True)
