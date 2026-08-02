import os
import re
import time
import threading
import logging
from datetime import date, timedelta, datetime
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
    return "Global Multi-Factor Risk Engine Online (Equities/ETFs/Options only)!", 200


# ============================================================
# CONFIG
# ============================================================

HISTORY_YEARS = 12
RISK_FREE_ANNUAL = 0.04
TRADING_DAYS = 252

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
TOP_N_SECTORS = 3

HEDGE_INSTRUMENTS = {
    "long_duration_bonds": "TLT",
    "gold": "GLD",
    "inverse_equity": "SH",
}
REGIME_BENCHMARK = "VOO"
VIX_TICKER = "^VIX"

SATELLITE_HEDGE_TOTAL = 1.0 - CORE_TOTAL_WEIGHT   # 0.30
MIN_HEDGE_WEIGHT = 0.02
MAX_HEDGE_WEIGHT = 0.07   # hedge < 7% of portfolio

# --- Options overlay config ---
OPTIONS_ENABLED = True
COVERED_CALL_UNDERLYING = "VOO"
PROTECTIVE_PUT_UNDERLYING = "VOO"
CALL_OTM_PCT = 0.03
PUT_OTM_PCT = 0.07
OPTIONS_MIN_DTE = 25
OPTIONS_MAX_DTE = 45
PUT_HEDGE_SHARE = 0.25
OPTIONS_RUN_INTERVAL_SECONDS = 24 * 3600
MAX_OPTIONS_PCT_OF_EQUITY = 2.0   # options <= 2% of equity

# Only these asset classes may ever receive an order from this bot.
SAFE_ASSET_CLASSES = {AssetClass.US_EQUITY, AssetClass.US_OPTION}

TRAILING_STOP_PCT = 5.0
REBALANCE_BAND = 0.05
LOOP_SLEEP_SECONDS = 300
WEEKEND_SLEEP_SECONDS = 1800

# Position caps
MAX_ETF_WEIGHT = 0.40       # no single ETF > 40%
MAX_SECTOR_WEIGHT = 0.15    # no single sector > 15%

# Option liquidity filters
MIN_OPTION_VOLUME = 500
MAX_OPTION_SPREAD_PCT = 0.15  # bid/ask spread <= 15% of mid

# Trade limit
MAX_TRADES_PER_DAY = 30
TRADES_TODAY = 0
LAST_TRADE_DAY = None


# ============================================================
# SAFETY GUARD — never trade currency/forex or anything off-universe
# ============================================================

def assert_safe_order(symbol: str, asset_class):
    if asset_class not in SAFE_ASSET_CLASSES:
        raise ValueError(f"Blocked order for {symbol}: asset class {asset_class} is not permitted (no forex/crypto).")


# ============================================================
# TRADE LIMIT / SUBMIT WRAPPER
# ============================================================

def reset_trade_counter_if_new_day():
    global TRADES_TODAY, LAST_TRADE_DAY
    today = date.today()
    if LAST_TRADE_DAY != today:
        LAST_TRADE_DAY = today
        TRADES_TODAY = 0


def can_trade_today() -> bool:
    reset_trade_counter_if_new_day()
    return TRADES_TODAY < MAX_TRADES_PER_DAY


def submit_order_safe(trading_client: TradingClient, order_data, label: str = ""):
    global TRADES_TODAY
    reset_trade_counter_if_new_day()
    if TRADES_TODAY >= MAX_TRADES_PER_DAY:
        log.warning(f"Trade limit reached ({MAX_TRADES_PER_DAY} per day). Skipping order: {label}")
        return
    try:
        trading_client.submit_order(order_data=order_data)
        TRADES_TODAY += 1
        log.info(f"Order submitted ({TRADES_TODAY}/{MAX_TRADES_PER_DAY} today): {label}")
    except Exception as e:
        log.exception(f"Order submission failed for {label}: {e}")


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
# 1. DATA ENGINE
# ============================================================

class DataEngine:
    def __init__(self, tickers, years=HISTORY_YEARS):
        self.tickers = list(dict.fromkeys(tickers))
        self.years = years
        self.close = pd.DataFrame()
        self.returns = pd.DataFrame()

    def fetch(self):
        log.info(f"Downloading {self.years}y of history for {len(self.tickers)} tickers...")
        raw = yf.download(
            self.tickers, period=f"{self.years}y",
            auto_adjust=True, progress=False, group_by="column", threads=True,
        )
        if isinstance(raw.columns, pd.MultiIndex):
            close = raw["Close"]
        else:
            close = raw[["Close"]]
            close.columns = self.tickers
        self.close = close.dropna(how="all")
        self.returns = self.close.pct_change().dropna(how="all")
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

    def covariance_matrix(self, tickers, lookback_days=756):
        sub = self.returns[tickers].dropna().tail(lookback_days)
        return sub.cov() * TRADING_DAYS


def get_underlying_vol(data_engine: DataEngine, underlying: str, default_sigma: float = 0.20) -> float:
    vol = data_engine.volatility().get(underlying)
    return vol if vol and vol > 0 else default_sigma


# ============================================================
# 2. FUNDAMENTAL ANALYZER
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
            info = yf.Ticker(ticker).info or {}
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
# 3. NEWS SENTIMENT
# ============================================================

class NewsSentimentAnalyzer:
    def __init__(self):
        self.vader = SentimentIntensityAnalyzer()

    def score(self, ticker, max_headlines=8):
        try:
            news_items = yf.Ticker(ticker).news or []
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
# 4. COMPOSITE SCORER
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
    fund = zscore({t: fundamentals.score(t) for t in tickers})
    sent = zscore({t: sentiment.score(t) for t in tickers})

    scores = {}
    for t in tickers:
        scores[t] = (
            weights["sharpe"] * sharpe.get(t, 0.0)
            + weights["momentum"] * momentum.get(t, 0.0)
            + weights["fundamentals"] * fund.get(t, 0.0)
            + weights["sentiment"] * sent.get(t, 0.0)
        )
    return scores


# ============================================================
# 5. MEAN-VARIANCE OPTIMIZER
# ============================================================

def optimize_weights(expected_returns, cov_matrix, total_weight, max_weight_per_asset=0.6):
    tickers = list(expected_returns.keys())
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
# 6. REGIME / RISK MANAGER
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
            vix_hist = yf.download(VIX_TICKER, period="6mo", progress=False, auto_adjust=True)["Close"]
            vix_level = float(vix_hist.iloc[-1])
            score_components.append(float(np.clip((vix_level - 15) / 20, 0, 1)))
        except Exception as e:
            log.warning(f"VIX fetch failed: {e}")
        return float(np.mean(score_components)) if score_components else 0.3

    def hedge_weight_breakdown(self, hedge_total_weight):
        vol = self.de.volatility()
        equity_vol = vol.get(REGIME_BENCHMARK, 0.15)
        if equity_vol > 0.22:
            split = {"inverse_equity": 0.45, "long_duration_bonds": 0.30, "gold": 0.25}
        else:
            split = {"inverse_equity": 0.25, "long_duration_bonds": 0.40, "gold": 0.35}
        return {name: hedge_total_weight * pct for name, pct in split.items()}


# ============================================================
# 7. OPTIONS ENGINE
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
        today = date.today()
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
        # Volume filter
        vol = getattr(contract, "volume", None)
        if vol is None or vol < MIN_OPTION_VOLUME:
            log.info(f"Skipping illiquid option {contract.symbol}: volume={vol}")
            return False
        # Spread filter
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

        T = (contract.expiration_date - date.today()).days / TRADING_DAYS
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

        T = (contract.expiration_date - date.today()).days / TRADING_DAYS
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


# ============================================================
# 8. EXECUTION LAYER
# ============================================================

def get_price(stock_data_client, ticker):
    req = StockLatestTradeRequest(symbol_or_symbols=ticker)
    trade = stock_data_client.get_stock_latest_trade(req)[ticker]
    return float(trade.price)


def rebalance_to_target(trading_client, stock_data_client, positions, total_equity, cash_available, ticker, target_weight, is_sector=False):
    # Enforce ETF and sector caps
    if is_sector:
        target_weight = min(target_weight, MAX_SECTOR_WEIGHT)
    else:
        target_weight = min(target_weight, MAX_ETF_WEIGHT)

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
    qty = int((target_value - current_value) // price)
    if qty > 0 and can_trade_today():
        assert_safe_order(ticker, AssetClass.US_EQUITY)
        order = MarketOrderRequest(symbol=ticker, qty=qty, side=OrderSide.BUY, time_in_force=TimeInForce.GTC)
        submit_order_safe(trading_client, order, label=f"BUY {ticker} toward {target_weight:.3f}")
        log.info(f"BUY {qty} {ticker} toward target weight {target_weight:.3f}")


def apply_trailing_stops(trading_client, positions):
    for symbol, pos in list(positions.items()):
        if pos.asset_class != AssetClass.US_EQUITY:
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


# ============================================================
# 9. MAIN LOOP
# ============================================================

def trading_bot_loop():
    api_key = os.environ.get("ALPACA_PAPER_KEY")
    secret_key = os.environ.get("ALPACA_PAPER_SECRET")

    if not api_key or not secret_key:
        log.error("Missing Alpaca credentials. Bot will not start.")
        return

    trading_client = TradingClient(api_key, secret_key, paper=True)
    stock_data_client = StockHistoricalDataClient(api_key, secret_key)
    option_data_client = OptionHistoricalDataClient(api_key, secret_key)

    all_tickers = list(CORE_ETFS.keys()) + SECTOR_TICKERS + list(HEDGE_INSTRUMENTS.values())
    fundamentals = FundamentalAnalyzer()
    sentiment = NewsSentimentAnalyzer()
    de = DataEngine(all_tickers)

    options_engine = OptionsEngine(trading_client, option_data_client, de)

    last_data_refresh = 0
    last_options_run = 0

    while True:
        try:
            reset_trade_counter_if_new_day()

            if time.gmtime().tm_wday >= 5:
                time.sleep(WEEKEND_SLEEP_SECONDS)
                continue

            if time.time() - last_data_refresh > 6 * 3600:
                de.fetch()
                last_data_refresh = time.time()

            account = trading_client.get_account()
            total_equity = float(account.portfolio_value)
            cash_available = float(account.cash)
            positions = {p.symbol: p for p in trading_client.get_all_positions()}

            apply_trailing_stops(trading_client, positions)

            # 1. CORE SLEEVE
            for ticker, weight in CORE_ETFS.items():
                rebalance_to_target(
                    trading_client, stock_data_client, positions,
                    total_equity, cash_available, ticker, weight, is_sector=False
                )

            # 2. REGIME / RISK SPLIT
            risk_mgr = RiskManager(de)
            risk_score = risk_mgr.regime_risk_score()
            hedge_weight = float(np.clip(
                MIN_HEDGE_WEIGHT + risk_score * (MAX_HEDGE_WEIGHT - MIN_HEDGE_WEIGHT),
                MIN_HEDGE_WEIGHT, MAX_HEDGE_WEIGHT,
            ))
            satellite_weight = SATELLITE_HEDGE_TOTAL - hedge_weight
            log.info(f"Risk score={risk_score:.2f} -> satellite={satellite_weight:.2%}, hedge={hedge_weight:.2%}")

            # 3. SATELLITE SLEEVE
            scores = composite_scores(de, fundamentals, sentiment, SECTOR_TICKERS)
            top_sectors = [t for t, _ in sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:TOP_N_SECTORS]]
            if top_sectors:
                momentum = de.momentum_12_1()
                expected_returns = {t: momentum.get(t, 0.0) for t in top_sectors}
                cov = de.covariance_matrix(top_sectors)
                sat_weights = optimize_weights(expected_returns, cov, satellite_weight)
                for ticker, weight in sat_weights.items():
                    rebalance_to_target(
                        trading_client, stock_data_client, positions,
                        total_equity, cash_available, ticker, weight, is_sector=True
                    )

            # 4. HEDGE SLEEVE
            etf_hedge_weight = hedge_weight * (1 - PUT_HEDGE_SHARE)
            put_budget = total_equity * hedge_weight * PUT_HEDGE_SHARE
            hedge_alloc = risk_mgr.hedge_weight_breakdown(etf_hedge_weight)
            for name, weight in hedge_alloc.items():
                ticker = HEDGE_INSTRUMENTS[name]
                rebalance_to_target(
                    trading_client, stock_data_client, positions,
                    total_equity, cash_available, ticker, weight, is_sector=False
                )

            # 5. OPTIONS OVERLAY
            if OPTIONS_ENABLED and time.time() - last_options_run > OPTIONS_RUN_INTERVAL_SECONDS:
                options_pct = check_options_exposure(positions, total_equity)
                if options_pct < MAX_OPTIONS_PCT_OF_EQUITY:
                    opt_summary = options_position_summary(list(positions.values()))
                    try:
                        spot = get_price(stock_data_client, COVERED_CALL_UNDERLYING)
                    except Exception as e:
                        log.warning(f"Spot price fetch failed for options overlay: {e}")
                        spot = None

                    if spot:
                        core_pos = positions.get(COVERED_CALL_UNDERLYING)
                        if core_pos:
                            shares_held = float(core_pos.qty)
                            short_calls = opt_summary.get((COVERED_CALL_UNDERLYING, "call"), 0.0)
                            existing_short = abs(short_calls) if short_calls < 0 else 0.0
                            options_engine.sell_covered_calls(COVERED_CALL_UNDERLYING, shares_held, existing_short, spot)

                        put_pos = positions.get(PROTECTIVE_PUT_UNDERLYING)
                        shares_to_hedge = float(put_pos.qty) if put_pos else 0.0
                        long_puts = opt_summary.get((PROTECTIVE_PUT_UNDERLYING, "put"), 0.0)
                        existing_long = long_puts if long_puts > 0 else 0.0
                        options_engine.buy_protective_puts(
                            PROTECTIVE_PUT_UNDERLYING, shares_to_hedge, existing_long, spot, put_budget
                        )
                last_options_run = time.time()

            # 6. FINAL OPTIONS EXPOSURE CHECK
            check_options_exposure(positions, total_equity)

        except Exception as e:
            log.exception(f"Main loop error: {e}")

        time.sleep(LOOP_SLEEP_SECONDS)


threading.Thread(target=trading_bot_loop, daemon=True).start()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))
