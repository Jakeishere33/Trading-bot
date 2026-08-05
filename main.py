import logging
import time
from datetime import datetime, timedelta, date
from zoneinfo import ZoneInfo
from math import sqrt, log, exp

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
log = logging.getLogger("lean_risk_engine")

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

SATELLITE_HEDGE_TOTAL = 1.0 - CORE_TOTAL_WEIGHT   # 0.30
HEDGE_SLEEVE_WEIGHT = 0.10
SATELLITE_SLEEVE_WEIGHT = SATELLITE_HEDGE_TOTAL - HEDGE_SLEEVE_WEIGHT  # 0.20

SHORT_SLEEVE_WEIGHT = 0.05
MAX_SHORT_POSITION_WEIGHT = 0.03
TOP_N_SHORTS = 3
SHORT_STOP_LOSS_PCT = 0.15

HEDGE_INSTRUMENTS = {
    "long_duration_bonds": "TLT",
    "gold": "GLD",
    "inverse_equity": "SH",
}

REGIME_BENCHMARK = "VOO"

OPTIONS_ENABLED = True
CALL_OTM_PCT = 0.03
PUT_OTM_PCT = 0.07
OPTIONS_MIN_DTE = 25
OPTIONS_MAX_DTE = 45
MAX_OPTIONS_PCT_OF_EQUITY = 2.0
COVERED_CALL_MIN_SHARES = 100

SAFE_ASSET_CLASSES = {AssetClass.US_EQUITY, AssetClass.US_OPTION}

TRAILING_STOP_PCT = 5.0
MAX_TRADES_PER_DAY = 500
TRADES_TODAY = 0
LAST_TRADE_DAY = None

NO_TRADE_BEFORE = (9, 45)   # 9:45 AM ET
NO_TRADE_AFTER = (15, 55)   # 3:55 PM ET

MIN_OPTION_VOLUME = 500
MAX_OPTION_SPREAD_PCT = 0.15

# ============================================================
# TIME / TRADE WINDOW HELPERS
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
        log.warning(f"Trade limit reached. Skipping order: {label}")
        return
    if in_no_trade_window():
        log.info(f"No-trade window active. Skipping order: {label}")
        return
    try:
        trading_client.submit_order(order_data=order_data)
        TRADES_TODAY += 1
        log.info(f"Order submitted ({TRADES_TODAY}/{MAX_TRADES_PER_DAY}): {label}")
    except Exception as e:
        log.exception(f"Order submission failed for {label}: {e}")


def assert_safe_order(symbol: str, asset_class):
    if asset_class not in SAFE_ASSET_CLASSES:
        raise ValueError(f"Blocked order for {symbol}: asset class {asset_class} not permitted.")


# ============================================================
# LIGHTWEIGHT DATA HELPERS (ALPACA ONLY)
# ============================================================

class LeanData:
    def __init__(self, stock_client: StockHistoricalDataClient):
        self.stock_client = stock_client

    def get_recent_bars(self, symbol: str, days: int = 30):
        end = now_et()
        start = end - timedelta(days=days * 2)  # buffer for non-trading days
        try:
            bars = self.stock_client.get_stock_bars(
                symbol_or_symbols=symbol,
                timeframe="1Day",
                start=start,
                end=end,
            )[symbol]
            return list(bars)
        except Exception as e:
            log.warning(f"Failed to fetch bars for {symbol}: {e}")
            return []

    def simple_momentum(self, symbol: str, lookback_days: int = 20):
        bars = self.get_recent_bars(symbol, days=lookback_days + 5)
        if len(bars) < lookback_days + 1:
            return 0.0
        p_now = float(bars[-1].close)
        p_then = float(bars[-lookback_days].close)
        if p_then <= 0:
            return 0.0
        return (p_now / p_then) - 1.0

    def simple_volatility(self, symbol: str, window: int = 14):
        bars = self.get_recent_bars(symbol, days=window + 10)
        if len(bars) < window + 1:
            return 0.0
        rets = []
        for i in range(1, len(bars)):
            p0 = float(bars[i - 1].close)
            p1 = float(bars[i].close)
            if p0 > 0:
                rets.append((p1 / p0) - 1.0)
        if len(rets) < window:
            return 0.0
        mean = sum(rets[-window:]) / window
        var = sum((r - mean) ** 2 for r in rets[-window:]) / window
        return sqrt(var) * sqrt(TRADING_DAYS)


# ============================================================
# SIMPLE REGIME / RISK MANAGER
# ============================================================

class LeanRiskManager:
    def __init__(self, data: LeanData):
        self.data = data

    def regime_risk_score(self):
        # 0 = bullish, 1 = bearish
        bars = self.data.get_recent_bars(REGIME_BENCHMARK, days=220)
        if len(bars) < 200:
            return 0.3
        closes = [float(b.close) for b in bars]
        sma200 = sum(closes[-200:]) / 200
        last = closes[-1]
        below_sma = 0.0 if last > sma200 else 1.0
        last_year = closes[-252:]
        dd = 1.0 - (last / max(last_year))
        dd_score = max(0.0, min(dd / 0.20, 1.0))
        return (below_sma + dd_score) / 2.0

    def hedge_weights(self):
        # Simple static split, scaled by hedge sleeve
        return {
            "inverse_equity": HEDGE_SLEEVE_WEIGHT * 0.4,
            "long_duration_bonds": HEDGE_SLEEVE_WEIGHT * 0.35,
            "gold": HEDGE_SLEEVE_WEIGHT * 0.25,
        }


# ============================================================
# BLACK–SCHOLES (FOR ROUGH FAIR VALUE)
# ============================================================

def black_scholes_price(S, K, T, r, sigma, option_type):
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return 0.0
    d1 = (log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    from math import erf

    def norm_cdf(x):
        return 0.5 * (1.0 + erf(x / sqrt(2.0)))

    if option_type == "call":
        return S * norm_cdf(d1) - K * exp(-r * T) * norm_cdf(d2)
    else:
        return K * exp(-r * T) * norm_cdf(-d2) - S * norm_cdf(-d1)


# ============================================================
# OPTIONS ENGINE (LEAN)
# ============================================================

class LeanOptionsEngine:
    def __init__(
        self,
        trading_client: TradingClient,
        option_data_client: OptionHistoricalDataClient,
        data: LeanData,
    ):
        self.trading = trading_client
        self.opt_data = option_data_client
        self.data = data

    def _find_contracts(self, underlying, contract_type):
        today = now_et().date()
        req = GetOptionContractsRequest(
            underlying_symbols=[underlying],
            status=AssetStatus.ACTIVE,
            type=contract_type,
            expiration_date_gte=today + timedelta(days=OPTIONS_MIN_DTE),
            expiration_date_lte=today + timedelta(days=OPTIONS_MAX_DTE),
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
            q = self.opt_data.get_option_latest_quote(req)[symbol]
            return float(q.bid_price), float(q.ask_price)
        except Exception as e:
            log.warning(f"Option quote failed for {symbol}: {e}")
            return None, None

    def _liquidity_ok(self, contract, bid, ask):
        vol = getattr(contract, "volume", None)
        if vol is None or vol < MIN_OPTION_VOLUME:
            return False
        if bid is None or ask is None or bid <= 0 or ask <= 0:
            return False
        mid = (bid + ask) / 2.0
        if mid <= 0:
            return False
        spread_pct = (ask - bid) / mid
        return spread_pct <= MAX_OPTION_SPREAD_PCT

    def _closest_strike(self, contracts, target_price):
        if not contracts:
            return None
        return sorted(
            contracts,
            key=lambda c: (abs(float(c.strike_price) - target_price), c.expiration_date),
        )[0]

    def sell_covered_calls(self, underlying, shares_held, spot_price):
        max_contracts = int(shares_held // COVERED_CALL_MIN_SHARES)
        if max_contracts <= 0 or spot_price <= 0:
            return
        contracts = self._find_contracts(underlying, ContractType.CALL)
        target = spot_price * (1 + CALL_OTM_PCT)
        contract = self._closest_strike(contracts, target)
        if not contract:
            return
        assert_safe_order(contract.symbol, AssetClass.US_OPTION)

        T = (contract.expiration_date - now_et().date()).days / TRADING_DAYS
        sigma = max(self.data.simple_volatility(underlying), 0.15)
        bs_price = black_scholes_price(
            S=spot_price,
            K=float(contract.strike_price),
            T=T,
            r=RISK_FREE_ANNUAL,
            sigma=sigma,
            option_type="call",
        )

        bid, ask = self._quote(contract.symbol)
        if not self._liquidity_ok(contract, bid, ask):
            return
        use_price = bid if bid and bid > 0 else bs_price
        if use_price <= 0:
            return

        order = LimitOrderRequest(
            symbol=contract.symbol,
            qty=max_contracts,
            side=OrderSide.SELL,
            type=OrderType.LIMIT,
            time_in_force=TimeInForce.DAY,
            limit_price=round(use_price, 2),
        )
        submit_order_safe(self.trading, order, label=f"SELL covered calls {contract.symbol}")

    def buy_protective_puts(self, underlying, shares_to_hedge, spot_price, dollar_budget):
        max_contracts = int(shares_to_hedge // 100)
        if max_contracts <= 0 or spot_price <= 0 or dollar_budget <= 0:
            return
        contracts = self._find_contracts(underlying, ContractType.PUT)
        target = spot_price * (1 - PUT_OTM_PCT)
        contract = self._closest_strike(contracts, target)
        if not contract:
            return
        assert_safe_order(contract.symbol, AssetClass.US_OPTION)

        T = (contract.expiration_date - now_et().date()).days / TRADING_DAYS
        sigma = max(self.data.simple_volatility(underlying), 0.15)
        bs_price = black_scholes_price(
            S=spot_price,
            K=float(contract.strike_price),
            T=T,
            r=RISK_FREE_ANNUAL,
            sigma=sigma,
            option_type="put",
        )

        bid, ask = self._quote(contract.symbol)
        if not self._liquidity_ok(contract, bid, ask):
            return
        use_price = ask if ask and ask > 0 else bs_price
        if use_price <= 0:
            return

        max_affordable = int(dollar_budget // (use_price * 100))
        qty = min(max_contracts, max_affordable)
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


# ============================================================
# MAIN TRADING ENGINE (LONGS, SHORTS, HEDGES, OPTIONS)
# ============================================================

class LeanTradingEngine:
    def __init__(
        self,
        trading_client: TradingClient,
        stock_client: StockHistoricalDataClient,
        option_client: OptionHistoricalDataClient,
    ):
        self.trading = trading_client
        self.data = LeanData(stock_client)
        self.risk = LeanRiskManager(self.data)
        self.options = LeanOptionsEngine(trading_client, option_client, self.data)

    # ---- portfolio helpers ----
    def get_equity(self):
        try:
            acct = self.trading.get_account()
            return float(acct.equity)
        except Exception as e:
            log.warning(f"Failed to fetch account equity: {e}")
            return 0.0

    def get_positions(self):
        try:
            return list(self.trading.get_all_positions())
        except Exception as e:
            log.warning(f"Failed to fetch positions: {e}")
            return []

    def get_position_map(self):
        pos = self.get_positions()
        out = {}
        for p in pos:
            out[p.symbol] = p
        return out

    # ---- core sleeve ----
    def run_core_etfs(self):
        equity = self.get_equity()
        if equity <= 0:
            return
        pos_map = self.get_position_map()
        for sym, target_weight in CORE_ETFS.items():
            target_value = equity * target_weight
            current_value = float(pos_map.get(sym).market_value) if sym in pos_map else 0.0
            diff = target_value - current_value
            if abs(diff) / equity < 0.01:
                continue
            side = OrderSide.BUY if diff > 0 else OrderSide.SELL
            qty = int(abs(diff) / self._latest_price(sym))
            if qty <= 0:
                continue
            assert_safe_order(sym, AssetClass.US_EQUITY)
            order = MarketOrderRequest(
                symbol=sym,
                qty=qty,
                side=side,
                time_in_force=TimeInForce.DAY,
            )
            submit_order_safe(self.trading, order, label=f"CORE rebalance {sym}")

    def _latest_price(self, symbol: str) -> float:
        try:
            req = StockLatestTradeRequest(symbol_or_symbols=symbol)
            trade = self.data.stock_client.get_stock_latest_trade(req)[symbol]
            return float(trade.price)
        except Exception:
            bars = self.data.get_recent_bars(symbol, days=5)
            return float(bars[-1].close) if bars else 0.0

    # ---- satellite sectors ----
    def run_satellite_sectors(self):
        equity = self.get_equity()
        if equity <= 0:
            return
        pos_map = self.get_position_map()
        # rank sectors by simple momentum
        scores = {t: self.data.simple_momentum(t, lookback_days=60) for t in SECTOR_TICKERS}
        ranked = sorted(SECTOR_TICKERS, key=lambda x: scores.get(x, -1e9), reverse=True)
        top = ranked[:3]
        per_weight = SATELLITE_SLEEVE_WEIGHT / max(len(top), 1)
        for sym in top:
            target_value = equity * per_weight
            current_value = float(pos_map.get(sym).market_value) if sym in pos_map else 0.0
            diff = target_value - current_value
            if abs(diff) / equity < 0.01:
                continue
            side = OrderSide.BUY if diff > 0 else OrderSide.SELL
            price = self._latest_price(sym)
            if price <= 0:
                continue
            qty = int(abs(diff) / price)
            if qty <= 0:
                continue
            assert_safe_order(sym, AssetClass.US_EQUITY)
            order = MarketOrderRequest(
                symbol=sym,
                qty=qty,
                side=side,
                time_in_force=TimeInForce.DAY,
            )
            submit_order_safe(self.trading, order, label=f"SATELLITE sector {sym}")

    # ---- short sleeve (simple) ----
    def run_shorts(self):
        equity = self.get_equity()
        if equity <= 0:
            return
        pos_map = self.get_position_map()
        # pick a small universe: sectors themselves, short weakest
        scores = {t: self.data.simple_momentum(t, lookback_days=60) for t in SECTOR_TICKERS}
        ranked = sorted(SECTOR_TICKERS, key=lambda x: scores.get(x, 1e9))
        shorts = ranked[:TOP_N_SHORTS]
        per_short_weight = SHORT_SLEEVE_WEIGHT / max(len(shorts), 1)
        for sym in shorts:
            target_value = equity * per_short_weight
            price = self._latest_price(sym)
            if price <= 0:
                continue
            qty = int(target_value / price)
            if qty <= 0:
                continue
            # if already long, skip; if already short, leave
            if sym in pos_map and float(pos_map[sym].qty) > 0:
                continue
            assert_safe_order(sym, AssetClass.US_EQUITY)
            order = MarketOrderRequest(
                symbol=sym,
                qty=qty,
                side=OrderSide.SELL,
                time_in_force=TimeInForce.DAY,
            )
            submit_order_safe(self.trading, order, label=f"SHORT sleeve {sym}")

    # ---- hedges ----
    def run_hedges(self):
        equity = self.get_equity()
        if equity <= 0:
            return
        pos_map = self.get_position_map()
        weights = self.risk.hedge_weights()
        for name, sym in HEDGE_INSTRUMENTS.items():
            target_weight = weights.get(name, 0.0)
            if target_weight <= 0:
                continue
            target_value = equity * target_weight
            current_value = float(pos_map.get(sym).market_value) if sym in pos_map else 0.0
            diff = target_value - current_value
            if abs(diff) / equity < 0.01:
                continue
            side = OrderSide.BUY if diff > 0 else OrderSide.SELL
            price = self._latest_price(sym)
            if price <= 0:
                continue
            qty = int(abs(diff) / price)
            if qty <= 0:
                continue
            assert_safe_order(sym, AssetClass.US_EQUITY)
            order = MarketOrderRequest(
                symbol=sym,
                qty=qty,
                side=side,
                time_in_force=TimeInForce.DAY,
            )
            submit_order_safe(self.trading, order, label=f"HEDGE {sym}")

    # ---- options overlay ----
    def run_options_overlay(self):
        if not OPTIONS_ENABLED:
            return
        equity = self.get_equity()
        if equity <= 0:
            return
        pos_map = self.get_position_map()
        # covered calls + protective puts on core ETF
        for sym in CORE_ETFS.keys():
            pos = pos_map.get(sym)
            if not pos:
                continue
            shares = float(pos.qty)
            price = self._latest_price(sym)
            if shares >= COVERED_CALL_MIN_SHARES:
                self.options.sell_covered_calls(sym, shares, price)
            # simple protective put budget
            put_budget = equity * 0.01
            self.options.buy_protective_puts(sym, shares, price, put_budget)

    # ---- main run ----
    def run_once(self):
        if not can_trade_today():
            return
        regime = self.risk.regime_risk_score()
        log.info(f"Regime risk score: {regime:.2f}")
        # core always
        self.run_core_etfs()
        # satellite + hedges
        self.run_satellite_sectors()
        self.run_hedges()
        # shorts (scaled by regime)
        if regime > 0.4:
            self.run_shorts()
        # options overlay
        self.run_options_overlay()


# ============================================================
# ENTRY POINT
# ============================================================

def main():
    API_KEY = "YOUR_ALPACA_KEY"
    API_SECRET = "YOUR_ALPACA_SECRET"
    PAPER = True

    trading_client = TradingClient(API_KEY, API_SECRET, paper=PAPER)
    stock_client = StockHistoricalDataClient(API_KEY, API_SECRET)
    option_client = OptionHistoricalDataClient(API_KEY, API_SECRET)

    engine = LeanTradingEngine(trading_client, stock_client, option_client)

    # For Render, you might call run_once on a schedule (cron/worker)
    while True:
        try:
            engine.run_once()
        except Exception as e:
            log.exception(f"Engine run failed: {e}")
        time.sleep(300)  # 5 minutes between cycles


if __name__ == "__main__":
    main()
