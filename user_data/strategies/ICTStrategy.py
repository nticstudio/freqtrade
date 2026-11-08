# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401
# isort: skip_file
"""
ICTStrategy — Smart Money / ICT-inspired strategy for Freqtrade.

Concepts:
  - Fair Value Gaps (FVG)
  - Order Blocks (OB)
  - Market structure (BOS / CHoCH)
  - Liquidity sweeps (equal highs / lows)
  - SMT divergence vs BTC (alts)

Timeframes:
  - Execution / entries: 15m only (selective)
  - Bias: 1h + 4h (must be aligned)

Trade management:
  - Stop: beyond the tapped OB / FVG (+ ATR buffer)
  - TP1 (50%): internal liquidity (last swing), min 1R — then stop to breakeven
  - TP2: external liquidity (equal highs / lows), fallback 2R
"""
from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
from pandas import DataFrame

import talib.abstract as ta
from freqtrade.persistence import Trade
from freqtrade.strategy import (
    BooleanParameter,
    CategoricalParameter,
    DecimalParameter,
    IStrategy,
    IntParameter,
    merge_informative_pair,
    stoploss_from_absolute,
    timeframe_to_prev_date,
)


def _confirmed_swings(df: DataFrame, left: int, right: int) -> DataFrame:
    """
    Swing highs/lows, reported on the bar where they become confirmed
    (pivot `right` bars ago), so no future candles are used.
    """
    win = left + right + 1
    roll_max = df["high"].rolling(win, min_periods=win).max()
    roll_min = df["low"].rolling(win, min_periods=win).min()
    cand_high = df["high"].shift(right)
    cand_low = df["low"].shift(right)
    df["swing_high"] = np.where(cand_high == roll_max, cand_high, np.nan)
    df["swing_low"] = np.where(cand_low == roll_min, cand_low, np.nan)
    return df


def _add_structure(df: DataFrame, swing_left: int, swing_right: int) -> DataFrame:
    """
    Market structure from confirmed swings.
    BOS = close crosses the last swing in the trend direction.
    CHoCH = close crosses the last swing against the current trend (trend flips).
    """
    df = _confirmed_swings(df.copy(), swing_left, swing_right)
    df["last_swing_high"] = df["swing_high"].ffill()
    df["last_swing_low"] = df["swing_low"].ffill()

    ref_high = df["last_swing_high"].shift(1).to_numpy(dtype=float)
    ref_low = df["last_swing_low"].shift(1).to_numpy(dtype=float)
    close = df["close"].to_numpy(dtype=float)
    n = len(df)

    trend = np.zeros(n, dtype=int)
    bos_bull = np.zeros(n, dtype=bool)
    bos_bear = np.zeros(n, dtype=bool)
    choch_bull = np.zeros(n, dtype=bool)
    choch_bear = np.zeros(n, dtype=bool)

    t = 0
    for i in range(1, n):
        rh, rl = ref_high[i], ref_low[i]
        if not np.isnan(rh) and close[i] > rh and close[i - 1] <= rh:
            if t < 0:
                choch_bull[i] = True
            else:
                bos_bull[i] = True
            t = 1
        elif not np.isnan(rl) and close[i] < rl and close[i - 1] >= rl:
            if t > 0:
                choch_bear[i] = True
            else:
                bos_bear[i] = True
            t = -1
        trend[i] = t

    df["ms_trend"] = trend
    df["bos_bull"] = bos_bull
    df["bos_bear"] = bos_bear
    df["choch_bull"] = choch_bull
    df["choch_bear"] = choch_bear
    return df


def _track_zones(new_top: np.ndarray, new_bot: np.ndarray, close: np.ndarray, invalidate: str):
    """
    Keep the latest zone alive until a candle closes through it.
    Value at index i = zone state after candle i closed.
    """
    n = len(close)
    tops = np.full(n, np.nan)
    bots = np.full(n, np.nan)
    ct = cb = np.nan
    for i in range(n):
        if not np.isnan(new_top[i]):
            ct, cb = new_top[i], new_bot[i]
        if not np.isnan(ct):
            if (invalidate == "below" and close[i] < cb) or (invalidate == "above" and close[i] > ct):
                ct = cb = np.nan
        tops[i] = ct
        bots[i] = cb
    return tops, bots


def _add_zones(df: DataFrame, min_gap_atr: float, impulse_atr: float) -> DataFrame:
    """FVG + Order Blocks, with tap detection on zones that existed before the candle."""
    atr = ta.ATR(df, timeperiod=14)
    close = df["close"].to_numpy(dtype=float)

    # FVG (3 candles): bullish = low[i] > high[i-2], bearish = high[i] < low[i-2]
    bull_fvg = (df["low"] - df["high"].shift(2)) > atr * min_gap_atr
    bear_fvg = (df["low"].shift(2) - df["high"]) > atr * min_gap_atr
    bft, bfb = _track_zones(
        np.where(bull_fvg, df["low"], np.nan),
        np.where(bull_fvg, df["high"].shift(2), np.nan),
        close,
        "below",
    )
    sft, sfb = _track_zones(
        np.where(bear_fvg, df["low"].shift(2), np.nan),
        np.where(bear_fvg, df["high"], np.nan),
        close,
        "above",
    )

    # OB: last opposite candle before a displacement candle
    body = (df["close"] - df["open"]).abs()
    impulse_up = (df["close"] > df["open"]) & (body > atr * impulse_atr) & (df["close"] > df["high"].shift(1))
    impulse_dn = (df["close"] < df["open"]) & (body > atr * impulse_atr) & (df["close"] < df["low"].shift(1))
    bull_ob = impulse_up & (df["close"].shift(1) < df["open"].shift(1))
    bear_ob = impulse_dn & (df["close"].shift(1) > df["open"].shift(1))
    bot_, bob = _track_zones(
        np.where(bull_ob, df["high"].shift(1), np.nan),
        np.where(bull_ob, df["low"].shift(1), np.nan),
        close,
        "below",
    )
    sot, sob = _track_zones(
        np.where(bear_ob, df["high"].shift(1), np.nan),
        np.where(bear_ob, df["low"].shift(1), np.nan),
        close,
        "above",
    )

    zones = pd.DataFrame(
        {
            "bull_fvg_top": bft,
            "bull_fvg_bottom": bfb,
            "bear_fvg_top": sft,
            "bear_fvg_bottom": sfb,
            "bull_ob_top": bot_,
            "bull_ob_bottom": bob,
            "bear_ob_top": sot,
            "bear_ob_bottom": sob,
        },
        index=df.index,
    )
    # Zones known before the current candle — a zone cannot be "tapped" on the candle creating it
    prev = zones.shift(1)

    low, high, cl = df["low"], df["high"], df["close"]
    in_bull_fvg = prev["bull_fvg_bottom"].notna() & (low <= prev["bull_fvg_top"]) & (cl >= prev["bull_fvg_bottom"])
    in_bear_fvg = prev["bear_fvg_top"].notna() & (high >= prev["bear_fvg_bottom"]) & (cl <= prev["bear_fvg_top"])
    in_bull_ob = prev["bull_ob_bottom"].notna() & (low <= prev["bull_ob_top"]) & (cl >= prev["bull_ob_bottom"])
    in_bear_ob = prev["bear_ob_top"].notna() & (high >= prev["bear_ob_bottom"]) & (cl <= prev["bear_ob_top"])

    out = pd.concat([df, zones], axis=1)
    out["atr"] = atr
    out["in_bull_fvg"] = in_bull_fvg
    out["in_bear_fvg"] = in_bear_fvg
    out["in_bull_ob"] = in_bull_ob
    out["in_bear_ob"] = in_bear_ob
    # Invalidation level of the tapped zone (OB preferred over FVG)
    out["zone_low"] = np.where(in_bull_ob, prev["bull_ob_bottom"], np.where(in_bull_fvg, prev["bull_fvg_bottom"], np.nan))
    out["zone_high"] = np.where(in_bear_ob, prev["bear_ob_top"], np.where(in_bear_fvg, prev["bear_fvg_top"], np.nan))
    return out


def _add_liquidity(df: DataFrame, eq_tol: float, lookback: int = 40) -> DataFrame:
    """Equal highs/lows (from confirmed swings) + sweeps (wick through, close back inside)."""
    n = len(df)
    eq_high = np.full(n, np.nan)
    eq_low = np.full(n, np.nan)

    for col, out, pick in (("swing_high", eq_high, max), ("swing_low", eq_low, min)):
        values = df[col].to_numpy(dtype=float)
        idx = np.where(~np.isnan(values))[0]
        for k, i in enumerate(idx):
            for j in idx[:k][::-1]:
                if i - j > lookback:
                    break
                if abs(values[i] - values[j]) / values[j] <= eq_tol:
                    out[i] = pick(values[i], values[j])
                    break

    df["eq_high"] = pd.Series(eq_high, index=df.index).ffill()
    df["eq_low"] = pd.Series(eq_low, index=df.index).ffill()
    prev_eq_high = df["eq_high"].shift(1)
    prev_eq_low = df["eq_low"].shift(1)
    df["liq_sweep_low"] = prev_eq_low.notna() & (df["low"] < prev_eq_low) & (df["close"] > prev_eq_low)
    df["liq_sweep_high"] = prev_eq_high.notna() & (df["high"] > prev_eq_high) & (df["close"] < prev_eq_high)
    # Extreme of the sweep candle (ICT invalidation reference)
    df["sweep_ext_low"] = np.where(df["liq_sweep_low"], df["low"], np.nan)
    df["sweep_ext_high"] = np.where(df["liq_sweep_high"], df["high"], np.nan)
    return df


def _enrich_ict(df: DataFrame, swing_left: int, swing_right: int, min_gap_atr: float, impulse_atr: float, eq_tol: float) -> DataFrame:
    df = _add_structure(df, swing_left, swing_right)
    df = _add_zones(df, min_gap_atr, impulse_atr)
    df = _add_liquidity(df, eq_tol)
    return df


class ICTStrategy(IStrategy):
    """
    ICT / Smart Money Concepts strategy (selective profile).

    Entries on 15m only. Bias requires 1h + 4h alignment. Score >= 4.
    Noisy alts (XRP/TAO/NEAR): SMT required, wider stop, higher RR, no CHoCH exit.
    NEAR: short only.
    Long & short. Requires futures config (trading_mode=futures). Leverage stays at 1x.
    """

    INTERFACE_VERSION = 3
    can_short = True

    # Selective: entries only on 15m (not 5m)
    timeframe = "15m"
    startup_candle_count = 300
    process_only_new_candles = True

    # Safety net only — exits are driven by TP1 / TP2 / stop
    minimal_roi = {"0": 0.12}

    stoploss = -0.25
    use_custom_stoploss = True
    trailing_stop = False

    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = True

    position_adjustment_enable = True
    max_entry_position_adjustment = 0

    # Pair-specific overrides for noisy alts
    NOISY_BASES = {"XRP", "TAO", "NEAR"}
    SHORT_ONLY_BASES = {"NEAR"}

    # --- Tunables (selective + stop/RR improvements) ---
    swing_left = IntParameter(2, 5, default=3, space="buy", optimize=True, load=True)
    swing_right = IntParameter(2, 5, default=3, space="buy", optimize=True, load=True)
    min_gap_atr = DecimalParameter(0.05, 0.4, default=0.12, decimals=2, space="buy", optimize=True, load=True)
    impulse_atr = DecimalParameter(0.8, 2.0, default=1.0, decimals=1, space="buy", optimize=True, load=True)
    eq_tol = DecimalParameter(0.0003, 0.002, default=0.001, decimals=4, space="buy", optimize=True, load=True)
    # Score = bias + zone + structure + sweep + SMT (max 5). Selective default = 4.
    min_score = IntParameter(3, 5, default=4, space="buy", optimize=True, load=True)
    require_smt = BooleanParameter(default=False, space="buy", optimize=True, load=True)
    bias_mode = CategoricalParameter(
        ["any_htf", "both_htf", "h1_only"], default="both_htf", space="buy", optimize=True, load=True
    )
    tp1_share = DecimalParameter(0.3, 0.7, default=0.5, decimals=1, space="sell", optimize=True, load=True)
    # Stop beyond sweep extreme + ATR buffer; skip if TP2 reward/risk < min_rr
    stop_atr_buf = DecimalParameter(0.4, 1.5, default=0.75, decimals=2, space="sell", optimize=True, load=True)
    min_rr = DecimalParameter(1.5, 3.5, default=2.0, decimals=1, space="sell", optimize=True, load=True)
    # Overrides for noisy alts (XRP/TAO/NEAR)
    noisy_stop_atr_buf = DecimalParameter(0.8, 2.0, default=1.25, decimals=2, space="sell", optimize=True, load=True)
    noisy_min_rr = DecimalParameter(2.0, 4.0, default=2.5, decimals=1, space="sell", optimize=True, load=True)

    order_types = {
        "entry": "limit",
        "exit": "limit",
        "stoploss": "market",
        "stoploss_on_exchange": False,
    }

    plot_config = {
        "main_plot": {
            "bull_fvg_top": {"color": "rgba(0,200,0,0.4)"},
            "bull_fvg_bottom": {"color": "rgba(0,200,0,0.4)"},
            "bear_fvg_top": {"color": "rgba(200,0,0,0.4)"},
            "bear_fvg_bottom": {"color": "rgba(200,0,0,0.4)"},
            "eq_high": {"color": "orange"},
            "eq_low": {"color": "orange"},
        },
        "subplots": {
            "score": {
                "long_score": {"color": "green"},
                "short_score": {"color": "red"},
            },
            "structure": {
                "ms_trend": {"color": "blue"},
                "ms_trend_1h": {"color": "purple"},
                "ms_trend_4h": {"color": "black"},
            },
        },
    }

    def _base(self, pair: str) -> str:
        return pair.split("/")[0].upper()

    def _is_noisy(self, pair: str) -> bool:
        return self._base(pair) in self.NOISY_BASES

    def _pair_rules(self, pair: str) -> dict[str, Any]:
        """Per-pair overrides. Noisy alts get stricter filters."""
        noisy = self._is_noisy(pair)
        return {
            "noisy": noisy,
            "stop_atr_buf": float(self.noisy_stop_atr_buf.value if noisy else self.stop_atr_buf.value),
            "min_rr": float(self.noisy_min_rr.value if noisy else self.min_rr.value),
            "require_smt": True if noisy else bool(self.require_smt.value),
            "allow_long": self._base(pair) not in self.SHORT_ONLY_BASES,
            "allow_short": True,
            "use_exit_signal": not noisy,
        }

    def _ref_pair(self) -> str:
        stake = self.config["stake_currency"]
        if self.config.get("trading_mode") == "futures":
            return f"BTC/{stake}:{stake}"
        return f"BTC/{stake}"

    def informative_pairs(self):
        whitelist = self.dp.current_whitelist() if self.dp else []
        ref = self._ref_pair()
        # Main TF is 15m — bias from 1h/4h, SMT vs BTC on 15m
        pairs = {(pair, tf) for pair in whitelist for tf in ("1h", "4h")}
        pairs.add((ref, self.timeframe))
        return list(pairs)

    def _params(self) -> dict[str, Any]:
        return {
            "swing_left": int(self.swing_left.value),
            "swing_right": int(self.swing_right.value),
            "min_gap_atr": float(self.min_gap_atr.value),
            "impulse_atr": float(self.impulse_atr.value),
            "eq_tol": float(self.eq_tol.value),
        }

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        p = self._params()
        pair = metadata["pair"]
        rules = self._pair_rules(pair)
        # Enrich ICT on the entry timeframe (15m)
        dataframe = _enrich_ict(dataframe, **p)

        for tf in ("1h", "4h"):
            inf = self.dp.get_pair_dataframe(pair=pair, timeframe=tf)
            if not inf.empty:
                inf = _add_structure(inf, p["swing_left"], p["swing_right"])[["date", "ms_trend"]]
                dataframe = merge_informative_pair(dataframe, inf, self.timeframe, tf, ffill=True)

        false_s = pd.Series(False, index=dataframe.index)
        zero_s = pd.Series(0, index=dataframe.index)

        def col(name: str, default: pd.Series) -> pd.Series:
            return dataframe[name].fillna(default) if name in dataframe else default

        # --- SMT vs BTC on 15m: alt sweeps liquidity while BTC does not ---
        ref = self._ref_pair()
        is_ref = pair == ref
        if not is_ref:
            btc = self.dp.get_pair_dataframe(pair=ref, timeframe=self.timeframe)
            if not btc.empty:
                btc = _add_structure(btc, p["swing_left"], p["swing_right"])
                btc = _add_liquidity(btc, p["eq_tol"])
                btc = btc[["date", "liq_sweep_low", "liq_sweep_high"]]
                dataframe = merge_informative_pair(
                    dataframe, btc, self.timeframe, self.timeframe, ffill=True, append_timeframe=False, suffix="btc"
                )
        btc_sweep_low = col("liq_sweep_low_btc", false_s).astype(bool)
        btc_sweep_high = col("liq_sweep_high_btc", false_s).astype(bool)
        smt_bull = (dataframe["liq_sweep_low"] & ~btc_sweep_low) if not is_ref else false_s
        smt_bear = (dataframe["liq_sweep_high"] & ~btc_sweep_high) if not is_ref else false_s

        # --- HTF bias (selective default: both 1h + 4h aligned) ---
        h1 = col("ms_trend_1h", zero_s)
        h4 = col("ms_trend_4h", zero_s)
        mode = self.bias_mode.value
        if mode == "both_htf":
            bias_bull = (h1 > 0) & (h4 > 0)
            bias_bear = (h1 < 0) & (h4 < 0)
        elif mode == "h1_only":
            bias_bull = h1 > 0
            bias_bear = h1 < 0
        else:
            bias_bull = ((h1 > 0) | (h4 > 0)) & (h1 >= 0)
            bias_bear = ((h1 < 0) | (h4 < 0)) & (h1 <= 0)

        # --- Structure on entry TF (15m) only ---
        struct_bull = dataframe["bos_bull"] | dataframe["choch_bull"]
        struct_bear = dataframe["bos_bear"] | dataframe["choch_bear"]

        zone_bull = dataframe["in_bull_fvg"] | dataframe["in_bull_ob"]
        zone_bear = dataframe["in_bear_fvg"] | dataframe["in_bear_ob"]

        atr = dataframe["atr"].fillna(dataframe["close"] * 0.01)
        buf = float(rules["stop_atr_buf"])
        # Stop beyond the sweep extreme (wider on noisy alts)
        sl_long = np.where(
            dataframe["liq_sweep_low"],
            dataframe["sweep_ext_low"] - atr * buf,
            np.nan,
        )
        sl_short = np.where(
            dataframe["liq_sweep_high"],
            dataframe["sweep_ext_high"] + atr * buf,
            np.nan,
        )

        entry = dataframe["close"]
        # Prospective TP2 = opposing equal high/low if further than entry; else 2R fallback
        risk_long = entry - sl_long
        risk_short = sl_short - entry
        eq_high = dataframe["eq_high"]
        eq_low = dataframe["eq_low"]
        tp2_long = np.where(
            eq_high.notna() & (eq_high > entry),
            eq_high,
            entry + 2.0 * risk_long,
        )
        tp2_short = np.where(
            eq_low.notna() & (eq_low < entry),
            eq_low,
            entry - 2.0 * risk_short,
        )
        rr_long = np.where(risk_long > 0, (tp2_long - entry) / risk_long, 0.0)
        rr_short = np.where(risk_short > 0, (entry - tp2_short) / risk_short, 0.0)

        extra = pd.DataFrame(
            {
                "smt_bull": smt_bull,
                "smt_bear": smt_bear,
                "smt_neutral": is_ref,
                "bias_bull": bias_bull,
                "bias_bear": bias_bear,
                "struct_bull": struct_bull,
                "struct_bear": struct_bear,
                "zone_bull": zone_bull,
                "zone_bear": zone_bear,
                "sl_long": sl_long,
                "sl_short": sl_short,
                "tp2_long": tp2_long,
                "tp2_short": tp2_short,
                "rr_long": rr_long,
                "rr_short": rr_short,
                "long_score": (
                    bias_bull.astype(int) + zone_bull.astype(int) + struct_bull.astype(int)
                    + dataframe["liq_sweep_low"].astype(int) + smt_bull.astype(int)
                ),
                "short_score": (
                    bias_bear.astype(int) + zone_bear.astype(int) + struct_bear.astype(int)
                    + dataframe["liq_sweep_high"].astype(int) + smt_bear.astype(int)
                ),
            },
            index=dataframe.index,
        )
        return pd.concat([dataframe, extra], axis=1)

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        rules = self._pair_rules(metadata["pair"])
        min_score = int(self.min_score.value)
        min_rr = float(rules["min_rr"])

        if rules["require_smt"]:
            smt_long = dataframe["smt_bull"] | dataframe["smt_neutral"]
            smt_short = dataframe["smt_bear"] | dataframe["smt_neutral"]
        else:
            smt_long = smt_short = pd.Series(True, index=dataframe.index)

        # zone AND sweep, plus min R:R on TP2 (stricter on noisy alts)
        long_cond = (
            dataframe["bias_bull"]
            & dataframe["zone_bull"]
            & dataframe["liq_sweep_low"]
            & (dataframe["long_score"] >= min_score)
            & smt_long
            & (dataframe["sl_long"] < dataframe["close"])
            & (dataframe["rr_long"] >= min_rr)
            & (dataframe["volume"] > 0)
        )
        short_cond = (
            dataframe["bias_bear"]
            & dataframe["zone_bear"]
            & dataframe["liq_sweep_high"]
            & (dataframe["short_score"] >= min_score)
            & smt_short
            & (dataframe["sl_short"] > dataframe["close"])
            & (dataframe["rr_short"] >= min_rr)
            & (dataframe["volume"] > 0)
            & ~long_cond
        )
        if not rules["allow_long"]:
            long_cond = pd.Series(False, index=dataframe.index)
        if not rules["allow_short"]:
            short_cond = pd.Series(False, index=dataframe.index)

        dataframe.loc[long_cond, "enter_long"] = 1
        dataframe.loc[long_cond, "enter_tag"] = "ict_long"
        dataframe.loc[short_cond, "enter_short"] = 1
        dataframe.loc[short_cond, "enter_tag"] = "ict_short"
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Noisy alts: disable CHoCH exit_signal (was 0% win there); keep stop/TP only
        if not self._pair_rules(metadata["pair"])["use_exit_signal"]:
            return dataframe
        dataframe.loc[dataframe["choch_bear"], "exit_long"] = 1
        dataframe.loc[dataframe["choch_bull"], "exit_short"] = 1
        return dataframe

    # ------------------------------------------------------------------ #
    # Trade management
    # ------------------------------------------------------------------ #
    def _signal_row(self, trade: Trade) -> pd.Series | None:
        """Candle that produced the entry signal (the last one closed before the entry order)."""
        dataframe, _ = self.dp.get_analyzed_dataframe(trade.pair, self.timeframe)
        if dataframe is None or dataframe.empty:
            return None
        entry_orders = [o for o in trade.orders if o.ft_order_side == trade.entry_side]
        order_time = entry_orders[0].order_date_utc if entry_orders else trade.open_date_utc
        candle_open = timeframe_to_prev_date(self.timeframe, order_time)
        rows = dataframe.loc[dataframe["date"] < candle_open]
        return None if rows.empty else rows.iloc[-1]

    def _levels(self, trade: Trade) -> dict | None:
        """Stop / TP1 / TP2 fixed once per trade and persisted in the trade's custom data."""
        levels = trade.get_custom_data("ict_levels")
        if levels:
            return levels
        row = self._signal_row(trade)
        if row is None:
            return None

        entry = trade.open_rate
        atr = float(row["atr"]) if not np.isnan(row.get("atr", np.nan)) else entry * 0.01
        sign = -1 if trade.is_short else 1
        stop = float(row["sl_short"] if trade.is_short else row["sl_long"])
        if np.isnan(stop) or sign * (entry - stop) <= 0:
            stop = entry - sign * float(self.stop_atr_buf.value) * 1.5 * atr
        risk = abs(entry - stop)

        internal = float(row["last_swing_low"] if trade.is_short else row["last_swing_high"])
        tp1 = internal if not np.isnan(internal) and sign * (internal - entry) >= risk else entry + sign * risk

        planned = row.get("tp2_short" if trade.is_short else "tp2_long")
        if planned is not None and not np.isnan(planned) and sign * (float(planned) - tp1) > 0:
            tp2 = float(planned)
        else:
            external = float(row["eq_low"] if trade.is_short else row["eq_high"])
            tp2 = external if not np.isnan(external) and sign * (external - tp1) > 0 else entry + sign * 2 * risk
            tp2 = tp2 if sign * (tp2 - tp1) > 0 else tp1 + sign * risk

        levels = {"stop": stop, "tp1": tp1, "tp2": tp2}
        trade.set_custom_data("ict_levels", levels)
        return levels

    def custom_stoploss(
        self,
        pair: str,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        after_fill: bool,
        **kwargs,
    ) -> float | None:
        levels = self._levels(trade)
        if not levels:
            return None
        # After TP1, the rest of the position runs risk-free
        stop = trade.open_rate if trade.nr_of_successful_exits > 0 else levels["stop"]
        return stoploss_from_absolute(stop, current_rate, is_short=trade.is_short, leverage=trade.leverage)

    def custom_exit(
        self,
        pair: str,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        **kwargs,
    ) -> str | bool | None:
        levels = self._levels(trade)
        if not levels:
            return None
        tp2 = levels["tp2"]
        if (trade.is_short and current_rate <= tp2) or (not trade.is_short and current_rate >= tp2):
            return "ict_tp2_liquidity"
        return None

    def adjust_trade_position(
        self,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        min_stake: float | None,
        max_stake: float,
        current_entry_rate: float,
        current_exit_rate: float,
        current_entry_profit: float,
        current_exit_profit: float,
        **kwargs,
    ) -> float | None | tuple[float | None, str | None]:
        if trade.nr_of_successful_exits > 0 or trade.has_open_orders:
            return None
        levels = self._levels(trade)
        if not levels:
            return None
        tp1 = levels["tp1"]
        if (trade.is_short and current_exit_rate <= tp1) or (not trade.is_short and current_exit_rate >= tp1):
            return -(trade.stake_amount * float(self.tp1_share.value)), "ict_tp1_partial"
        return None
