from datetime import datetime
import math
import numpy as np
import pandas as pd
import yfinance as yf

from database import (
    create_connection,
    get_historical_iv_snapshots,
    get_zero_rank_snapshots,
    insert_volatility,
    update_snapshot_iv_rank,
)


def norm_cdf(x: float) -> float:
    """Standard normal cumulative distribution function."""
    return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0


def norm_pdf(x: float) -> float:
    """Standard normal probability density function."""
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def black_scholes_call(
    spot: float,
    strike: float,
    dte: int | float,
    volatility: float,
    risk_free_rate: float = 0.045
) -> dict:
    """
    Computes theoretical call price and Greeks (Delta, Gamma, Theta, Vega)
    using the Black-Scholes analytical model.
    """
    s = float(spot)
    k = float(strike)
    r = float(risk_free_rate)
    intrinsic = max(0.0, s - k)

    if dte <= 0:
        return {
            "price": round(intrinsic, 2),
            "delta": 1.0 if s > k else 0.0,
            "gamma": 0.0,
            "theta": 0.0,
            "vega": 0.0,
        }

    t = max(float(dte), 0.5) / 365.0
    sigma = max(float(volatility), 0.02)

    try:
        d1 = (math.log(s / k) + (r + 0.5 * sigma * sigma) * t) / (sigma * math.sqrt(t))
        d2 = d1 - sigma * math.sqrt(t)

        price = s * norm_cdf(d1) - k * math.exp(-r * t) * norm_cdf(d2)
        delta = norm_cdf(d1)
        gamma = norm_pdf(d1) / (s * sigma * math.sqrt(t))
        # Daily theta decay (per calendar day)
        theta = (-(s * norm_pdf(d1) * sigma) / (2.0 * math.sqrt(t)) - r * k * math.exp(-r * t) * norm_cdf(d2)) / 365.0
        # Vega per 1.0% (0.01) change in IV
        vega = (s * math.sqrt(t) * norm_pdf(d1)) / 100.0

        # An American/European call on non-dividend stock must be >= intrinsic value
        fair_price = max(round(price, 2), round(intrinsic, 2))

        return {
            "price": fair_price,
            "delta": round(float(delta), 3),
            "gamma": round(float(gamma), 4),
            "theta": round(float(theta), 3),
            "vega": round(float(vega), 3),
        }
    except Exception:
        return {
            "price": round(intrinsic, 2),
            "delta": 1.0 if s > k else 0.0,
            "gamma": 0.0,
            "theta": 0.0,
            "vega": 0.0,
        }


def calc_20day_hv(ticker: str) -> float | None:
    """Calculates the 20-day annualized historical volatility from SQLite."""
    connection = None
    try:
        connection = create_connection()
        query = """
            SELECT close FROM stock_prices
            WHERE ticker = ?
            ORDER BY date DESC
            LIMIT 21
        """
        df = pd.read_sql_query(query, connection, params=(ticker,))

        if len(df) < 21:
            return None

        close = df['close'].values[::-1]
        if np.any(close <= 0):
            return None

        log_returns = np.log(np.array(close[1:]) / np.array(close[:-1]))
        daily_std = np.std(log_returns, ddof=1)
        if np.isnan(daily_std) or daily_std <= 0:
            return None

        annualized_hv = daily_std * math.sqrt(252)
        if math.isnan(annualized_hv) or math.isinf(annualized_hv):
            return None

        return float(annualized_hv)
    except Exception as e:
        print(f"Error calculating 20D HV for {ticker}: {e}")
        return None
    finally:
        if connection:
            connection.close()


def calc_iv_rank_proxy(ticker: str, current_iv: float) -> float | None:
    """Computes rolling 20-day HV range across up to 252 trading days to proxy IV Rank."""
    if current_iv is None or pd.isna(current_iv) or current_iv <= 0:
        return None

    connection = None
    try:
        connection = create_connection()
        query = """
            SELECT date, close FROM stock_prices
            WHERE ticker = ?
            ORDER BY date DESC
            LIMIT 272
        """
        df = pd.read_sql_query(query, connection, params=(ticker,))

        if len(df) < 60:
            return None

        df = df.sort_values("date").reset_index(drop=True)
        df["log_return"] = np.log(df["close"] / df["close"].shift(1))
        df["hv_20"] = df["log_return"].rolling(window=20).std(ddof=1) * math.sqrt(252)

        hv_series = df["hv_20"].dropna()
        if len(hv_series) < 20:
            return None

        min_hv = float(hv_series.min())
        max_hv = float(hv_series.max())

        if max_hv <= min_hv or pd.isna(max_hv):
            return 50.0

        iv_rank = ((current_iv - min_hv) / (max_hv - min_hv)) * 100.0
        return float(np.clip(iv_rank, 0.0, 100.0))

    except Exception as e:
        print(f"Error calculating IV Rank proxy for {ticker}: {e}")
        return None
    finally:
        if connection:
            connection.close()


def calc_iv_rank(ticker: str, current_iv: float) -> float | None:
    """
    Computes IV Rank consistently:
    1. If >= 20 historical IV snapshots exist, calculates empirical percentile.
    2. Otherwise falls back to 252-day rolling 20D HV proxy.
    """
    if current_iv is None or pd.isna(current_iv) or current_iv <= 0.01:
        return None

    history = get_historical_iv_snapshots(ticker)
    if len(history) >= 20:
        sample = history + [float(current_iv)]
        min_iv = min(sample)
        max_iv = max(sample)
        if max_iv > min_iv:
            rank = ((current_iv - min_iv) / (max_iv - min_iv)) * 100.0
            return float(np.clip(round(rank, 2), 0.0, 100.0))
        return 50.0

    proxy = calc_iv_rank_proxy(ticker, current_iv)
    return round(proxy, 2) if proxy is not None else None


def atm_iv(ticker: str) -> float | None:
    """Fetches live front-month ATM call implied volatility from yfinance, with robust after-hours fallback."""
    try:
        stock = yf.Ticker(ticker)

        spot_price = None
        try:
            spot_price = stock.fast_info['lastPrice']
        except Exception:
            pass

        if not spot_price or pd.isna(spot_price) or spot_price <= 0:
            connection = create_connection()
            cursor = connection.cursor()
            cursor.execute("SELECT close FROM stock_prices WHERE ticker = ? ORDER BY date DESC LIMIT 1", (ticker,))
            row = cursor.fetchone()
            connection.close()
            if row and row[0] and float(row[0]) > 0:
                spot_price = float(row[0])

        if not spot_price or spot_price <= 0:
            return None

        spot_price = float(spot_price)
        expirations = stock.options
        if not expirations:
            return None

        today = datetime.now().date()
        exp_candidates = []
        for exp in expirations:
            try:
                exp_dt = datetime.strptime(exp, "%Y-%m-%d").date()
                dte = (exp_dt - today).days
                if dte >= 2:
                    exp_candidates.append((exp, dte))
            except Exception:
                continue

        if not exp_candidates:
            target_expiration = expirations[0]
            dte = max(1, (datetime.strptime(target_expiration, "%Y-%m-%d").date() - today).days)
        else:
            front_month = [x for x in exp_candidates if 14 <= x[1] <= 45]
            if front_month:
                target_expiration, dte = front_month[0]
            else:
                target_expiration, dte = exp_candidates[0]

        options_chain = stock.option_chain(target_expiration)
        calls = options_chain.calls

        if calls is None or calls.empty:
            return None

        if isinstance(calls.columns, pd.MultiIndex):
            calls.columns = calls.columns.get_level_values(0)

        calls = calls.copy()
        calls['strike_diff'] = (calls['strike'] - spot_price).abs()
        atm_option = calls.loc[calls['strike_diff'].idxmin()]

        raw_iv = float(atm_option.get('impliedVolatility', 0.0) or 0.0)

        # After-hours or closed-market fallback when bid/ask/IV collapse
        if raw_iv < 0.05:
            last_p = float(atm_option.get('lastPrice', 0.0) or 0.0)
            strike = float(atm_option['strike'])
            if last_p > 0:
                t_years = max(dte, 1) / 365.0
                extrinsic = max(0.01, last_p - max(0.0, spot_price - strike))
                approx_iv = (extrinsic / spot_price) * math.sqrt(2 * math.pi / t_years)
                if 0.05 <= approx_iv <= 3.0:
                    raw_iv = approx_iv

        # Never return bogus zero or negative IV (must be tradeable IV > 1%)
        if raw_iv <= 0.01 or pd.isna(raw_iv) or math.isinf(raw_iv):
            return None

        return round(raw_iv, 4)

    except Exception as e:
        print(f"[{ticker}] Error fetching ATM IV: {e}")
        return None


def vol_scan(ticker: str, verbose: bool = True) -> dict | None:
    """Calculates 20D HV, ATM IV, and computes real IV Rank; stores snapshot in database."""
    hv_20 = calc_20day_hv(ticker)
    iv_atm = atm_iv(ticker)

    if hv_20 is None or iv_atm is None or iv_atm <= 0.01:
        if verbose:
            print(f"Skipping {ticker}: missing metrics (20D HV: {hv_20}, ATM IV: {iv_atm})")
        return None

    iv_rank = calc_iv_rank(ticker, iv_atm)
    if iv_rank is None:
        iv_rank = 50.0

    iv_rank = round(iv_rank, 2)
    insert_volatility(ticker, hv_20, iv_atm, iv_rank)
    if verbose:
        print(f"[{ticker}] Snapshot saved -> 20D HV: {hv_20:.2%}, ATM IV: {iv_atm:.2%}, IV Rank: {iv_rank:.1f}%")

    return {
        "ticker": ticker,
        "hv_20day": hv_20,
        "atm_iv": iv_atm,
        "iv_rank": iv_rank,
        "spread": iv_atm - hv_20
    }


def get_deep_itm_call(
    ticker: str,
    target_dte: tuple[int, int] = (60, 90),
    target_delta: float = 0.80
) -> dict | None:
    """
    Recommends a Phase 1 deep ITM call option (~0.80 Delta, 60-90 DTE).
    Computes theoretical pricing and Greeks (Delta, Theta, Vega) via Black-Scholes.
    """
    try:
        stock = yf.Ticker(ticker)
        spot_price = None
        try:
            spot_price = stock.fast_info['lastPrice']
        except Exception:
            pass

        if not spot_price or pd.isna(spot_price) or spot_price <= 0:
            connection = create_connection()
            cursor = connection.cursor()
            cursor.execute("SELECT close FROM stock_prices WHERE ticker = ? ORDER BY date DESC LIMIT 1", (ticker,))
            row = cursor.fetchone()
            connection.close()
            if row and row[0] and float(row[0]) > 0:
                spot_price = float(row[0])

        if not spot_price or spot_price <= 0:
            return None

        spot_price = float(spot_price)
        expirations = stock.options
        if not expirations:
            return None

        today = datetime.now().date()
        exp_candidates = []
        for exp in expirations:
            try:
                exp_dt = datetime.strptime(exp, "%Y-%m-%d").date()
                dte = (exp_dt - today).days
                if dte > 0:
                    exp_candidates.append((exp, dte))
            except Exception:
                continue

        if not exp_candidates:
            return None

        in_range = [x for x in exp_candidates if target_dte[0] <= x[1] <= target_dte[1]]
        target_mid = (target_dte[0] + target_dte[1]) / 2.0
        if in_range:
            chosen_exp, chosen_dte = min(in_range, key=lambda x: abs(x[1] - target_mid))
        else:
            # Fall back to expiration >= 45 DTE closest to target_mid
            longer_candidates = [x for x in exp_candidates if x[1] >= 45]
            if longer_candidates:
                chosen_exp, chosen_dte = min(longer_candidates, key=lambda x: abs(x[1] - target_mid))
            else:
                chosen_exp, chosen_dte = min(exp_candidates, key=lambda x: abs(x[1] - target_mid))

        opt_chain = stock.option_chain(chosen_exp)
        calls = opt_chain.calls
        if calls is None or calls.empty:
            return None

        if isinstance(calls.columns, pd.MultiIndex):
            calls.columns = calls.columns.get_level_values(0)

        # Baseline implied volatility for Greeks calculation
        base_iv = atm_iv(ticker) or 0.25

        candidate_rows = []
        for _, call in calls.iterrows():
            strike = float(call["strike"])
            # Focus on ITM strikes (strike < spot_price)
            if strike >= spot_price:
                continue

            contract_iv = float(call.get("impliedVolatility", 0.0) or 0.0)
            use_iv = contract_iv if contract_iv > 0.02 else base_iv

            bs = black_scholes_call(spot_price, strike, chosen_dte, use_iv)
            delta_diff = abs(bs["delta"] - target_delta)
            candidate_rows.append((delta_diff, call, bs, use_iv))

        if not candidate_rows:
            # If no strikes < spot_price, fall back to lowest available strike
            for _, call in calls.iterrows():
                strike = float(call["strike"])
                contract_iv = float(call.get("impliedVolatility", 0.0) or 0.0)
                use_iv = contract_iv if contract_iv > 0.02 else base_iv
                bs = black_scholes_call(spot_price, strike, chosen_dte, use_iv)
                candidate_rows.append((abs(strike - spot_price * 0.88), call, bs, use_iv))

        if not candidate_rows:
            return None

        # Select call minimizing distance to target delta (0.80)
        _, best_call, bs, final_iv = min(candidate_rows, key=lambda x: x[0])

        strike = float(best_call["strike"])
        bid = float(best_call.get("bid", 0.0) or 0.0)
        ask = float(best_call.get("ask", 0.0) or 0.0)
        last_price = float(best_call.get("lastPrice", 0.0) or 0.0)

        intrinsic = max(0.0, spot_price - strike)
        if ask > 0 and bid > 0:
            est_cost = (bid + ask) / 2.0
        elif ask > 0:
            est_cost = ask
        elif last_price > 0 and last_price >= intrinsic:
            est_cost = last_price
        else:
            est_cost = bs["price"]

        est_cost = max(est_cost, intrinsic)

        return {
            "ticker": ticker,
            "contract_symbol": str(best_call.get("contractSymbol", f"{ticker}-{chosen_exp}-{strike}")),
            "contract_type": "CALL",
            "expiry": chosen_exp,
            "dte": chosen_dte,
            "strike": strike,
            "spot_price": round(spot_price, 2),
            "bid": round(bid, 2),
            "ask": round(ask, 2),
            "last_price": round(last_price, 2),
            "estimated_cost": round(est_cost, 2),
            "theoretical_price": bs["price"],
            "delta": bs["delta"],
            "gamma": bs["gamma"],
            "theta": bs["theta"],
            "vega": bs["vega"],
            "implied_volatility": round(final_iv, 4),
            "itm_pct": round(((spot_price - strike) / spot_price) * 100.0, 2),
        }
    except Exception as e:
        print(f"[{ticker}] Error recommending deep ITM call: {e}")
        return None


def get_bloated_otm_call(
    ticker: str,
    target_dte: tuple[int, int] = (14, 35),
    target_delta: float = 0.30
) -> dict | None:
    """
    Recommends a Phase 2 bloated OTM call option (~0.30 Delta, 14-35 DTE) to sell as a hedge.
    Computes theoretical pricing and Greeks (Delta, Theta decay, Vega) via Black-Scholes.
    """
    try:
        stock = yf.Ticker(ticker)
        spot_price = None
        try:
            spot_price = stock.fast_info['lastPrice']
        except Exception:
            pass

        if not spot_price or pd.isna(spot_price) or spot_price <= 0:
            connection = create_connection()
            cursor = connection.cursor()
            cursor.execute("SELECT close FROM stock_prices WHERE ticker = ? ORDER BY date DESC LIMIT 1", (ticker,))
            row = cursor.fetchone()
            connection.close()
            if row and row[0] and float(row[0]) > 0:
                spot_price = float(row[0])

        if not spot_price or spot_price <= 0:
            return None

        spot_price = float(spot_price)
        expirations = stock.options
        if not expirations:
            return None

        today = datetime.now().date()
        exp_candidates = []
        for exp in expirations:
            try:
                exp_dt = datetime.strptime(exp, "%Y-%m-%d").date()
                dte = (exp_dt - today).days
                if dte > 0:
                    exp_candidates.append((exp, dte))
            except Exception:
                continue

        if not exp_candidates:
            return None

        in_range = [x for x in exp_candidates if target_dte[0] <= x[1] <= target_dte[1]]
        target_mid = (target_dte[0] + target_dte[1]) / 2.0
        if in_range:
            chosen_exp, chosen_dte = min(in_range, key=lambda x: abs(x[1] - target_mid))
        else:
            chosen_exp, chosen_dte = min(exp_candidates, key=lambda x: abs(x[1] - target_mid))

        opt_chain = stock.option_chain(chosen_exp)
        calls = opt_chain.calls
        if calls is None or calls.empty:
            return None

        if isinstance(calls.columns, pd.MultiIndex):
            calls.columns = calls.columns.get_level_values(0)

        base_iv = atm_iv(ticker) or 0.25

        candidate_rows = []
        for _, call in calls.iterrows():
            strike = float(call["strike"])
            # Focus on OTM strikes (strike > spot_price)
            if strike <= spot_price:
                continue

            contract_iv = float(call.get("impliedVolatility", 0.0) or 0.0)
            use_iv = contract_iv if contract_iv > 0.02 else base_iv

            bs = black_scholes_call(spot_price, strike, chosen_dte, use_iv)
            delta_diff = abs(bs["delta"] - target_delta)
            candidate_rows.append((delta_diff, call, bs, use_iv))

        if not candidate_rows:
            # Fall back to strike nearest spot * 1.07
            target_strike = spot_price * 1.07
            for _, call in calls.iterrows():
                strike = float(call["strike"])
                contract_iv = float(call.get("impliedVolatility", 0.0) or 0.0)
                use_iv = contract_iv if contract_iv > 0.02 else base_iv
                bs = black_scholes_call(spot_price, strike, chosen_dte, use_iv)
                candidate_rows.append((abs(strike - target_strike), call, bs, use_iv))

        if not candidate_rows:
            return None

        # Select call minimizing distance to target delta (0.30)
        _, best_call, bs, final_iv = min(candidate_rows, key=lambda x: x[0])

        strike = float(best_call["strike"])
        bid = float(best_call.get("bid", 0.0) or 0.0)
        ask = float(best_call.get("ask", 0.0) or 0.0)
        last_price = float(best_call.get("lastPrice", 0.0) or 0.0)

        if bid > 0 and ask > 0:
            est_credit = (bid + ask) / 2.0
        elif bid > 0:
            est_credit = bid
        elif last_price > 0:
            est_credit = last_price
        else:
            est_credit = bs["price"]

        return {
            "ticker": ticker,
            "contract_symbol": str(best_call.get("contractSymbol", f"{ticker}-{chosen_exp}-{strike}")),
            "contract_type": "CALL",
            "expiry": chosen_exp,
            "dte": chosen_dte,
            "strike": strike,
            "spot_price": round(spot_price, 2),
            "bid": round(bid, 2),
            "ask": round(ask, 2),
            "last_price": round(last_price, 2),
            "estimated_credit": round(est_credit, 2),
            "theoretical_price": bs["price"],
            "delta": bs["delta"],
            "gamma": bs["gamma"],
            "theta": bs["theta"],
            "vega": bs["vega"],
            "implied_volatility": round(final_iv, 4),
            "otm_pct": round(((strike - spot_price) / spot_price) * 100.0, 2),
        }
    except Exception as e:
        print(f"[{ticker}] Error recommending bloated OTM call: {e}")
        return None


def backfill_all_iv_ranks(verbose: bool = True) -> int:
    """Backfills missing or zero IV ranks across all historical volatility snapshots."""
    zero_snaps = get_zero_rank_snapshots()
    if not zero_snaps:
        if verbose:
            print("No zero-rank snapshots found. All records are up to date.")
        return 0

    if verbose:
        print(f"Backfilling IV ranks for {len(zero_snaps)} snapshots...")

    updated_count = 0
    for snap in zero_snaps:
        s_id = snap["id"]
        ticker = snap["ticker"]
        atm_vol = float(snap["atm_iv"])

        rank = calc_iv_rank_proxy(ticker, atm_vol)
        if rank is not None:
            if update_snapshot_iv_rank(s_id, rank):
                updated_count += 1

    if verbose:
        print(f"Successfully backfilled {updated_count}/{len(zero_snaps)} snapshot IV ranks.")

    return updated_count
