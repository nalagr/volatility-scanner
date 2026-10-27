from datetime import datetime, timedelta
import math
import numpy as np
import pandas as pd
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from database import create_connection
from fetch import getData
from volatility import black_scholes_call

console = Console()


def run_backtest(
    ticker: str,
    initial_capital: float = 10000.0,
    risk_free_rate: float = 0.045
) -> dict:
    """
    Simulates the 2-phase dynamic volatility spread strategy on historical price data.
    - Phase 1: Buys ~0.80 delta 75 DTE call when rolling IV Rank <= 35%.
    - Phase 2: Sells ~0.30 delta 25 DTE call when rolling IV Rank surges >= 70%.
    - Manages rolls, expirations, cumulative cost basis reductions, and compares vs buy-and-hold.
    """
    # Guarantee price data is present
    getData(ticker, verbose=False)

    conn = create_connection()
    df = pd.read_sql_query(
        "SELECT date, close FROM stock_prices WHERE ticker = ? ORDER BY date ASC",
        conn,
        params=(ticker,)
    )
    conn.close()

    if len(df) < 80:
        console.print(f"[red]Insufficient historical data ({len(df)} bars) for backtesting {ticker}. Requires >= 80 days.[/red]")
        return {}

    # Compute rolling 20-day historical volatility
    df["log_ret"] = np.log(df["close"] / df["close"].shift(1))
    df["hv_20"] = df["log_ret"].rolling(window=20).std(ddof=1) * math.sqrt(252)

    # Compute rolling 252-day IV rank proxy
    df["min_hv"] = df["hv_20"].rolling(window=252, min_periods=40).min()
    df["max_hv"] = df["hv_20"].rolling(window=252, min_periods=40).max()
    df["iv_rank"] = ((df["hv_20"] - df["min_hv"]) / (df["max_hv"] - df["min_hv"] + 1e-6)) * 100.0
    df["iv_rank"] = df["iv_rank"].clip(0.0, 100.0)

    # Backtest state
    cash = initial_capital
    portfolio_values = []
    trades = []

    pos_state = "NONE"  # "NONE", "PHASE1_OPEN", "PHASE2_HEDGED"
    active_trade = {}

    for i in range(50, len(df)):
        row = df.iloc[i]
        date_str = str(row["date"])[:10]
        spot = float(row["close"])
        hv = float(row["hv_20"]) if not pd.isna(row["hv_20"]) else 0.25
        iv_rank = float(row["iv_rank"]) if not pd.isna(row["iv_rank"]) else 50.0

        # Update active position DTE
        if pos_state in ("PHASE1_OPEN", "PHASE2_HEDGED"):
            active_trade["long_dte"] -= 1
            if pos_state == "PHASE2_HEDGED":
                active_trade["short_dte"] -= 1

                # Check short leg expiration
                if active_trade["short_dte"] <= 0:
                    short_strike = active_trade["short_strike"]
                    # If spot > short strike, pay intrinsic settlement
                    loss_on_short = max(0.0, spot - short_strike)
                    cash -= loss_on_short * 100.0
                    active_trade["notes"].append(f"Short call expired. Cash settlement: -${loss_on_short:.2f}/sh")
                    # Revert to PHASE1_OPEN
                    pos_state = "PHASE1_OPEN"
                    active_trade["short_strike"] = None

            # Check long leg expiration (or closing when DTE <= 10)
            if active_trade["long_dte"] <= 10:
                # Sell/exercise remaining long call for fair value
                bs_long = black_scholes_call(spot, active_trade["long_strike"], max(1, active_trade["long_dte"]), hv, risk_free_rate)
                exit_price = bs_long["price"]
                cash += exit_price * 100.0

                net_pnl = (exit_price - active_trade["net_cost_basis"]) * 100.0
                trades.append({
                    "entry_date": active_trade["entry_date"],
                    "exit_date": date_str,
                    "initial_cost": active_trade["initial_cost"],
                    "net_cost_basis": active_trade["net_cost_basis"],
                    "credits_collected": active_trade["credits_collected"],
                    "exit_price": exit_price,
                    "pnl": net_pnl,
                    "pct_gain": (net_pnl / (active_trade["initial_cost"] * 100.0)) * 100.0 if active_trade["initial_cost"] > 0 else 0.0
                })
                pos_state = "NONE"
                active_trade = {}

        # Trade Entry Rules
        if pos_state == "NONE":
            # Phase 1 Entry: IV Rank <= 35%
            if iv_rank <= 35.0 and cash >= (spot * 0.15 * 100):
                long_strike = round(spot * 0.88, 2)
                long_dte = 75
                bs = black_scholes_call(spot, long_strike, long_dte, hv, risk_free_rate)
                long_cost = bs["price"]

                cash -= long_cost * 100.0
                pos_state = "PHASE1_OPEN"
                active_trade = {
                    "entry_date": date_str,
                    "spot_entry": spot,
                    "long_strike": long_strike,
                    "long_dte": long_dte,
                    "initial_cost": long_cost,
                    "net_cost_basis": long_cost,
                    "credits_collected": 0.0,
                    "short_strike": None,
                    "short_dte": 0,
                    "notes": []
                }

        elif pos_state == "PHASE1_OPEN":
            # Phase 2 Hedge Harvest: IV Rank >= 70%
            if iv_rank >= 70.0 and active_trade["long_dte"] > 30:
                short_strike = round(spot * 1.06, 2)
                short_dte = 25
                bs_short = black_scholes_call(spot, short_strike, short_dte, hv, risk_free_rate)
                short_credit = bs_short["price"]

                cash += short_credit * 100.0
                active_trade["credits_collected"] += short_credit
                active_trade["net_cost_basis"] = round(active_trade["net_cost_basis"] - short_credit, 2)
                active_trade["short_strike"] = short_strike
                active_trade["short_dte"] = short_dte
                pos_state = "PHASE2_HEDGED"

        # Record current portfolio equity
        open_val = 0.0
        if pos_state in ("PHASE1_OPEN", "PHASE2_HEDGED"):
            long_val = black_scholes_call(spot, active_trade["long_strike"], max(1, active_trade["long_dte"]), hv, risk_free_rate)["price"]
            short_val = 0.0
            if pos_state == "PHASE2_HEDGED" and active_trade["short_strike"]:
                short_val = black_scholes_call(spot, active_trade["short_strike"], max(1, active_trade["short_dte"]), hv, risk_free_rate)["price"]
            open_val = (long_val - short_val) * 100.0

        portfolio_values.append(cash + open_val)

    # Benchmark: Buy and Hold underlying stock
    first_close = float(df.iloc[50]["close"])
    last_close = float(df.iloc[-1]["close"])
    bh_return_pct = ((last_close - first_close) / first_close) * 100.0

    final_equity = portfolio_values[-1] if portfolio_values else initial_capital
    strategy_return_pct = ((final_equity - initial_capital) / initial_capital) * 100.0

    # Max Drawdown
    equity_series = pd.Series(portfolio_values)
    roll_max = equity_series.cummax()
    drawdown = (equity_series - roll_max) / roll_max
    max_dd_pct = float(drawdown.min()) * 100.0 if not drawdown.empty else 0.0

    wins = [t for t in trades if t["pnl"] > 0]
    win_rate = (len(wins) / len(trades)) * 100.0 if trades else 0.0
    avg_credit = float(np.mean([t["credits_collected"] for t in trades])) if trades else 0.0
    avg_basis_reduction = float(np.mean([(t["credits_collected"] / t["initial_cost"]) * 100.0 for t in trades if t["initial_cost"] > 0])) if trades else 0.0

    # Display Report Table
    table = Table(
        title=f"Quantitative Backtest Performance: {ticker} ({len(trades)} Trades)",
        header_style="bold green",
        show_lines=True
    )
    table.add_column("Metric", style="bold white")
    table.add_column("Strategy Value", justify="right", style="bold cyan")
    table.add_column("Benchmark (Buy & Hold)", justify="right", style="bold yellow")

    table.add_row("Initial Capital", f"${initial_capital:,.2f}", f"${initial_capital:,.2f}")
    table.add_row("Final Equity", f"${final_equity:,.2f}", f"${initial_capital * (1 + bh_return_pct/100.0):,.2f}")
    table.add_row("Total Return", f"{strategy_return_pct:+.2f}%", f"{bh_return_pct:+.2f}%")
    table.add_row("Maximum Drawdown", f"{max_dd_pct:.2f}%", "-")
    table.add_row("Total Trades", str(len(trades)), "1 (Buy & Hold)")
    table.add_row("Win Rate", f"{win_rate:.1f}%", "-")
    table.add_row("Avg Credit Harvested / Trade", f"${avg_credit:.2f}/sh", "$0.00")
    table.add_row("Avg Cost Basis Reduction", f"-{avg_basis_reduction:.1f}%", "0.0%")

    console.print(table)

    return {
        "ticker": ticker,
        "initial_capital": initial_capital,
        "final_equity": round(final_equity, 2),
        "strategy_return_pct": round(strategy_return_pct, 2),
        "benchmark_return_pct": round(bh_return_pct, 2),
        "max_drawdown_pct": round(max_dd_pct, 2),
        "total_trades": len(trades),
        "win_rate": round(win_rate, 2),
        "avg_basis_reduction_pct": round(avg_basis_reduction, 2),
        "trades": trades
    }
