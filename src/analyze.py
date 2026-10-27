import csv
from datetime import datetime
import json
import os
import sys
import urllib.request
import pandas as pd
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
import yfinance as yf

# Ensure UTF-8 output on Windows consoles
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

console = Console()

from database import (
    create_connection,
    get_hedged_positions,
    get_open_positions,
    get_positions,
    insert_position,
    update_position_hedge,
    roll_position_hedge,
    unhedge_position,
    close_position,
    update_snapshot_iv_rank,
)
from volatility import (
    atm_iv,
    calc_20day_hv,
    calc_iv_rank,
    get_bloated_otm_call,
    get_deep_itm_call,
)

console = Console()


def get_next_earnings(ticker: str) -> str:
    """Extracts next upcoming earnings date from yfinance calendar."""
    try:
        stock = yf.Ticker(ticker)
        cal = stock.calendar
        if cal is None:
            return "N/A"

        if isinstance(cal, dict) and "Earnings Date" in cal and cal["Earnings Date"]:
            ed = cal["Earnings Date"]
            if isinstance(ed, list) and len(ed) > 0:
                return str(ed[0])
            return str(ed)
        elif isinstance(cal, pd.DataFrame):
            if "Earnings Date" in cal.index:
                val = cal.loc["Earnings Date"].iloc[0]
                return str(val)[:10]
        return "N/A"
    except Exception:
        return "N/A"


def send_webhook_alert(webhook_url: str, title: str, message: str, color: int = 0x00FF00):
    """Sends a rich webhook payload (Discord / Slack compatible) for trade alerts."""
    try:
        payload = {
            "embeds": [{
                "title": title,
                "description": message,
                "color": color,
                "timestamp": datetime.utcnow().isoformat()
            }]
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            webhook_url,
            data=data,
            headers={"Content-Type": "application/json", "User-Agent": "VolatilityArbitrage/1.0"}
        )
        with urllib.request.urlopen(req, timeout=5) as response:
            pass
        console.print("[green]Webhook notification sent successfully.[/green]")
    except Exception as e:
        console.print(f"[yellow]Failed to send webhook notification: {e}[/yellow]")


def screen_opportunities(
    phase: str = "all",
    limit: int = 10,
    export_csv: str | None = None
) -> list[dict]:
    """
    Screens stocks from volatility snapshots for Phase 1 (quiet accumulation)
    and Phase 2 (volatility surge / harvest sell) opportunities.
    """
    connection = None
    records = []
    try:
        connection = create_connection()
        query = """
            WITH latest_snap AS (
                SELECT ticker, MAX(id) as max_id
                FROM volatility_snapshots
                GROUP BY ticker
            )
            SELECT v.id, v.ticker, v.timestamp, v.hv_20day, v.atm_iv, v.iv_rank, (v.atm_iv - v.hv_20day) as vol_spread
            FROM volatility_snapshots v
            JOIN latest_snap ls ON v.id = ls.max_id
            WHERE v.hv_20day IS NOT NULL AND v.atm_iv IS NOT NULL AND v.atm_iv > 0.01
        """
        rows = connection.execute(query).fetchall()

        for r in rows:
            ticker = r["ticker"]
            snap_id = r["id"]
            hv = float(r["hv_20day"])
            iv = float(r["atm_iv"])
            iv_rk = float(r["iv_rank"]) if r["iv_rank"] is not None else 0.0

            # Heal and persist 0.0 IV ranks using proxy if needed
            if iv_rk == 0.0:
                computed_rank = calc_iv_rank(ticker, iv)
                if computed_rank is not None:
                    iv_rk = computed_rank
                    update_snapshot_iv_rank(snap_id, iv_rk)

            spread = iv - hv
            records.append({
                "ticker": ticker,
                "timestamp": r["timestamp"],
                "hv_20day": hv,
                "atm_iv": iv,
                "iv_rank": iv_rk,
                "vol_spread": spread,
            })
    finally:
        if connection:
            connection.close()

    if not records:
        console.print("[yellow]No valid volatility snapshots found in database. Run 'main.py scan' first.[/yellow]")
        return []

    df = pd.DataFrame(records)

    phase1_candidates = []
    phase2_candidates = []

    # Phase 1: Underpriced options during quiet periods (IV Rank <= 35% AND vol_spread <= 0.05)
    p1_filter = (df["iv_rank"] <= 35.0) & (df["vol_spread"] <= 0.05)
    p1_df = df[p1_filter].sort_values(by=["vol_spread", "iv_rank"], ascending=[True, True]).head(limit)

    # Phase 2: Bloated options during volatility surges (IV Rank >= 70% OR vol_spread >= 0.15)
    p2_filter = (df["iv_rank"] >= 70.0) | (df["vol_spread"] >= 0.15)
    p2_df = df[p2_filter].sort_values(by=["vol_spread", "iv_rank"], ascending=[False, False]).head(limit)

    all_exported = []

    # Display Phase 1
    if phase in ("1", "all"):
        p1_table = Table(
            title=f"Phase 1: Accumulation Screener (Low IV Rank <= 35%, Low Spread <= 0.05) [Top {len(p1_df)}]",
            header_style="bold cyan",
            show_lines=True
        )
        p1_table.add_column("Ticker", style="bold white")
        p1_table.add_column("Spot", justify="right")
        p1_table.add_column("20D HV", justify="right")
        p1_table.add_column("ATM IV", justify="right")
        p1_table.add_column("Spread", justify="right")
        p1_table.add_column("IV Rank", justify="right")
        p1_table.add_column("Earnings", justify="center")
        p1_table.add_column("Rec Deep ITM Call", justify="left")
        p1_table.add_column("DTE", justify="right")
        p1_table.add_column("Strike", justify="right")
        p1_table.add_column("Est Cost", justify="right")
        p1_table.add_column("Delta", justify="right")
        p1_table.add_column("Theta/day", justify="right")

        console.print("\n[bold cyan]Fetching contract recommendations for Phase 1 candidates...[/bold cyan]")
        for _, row in p1_df.iterrows():
            ticker = row["ticker"]
            earnings_date = get_next_earnings(ticker)
            contract = get_deep_itm_call(ticker)

            spot_str = f"${contract['spot_price']:.2f}" if contract else "-"
            contract_sym = contract["contract_symbol"] if contract else "N/A"
            dte_str = str(contract["dte"]) if contract else "-"
            strike_str = f"${contract['strike']:.2f}" if contract else "-"
            cost_str = f"${contract['estimated_cost']:.2f}" if contract else "-"
            delta_str = f"{contract['delta']:.2f}" if contract else "-"
            theta_str = f"${contract['theta']:.2f}" if contract else "-"

            p1_table.add_row(
                ticker,
                spot_str,
                f"{row['hv_20day']:.1%}",
                f"{row['atm_iv']:.1%}",
                f"{row['vol_spread']:+.1%}",
                f"{row['iv_rank']:.1f}%",
                earnings_date,
                contract_sym,
                dte_str,
                strike_str,
                cost_str,
                delta_str,
                theta_str
            )

            rec = {
                "phase": "PHASE1_BUY",
                "ticker": ticker,
                "spot": contract["spot_price"] if contract else None,
                "hv_20day": row["hv_20day"],
                "atm_iv": row["atm_iv"],
                "vol_spread": row["vol_spread"],
                "iv_rank": row["iv_rank"],
                "next_earnings": earnings_date,
                "rec_contract": contract_sym,
                "dte": contract["dte"] if contract else None,
                "strike": contract["strike"] if contract else None,
                "est_price": contract["estimated_cost"] if contract else None,
                "delta": contract["delta"] if contract else None,
                "theta": contract["theta"] if contract else None,
                "vega": contract["vega"] if contract else None,
            }
            phase1_candidates.append(rec)
            all_exported.append(rec)

        console.print(p1_table)

    # Display Phase 2
    if phase in ("2", "all"):
        p2_table = Table(
            title=f"Phase 2: Harvest Screener (High IV Rank >= 70% OR High Spread >= 15%) [Top {len(p2_df)}]",
            header_style="bold magenta",
            show_lines=True
        )
        p2_table.add_column("Ticker", style="bold white")
        p2_table.add_column("Spot", justify="right")
        p2_table.add_column("20D HV", justify="right")
        p2_table.add_column("ATM IV", justify="right")
        p2_table.add_column("Spread", justify="right")
        p2_table.add_column("IV Rank", justify="right")
        p2_table.add_column("Earnings", justify="center")
        p2_table.add_column("Rec Bloated OTM Call", justify="left")
        p2_table.add_column("DTE", justify="right")
        p2_table.add_column("Strike", justify="right")
        p2_table.add_column("Est Credit", justify="right")
        p2_table.add_column("Delta", justify="right")
        p2_table.add_column("Theta/day", justify="right")

        console.print("\n[bold magenta]Fetching contract recommendations for Phase 2 candidates...[/bold magenta]")
        for _, row in p2_df.iterrows():
            ticker = row["ticker"]
            earnings_date = get_next_earnings(ticker)
            contract = get_bloated_otm_call(ticker)

            spot_str = f"${contract['spot_price']:.2f}" if contract else "-"
            contract_sym = contract["contract_symbol"] if contract else "N/A"
            dte_str = str(contract["dte"]) if contract else "-"
            strike_str = f"${contract['strike']:.2f}" if contract else "-"
            credit_str = f"${contract['estimated_credit']:.2f}" if contract else "-"
            delta_str = f"{contract['delta']:.2f}" if contract else "-"
            theta_str = f"${contract['theta']:.2f}" if contract else "-"

            p2_table.add_row(
                ticker,
                spot_str,
                f"{row['hv_20day']:.1%}",
                f"{row['atm_iv']:.1%}",
                f"{row['vol_spread']:+.1%}",
                f"{row['iv_rank']:.1f}%",
                earnings_date,
                contract_sym,
                dte_str,
                strike_str,
                credit_str,
                delta_str,
                theta_str
            )

            rec = {
                "phase": "PHASE2_SELL",
                "ticker": ticker,
                "spot": contract["spot_price"] if contract else None,
                "hv_20day": row["hv_20day"],
                "atm_iv": row["atm_iv"],
                "vol_spread": row["vol_spread"],
                "iv_rank": row["iv_rank"],
                "next_earnings": earnings_date,
                "rec_contract": contract_sym,
                "dte": contract["dte"] if contract else None,
                "strike": contract["strike"] if contract else None,
                "est_price": contract["estimated_credit"] if contract else None,
                "delta": contract["delta"] if contract else None,
                "theta": contract["theta"] if contract else None,
                "vega": contract["vega"] if contract else None,
            }
            phase2_candidates.append(rec)
            all_exported.append(rec)

        console.print(p2_table)

    if export_csv and all_exported:
        export_df = pd.DataFrame(all_exported)
        export_df.to_csv(export_csv, index=False)
        console.print(f"\n[green]Successfully exported {len(all_exported)} opportunities to {export_csv}[/green]")

    return all_exported


def check_position_alerts(webhook_url: str | None = None):
    """
    Monitors all open Phase 1 positions against live market implied volatility
    and Phase 2 hedged positions for expiring short calls.
    """
    target_webhook = webhook_url or os.getenv("ALERT_WEBHOOK_URL")

    # 1. Evaluate Phase 1 positions for IV surge harvest triggers
    open_positions = get_open_positions()
    hedged_positions = get_hedged_positions()

    if not open_positions and not hedged_positions:
        console.print(Panel(
            "[yellow]No active Phase 1 or Phase 2 positions found to monitor.[/yellow]\n"
            "Log a position with: [cyan]python src/main.py track add --ticker <TICKER> --strike <STRIKE> --expiry <EXPIRY> --cost <COST>[/cyan]",
            title="Position Tracker Alert Monitor",
            border_style="yellow"
        ))
        return

    today = datetime.now().date()

    if open_positions:
        console.print(f"\n[bold]Checking {len(open_positions)} open Phase 1 position(s) for IV surge harvest triggers...[/bold]\n")

        for pos in open_positions:
            ticker = pos["ticker"]
            pos_id = pos["id"]
            long_strike = pos["long_strike"]
            long_expiry = pos["long_expiry"]
            long_cost = pos["long_cost"]

            # Fetch live spot price with SQLite fallback
            stock = yf.Ticker(ticker)
            spot_price = None
            try:
                spot_price = stock.fast_info['lastPrice']
            except Exception:
                pass

            if not spot_price or pd.isna(spot_price) or spot_price <= 0:
                conn = create_connection()
                row = conn.execute("SELECT close FROM stock_prices WHERE ticker = ? ORDER BY date DESC LIMIT 1", (ticker,)).fetchone()
                conn.close()
                if row and row[0]:
                    spot_price = float(row[0])

            hv_20 = calc_20day_hv(ticker)
            iv = atm_iv(ticker)

            if hv_20 is None or iv is None:
                console.print(f"[yellow][Pos #{pos_id} {ticker}][/yellow] Insufficient market metrics to evaluate harvest alert.")
                continue

            vol_spread = iv - hv_20
            iv_rank = calc_iv_rank(ticker, iv) or 50.0

            is_harvest_trigger = (iv_rank >= 70.0) or (vol_spread >= 0.15)
            spot_display = f"${spot_price:.2f}" if spot_price else "N/A"

            if is_harvest_trigger:
                otm_call = get_bloated_otm_call(ticker)
                credit = otm_call["estimated_credit"] if otm_call else 0.0
                new_net_cost = round(long_cost - credit, 2)
                pct_reduction = (credit / long_cost) * 100.0 if long_cost > 0 else 0.0

                if otm_call:
                    contract_info = (
                        f"Recommended Short Leg: [bold yellow]Sell {otm_call['contract_symbol']}[/bold yellow]\n"
                        f"  Strike: ${otm_call['strike']:.2f} | Expiry: {otm_call['expiry']} ({otm_call['dte']} DTE) | Est. Credit: [bold green]${credit:.2f}[/bold green]\n"
                        f"  Greeks: Delta {otm_call['delta']} | Theta: +${abs(otm_call['theta']):.2f}/day decay | Vega: ${otm_call['vega']:.2f}\n\n"
                        f"Projected Net Cost Basis: [bold cyan]${new_net_cost:.2f}[/bold cyan] ([bold green]-{pct_reduction:.1f}%[/bold green] basis reduction)\n\n"
                        f"[bold white]Action Command to Lock In Hedge:[/bold white]\n"
                        f"[cyan]python src/main.py track hedge --id {pos_id} --strike {otm_call['strike']} --expiry {otm_call['expiry']} --credit {credit:.2f}[/cyan]"
                    )
                else:
                    contract_info = (
                        f"[yellow]High IV surge detected, but no live option quotes were returned for {ticker}. Check options chain manually.[/yellow]"
                    )

                alert_msg = (
                    f"[bold red]HARVEST SELL TRIGGER MET![/bold red]\n"
                    f"Position ID: [bold]{pos_id}[/bold] | Ticker: [bold white]{ticker}[/bold white]\n"
                    f"Current Spot: {spot_display} | 20D HV: {hv_20:.1%} | Live ATM IV: {iv:.1%}\n"
                    f"Vol Spread: [bold green]{vol_spread:+.1%}[/bold green] | IV Rank: [bold green]{iv_rank:.1f}%[/bold green]\n\n"
                    f"Active Long Leg: Strike ${long_strike:.2f} Exp {long_expiry} (Cost Basis: ${long_cost:.2f})\n\n"
                    f"{contract_info}"
                )
                console.print(Panel(alert_msg, title=f"ALERT: Phase 2 Harvest Opportunity for {ticker}", border_style="red"))

                if target_webhook:
                    plain_msg = (
                        f"**HARVEST SELL TRIGGER MET for {ticker} (Pos #{pos_id})**\n"
                        f"Spot: {spot_display} | IV: {iv:.1%} | 20D HV: {hv_20:.1%} | Spread: {vol_spread:+.1%} | IV Rank: {iv_rank:.1f}%\n"
                        f"Long Leg: ${long_strike:.2f} Exp {long_expiry} (Cost: ${long_cost:.2f})\n"
                        f"Action: Sell OTM Call to lock in cost reduction."
                    )
                    send_webhook_alert(target_webhook, f"Trading Alert: Harvest Trigger for {ticker}", plain_msg, color=0xFF0000)
            else:
                status_msg = (
                    f"Position #{pos_id} [{ticker}]: Status: [cyan]HOLDING PHASE 1[/cyan] | "
                    f"Spot: {spot_display} | 20D HV: {hv_20:.1%} | ATM IV: {iv:.1%} | Spread: {vol_spread:+.1%} | IV Rank: {iv_rank:.1f}%\n"
                    f"Threshold not reached yet (Requires IV Rank >= 70% or Spread >= +15%)."
                )
                console.print(Panel(status_msg, border_style="dim"))

    # 2. Evaluate Phase 2 hedged positions for expiring short calls
    if hedged_positions:
        console.print(f"\n[bold]Checking {len(hedged_positions)} hedged Phase 2 position(s) for expiration & roll opportunities...[/bold]\n")

        for pos in hedged_positions:
            ticker = pos["ticker"]
            pos_id = pos["id"]
            short_strike = pos["short_strike"]
            short_expiry_str = pos["short_expiry"]
            short_credit = pos["short_credit"] or 0.0
            net_basis = pos["net_cost_basis"]

            if not short_expiry_str:
                continue

            try:
                short_exp_dt = datetime.strptime(short_expiry_str[:10], "%Y-%m-%d").date()
                dte_left = (short_exp_dt - today).days

                if dte_left <= 0:
                    # Short call expired
                    expired_msg = (
                        f"[bold yellow]SHORT CALL HEDGE HAS EXPIRED![/bold yellow]\n"
                        f"Position ID: [bold]{pos_id}[/bold] | Ticker: [bold white]{ticker}[/bold white]\n"
                        f"Short Leg: Strike ${short_strike:.2f} expired on {short_expiry_str} ({dte_left} DTE)\n"
                        f"Current Net Cost Basis: [bold green]${net_basis:.2f}[/bold green]\n\n"
                        f"[bold white]Options to take:[/bold white]\n"
                        f"1. Return to Phase 1 accumulation:\n"
                        f"   [cyan]python src/main.py track unhedge --id {pos_id}[/cyan]\n"
                        f"2. Roll into a new monthly short call:\n"
                        f"   [cyan]python src/main.py track roll --id {pos_id} --strike <NEW_STRIKE> --expiry <NEW_EXPIRY> --credit <NEW_CREDIT>[/cyan]"
                    )
                    console.print(Panel(expired_msg, title=f"HEDGE EXPIRED: {ticker} (Pos #{pos_id})", border_style="yellow"))
                    if target_webhook:
                        send_webhook_alert(target_webhook, f"Hedge Expired for {ticker}", f"Short call expired. Run track unhedge or track roll for Pos #{pos_id}.", color=0xFFFF00)
                elif dte_left <= 7:
                    # Approaching expiration
                    expiring_msg = (
                        f"[bold cyan]SHORT CALL HEDGE EXPIRING SOON ({dte_left} DTE)[/bold cyan]\n"
                        f"Position ID: [bold]{pos_id}[/bold] | Ticker: [bold white]{ticker}[/bold white]\n"
                        f"Short Leg: Strike ${short_strike:.2f} exp {short_expiry_str}\n"
                        f"Time to evaluate rolling to the next monthly cycle:\n"
                        f"[cyan]python src/main.py track roll --id {pos_id} --strike <NEW_STRIKE> --expiry <NEW_EXPIRY> --credit <NEW_CREDIT>[/cyan]"
                    )
                    console.print(Panel(expiring_msg, title=f"HEDGE EXPIRING: {ticker} ({dte_left} DTE Left)", border_style="cyan"))
                else:
                    console.print(
                        f"Position #{pos_id} [{ticker}]: Status: [magenta]PHASE 2 HEDGED[/magenta] | "
                        f"Short Call ${short_strike:.2f} exp {short_expiry_str} ({dte_left} DTE remaining) | Net Basis: ${net_basis:.2f}"
                    )
            except Exception as e:
                console.print(f"[yellow]Could not parse expiry '{short_expiry_str}' for Pos #{pos_id}: {e}[/yellow]")


def validate_date(date_str: str) -> bool:
    """Validates that a date string matches YYYY-MM-DD."""
    try:
        datetime.strptime(date_str, "%Y-%m-%d")
        return True
    except Exception:
        return False


def add_position_cmd(
    ticker: str,
    strike: float,
    expiry: str,
    cost: float,
    notes: str | None = None
):
    """Logs a new Phase 1 long position."""
    if strike <= 0:
        console.print("[red]Error: Strike price must be positive.[/red]")
        return
    if cost <= 0:
        console.print("[red]Error: Option cost premium must be positive.[/red]")
        return
    if not validate_date(expiry):
        console.print("[red]Error: Expiry must be formatted as YYYY-MM-DD.[/red]")
        return

    entry_date = datetime.now().strftime("%Y-%m-%d")
    pos_id = insert_position(ticker, entry_date, strike, expiry, cost, notes)
    console.print(Panel(
        f"[green]Successfully logged Phase 1 position #{pos_id}[/green]\n"
        f"Ticker: [bold white]{ticker}[/bold white]\n"
        f"Entry Date: {entry_date}\n"
        f"Long Call: Strike ${strike:.2f} | Expiry {expiry} | Cost: ${cost:.2f}\n"
        f"Net Cost Basis: ${cost:.2f}\n"
        f"Notes: {notes or 'None'}",
        title="Position Added",
        border_style="green"
    ))


def hedge_position_cmd(
    position_id: int,
    strike: float,
    expiry: str,
    credit: float,
    notes: str | None = None
):
    """Logs a Phase 2 short call hedge and updates net cost basis."""
    if strike <= 0:
        console.print("[red]Error: Strike price must be positive.[/red]")
        return
    if credit <= 0:
        console.print("[red]Error: Short call credit collected must be positive.[/red]")
        return
    if not validate_date(expiry):
        console.print("[red]Error: Expiry must be formatted as YYYY-MM-DD.[/red]")
        return

    positions = get_positions()
    pos = next((p for p in positions if p["id"] == position_id), None)
    if not pos:
        console.print(f"[red]Error: Position #{position_id} not found.[/red]")
        return

    long_exp = str(pos["long_expiry"])[:10]
    if expiry > long_exp:
        console.print(f"[yellow]Warning: Short call expiry ({expiry}) is after long call expiry ({long_exp}). Diagonal spreads typically require short expiry <= long expiry.[/yellow]")

    success = update_position_hedge(position_id, strike, expiry, credit, notes)
    if success:
        updated_pos = next((p for p in get_positions() if p["id"] == position_id), None)
        long_cost = updated_pos["long_cost"] if updated_pos else pos["long_cost"]
        net_basis = updated_pos["net_cost_basis"] if updated_pos else (long_cost - credit)
        total_credit_collected = long_cost - net_basis
        pct_reduced = (total_credit_collected / long_cost) * 100.0 if long_cost > 0 else 0.0

        console.print(Panel(
            f"[green]Successfully hedged Position #{position_id} into PHASE 2![/green]\n"
            f"Short Call: Strike ${strike:.2f} | Expiry {expiry} | Credit Collected: ${credit:.2f}\n"
            f"Original Long Cost: ${long_cost:.2f}\n"
            f"New Net Cost Basis: [bold cyan]${net_basis:.2f}[/bold cyan] ([bold green]-{pct_reduced:.1f}%[/bold green] cumulative reduction)\n"
            f"Notes: {updated_pos['notes'] if updated_pos else notes}",
            title="Phase 2 Hedge Recorded",
            border_style="green"
        ))
    else:
        console.print(f"[red]Error: Could not update position #{position_id}[/red]")


def roll_position_cmd(
    position_id: int,
    strike: float,
    expiry: str,
    credit: float,
    notes: str | None = None
):
    """Rolls a Phase 2 short call hedge to a new strike/expiry and accumulates credit."""
    if strike <= 0 or credit <= 0 or not validate_date(expiry):
        console.print("[red]Error: Invalid strike, credit, or date format.[/red]")
        return

    success = roll_position_hedge(position_id, strike, expiry, credit, notes)
    if success:
        updated_pos = next((p for p in get_positions() if p["id"] == position_id), None)
        long_cost = updated_pos["long_cost"] if updated_pos else 0.0
        net_basis = updated_pos["net_cost_basis"] if updated_pos else 0.0
        total_reduced = (long_cost - net_basis)
        pct_reduced = (total_reduced / long_cost) * 100.0 if long_cost > 0 else 0.0

        console.print(Panel(
            f"[green]Successfully rolled hedge for Position #{position_id}![/green]\n"
            f"New Short Leg: Strike ${strike:.2f} | Expiry {expiry} | Credit Collected: ${credit:.2f}\n"
            f"New Net Cost Basis: [bold cyan]${net_basis:.2f}[/bold cyan] ([bold green]-{pct_reduced:.1f}%[/bold green] cumulative reduction)\n"
            f"Notes: {updated_pos['notes'] if updated_pos else ''}",
            title="Hedge Rolled Forward",
            border_style="green"
        ))
    else:
        console.print(f"[red]Error: Could not roll position #{position_id}[/red]")


def unhedge_position_cmd(position_id: int, notes: str | None = None):
    """Returns a hedged position back to PHASE1_OPEN when short call expires."""
    success = unhedge_position(position_id, notes)
    if success:
        updated_pos = next((p for p in get_positions() if p["id"] == position_id), None)
        net_basis = updated_pos["net_cost_basis"] if updated_pos else 0.0
        console.print(Panel(
            f"[green]Successfully unhedged Position #{position_id}![/green]\n"
            f"Status returned to: [bold cyan]PHASE1_OPEN[/bold cyan]\n"
            f"Net Cost Basis preserved at: [bold green]${net_basis:.2f}[/bold green]\n"
            f"The position is now ready to harvest the next IV surge.",
            title="Position Unhedged",
            border_style="green"
        ))
    else:
        console.print(f"[red]Error: Could not unhedge position #{position_id}[/red]")


def close_position_cmd(position_id: int, notes: str | None = None):
    """Closes an active position."""
    success = close_position(position_id, notes)
    if success:
        console.print(f"[green]Successfully closed position #{position_id}[/green]")
    else:
        console.print(f"[red]Error: Could not close position #{position_id}[/red]")


def list_positions_cmd(status: str | None = None):
    """Displays all tracked positions in a formatted table."""
    filter_status = None
    if status and status.lower() != "all":
        status_map = {
            "open": "PHASE1_OPEN",
            "hedged": "PHASE2_HEDGED",
            "closed": "CLOSED"
        }
        filter_status = status_map.get(status.lower(), status.upper())

    positions = get_positions(filter_status)
    if not positions:
        console.print("[yellow]No positions found matching criteria.[/yellow]")
        return

    table = Table(
        title=f"Tracked Strategy Positions [{len(positions)} Total]",
        header_style="bold blue",
        show_lines=True
    )
    table.add_column("ID", justify="right")
    table.add_column("Ticker", style="bold white")
    table.add_column("Status", justify="center")
    table.add_column("Entry Date", justify="center")
    table.add_column("Long Call Leg", justify="left")
    table.add_column("Long Cost", justify="right")
    table.add_column("Short Call Leg", justify="left")
    table.add_column("Short Credit", justify="right")
    table.add_column("Net Basis", justify="right", style="bold green")
    table.add_column("% Saved", justify="right")
    table.add_column("Notes", justify="left")

    for p in positions:
        st = p["status"]
        st_style = "cyan" if st == "PHASE1_OPEN" else ("magenta" if st == "PHASE2_HEDGED" else "dim")

        long_str = f"${p['long_strike']:.2f} ({p['long_expiry']})"
        short_str = f"${p['short_strike']:.2f} ({p['short_expiry']})" if p["short_strike"] else "-"
        credit_str = f"${p['short_credit']:.2f}" if p["short_credit"] else "-"

        long_c = p["long_cost"]
        net_b = p["net_cost_basis"] if p["net_cost_basis"] is not None else long_c
        total_credit = long_c - net_b
        pct_saved = f"{(total_credit / long_c) * 100.0:.1f}%" if long_c > 0 and total_credit > 0 else "-"

        table.add_row(
            str(p["id"]),
            p["ticker"],
            f"[{st_style}]{st}[/{st_style}]",
            str(p["entry_date"]),
            long_str,
            f"${long_c:.2f}",
            short_str,
            credit_str,
            f"${net_b:.2f}",
            pct_saved,
            p["notes"] or ""
        )

    console.print(table)
