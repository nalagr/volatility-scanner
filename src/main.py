import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import io
import sys

# Ensure UTF-8 output on Windows consoles
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import pandas as pd
import requests

from analyze import (
    add_position_cmd,
    check_position_alerts,
    close_position_cmd,
    hedge_position_cmd,
    list_positions_cmd,
    roll_position_cmd,
    screen_opportunities,
    unhedge_position_cmd,
)
from backtest import run_backtest
from database import create_tables
from fetch import getData
from volatility import backfill_all_iv_ranks, vol_scan


def get_sp500_tickers() -> list[str]:
    """Fetches current S&P 500 tickers from Wikipedia, with resilient fallback."""
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36"
        )
    }

    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        html_data = io.StringIO(response.text)
        tables = pd.read_html(html_data)

        sp500_df = tables[0]
        tickers = sp500_df["Symbol"].str.replace(".", "-", regex=False).tolist()
        return tickers

    except Exception as e:
        print(f"Error fetching S&P 500 tickers from Wikipedia: {e}. Falling back to default list.")
        return [
            "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "AMD",
            "TSLA", "AVGO", "JPM", "V", "UNH", "ABBV", "COST", "NFLX"
        ]


def process_single_ticker(ticker: str, force_full: bool = False) -> str:
    """Worker function to incrementally fetch prices and compute volatility metrics."""
    try:
        # 1. Fetch prices (smart incremental, thread-safe silent mode)
        getData(ticker, force_full=force_full, verbose=False)

        # 2. Calculate volatility metrics (HV20, ATM IV, IV Rank, silent mode)
        res = vol_scan(ticker, verbose=False)
        if res:
            return f"[SUCCESS] {ticker} (HV: {res['hv_20day']:.1%}, IV: {res['atm_iv']:.1%}, IV Rank: {res['iv_rank']:.1f}%)"
        return f"[PARTIAL] {ticker} - Metrics unavailable (missing option quotes)"
    except Exception as e:
        return f"[ERROR] {ticker}: {e}"


def handle_scan(args):
    """Executes the S&P 500 data ingestion and volatility calculation scan."""
    print("=" * 65)
    print("Volatility Arbitrage Platform: Data Ingestion & Volatility Engine")
    print("=" * 65)
    print("Initializing Database...")
    create_tables()

    if getattr(args, "backfill_ranks", False):
        print("Running IV rank backfill across existing snapshots...")
        backfill_all_iv_ranks(verbose=True)

    print("Fetching S&P 500 tickers...")
    tickers = get_sp500_tickers()
    if args.limit:
        tickers = tickers[:args.limit]
        print(f"Limited run: processing first {len(tickers)} tickers.")

    total_tickers = len(tickers)
    max_workers = args.workers

    print(
        f"Starting parallel scan across {total_tickers} tickers "
        f"using {max_workers} worker threads (force_full={args.full})...\n"
    )

    completed = 0
    success_count = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_ticker = {
            executor.submit(process_single_ticker, ticker, args.full): ticker
            for ticker in tickers
        }

        for future in as_completed(future_to_ticker):
            completed += 1
            ticker = future_to_ticker[future]
            try:
                result = future.result()
                if "[SUCCESS]" in result:
                    success_count += 1
                print(f"[{completed}/{total_tickers}] {result}")
            except Exception as e:
                print(f"[{completed}/{total_tickers}] [FAILED] {ticker}: {e}")

    print(f"\nScan completed: {success_count}/{total_tickers} tickers successfully processed.")


def handle_screen(args):
    """Executes Phase 1 and/or Phase 2 market screening."""
    screen_opportunities(
        phase=args.phase,
        limit=args.limit,
        export_csv=args.csv
    )


def handle_track(args):
    """Handles position tracking subcommands."""
    if not args.track_action:
        print("Please specify a track action: list, add, hedge, roll, unhedge, close, check. Run with --help for details.")
        return

    if args.track_action == "list":
        list_positions_cmd(status=args.status)
    elif args.track_action == "add":
        add_position_cmd(
            ticker=args.ticker.upper(),
            strike=args.strike,
            expiry=args.expiry,
            cost=args.cost,
            notes=args.notes
        )
    elif args.track_action == "hedge":
        hedge_position_cmd(
            position_id=args.id,
            strike=args.strike,
            expiry=args.expiry,
            credit=args.credit,
            notes=args.notes
        )
    elif args.track_action == "roll":
        roll_position_cmd(
            position_id=args.id,
            strike=args.strike,
            expiry=args.expiry,
            credit=args.credit,
            notes=args.notes
        )
    elif args.track_action == "unhedge":
        unhedge_position_cmd(
            position_id=args.id,
            notes=args.notes
        )
    elif args.track_action == "close":
        close_position_cmd(
            position_id=args.id,
            notes=args.notes
        )
    elif args.track_action == "check":
        check_position_alerts(webhook_url=getattr(args, "webhook", None))
    else:
        print("Invalid track action. Run 'python src/main.py track --help'.")


def handle_backtest(args):
    """Executes historical backtest of the two-phase diagonal volatility spread strategy."""
    run_backtest(
        ticker=args.ticker.upper(),
        initial_capital=args.capital
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="S&P 500 Volatility Arbitrage Screener, Position Tracker & Quantitative Simulator",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    subparsers = parser.add_subparsers(dest="subcommand", help="Platform commands")

    # 1. scan
    scan_parser = subparsers.add_parser("scan", help="Sync price data and calculate volatility metrics")
    scan_parser.add_argument("--workers", type=int, default=7, help="Worker threads for parallel sync (default: 7)")
    scan_parser.add_argument("--full", action="store_true", help="Force full 400-day re-download of price history")
    scan_parser.add_argument("--limit", type=int, default=None, help="Limit scan to first N tickers (for testing)")
    scan_parser.add_argument("--backfill-ranks", action="store_true", help="Backfill missing IV ranks across snapshots")
    scan_parser.set_defaults(func=handle_scan)

    # 2. screen
    screen_parser = subparsers.add_parser("screen", help="Screen market for Phase 1 (buys) and Phase 2 (sells)")
    screen_parser.add_argument(
        "--phase",
        choices=["1", "2", "all"],
        default="all",
        help="Strategy phase to screen: '1' (accumulation), '2' (harvest), or 'all' (default: all)"
    )
    screen_parser.add_argument("--limit", type=int, default=10, help="Maximum candidates per phase (default: 10)")
    screen_parser.add_argument("--csv", type=str, default=None, help="Export screened opportunities to CSV file")
    screen_parser.set_defaults(func=handle_screen)

    # 3. track
    track_parser = subparsers.add_parser("track", help="Manage and monitor active strategy positions")
    track_subparsers = track_parser.add_subparsers(dest="track_action", help="Position actions")

    # track list
    list_p = track_subparsers.add_parser("list", help="List tracked positions")
    list_p.add_argument(
        "--status",
        choices=["all", "open", "hedged", "closed"],
        default="all",
        help="Filter by status (default: all)"
    )

    # track add
    add_p = track_subparsers.add_parser("add", help="Log a new Phase 1 long call position")
    add_p.add_argument("--ticker", type=str, required=True, help="Stock ticker symbol (e.g. AAPL)")
    add_p.add_argument("--strike", type=float, required=True, help="Long call strike price")
    add_p.add_argument("--expiry", type=str, required=True, help="Long call expiration date (YYYY-MM-DD)")
    add_p.add_argument("--cost", type=float, required=True, help="Long call entry premium cost per share")
    add_p.add_argument("--notes", type=str, default=None, help="Optional trade thesis or notes")

    # track hedge
    hedge_p = track_subparsers.add_parser("hedge", help="Log a Phase 2 short call hedge against an open position")
    hedge_p.add_argument("--id", type=int, required=True, help="Position ID to hedge")
    hedge_p.add_argument("--strike", type=float, required=True, help="Short call strike price")
    hedge_p.add_argument("--expiry", type=str, required=True, help="Short call expiration date (YYYY-MM-DD)")
    hedge_p.add_argument("--credit", type=float, required=True, help="Short call premium credit collected per share")
    hedge_p.add_argument("--notes", type=str, default=None, help="Optional hedge notes")

    # track roll
    roll_p = track_subparsers.add_parser("roll", help="Roll an existing short call hedge to a new month/strike")
    roll_p.add_argument("--id", type=int, required=True, help="Position ID to roll")
    roll_p.add_argument("--strike", type=float, required=True, help="New short call strike price")
    roll_p.add_argument("--expiry", type=str, required=True, help="New short call expiration date (YYYY-MM-DD)")
    roll_p.add_argument("--credit", type=float, required=True, help="New short call premium credit collected")
    roll_p.add_argument("--notes", type=str, default=None, help="Optional roll notes")

    # track unhedge
    unhedge_p = track_subparsers.add_parser("unhedge", help="Unhedge a position when short call expires worthless")
    unhedge_p.add_argument("--id", type=int, required=True, help="Position ID to unhedge")
    unhedge_p.add_argument("--notes", type=str, default=None, help="Optional remarks")

    # track close
    close_p = track_subparsers.add_parser("close", help="Close an open or hedged position")
    close_p.add_argument("--id", type=int, required=True, help="Position ID to close")
    close_p.add_argument("--notes", type=str, default=None, help="Closing notes / final PnL remarks")

    # track check
    check_p = track_subparsers.add_parser("check", help="Check positions against live market for IV surges and expiring hedges")
    check_p.add_argument("--webhook", type=str, default=None, help="Optional Discord/Slack webhook URL for push notifications")

    track_parser.set_defaults(func=handle_track)

    # 4. check (shortcut for track check)
    check_parser = subparsers.add_parser("check", help="Check open positions for harvest sell alerts")
    check_parser.add_argument("--webhook", type=str, default=None, help="Optional Discord/Slack webhook URL")
    check_parser.set_defaults(func=lambda args: check_position_alerts(webhook_url=getattr(args, "webhook", None)))

    # 5. backtest
    backtest_parser = subparsers.add_parser("backtest", help="Simulate quantitative performance of strategy on historical data")
    backtest_parser.add_argument("--ticker", type=str, default="AAPL", help="Stock ticker symbol to backtest (default: AAPL)")
    backtest_parser.add_argument("--capital", type=float, default=10000.0, help="Initial simulation capital (default: $10,000)")
    backtest_parser.set_defaults(func=handle_backtest)

    return parser


def main():
    parser = build_parser()
    if len(sys.argv) == 1:
        print("""
S&P 500 Volatility Scanner

Quick Commands:
  python src/main.py scan    -> Update prices & volatility snapshots
  python src/main.py screen  -> Find today's buys (low IV) and sells (high IV)
  python src/main.py check   -> Check open positions for harvest alerts
  python src/main.py track   -> Manage positions (add, hedge, roll, list)

Run any command with --help for options (e.g. 'python src/main.py screen --help').
""")
        sys.exit(0)

    args = parser.parse_args()
    if not hasattr(args, "func"):
        parser.print_help()
        sys.exit(0)

    args.func(args)


if __name__ == "__main__":
    main()