# S&P 500 Volatility Scanner

Pulls S&P 500 stocks, tracks historical volatility against live options implied volatility, and alerts when to trade dynamic diagonal spreads to lower your cost basis.

---

## How the Strategy Works

Most options spreads enter both legs at the same time. This strategy splits them into two steps:

1. **Step 1: Buy an ITM call when volatility is quiet**
   - When a stock has been calm (low IV rank, low IV relative to 20-day historical volatility), options are cheap.
   - You buy a deep in-the-money call (~0.80 delta, 60–90 days out). Because it is deep ITM, it moves closely with the stock and has low vega (minimal risk from volatility drops).

2. **Step 2: Sell an OTM call when volatility spikes**
   - Ahead of an event (like earnings or major news), implied volatility jumps as traders bid up options.
   - When IV surges (IV rank > 70% or IV significantly above historical volatility), you sell an out-of-the-money call (~0.30 delta, front-month 14–35 days out).
   - Because it is OTM, its price is almost entirely inflated extrinsic value. Selling it brings in a cash credit that cuts your cost basis on the original position.

3. **Step 3: Post-catalyst IV crush**
   - Once the event passes, implied volatility collapses. The short call drops in value and can be bought back cheaply or left to expire worthless.

---

## What the Code Does

- **Pulls S&P 500 tickers**: Scrapes current constituents from Wikipedia.
- **Incremental price updates**: Checks SQLite first and only downloads missing days via `yfinance`, avoiding API rate limits.
- **Volatility calculations**: Calculates 20-day annualized historical volatility, front-month ATM implied volatility, and rolling IV rank.
- **Greeks & contract matching**: Uses Black-Scholes to find contracts matching target deltas (~0.80 ITM for buys, ~0.30 OTM for sells).
- **Position tracker**: SQLite database (`data/trading.db`, WAL mode) tracks trades through their full lifecycle (open, hedged, rolled, closed) and logs your net cost basis after each credit.
- **Alerts**: Checks open positions against live market IV and flags when an IV spike gives you an opportunity to sell the short leg. Supports Discord/Slack webhooks.
- **Backtesting**: Runs the two-phase logic over historical daily prices to see returns, win rate, and average cost-basis reduction against buy-and-hold.

---

## Setup

Requires Python 3.13+.

```bash
git clone https://github.com/nalagr/trading-project.git
cd trading-project

# using uv
uv venv
uv pip install -r pyproject.toml

# or standard venv
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e .
```

---

## Usage

### 1. Fetch data
Pulls price history and saves volatility snapshots to the database:

```bash
# Standard daily update
python src/main.py scan

# Adjust thread count (default is 7 to avoid rate limits)
python src/main.py scan --workers 5

# Test run with first 10 tickers
python src/main.py scan --limit 10
```

### 2. Screen for trades
Scans the database and live options chains for setups:

```bash
# View top candidates for both phases
python src/main.py screen --phase all --limit 5

# Only phase 1 buys (low IV rank, cheap options)
python src/main.py screen --phase 1 --limit 10

# Only phase 2 sells (high IV rank, bloated options)
python src/main.py screen --phase 2 --limit 10

# Export to CSV
python src/main.py screen --phase all --csv data/results.csv
```

### 3. Track positions & get alerts

```bash
# Log an initial ITM call purchase
python src/main.py track add --ticker AMD --strike 430.00 --expiry 2027-01-15 --cost 110.05 --notes "Low IV entry"

# Check if any open positions have hit their IV spike harvest threshold
python src/main.py check

# Send alerts to Discord / Slack
python src/main.py check --webhook "https://discord.com/api/webhooks/..."

# Log selling the short call when an alert fires
python src/main.py track hedge --id 1 --strike 550.00 --expiry 2026-11-20 --credit 14.50

# Roll the short call to next month
python src/main.py track roll --id 1 --strike 560.00 --expiry 2026-12-18 --credit 6.20

# If the short call expires worthless, unhedge to wait for the next spike
python src/main.py track unhedge --id 1

# List all open positions and net cost basis
python src/main.py track list

# Close position
python src/main.py track close --id 1 --notes "Took profit"
```

### 4. Backtest

Simulate how the strategy would have performed on a specific stock:

```bash
python src/main.py backtest --ticker NVDA
python src/main.py backtest --ticker AAPL --capital 15000
```

---

## Testing

```bash
python -m unittest discover -s tests -v
```
