# swing- : a daily swing-trading bot

A small Python bot that trades stocks on daily bars. It includes:

- **Strategy**: trend-following (`swing_bot/strategy.py`). It enters when the 20-day EMA crosses above the 50-day SMA, but only while price is above the 200-day SMA and RSI(14) < 70. It exits on the reverse cross or when RSI > 80. Each trade has an ATR stop (2×ATR) and a profit target (4×ATR).
- **Risk sizing** (`risk.py`): each trade risks 1% of equity. A single position is capped at 20% of equity, and at most 5 positions are open at once.
- **Backtester** (`backtest.py`): it trades a portfolio across many symbols. Signals are taken at the close and filled at the next open, with slippage. Stops and targets are checked against each day's high and low, and a gap through a level fills at the open.
- **Brokers** (`broker.py`):
  - `PaperBroker`: a simulated account saved to `paper_state.json`. It needs no keys.
  - `AlpacaBroker`: places real bracket orders through [Alpaca](https://alpaca.markets). It uses the **paper account by default**.

> ⚠️ This is educational software, not financial advice. Backtest results don't predict future returns. Paper-trade for a long time before you risk real money.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

```bash
# Backtest on Yahoo Finance data
python -m swing_bot backtest --symbols AAPL MSFT NVDA SPY --start 2018-01-01 --trades-csv trades.csv

# Offline backtest on synthetic data
python -m swing_bot backtest --source synthetic

# Show what the bot would do today, without placing orders
python -m swing_bot run --dry-run

# Paper-trade with the local simulated account, then check it
python -m swing_bot run
python -m swing_bot status

# Trade through Alpaca (export the keys from .env.example first)
python -m swing_bot run --broker alpaca --dry-run
python -m swing_bot run --broker alpaca
```

Useful flags: `--risk 0.005`, `--max-positions 8`, `--cash 25000`, `-v` (verbose logging).

### Run it daily

The bot is meant to run once a day after the US market closes. Example cron entry (Mon–Fri, 4:30pm New York):

```
30 16 * * 1-5  cd /path/to/swing- && .venv/bin/python -m swing_bot run >> bot.log 2>&1
```

With Alpaca, orders placed after the close fill at the next open, which matches the backtest. The local paper broker fills at the last close as an approximation.

## Customizing

- Change parameters in `swing_bot/config.py`.
- To add a strategy, subclass `Strategy` in `strategy.py`. Implement `generate(df)`, which returns a `signal` column (+1 enter, -1 exit, 0 hold) plus `atr`, and `stop_and_target(...)`. Then register the class in `STRATEGIES`.

## Tests

```bash
pytest -q
```

The tests cover indicators, sizing, fill and stop mechanics, the paper bot cycle, and a **no-look-ahead** check. That check confirms that adding future bars never changes past signals.

---

# Day trading (cash account)

There are two separate systems, because most small caps have no tradeable options.

| | System 1: options | System 2: small-cap shares |
|---|---|---|
| Stocks | Liquid names with tight option spreads | $1–$20 stocks gapping 10% or more |
| Direction | Calls on gap-ups, puts on gap-downs | Long only (a cash account can't short) |
| Entry | First 1-min close beyond the premarket high or low, on the right side of VWAP | Same, upside only |
| Exits | Premium -30% stop, +100% target, trailing stop after +30%, stock loses VWAP, 3:45pm | Stop below the signal bar or VWAP (5% max), breakeven at +1R, 2R target, loses VWAP, 3:45pm |
| Sizing | 10% of settled cash per trade | Risk 2% of settled cash, at most 25% in one position |

Both systems trade only with **settled** cash (T+1), so they can't cause good-faith violations. Each is limited to 3 trades a day and stops trading for the day after a set loss.

```bash
python -m swing_bot premarket --cash 5000         # morning scan for both systems, with suggested contracts
python -m swing_bot options-trade --dry-run       # System 1; loops from the open (--once for a single check)
python -m swing_bot stocks-trade --dry-run        # System 2
python -m swing_bot options-trade                 # paper trade (local simulated cash account)
python -m swing_bot options-trade --broker alpaca # Alpaca paper account
python -m swing_bot accounts                      # paper account balances, positions, and log
python -m swing_bot gap-backtest --system options # test the signal on the last ~60 days
```

**Data limits:** Yahoo's free data is delayed, doesn't report premarket volume, and its movers lists don't show fresh premarket gappers. For live use, switch the data source to Alpaca or another real-time feed.
