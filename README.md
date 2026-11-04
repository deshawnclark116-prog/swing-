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
| Filter | Calls: gap up 3% or more AND opening volume at least 2.5x normal. Puts: gap down 5% or more | Gap up 10% or more |
| Entry | After 9:45, a 1-min close beyond the 15-minute opening range, on the right side of VWAP | 1-min close above the premarket high and VWAP |
| Exits | Stock moves 1.5% against the entry, calls closed by noon, puts by 3:45. Safety nets on the option: -50% stop, +200% target | Stop below the signal bar or VWAP (5% max), breakeven at +1R, 2R target, loses VWAP, 3:45pm |
| Status | Selected by out-of-sample testing (see below) | **Not validated.** Free data can't find enough small-cap runners. Paper only |
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

## How the options rules were chosen (and how much to trust them)

`research/gap_variants.py` tested 96 variants (entry type, gap size, relative volume, SPY filter, exit style) on about 60 days of 5-minute data for 289 liquid names. Each variant was tuned on the first half of the days and then judged on the second half.

- Averaged across all 96 variants, the result on the unseen half was about **0%**. Most setups that look good in testing don't hold up.
- Two setups held up, and both match standard momentum-trading practice. **Calls:** opening-range breakout with heavy opening volume (2.5x or more), out by noon. **Puts:** opening-range breakdown on gaps of 5% or more.
- With the bot's exact rules (`gap-backtest`), the result is 180 trades, a 49% win rate, and **+0.77% average stock move per trade**, with **+0.59%** on the unseen half. A move of about +0.5% is needed to cover option spreads and time decay. **The edge is thin.**
- **Caveats:** 60 days is short and covers one market environment. The universe is today's movers (selection bias), and the option profit and loss is not simulated. Paper trade for at least a month and compare against this backtest before risking money.

## Automatic paper trading on GitHub

`.github/workflows/paper-trade.yml` runs **both bots** with **fake money** every weekday:
- They start at 9:20am New York time, scan the premarket, and trade.
- **Each bot has its own $100**, with up to 50% per trade.
- Calls close by noon and everything is closed by 3:00pm.
- The accounts are saved to `paper/options_state.json` and `paper/stocks_state.json`, and the combined results go to **`paper/REPORT.md`**.

**Why the bots don't buy premarket:** `research/premarket_shares.py` tested buying small caps that were already up 10% or more on a new premarket high. It tried 108 variants, starting from 4am, 7am, or 8am, with different stops and exits. **Every variant lost money**, about -1% per trade even before premarket spreads. Options also don't trade before 9:30, so the bots use the premarket to pick stocks and trade after the open.

To check results, open `paper/REPORT.md` on GitHub. To run it by hand, go to **Actions → Paper trade → Run workflow**.
To start over, delete the `paper/*_state.json` files.
