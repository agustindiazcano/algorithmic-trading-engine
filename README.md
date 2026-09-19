# Algorithmic Trading Engine

A trading engine for developing, backtesting, and running algorithmic cryptocurrency trading strategies. The current focus is Binance spot trading.

## Status

Early stage. The repository currently holds a legacy standalone bot (kept for reference and as a starting point) while the engine itself is rebuilt from scratch.

## Repository Layout

- `legacy-experimental-phase/` — an earlier, standalone Binance trading bot ("Weyland Yutani / Prometheus"), kept for reference. Originally written with Spanish comments and identifiers; fully translated to English.
  - `wy_function_winners_list_v03.py` — connects to Binance's `!ticker@arr` WebSocket stream and maintains rolling `WINNERS` / `LOSERS` / `SYMBOLS` lists based on 24h price change.
  - `wy_multicoin_engine_v47.py` — the main multi-coin trading bot: technical indicators (ADX, RSI, Bollinger Bands, MACD, ATR), a scoring/ranking system, buy/sell execution via `python-binance`, Telegram notifications, and JSON/CSV persistence of trade history and candle data.

## Known Issues (legacy code, not yet fixed)

- `wy_multicoin_engine_v47.py` imports `load_trades_from_file`, `run`, and `get_conditions` from a `wy_autosell_module_test` module that does not exist in this repo. The file cannot run standalone until that module is restored or the dependency is removed.
- `TRADE_HISTORY_FILE`, `OPEN_TRADES_FILE`, and `DATAFRAMES_PATH` reference `v42` filenames despite living inside the `v47` engine file.
- No dependency manifest (`requirements.txt`) yet. At minimum the legacy bot needs: `websocket-client`, `pandas`, `python-binance`, `python-telegram-bot`, `python-dotenv`.
- No automated tests.

## Configuration

The legacy bot reads credentials from a `.env` file (not committed):

```
BINANCE_API_TEST_KEY=...
BINANCE_API_TEST_SECRET=...
TELEGRAM_BOT_TOKEN=...
TELEGRAM_CHAT_ID=...
```

`REAL_TRADES` (inside `wy_multicoin_engine_v47.py`) defaults to `False`, meaning trades are simulated. Setting it to `True` places real market orders on Binance with real funds.

## Getting Started

_TBD — setup and usage instructions will be added as the engine is rebuilt._

## License

_TBD_
