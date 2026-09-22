# AI Crypto Trading Agent

A quantitative trading system for Binance that starts as a rule-based bot and is built to grow into a decision-making agent: real-time market ingestion via WebSockets, a technical-signal engine, risk management, and order execution today, with sentiment-aware and self-tuning strategy agents on the roadmap -- evolving from a set of standalone scripts into a service architecture with transactional persistence, cached state, and a control panel.

Current status: project under active refactoring. `REAL_TRADES=False` (simulation mode) until Phase 1 is complete and the strategy has been validated through backtesting. Do not trade real capital before then.

---

## Table of contents

1. [Overview](#overview)
2. [Trading strategy](#trading-strategy)
3. [Technology stack](#technology-stack)
4. [Phased roadmap](#phased-roadmap)
5. [Known bugs (Phase 1)](#known-bugs-phase-1)
6. [Target folder structure](#target-folder-structure)
7. [Local setup](#local-setup)

---

## Overview

The system today consists of two independent processes that communicate through module import (not over the network), running as threads inside the same Python process and sharing state through global variables:

| Component | Current file | Responsibility |
|---|---|---|
| Market Scanner | `wy_function_winners_list_v03.py` | Subscribes to Binance's `!ticker@arr` stream, ranks all USDT pairs by 24h volume (`quoteVolume`), and exposes the Top N via `get_updated_symbols()`. |
| Trading Engine | `wy_multicoin__v47.py` | Consumes the symbol list, maintains 1m candles per symbol (klines plus tick-by-tick trades over a multi-stream channel), computes technical indicators, ranks by score, executes buys/sells, manages stops, and sends Telegram notifications. |
| Emergency Brake | `wy_autosell_module_test.py` (imported, not included in this repo) | Exposes `get_conditions()` to force manual stop-loss/trailing-stop closes from outside the main engine. |

---

## Trading strategy

### Entry (buy)

- Trading universe: Top 20 USDT symbols by 24h volume (liquidity filter), refreshed in real time by the Market Scanner.
- Entry signal: bullish MACD crossover over the signal line (DEA) -- MACD moves from below to above DEA on the most recently closed candle (`get_macd_crossover_signal`).
- Entry is only evaluated for symbols within the Top 3 of the score ranking at the moment their candle closes (`rankings[:3]`).
- Position size: `CAPITAL / MAX_TRADES_OPEN`, minus a 1.5% fee, rounded down to Binance's minimum lot size.

### Scoring system (`calculate_score`)

Multiplicative/additive scoring based on:

- ADX (trend strength): `>50` x3, `>30` x2, `>25` x1, otherwise discard (x0).
- +DI vs -DI: discard if there is no bullish predominance.
- RSI: bonus for the neutral zone (30-70) x2, penalty for overbought/oversold x0.5.
- Bollinger slope (SMA): discard if the slope isn't positive; a strong slope (`>2`) additionally requires ADX above the minimum and RSI below 70.
- Relative volume: +1 if current volume exceeds the 20-candle average by more than 20%.
- MACD/DEA: x1.2 if MACD>DEA and MACD>0; x0.9 if MACD<0.
- ATR: x0.95 if ATR>3 (penalizes excessive volatility).

Note: the `min_score=7` threshold is not empirically calibrated -- it is an arbitrary value pending validation through backtesting (see Phase 3).

### Exit (sell)

The position is closed if any of the following occurs:

1. Price touches the Bollinger upper band (take profit).
2. RSI(12) > 70 (overbought).
3. ADX < 20 and price near/below the SMA (loss of trend).
4. Fixed stop-loss: -2% from entry price (`STOP_LOSS_PERCENT`).
5. Trailing stop: -5% from the highest price reached since entry (`TRAILING_STOP_PERCENT`).
6. External manual conditions (`get_conditions()` via `wy_autosell_module_test`).
7. (Optional, currently disabled via flags) Live tick-based stop-loss, two variants of a live staged trailing stop, dynamic ATR-based stop (`ENABLE_LIVE_STOPLOSS`, `ENABLE_LIVE_TRAILING_STOP_1/2` -- currently `False`).

### Planned extensions (Phase 3)

- Market regime detector (ranging vs. trending) using ADX on a higher timeframe (1h/4h) -- MACD wins in trending markets, Bollinger wins in ranging markets; the detector decides which to use at each moment.
- Beta vs. BTC: relative risk filter; block altcoin buys with high beta when BTC is falling sharply.
- Backtesting engine: validate the strategy and calibrate `min_score` and risk parameters against historical data before risking real capital.

---

## Technology stack

### Current

- Python 3, `websocket-client`, `pandas`, `python-binance`, `python-telegram-bot`, `python-dotenv`
- Persistence: JSON files (`trade_history`, `open_trades`, `error_log`, `real_operations_errors`) and CSV (candles, in `dataframes__v47/`)
- Dependency management: no formal lockfile

### Target (Phase 2+)

| Piece | Use | Why |
|---|---|---|
| Python 3.11+ / asyncio | Central engine | Unifies WebSockets, strategy execution, and the API without blocking between tasks (today `send_telegram_message_sync` creates a new event loop on every call from a synchronous thread). |
| FastAPI | Control panel | Endpoints to view open trades, pause/resume the bot, adjust parameters without restarting the process. Requires authentication from day one -- it exposes control over real orders. |
| PostgreSQL + SQLAlchemy/SQLModel | Transactional persistence | Trade history, error logs, performance metrics (PnL, win rate), historical candles for backtesting. |
| Redis | Shared state across processes | Live prices and open trades, designed for when the scanner and the strategy engine run as separate processes (not just threads within one process) -- that is where it stops being cosmetic and actually resolves the decoupling, and eliminates the current race condition on `open_trades`/`dataframes`. |
| Docker / docker-compose | Reproducible environment | Bring up Postgres + Redis + the app with a single command, same environment on any machine. |
| `pyproject.toml` | Dependency and build management | Replaces a loose `requirements.txt`; reproducible environment with a lockfile (`uv`/`poetry`), project metadata, and tool configuration (ruff, pytest) in a single file. |
| `logging` / `loguru` | Observability | Replaces the ~60 `print()` calls in the current code; levels, file rotation, structured logs. |

### Target (Phase 4-5, experimental)

| Piece | Use | Why |
|---|---|---|
| `pgvector` (reuses the Phase 2 Postgres instance) | Sentiment/news vector store | Embeds ingested news, social posts, and on-chain alerts so the RAG pipeline can retrieve the most relevant context for a symbol before scoring its sentiment -- no separate vector database service to operate. |
| LLM provider (pluggable via an `LLM_PROVIDER`-style factory, same pattern as the exchange client) | Sentiment classification and the strategy decision agent | Classifies retrieved text into a sentiment signal and, separately, chooses which strategy component should be active for current conditions. Kept behind a factory so the provider can change without touching `trading/`. |
| News/social/on-chain APIs (e.g. a crypto news aggregator, a fear-and-greed index endpoint) | Sentiment data ingestion | Raw input for the RAG pipeline; each source is fetched, chunked, and embedded before being queried. |
| `DEAP` (or a hand-rolled genetic algorithm module) | Strategy parameter evolution | Evolves strategy parameters (score weights, stop-loss/trailing-stop percentages, indicator periods) against the Phase 3 backtesting engine as the fitness function -- a parsimonious choice over a full ML framework, since the search space here is a fixed set of numeric parameters, not a learned function. |

---

## Phased roadmap

### Phase 1 -- Stabilization (bugs first)

Goal: make the current system correct and safe in simulation mode before touching the architecture. See [Known bugs](#known-bugs-phase-1). No new infrastructure is added until this phase is closed.

### Phase 2 -- Refactoring and infrastructure

- Migrate to `pyproject.toml`.
- Replace `print()` with `logging`.
- Remove `generate_and_run_crypto_script` (script generation via `subprocess`).
- Introduce `threading.Lock` (or migrate to `asyncio`) for shared state (`open_trades`, `dataframes`, `SYMBOLS`).
- `docker-compose.yml` with Postgres + Redis.
- Data models in Postgres (trades, logs, metrics) and migration from JSON/CSV.
- Redis layer for live prices and open trades, with the scanner and the engine as separate processes.
- FastAPI with basic authentication: status, open trades, pause/resume, adjust parameters.
- WebSocket reconnection with exponential backoff in both scripts (the main engine already reconnects with a fixed 5s retry; it still needs exponential backoff and a unified approach with the scanner, which currently does not reconnect at all).

### Phase 3 -- Market intelligence and validation

- Regime detector (ranging vs. trending).
- Beta vs. BTC calculation as a risk filter.
- Backtesting engine over historical data in Postgres.
- Empirical calibration of `min_score` and risk parameters against backtesting results.

### Phase 4 -- Market sentiment intelligence (RAG + decision agents), experimental

Goal: give the bot a signal that plain price/volume indicators cannot see -- news, social sentiment, and on-chain events -- and let an agent act on it, without letting that agent bypass the existing risk layer.

- Sentiment ingestion: scheduled API calls to news/social/on-chain sources per tracked symbol, stored as raw text with a timestamp and source.
- Physics-Based RAG & Vector Search: Designed a dynamic market-regime detector that maps multidimensional (3D) market structure into vector embeddings (pgvector), retrieves the physical model matched to the closest historical regime, and feeds that context to an LLM strategy-selection agent that decides which numerical strategy to deploy.
- Sentiment scoring: an LLM call classifies the retrieved context into a sentiment signal (e.g. bullish/neutral/bearish, or a bounded numeric score) for the symbol under evaluation.
- Circuit-breaker agent: monitors sentiment and news-event signals and can veto new entries or force-close existing positions on a symbol -- for example, on a detected exchange hack, a regulatory action, a stablecoin depeg, or a sharp sentiment-price divergence. This agent can only ever make trading *more* conservative (block or close), never open a position or override the risk layer's caps; every halt decision is logged with its trigger and reasoning, the same way a rejected order is logged today.
- Strategy decision agent: given the current regime (Phase 3), the indicator scores (existing `strategy.py`), and the sentiment signal (this phase), chooses which strategy component should be active (e.g. MACD-trend vs. Bollinger-range vs. sitting out) instead of that choice being hardcoded. Its output is a strategy selection, not an order -- it still goes through the normal entry/exit and risk logic.
- Both agents are advisory/gating signals that feed into the existing `trading/strategy.py` and `trading/risk.py` layer; neither one calls the exchange client directly, following the same separation-of-concerns rule as the rest of the trading layer.
- This phase is design/prototype status, same as Phase 3's regime detector was before validation -- ship the ingestion and scoring first, validate the sentiment signal's actual predictive value against historical data, and only then wire the circuit-breaker into live decisions.

### Phase 5 -- Genetic algorithm strategy optimization, experimental

Goal: replace hand-tuned strategy parameters (the currently arbitrary `min_score=7`, the fixed 2%/5% stop-loss/trailing-stop, indicator periods) with values found by evolving them against real backtest performance.

- Encode a strategy configuration (score weights, thresholds, stop-loss/trailing-stop percentages, indicator periods) as a genome.
- Fitness function: run the Phase 3 backtesting engine on historical data for a candidate genome and score it on a risk-adjusted metric (e.g. profit factor or Sharpe ratio adjusted for max drawdown, not raw return alone -- raw return rewards reckless parameter sets).
- Evolve a population of configurations over generations (selection, crossover, mutation) using `DEAP` or an equivalent library.
- Runs entirely offline against historical data, out-of-band from live trading, mirroring how the ML comparison track is meant to work in the other project's roadmap -- its output is a candidate parameter set for a human to review, never a set of parameters auto-applied to the live bot.
- Any winning configuration must still pass the same Phase 3 validation step (backtest report reviewed before use) before being adopted -- a genetic algorithm optimizing against historical data can overfit to that specific history, so a proposed configuration is a hypothesis to validate out-of-sample, not a result to trust directly.

---

## Known bugs (Phase 1)

### `wy_multicoin__v47.py` (Trading Engine)

1. `calculate_total_profit` never reads the correct file. It is called with `file_path = "TRADE_HISTORY_FILE"` (a string literal, line 374) instead of the `TRADE_HISTORY_FILE` variable. It always fails with "File does not exist" on module startup.
2. `execute_trade` extracts the symbol incorrectly in the SELL branch without a quantity. Line 909: calls `get_sell_quantity(symbol.split('USDT')[0])`, but `get_sell_quantity` already does `symbol.replace("USDT", "")` internally -- passing the already-stripped asset (e.g. `"BTC"` instead of `"BTCUSDT"`) corrupts the balance lookup. (Note: inside `execute_trade`, in the `REAL_TRADES` branch for SELL, line 933, the same incorrect pattern is repeated.)
3. `round_quantity` is defined but never used. Quantities are never rounded to Binance's `stepSize` before sending the order -- a real risk of order rejection due to invalid precision when `REAL_TRADES=True`. Only `minQty` (minimum lot) is validated, not `stepSize`.
4. RSI can divide by zero. In `calculate_rsi` (line 613), if `losses == 0` within the window, `rs = gains / losses` produces `inf` or `NaN` with no explicit handling, and `rsi.iloc[-1]` can propagate `NaN` into `calculate_score`.
5. `calculate_score` treats `rsi == 0` as falsy. The condition `if rsi:` (line 679) skips the block when `rsi` is exactly `0`, treating it as "no data" instead of a valid extreme-oversold value.
6. Inconsistent discard pattern in the score. It mixes `return 0.0` (short-circuits the whole function, lines 695/701) with `score *= 0` (a multiplication that could theoretically be "revived" if the evaluation order changed, lines 667/675) for the same type of discard condition -- inconsistent behavior that is hard to audit or test.
7. `close_all_open_trades` closes at the entry price, not the current price. Line 1440: uses `trade["price"]` as the close price, which reports a false PnL (~0) on emergency closes triggered by a WebSocket failure. In addition, this function is not currently called from `on_close` (it is commented out on line 1486), so a connection close no longer forces positions to close -- this needs to be a deliberate decision, not an accident.
8. No lock between threads for shared state. `open_trades`, `dataframes`, and `trade_history` are read and written from the main WebSocket thread and the volume-scanner thread without a `threading.Lock`, exposed to race conditions (currently mitigated only by the GIL, not by design).
9. Disk I/O in the hot path of price updates. `check_stops_for_trade` calls `save_trade_history()` (a full JSON write, line 1117) every time `highest_price` is updated, which can happen on every closed kline -- unnecessary performance impact and disk wear.
10. The scanner's reconnection thread is duplicated on every retry. `run_bot()` (line 1501) relaunches `threading.Thread(target=start_websocket, daemon=True).start()` every time it is called, including on every reconnection triggered from `on_close()` -- accumulating duplicate volume-scanner threads across successive reconnections of the main WebSocket.
11. `send_telegram_message_sync` creates a new event loop on every call (`asyncio.run(...)`, line 400) from the WebSocket thread -- unnecessary overhead and a risk of momentarily blocking the thread that processes market messages right when a position opens or closes.
12. `generate_and_run_crypto_script` (line 200) spawns a new Python process per symbol by writing a `.py` file to disk and running it via `subprocess.Popen`. It is not active in the current flow (calls are commented out on lines 923 and 947), but it is technical debt and a real risk surface if reactivated unintentionally.
13. No handling of Binance REST API rate limits in synchronous calls within hot paths (`get_symbol_info`, `get_asset_balance`, called on every `execute_trade`/`open_position`).
14. `retries` is a global counter that is never reset after a successful reconnection. Line 1469: once the `MAX_RETRIES=60` reconnection attempts accumulated over the entire lifetime of the process are exhausted (not per disconnection episode), the bot stops reconnecting permanently, even if it ran stably for days in between.

### `wy_function_winners_list_v03.py` (Market Scanner)

15. `on_close` does not reconnect. It only does `print("WebSocket closed")`. If Binance cuts the connection (which it routinely does every 24h), the Top N stays frozen indefinitely and the main engine operates with a stale symbol list, with no alert.
16. `update_every_hour()` is redundant dead logic. `SYMBOLS` is already updated on every WebSocket message via `process_data()` -> `update_symbols()`; the hourly loop serves no real purpose other than keeping the main thread alive, which can be achieved more cleanly (e.g. `threading.Event().wait()`).
17. No lock for `SYMBOLS` / `RANKED_BY_VOLUME` shared between the WebSocket thread and the main thread (same pattern as bug #8 in the main engine).
18. No retry handling for the subscription message if the initial `SUBSCRIBE` call fails silently (there is no confirmation that Binance accepted the subscription to `!ticker@arr`).

---

## Strategy Documentation and Research

The trading rules summarized above are the ones currently active. Detailed
write-ups of each strategy component -- including the physical mechanism
each formula is meant to capture, its valid regime, and known failure modes
-- live in `docs/strategies/`.

This project also builds on a series of earlier prototypes (rule-based
physics-inspired models, and an unfinished 3D shape-recognition neural
network blocked by local compute) kept in `legacy/` with their own
post-mortem in `docs/research.md`. They did not reach production, but each
one narrowed down what does and doesn't hold up when the market's regime
shifts -- see that log for the reasoning behind design choices in this
README's Phase 3 and Phase 4 sections.

---

## Target folder structure

```
crypto_bot_project/
├── pyproject.toml
├── docker-compose.yml
├── Dockerfile
├── .env.example
├── README.md
│
├── legacy-experimental-phase/       # early prototypes, original scripts and experiments
│   ├── wy_multicoin_engine_v47.py
│   ├── wy_function_winners_list_v03.py
│   └── experiments/
│       ├── VLU_test01/              # C++/Python optimization & benchmark tests
│       ├── VLU_test02/              # neuro-plastic & symbolic trading experiments
│       └── VLU_test03/              # high-frequency & volatility trading models
│
├── docs/
│   ├── strategies/                  # one doc per model/strategy
│   │   ├── macd-dea-crossover.md
│   │   └── bollinger-lateral.md
│   └── research.md                  # log of prototypes: what worked, what failed, why
│
├── src/
│   ├── main.py                      # Entry point (FastAPI + background tasks)
│   │
│   ├── api/
│   │   └── routes.py                # GET /status, GET /trades, POST /pause, etc.
│   │
│   ├── core/
│   │   ├── config.py                # Pydantic Settings (.env)
│   │   ├── exceptions.py
│   │   ├── logger.py
│   │   └── trade_history.py         # Total profit calculation and file persistence
│   │
│   ├── db/
│   │   ├── models.py                # TradeHistory, SystemLog, Candle
│   │   └── database.py              # SQLAlchemy connection / session
│   │
│   ├── state/
│   │   └── redis_client.py          # Live prices, open trades
│   │
│   ├── services/
│   │   ├── binance_api.py           # REST client: orders, balances, lot size
│   │   ├── telegram.py              # Notifications
│   │   └── websockets.py            # WSS connections (klines, trades, ticker) + reconnection
│   │
│   ├── market_context/
│   │   ├── regime.py                # Ranging vs. trending (macro ADX)
│   │   └── beta_calc.py             # Altcoin beta vs. BTC
│   │
│   ├── trading/
│   │   ├── indicators.py            # MACD, RSI, Bollinger, ADX, ATR
│   │   ├── strategy.py              # Scoring and entry/exit rules
│   │   ├── risk.py                  # Stop-loss, trailing stop, position sizing
│   │   └── executor.py              # Orchestrates buy/sell based on the rules
│   │
│   ├── backtesting/
│   │   ├── data_loader.py           # Loads historical candles into Postgres
│   │   ├── engine.py                # Historical simulation of the strategy
│   │   └── reports.py               # Metrics: drawdown, win rate, profit factor
│   │
│   ├── sentiment/                   # Phase 4 (experimental): RAG ingestion and scoring
│   │   ├── ingestion.py             # Scheduled pulls from news/social/on-chain APIs
│   │   ├── rag_pipeline.py          # Chunking, embedding, and retrieval over pgvector
│   │   └── scoring.py               # LLM-based sentiment classification
│   │
│   ├── agents/                      # Phase 4 (experimental): advisory/gating agents
│   │   ├── circuit_breaker.py       # Can veto entries or force-close on adverse signals
│   │   └── strategy_selector.py     # Chooses active strategy component (trend/range/sentiment)
│   │
│   └── optimization/                # Phase 5 (experimental): genetic algorithm search
│       ├── genome.py                # Strategy configuration encoded as a genome
│       ├── fitness.py               # Wraps the backtesting engine as a fitness function
│       └── evolve.py                # Selection, crossover, mutation loop (DEAP-based)
│
└── tests/
    ├── unit/
    │   ├── core/
    │   │   └── test_trade_history.py
    │   ├── trading/
    │   └── market_context/
    ├── integration/
    └── performance/
```

---

## Local setup

```bash
# Clone and enter the project
git clone <repo>
cd crypto_bot_project

# Install dependencies (with uv or poetry, to be decided)
uv sync

# Bring up infrastructure (Postgres + Redis)
docker compose up -d

# Configure environment variables
cp .env.example .env
# fill in BINANCE_API_TEST_KEY, BINANCE_API_TEST_SECRET, TELEGRAM_BOT_TOKEN, etc.

# Run in development mode
python -m src.main
```

`REAL_TRADES=False` by default. Do not switch to `True` without having completed Phase 1 and run the Phase 3 backtesting.