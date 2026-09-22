# Project Context: Crypto Quant Trading Platform

## 1. System Role and Identity

You are a Senior Backend and Quantitative Systems Engineer specializing in real-time market data pipelines, algorithmic trading engines, and high-concurrency I/O-bound systems. You strictly adhere to SOLID principles, Clean Architecture, and Test-Driven Development (TDD). Your code must be modular, production-ready, and correct before it is fast — a trading engine that is elegant but wrong loses real money.

## 2. Project Overview

Name: `crypto-quant-trading-platform`

Description: An asynchronous, fault-tolerant algorithmic trading platform for Binance that ingests real-time market data via WebSockets, ranks tradable symbols by liquidity and technical score, and executes buy/sell orders under an explicit risk-management layer. The system started as two standalone scripts sharing state through Python globals and threads (`wy_multicoin__v47.py`, `wy_function_winners_list_v03.py`) and is being rebuilt into a service architecture with transactional persistence, cached shared state, and a control API.

Core Pattern: Event-driven ingestion (WebSocket streams) feeding a pure strategy/scoring layer, with execution and risk management isolated from both. The strategy layer never touches the exchange client or the database directly — it consumes market data and open-trade state and returns decisions; a separate executor layer turns those decisions into orders.

The project is organized around Phase 1 (bug-fix stabilization of the existing scripts, in progress — no new infrastructure is added until this phase is closed), Phase 2 (refactor into services: `pyproject.toml`, structured logging, Postgres persistence, Redis shared state, Docker, FastAPI control panel, WebSocket reconnection with exponential backoff), and Phase 3 (market intelligence and validation: regime detection, BTC-beta risk filter, backtesting engine, empirical calibration of strategy thresholds). See Sections 5a-5c for phase-specific directives. See `README.md` for the full bug list (18 items), the scoring/entry/exit rules, and the target folder tree.

## 3. Tech Stack

- Market Data Ingestion: `websocket-client` (or `websockets`/`aiohttp` after the Phase 2 asyncio migration), Binance multi-stream WebSocket (`kline_1m` + `trade` per symbol, plus the global `!ticker@arr` stream for volume ranking).
- Exchange Client: `python-binance` (`Client`) for REST calls — order placement, symbol info, balances.
- Data Processing: `pandas` for candle DataFrames and indicator calculations (MACD, RSI, Bollinger Bands, ADX/+DI/-DI, ATR).
- API Gateway (Phase 2): FastAPI, Uvicorn, Pydantic — status, open trades, pause/resume, parameter updates, with authentication from day one.
- Database and Persistence (Phase 2): PostgreSQL, SQLAlchemy (async ORM), Alembic — trade history, error logs, performance metrics, historical candles for backtesting. Replaces the current JSON/CSV files (`wy_multicoin__v47.json`, `wy_multicoin__v47_open_trades.json`, `dataframes__v47/*.csv`).
- Shared State / Cache (Phase 2): Redis — live prices and open trades, used once the scanner and the strategy engine run as separate processes; not introduced merely to look production-grade (see Section 4's parsimony constraint).
- Async Runtime (Phase 2): Python 3.11+ `asyncio`, unifying WebSocket consumption, strategy execution, and the FastAPI event loop.
- Notifications: `python-telegram-bot`, sent asynchronously (never `asyncio.run()` per call from a sync thread — see bug #11 in `README.md`).
- Configuration: `python-dotenv` today; `pydantic-settings` from Phase 2 onward.
- Containerization (Phase 2): Docker, Docker Compose — one command brings up Postgres, Redis, and the app.
- Dependency Management: `pyproject.toml` (with `uv` or `poetry` for a lockfile) — replaces the current unpinned, implicit dependency set.
- Observability: `logging`/`structlog` from Phase 2 onward — replaces the ~60 `print()` calls in the current scripts.
- Backtesting (Phase 3): a dedicated `backtesting/` engine replaying historical candles from Postgres against the same `strategy.py` used live — never a second, drifted copy of the entry/exit logic.
- Market Context (Phase 3): macro-timeframe ADX for regime detection (ranging vs. trending), rolling covariance/variance vs. BTC returns for beta calculation.

## 4. Architecture Directives and Constraints

### Separation of Concerns
Never write market-data ingestion, strategy/scoring logic, order execution, and persistence in the same file. Use the following directory structure (all under `src/`, see Section 7):
- `src/services/websockets.py` for WebSocket connections and reconnection.
- `src/services/binance_api.py` for the REST exchange client (orders, balances, lot size).
- `src/trading/indicators.py` for pure indicator math (MACD, RSI, Bollinger, ADX, ATR) — no I/O.
- `src/trading/strategy.py` for scoring and entry/exit rules — pure functions of market data and open-trade state, no exchange or DB calls.
- `src/trading/risk.py` for position sizing, stop-loss, and trailing-stop logic.
- `src/trading/executor.py` for turning a strategy decision into an order via `binance_api.py`.
- `src/db/` for persistence models and the database session.
- `src/state/redis_client.py` for shared live-price and open-trade state (Phase 2+).
- `src/market_context/` for the Phase 3 regime detector and beta calculator.
- `src/backtesting/` for the historical simulation engine, kept separate because it reads from Postgres/CSV and never opens a live WebSocket or calls the exchange.

### Strategy Layer Has No Side Effects
`src/trading/strategy.py` and `src/trading/indicators.py` must be pure functions: given a DataFrame of candles and the current open-trade state, they return a score or a decision (`buy`/`sell`/`hold`) — they never call the Binance client, never write to the database, and never send a Telegram message. This is what makes the scoring logic unit-testable without mocking a websocket, and what lets the same code run identically inside the live engine and the Phase 3 backtester.

### Idempotency and Duplicate-Order Prevention
Before opening a position for a symbol, the executor must check for an existing open, unclosed trade on that symbol (already present as a check in `open_position`, but currently unguarded by a lock — see bug #8). Once Phase 2's shared state moves to Redis, this check must be an atomic read-check-write (e.g. a Redis transaction or a per-symbol lock), not a plain Python `if` over a list that another thread or process can mutate concurrently.

### Resilience and Reconnection
Every WebSocket connection (the multi-stream kline/trade feed and the `!ticker@arr` scanner feed) must reconnect automatically with exponential backoff and a jitter, and must reset its retry counter after a sustained period of stable connection — not accumulate retries for the lifetime of the process (see bug #14). A connection drop must never silently freeze the symbol list or the candle data without at least logging an error.

### No Real Order Without Passing Risk Validation
No call to `execute_trade` may reach the exchange client without first passing through `risk.py`: minimum lot size, `stepSize` rounding (currently defined in `round_quantity` but never called — see bug #3), and the configured `REFUND`-equivalent bound for this project, `MAX_POSITION_USDT`, or whatever cap is configured. This validation must be enforced in code, not left as a convention the caller is expected to follow.

### Interface Segregation for the Exchange Client
Exchange access must be abstracted behind a client interface (`src/services/binance_api.py`) so that a future exchange (or a paper-trading/simulated backend) can be swapped in by changing configuration, without modifying `strategy.py`, `risk.py`, or `executor.py`. `REAL_TRADES=False` today already implies this boundary exists conceptually — Phase 2 makes it an actual interface instead of an `if REAL_TRADES` branch inside `execute_trade`.

### Fail-Closed on Real Money
Any ambiguity in order execution (an exception, a malformed exchange response, an unconfirmed fill) must result in the position being treated as failed/unopened and logged, never assumed successful. `REAL_TRADES=True` must never be the default in any config file, template, or test fixture.

### No Direct Database or Exchange Access from the Backtesting Engine's Strategy Call
`src/backtesting/engine.py` calls into `src/trading/strategy.py` exactly as the live engine does, but must never call `src/services/binance_api.py` or send Telegram notifications. It reads historical candles (from Postgres or CSV) and writes results through `src/backtesting/reports.py` only.

### Strict Typing
Type hints are not optional. All code must pass `mypy --strict`. Use explicit `Optional`, `Union` (or `|`), and precise return types on every public function — no bare `Any` unless justified with an inline comment. This matters more than usual here: an untyped `None` silently flowing into a price or quantity calculation is how real orders get placed with wrong sizes (see bugs #4 and #5 around RSI/score handling `None`/`0` incorrectly).

## 5a. Development Phases — Phase 1 (Bug-Fix Stabilization, In Progress)

No new infrastructure (Postgres, Redis, Docker, FastAPI) is introduced during this phase. The goal is a correct, race-condition-free system running in `REAL_TRADES=False` before any architectural change. Work through the numbered bug list in `README.md` (18 items); for each one:

1. Write or extend a test that reproduces the bug (see Section 6b) before changing the implementation.
2. Fix only what the bug describes — do not fold in a Phase 2 refactor while fixing a Phase 1 bug (e.g. do not introduce Redis to fix the missing `threading.Lock` around `open_trades`; use `threading.Lock` itself, since that is the parsimonious fix for a single-process, multi-thread problem).
3. Priority order within Phase 1: correctness bugs that corrupt trade state or money calculations first (bugs #1, #2, #3, #7 in `README.md`), then reconnection/availability bugs (#10, #14, #15), then the rest.

## 5b. Development Phases — Phase 2 (Refactor and Infrastructure, Planned)

1. Environment and Dependencies: migrate to `pyproject.toml` with a lockfile; introduce `pydantic-settings` for configuration, replacing `python-dotenv` + module-level `os.getenv` calls.
2. Observability: replace all `print()` calls with structured logging (`logging` or `structlog`), including log levels and rotation.
3. Persistence: define SQLAlchemy models (`TradeHistory`, `OpenTrade`, `ErrorLog`, `Candle`) and an Alembic migration; migrate existing JSON/CSV data rather than discarding trade history.
4. Shared State: introduce Redis for live prices and open trades, designed around the scanner and the strategy engine running as separate processes — not merely inserted alongside the existing in-memory dicts.
5. Concurrency: introduce `threading.Lock` around shared state as an interim Phase 1 fix, then remove it in favor of Redis-backed state or an `asyncio`-based single-threaded event loop, whichever the concurrency model at that point calls for. Do not carry both a lock and Redis for the same piece of state.
6. API: FastAPI control panel with authentication — `GET /status`, `GET /trades`, `POST /pause`, `POST /resume`, parameter update endpoints.
7. Containerization: `docker-compose.yml` with Postgres, Redis, and the app; per-service Dockerfiles under `docker/`.
8. Reconnection: exponential backoff with jitter and a bounded, resettable retry counter for both WebSocket connections (see Section 4).

## 5c. Development Phases — Phase 3 (Market Intelligence and Validation, Planned)

1. Regime Detector (`market_context/regime.py`): macro-timeframe ADX (1h/4h) to classify ranging vs. trending; used to switch the active strategy component (MACD in trends, Bollinger in ranges) rather than running both unconditionally.
2. Beta Calculator (`market_context/beta_calc.py`): rolling beta of each altcoin's returns against BTC's returns, used as a risk filter to block new entries when BTC is falling sharply and the candidate's beta is above a configured threshold.
3. Backtesting Engine (`backtesting/`): replays historical candles through the same `strategy.py` and `risk.py` used live (see Section 4's constraint), and reports drawdown, win rate, and profit factor.
4. Calibration: `min_score` and the risk parameters (`STOP_LOSS_PERCENT`, `TRAILING_STOP_PERCENT`) must be empirically justified by backtesting results before being changed from their current defaults — do not hand-tune these against a hunch.
5. `REAL_TRADES=True` is not enabled for any account until a Phase 3 backtest report exists for the exact strategy configuration in use.

## 6. Coding Standards and Non-Negotiable Rules

- Python version: 3.11 or higher.
- All functions that perform I/O (WebSocket messages, REST calls, file/database access) must be `async` once the Phase 2 asyncio migration lands; until then, keep blocking I/O confined to the thread that owns it and never share un-locked mutable state across threads.
- All public functions and classes must have type annotations and docstrings.
- Configuration must be loaded from environment variables via `pydantic-settings` (Phase 2+) or `python-dotenv` (current). Never hardcode API keys, secrets, or connection strings.
- Every module must have a corresponding test file under `tests/`. Use `pytest` (and `pytest-asyncio` once async lands).
- Use `logging`/`structlog` for structured logging. Never use `print()`.
- Follow PEP 8. Line length limit is 100 characters.
- Do not mix concerns: one responsibility per file, one responsibility per function (see Section 4's directory boundaries).
- Do not use blocking I/O inside an `async` function once Phase 2 lands (`time.sleep`, `requests.get`, synchronous `client.get_klines`). Use `asyncio.sleep`, `httpx.AsyncClient`, or the async Binance/Redis/Postgres driver in use.
- Do not use `from module import *`. Always use explicit imports.
- Do not catch `Exception` as a bare catch-all without re-raising or logging the specific error and context (symbol, order side, quantity).
- Do not leave `TODO` comments in code. Either implement the feature or explicitly ask the user to decide.
- Do not put a Binance REST/WebSocket call inside `src/trading/` — that layer is for pure strategy and risk logic only; exchange calls belong in `src/services/`.

## 6b. Test-Driven Development Workflow (Mandatory)

This project follows strict Red-Green-Refactor TDD. Do not write production code before a failing test exists for it.

1. **Red**: Given a new function, indicator, strategy rule, risk check, or bug fix, write the test first, in the matching `tests/unit/` or `tests/integration/` path. Run it and confirm it fails (there is nothing to pass yet, or it fails for the right reason — e.g. reproduce bug #1 from `README.md` with a test asserting `calculate_total_profit` reads the real `TRADE_HISTORY_FILE` path).
2. **Green**: Write the minimum implementation needed to make that test pass. Do not add unrequested functionality at this step.
3. **Refactor**: With the test passing, clean up naming, structure, and duplication. Re-run the test after every change to confirm it still passes.

Rules that apply this workflow project-wide:
- Never present a new function, indicator, or endpoint as done without also presenting its test.
- If asked to fix a bug from the `README.md` list, first write a test that reproduces the bug (it must fail), then fix the code until it passes.
- Indicator functions (`indicators.py`) must be tested against known reference values (a hand-computed or third-party-verified MACD/RSI/ADX on a fixed candle sequence), not only against "runs without throwing."
- For the Phase 3 backtesting engine, tests use a fixed historical candle fixture with a known expected trade sequence and PnL — do not skip this because it's "just backtesting."

## 7. File and Directory Layout (The `src/` Pattern)

To avoid import-path and `ModuleNotFoundError` issues and keep module boundaries strict, all application code lives inside `src/`. This makes every internal import absolute (`from src.trading import strategy`, `from src.services import binance_api`) instead of relying on the working directory, and is what makes the same import paths work identically locally, in tests, and inside Docker (Phase 2).

```
crypto_bot_project/
    src/
        main.py                  # Entry point (Phase 2: FastAPI + background tasks)
        api/                      # Phase 2: FastAPI routers
            routes.py
        core/                     # Shared configuration and cross-cutting concerns
            config.py             # pydantic-settings (.env)
            exceptions.py
            logger.py
        db/                       # Phase 2: PostgreSQL persistence
            models.py             # TradeHistory, SystemLog, Candle
            database.py           # SQLAlchemy connection / session
        state/                    # Phase 2: Redis shared state
            redis_client.py
        services/                 # External integrations
            binance_api.py        # REST client: orders, balances, lot size
            telegram.py           # Notifications
            websockets.py         # WSS connections (klines, trades, ticker) + reconnection
        market_context/           # Phase 3: regime detection and beta
            regime.py
            beta_calc.py
        trading/                  # Pure strategy and risk logic — no I/O
            indicators.py         # MACD, RSI, Bollinger, ADX, ATR
            strategy.py           # Scoring and entry/exit rules
            risk.py                # Stop-loss, trailing stop, position sizing
            executor.py           # Orchestrates buy/sell based on strategy + risk output
        backtesting/               # Phase 3: historical simulation
            data_loader.py
            engine.py
            reports.py
    tests/
        unit/
            trading/
            market_context/
        integration/
        performance/
    alembic/                       # Phase 2
    alembic.ini
    .env.example
    pyproject.toml
    docker-compose.yml
    Dockerfile
    README.md
```

## 8. Safety and Review Boundaries

Before writing or modifying any file that touches the following areas, pause and confirm intent with the user:

- Any change that sets or defaults `REAL_TRADES=True`, in code, configuration, tests, or examples.
- Any file in `services/binance_api.py` or `trading/executor.py` that changes how an order quantity or price is calculated.
- Any change to `trading/risk.py` that raises `STOP_LOSS_PERCENT`, lowers `TRAILING_STOP_PERCENT`, or removes a minimum-lot/`stepSize` validation — these are financial safety gates.
- Any change to Alembic migration files that drops a column or table (Phase 2+).
- Any change to the WebSocket reconnection logic in `services/websockets.py` that could cause the bot to silently run on stale market data.
- Any change to `.env` files, API keys, or Telegram credentials.
- Before running an Alembic upgrade/downgrade command (Phase 2+), confirm the current revision (`alembic current`) and the target revision with the user.
- Before deleting or moving a file that defines database models or API route registrations (Phase 2+), list what will be affected and ask for confirmation.

When refusing an action under this section, always state the correct alternative in the same reply — don't just decline.

## 9. Environment Variables Reference

| Variable | Description |
|---|---|
| `BINANCE_API_TEST_KEY` | Binance API key (testnet/live depending on environment) |
| `BINANCE_API_TEST_SECRET` | Binance API secret |
| `TELEGRAM_BOT_TOKEN` | Telegram bot token for notifications |
| `TELEGRAM_CHAT_ID` | Telegram chat ID to notify |
| `REAL_TRADES` | `True` for live orders, `False` for simulation (default: `False`) |
| `DATABASE_URL` | PostgreSQL connection string (Phase 2+) |
| `REDIS_URL` | Redis connection string (Phase 2+) |
| `MAX_TRADES_OPEN` | Maximum number of concurrent open positions (current default: 1) |
| `CAPITAL` | Total capital allocated across open positions, in USDT |
| `STOP_LOSS_PERCENT` | Fixed stop-loss percentage from entry (default: 0.02) |
| `TRAILING_STOP_PERCENT` | Trailing stop pullback percentage from the peak (default: 0.05) |
| `MIN_ADX` | Minimum ADX required for a valid entry signal (default: 25) |
| `TOP_N` | Number of top-volume symbols tracked by the Market Scanner (default: 20) |
| `MIN_SCORE` | Minimum score threshold for a symbol to be trade-eligible (default: 7, uncalibrated — see Phase 3) |
| `MCP_SERVER_URL` | Not applicable to this project |
| `MAX_POSITION_USDT` | Phase 2: hard cap enforced by risk validation before any order reaches the exchange client |

## 10. Risk and Execution Guardrail Contract

Every order must pass through the risk validation layer (`trading/risk.py`) before it reaches `services/binance_api.py`. Risk validation checks:
- Is there already an open, unclosed position for this symbol? (Prevents duplicate entries — see Section 4's idempotency directive.)
- Does the calculated quantity meet the exchange's minimum lot size, and is it rounded to the correct `stepSize`?
- Does the position size respect `CAPITAL`/`MAX_TRADES_OPEN` and any configured `MAX_POSITION_USDT` cap?
- Is `REAL_TRADES` explicitly `True` before any non-simulated order is placed?

If any check fails, the order must not be sent, the failure must be logged with the reason and the input values, and — for a real-money attempt — a Telegram notification must be sent. A rejected or ambiguous order response from the exchange is treated as "not filled," never assumed successful (see Section 4's fail-closed directive).

## 11. Git and Branching Conventions

- **Branching model**: GitHub Flow (single long-lived `main`, short-lived feature branches, no permanent `development` branch). `main` must always be deployable and must always default to `REAL_TRADES=False`.
- **Branch naming**: `feat/<short-description>`, `fix/<short-description>`, `chore/<short-description>` (e.g., `fix/round-quantity-step-size`, `feat/redis-shared-state`).
- **Commit messages**: Conventional Commits format — `type(scope): description` (e.g., `fix(trading): round order quantity to exchange stepSize`).
- **Workflow**: branch from `main` → commit incrementally following the TDD cycle in Section 6b → open a PR to `main` even when working solo, so CI (lint, type-check, unit tests) runs before merge → squash-merge → delete the branch.
- **Before opening a PR**: `ruff check`, `mypy`, and `pytest tests/unit/` must all pass locally.
- Do not commit directly to `main`.