# Repository Guide for AI Assistants

This file gives an AI coding assistant the context needed to work in this repository. Its content is intentionally duplicated in `GEMINI.md` — if you update one, update the other so they stay identical.

## Project

Algorithmic trading engine for cryptocurrency strategies, currently focused on Binance spot trading. The repository is in an early stage: the code under `legacy-experimental-phase/` is a legacy standalone bot kept for reference while the engine itself is rebuilt.

## Conventions

- All code, comments, docstrings, and log/user-facing strings must be in English. The original codebase was written in Spanish; it has been fully translated (comments, docstrings, print/log/Telegram messages, and identifiers). Keep all new code English-only.
- There is no dependency manifest yet. If you add a new dependency, create/update `requirements.txt` rather than assuming a package is installed.
- Prefer editing existing files over creating new ones; avoid introducing new abstractions unless the task requires them.

## Repository Layout

- `legacy-experimental-phase/` — legacy scripts, kept for reference:
  - `wy_function_winners_list_v03.py` — Binance 24h ticker WebSocket listener; maintains `WINNERS` / `LOSERS` / `SYMBOLS` lists.
  - `wy_multicoin_engine_v47.py` — the full multi-coin bot: indicators, scoring, buy/sell execution, Telegram alerts, JSON/CSV persistence. Imports from `wy_function_winners_list_v03.py`.

## Known Issues / Gotchas

- `wy_multicoin_engine_v47.py` imports a `wy_autosell_module_test` module (`load_trades_from_file`, `run`, `get_conditions`) that does not exist in this repo. The file will not run standalone until this is resolved.
- `TRADE_HISTORY_FILE`, `OPEN_TRADES_FILE`, and `DATAFRAMES_PATH` reference `v42` filenames despite living in the `v47` file — a known inconsistency, not yet fixed.
- `REAL_TRADES = False` gates real vs. simulated order execution. Treat any change to `True` as high risk — it places live market orders on Binance with real funds. Never flip this flag without explicit user instruction.
- Requires a `.env` file with `BINANCE_API_TEST_KEY`, `BINANCE_API_TEST_SECRET`, `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`.
- No automated test suite exists yet.

## Working in This Repo

- When editing the legacy scripts, keep changes minimal and consistent with the translation pass already done — don't silently "fix" the known issues above unless explicitly asked.
- There's no test suite, so verify Python changes at minimum with `python -m py_compile <file>`.
- This is a trading bot: be conservative with any change that could affect order execution, position sizing, or risk parameters (stop-loss/trailing-stop percentages, `MAX_TRADES_OPEN`, `CAPITAL`). Flag anything ambiguous rather than guessing.
