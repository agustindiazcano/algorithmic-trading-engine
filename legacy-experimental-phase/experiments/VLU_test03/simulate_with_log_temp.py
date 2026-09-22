from datetime import datetime, timezone

def ms_to_utc(ms: int) -> str:
    """Convert millisecond timestamp to UTC ISO string."""
    return datetime.fromtimestamp(ms/1000, tz=timezone.utc).isoformat(timespec="seconds")

def simulate_with_log(genome, threshold, inputs, prices, times_ms, TRADING_MODE="BOTH",
                      INITIAL_BALANCE=10000.0, MAX_BET=3333.0, SLIPPAGE=0.001,
                      FEE=0.0005, LEVERAGE=50.0):
    """
    Run simulation and log all trade events (OPEN/CLOSE/LIQUIDATION) with full context.
    
    Returns:
        balance (float): Final balance
        trades (int): Number of trades executed
        log (list): List of dicts, each representing a trade event
    """
    brain = DynamicCausalBrain(n_neurons=genome.shape[0], A_dc=1.618)
    brain.genome = genome

    balance = INITIAL_BALANCE
    position = 0
    entry_price = 0.0
    entry_t = None
    current_bet = 0.0
    trades = 0
    log = []

    for t in range(len(inputs)):
        signal = brain.predict([inputs[t]])
        price = float(prices[t])
        ts = int(times_ms[t])

        # --- CLOSE (flip signal) ---
        if (signal > threshold and position == -1) or (signal < -threshold and position == 1):
            pnl_pct = ((price - entry_price) / entry_price * LEVERAGE) if position == 1 else ((entry_price - price) / entry_price * LEVERAGE)
            profit = current_bet * pnl_pct
            fee_exit = (current_bet * LEVERAGE) * FEE

            balance += (profit - fee_exit)

            log.append({
                "t": t, "time": ms_to_utc(ts),
                "event": "CLOSE",
                "pos": position,
                "entry_t": entry_t, "entry_price": entry_price,
                "exit_price": price,
                "signal": float(signal), "thr": float(threshold),
                "p_change": float(inputs[t][0]), "vol_rel": float(inputs[t][1]), "volat": float(inputs[t][2]),
                "bet": float(current_bet),
                "pnl_pct": float(pnl_pct),
                "fee_exit": float(fee_exit),
                "balance": float(balance),
            })

            position = 0
            current_bet = 0.0
            entry_price = 0.0
            entry_t = None

        # --- OPEN ---
        if position == 0:
            if signal > threshold and TRADING_MODE in ["LONG", "BOTH"]:
                position = 1
                entry_price = price * (1 + SLIPPAGE)
                entry_t = t
                current_bet = min(balance, MAX_BET)
                fee_entry = (current_bet * LEVERAGE) * FEE
                balance -= fee_entry
                trades += 1

                log.append({
                    "t": t, "time": ms_to_utc(ts),
                    "event": "OPEN_LONG",
                    "pos": position,
                    "entry_t": entry_t, "entry_price": entry_price,
                    "signal": float(signal), "thr": float(threshold),
                    "p_change": float(inputs[t][0]), "vol_rel": float(inputs[t][1]), "volat": float(inputs[t][2]),
                    "bet": float(current_bet),
                    "fee_entry": float(fee_entry),
                    "balance": float(balance),
                })

            elif signal < -threshold and TRADING_MODE in ["SHORT", "BOTH"]:
                position = -1
                entry_price = price * (1 - SLIPPAGE)
                entry_t = t
                current_bet = min(balance, MAX_BET)
                fee_entry = (current_bet * LEVERAGE) * FEE
                balance -= fee_entry
                trades += 1

                log.append({
                    "t": t, "time": ms_to_utc(ts),
                    "event": "OPEN_SHORT",
                    "pos": position,
                    "entry_t": entry_t, "entry_price": entry_price,
                    "signal": float(signal), "thr": float(threshold),
                    "p_change": float(inputs[t][0]), "vol_rel": float(inputs[t][1]), "volat": float(inputs[t][2]),
                    "bet": float(current_bet),
                    "fee_entry": float(fee_entry),
                    "balance": float(balance),
                })

            if balance <= 100:  # Game Over
                break

        # --- LIQUIDATION ---
        if position != 0:
            unrealized = ((price - entry_price) / entry_price * LEVERAGE) if position == 1 else ((entry_price - price) / entry_price * LEVERAGE)
            if unrealized <= -0.9:
                balance -= current_bet
                log.append({
                    "t": t, "time": ms_to_utc(ts),
                    "event": "LIQUIDATION",
                    "pos": position,
                    "entry_t": entry_t, "entry_price": entry_price,
                    "mark_price": price,
                    "unrealized_pnl": float(unrealized),
                    "bet": float(current_bet),
                    "balance": float(balance),
                })
                position = 0
                current_bet = 0.0
                entry_price = 0.0
                entry_t = None

    # Final close if still in position
    if position != 0:
        price = float(prices[-1])
        ts = int(times_ms[-1])
        pnl_pct = ((price - entry_price) / entry_price * LEVERAGE) if position == 1 else ((entry_price - price) / entry_price * LEVERAGE)
        profit = current_bet * pnl_pct
        fee_exit = (current_bet * LEVERAGE) * FEE
        balance += (profit - fee_exit)
        
        log.append({
            "t": len(inputs)-1, "time": ms_to_utc(ts),
            "event": "FINAL_CLOSE",
            "pos": position,
            "entry_t": entry_t, "entry_price": entry_price,
            "exit_price": price,
            "pnl_pct": float(pnl_pct),
            "fee_exit": float(fee_exit),
            "balance": float(balance),
        })

    return balance, trades, log
