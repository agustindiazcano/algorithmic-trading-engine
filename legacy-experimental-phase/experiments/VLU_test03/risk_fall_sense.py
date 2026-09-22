import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional

# ==============================================================================
# 🤖 ROBOTIC FALL SENSING MODULE (V10)
# ==============================================================================
# Deterministic telemetry controller. No statistics. No predictions.
# Just pure "body balance" sensing: Liquidation Distance, Fall Rate, Pressure.

class SuggestedAction(Enum):
    OK = auto()             # Balance is stable
    REDUCE_SIZE = auto()    # Slight wobble, reduce exposure
    NO_NEW_TRADES = auto()  # Unstable, don't add weight
    EXIT_NOW = auto()       # FALLING! Emergency eject
    COOLDOWN = auto()       # Recovering from fall

@dataclass
class FallSenseState:
    margin: float       # "Distance to edge" (Positive = Safe, Negative = Danger)
    fall_rate: float    # "Velocity towards edge" (Negative = Falling)
    pressure: float     # External stress (MAE, Spread, etc)
    fall_score: float   # 0.0 (Stable) -> 1.0 (Critical)
    action: SuggestedAction
    scale_factor: float # 0.0 -> 1.0 multiplier for new entries

@dataclass
class FallSenseConfig:
    # 📐 Support Polygon Weights (k factors)
    k_stop_risk: float = 1.0    # Weight for stop loss proximity
    k_mae: float = 2.0          # Weight for Adverse Excursion
    k_spread: float = 5.0       # Weight for Spread (Liquidity risk)
    k_leverage: float = 0.5     # Weight for Leverage risk
    
    # ⚡ Reflex Thresholds
    min_liq_buffer: float = 0.005 # 0.5% distance minimum to liquidation
    hard_exit_score: float = 0.8  # FallScore > 0.8 -> EXIT
    reduce_score: float = 0.4     # FallScore > 0.4 -> REDUCE
    
    # ⏱️ Dynamics
    fall_rate_window: int = 3   # Simple moving average for smooth rate

class FallSenseController:
    def __init__(self, config: Optional[FallSenseConfig] = None):
        self.config = config if config else FallSenseConfig()
        self.history_margin: List[float] = []
        self.last_action = SuggestedAction.OK
        self.cooldown_counter = 0

    def reset(self):
        self.history_margin = []
        self.last_action = SuggestedAction.OK
        self.cooldown_counter = 0

    def update(self, 
               mark_price: float, 
               liquidation_price: float, 
               position_side: str, 
               stop_price: Optional[float] = None, 
               mae_pct: float = 0.0, 
               spread_pct: float = 0.0, 
               leverage: float = 1.0) -> FallSenseState:
        
        # 0. Handling Cooldown
        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
            return FallSenseState(0.0, 0.0, 0.0, 1.0, SuggestedAction.COOLDOWN, 0.0)

        # 1. Calculate Liquidation Distance (The "Edge")
        if position_side == "LONG":
            if liquidation_price <= 0: dist_liq_pct = 1.0 # Safe/Infinite
            else: dist_liq_pct = (mark_price - liquidation_price) / mark_price
        elif position_side == "SHORT":
            if liquidation_price <= 0: dist_liq_pct = 1.0
            else: dist_liq_pct = (liquidation_price - mark_price) / mark_price
        else:
            # Flat
            return FallSenseState(1.0, 0.0, 0.0, 0.0, SuggestedAction.OK, 1.0)

        # 2. Calculate Stop Distance (Secondary support)
        dist_stop_pct = 1.0 # Default safe
        if stop_price:
            if position_side == "LONG":
                dist_stop_pct = max(0, (mark_price - stop_price) / mark_price)
            elif position_side == "SHORT":
                dist_stop_pct = max(0, (stop_price - mark_price) / mark_price)
        
        # Penalize if stop is missing or too close
        stop_risk = max(0, 1.0 - (dist_stop_pct / 0.01)) # 1% reference
        if not stop_price: stop_risk = 2.0 # High penalty for no stop

        # 3. Calculate "Margin" (Stability Metric)
        # Margin = Dist_Liq - Penalties. 
        # If Margin < 0, we are outside stable polygon.
        
        # Normalize MAE and Spread to similar scales (0..1 approx)
        norm_mae = min(mae_pct / 0.02, 1.0) # 2% MAE is bad
        norm_spread = min(spread_pct / 0.001, 1.0) # 0.1% spread is bad
        
        margin = dist_liq_pct \
                 - (self.config.k_stop_risk * stop_risk * 0.001) \
                 - (self.config.k_mae * norm_mae * 0.01) \
                 - (self.config.k_spread * norm_spread * 0.01)

        # 4. Calculate Fall Rate (Dynamics)
        self.history_margin.append(margin)
        if len(self.history_margin) > self.config.fall_rate_window:
            self.history_margin.pop(0)
            
        fall_rate = 0.0
        if len(self.history_margin) > 1:
            # Delta positive = improving stability. Negative = falling.
            # Using simple difference of last vs first in window for trend
            fall_rate = self.history_margin[-1] - self.history_margin[0]

        # 5. Composite Fall Score
        # We want a score 0..1 where 1 is falling face flat.
        
        # Component A: Proximity to death (0 if > 1% away, 1 if < 0.1% away)
        score_prox = 0.0
        if dist_liq_pct < 0.01:
            score_prox = max(0.0, 1.0 - (dist_liq_pct / 0.01))
            
        # Component B: Velocity of fall
        score_vel = 0.0
        if fall_rate < 0:
            # If falling fast (e.g. -0.5% margin per step), panic.
            score_vel = min(abs(fall_rate) / 0.005, 1.0) 
            
        # Component C: Pressure (MAE/Spread)
        score_press = (norm_mae * 0.5) + (norm_spread * 0.5)
        
        # Fusion
        raw_score = (score_prox * 0.5) + (score_vel * 0.3) + (score_press * 0.2)
        
        # HARD REFLEXES (Overrides)
        # If we are literally about to die (0.5% buffer), score is 1.0
        if dist_liq_pct < self.config.min_liq_buffer:
            raw_score = 1.0 
            
        fall_score = max(0.0, min(1.0, raw_score))

        # 6. Determine Action
        action = SuggestedAction.OK
        scale = 1.0
        
        if fall_score > self.config.hard_exit_score:
            action = SuggestedAction.EXIT_NOW
            self.cooldown_counter = 10 # 10 ticks penalty
            scale = 0.0
        elif fall_score > 0.6: # Configurable? using hardcoded for now as per "NO_NEW_TRADES" tier
            action = SuggestedAction.NO_NEW_TRADES
            scale = 0.0
        elif fall_score > self.config.reduce_score:
            action = SuggestedAction.REDUCE_SIZE
            scale = 0.5
        
        self.last_action = action
        
        return FallSenseState(
            margin=margin,
            fall_rate=fall_rate,
            pressure=score_press,
            fall_score=fall_score,
            action=action,
            scale_factor=scale
        )

# ==============================================================================
# 🧪 SIMPLE UNIT TEST
# ==============================================================================
if __name__ == "__main__":
    import sys
    # Configurar UTF-8 para Windows
    if sys.platform.startswith("win"):
        sys.stdout.reconfigure(encoding='utf-8')
        
    print("🤖 TESTING FALL SENSE CONTROLLER...")
    ctrl = FallSenseController()
    
    print("\n--- TEST 1: SAFE ZONE ---")
    st = ctrl.update(mark_price=100.0, liquidation_price=80.0, position_side="LONG", 
                     stop_price=98.0, mae_pct=0.0, spread_pct=0.0001)
    print(f"DistLiq: {(100-80)/100:.2%} | Margin: {st.margin:.4f} | Score: {st.fall_score:.2f} | Action: {st.action.name}")
    
    print("\n--- TEST 2: HIGH SPREAD + NO STOP ---")
    # No stop penalty + high spread penalty
    st = ctrl.update(mark_price=100.0, liquidation_price=80.0, position_side="LONG", 
                     stop_price=None, mae_pct=0.005, spread_pct=0.002) 
    print(f"Margin: {st.margin:.4f} | Score: {st.fall_score:.2f} | Action: {st.action.name}")

    print("\n--- TEST 3: CRITICAL FALL (Approaching Liq Fast) ---")
    # T0: Far
    ctrl.update(mark_price=85.0, liquidation_price=80.0, position_side="LONG")
    # T1: Closer
    ctrl.update(mark_price=82.0, liquidation_price=80.0, position_side="LONG")
    # T2 - CRASH: Very close (0.5 away from 80 is 80.5) -> dist = 0.5/80.5 ~ 0.0062 (0.6%)
    st = ctrl.update(mark_price=80.45, liquidation_price=80.0, position_side="LONG") 
    
    # Check if buffer warning triggers (0.45/80.45 = 0.0055 > 0.005, but fall rate is huge)
    print(f"DistLiq: {(80.45-80)/80.45:.4f} | FallRate: {st.fall_rate:.4f} | Score: {st.fall_score:.2f} | Action: {st.action.name}")
    
    # T3 - DEAD: Inside buffer (< 0.5%)
    st = ctrl.update(mark_price=80.20, liquidation_price=80.0, position_side="LONG")
    print(f"DistLiq: {(80.20-80)/80.20:.4f} | Score: {st.fall_score:.2f} | Action: {st.action.name}")
