#!/usr/bin/env python3
"""
=============================================================================
TITAN V700 - SOVEREIGN EDITION (AGGRESSIVE ACCURACY FIX)
=============================================================================
Logic Stack:
1. CORE ENGINES: Trend, Reversion, Neuren (Velocity), Qaum (Chaos)
2. FRACTAL LAYER: Historical Pattern Matching (Replay)
3. SUPERVISOR 1 (Monitor): RELAXED Volatility Check (Less Skips)
4. SUPERVISOR 2 (Manager): INCREASED Tolerance (Bans after 5 losses)
5. GHOST PROTOCOL: Inversion Logic (Bet Opposite on losing streaks)
6. DYNAMIC MONEY: House Money vs. Defensive Mode
=============================================================================
"""

import math
import statistics
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum

# --- CONFIGURATION ---
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger("TITAN_V700")

class TradeDecision(Enum):
    BIG = "BIG"
    SMALL = "SMALL"
    SKIP = "SKIP"

@dataclass
class RiskProfile:
    base_risk_percent: float = 0.05       
    max_risk_percent: float = 0.15        
    min_bet_amount: float = 10.0
    max_bet_amount: float = 50000.0
    # FIX 1: Increased tolerance. Engines stay alive longer.
    stop_loss_streak: int = 5             
    martingale_multiplier: float = 2.0    

@dataclass
class EngineState:
    """Tracks the health of a specific engine"""
    name: str
    weight: float
    consecutive_losses: int = 0
    is_active: bool = True
    last_vote: Optional[TradeDecision] = None 

class GlobalState:
    """Persistent State across prediction cycles"""
    def __init__(self):
        # Streak Tracking
        self.loss_streak: int = 0
        self.total_wins: int = 0
        self.total_losses: int = 0
        
        # Volatility Management
        self.cooling_off_counter: int = 0  # Rounds to force SKIP
        
        # Ghost Protocol (Inversion)
        self.inversion_mode: bool = False
        self.consecutive_fails: int = 0    # Fails of "Normal" logic
        
        # Session Money Management
        self.session_start_bankroll: float = 0.0
        self.current_profit: float = 0.0
        
        # Engine Manager (The Council)
        self.engines: Dict[str, EngineState] = {
            'trend': EngineState('trend', 1.2),
            'reversion': EngineState('reversion', 1.5),
            'neuren': EngineState('neuren', 3.0),
            'qaum': EngineState('qaum', 3.5),
            'fractal': EngineState('fractal', 2.5) # High weight for patterns
        }
        
        self.last_prediction: Optional[TradeDecision] = None
        self.last_confidence: float = 0.0

# Initialize Global State
state = GlobalState()
config = RiskProfile()

# =============================================================================
# SECTION 1: MATH & PATTERN LIBRARY
# =============================================================================

class MathLib:
    @staticmethod
    def safe_float(value: Any) -> Optional[float]:
        try:
            return float(value) if value is not None else None
        except: return None

    @staticmethod
    def get_z_score(data: List[float]) -> float:
        if len(data) < 2: return 0.0
        try:
            mean = statistics.mean(data)
            stdev = statistics.pstdev(data)
            return (data[-1] - mean) / stdev if stdev != 0 else 0.0
        except: return 0.0

    @staticmethod
    def get_derivative(data: List[float], order: int = 1) -> float:
        if len(data) < order + 1: return 0.0
        current = data
        for _ in range(order):
            current = [current[i+1] - current[i] for i in range(len(current) - 1)]
        return current[-1] if current else 0.0

class FractalLib:
    @staticmethod
    def search_pattern(outcomes: List[str], full_history: List[str]) -> float:
        """Scans full history for the last 6 outcomes."""
        PATTERN_SIZE = 6
        if len(outcomes) < PATTERN_SIZE or len(full_history) < 100:
            return 0.0
            
        current_pattern = outcomes[-PATTERN_SIZE:] 
        match_count = 0
        big_next = 0
        
        # Limit scan to last 2000 rounds for speed
        scan_history = full_history[-2000:]
        
        for i in range(len(scan_history) - PATTERN_SIZE - 1):
            if scan_history[i : i+PATTERN_SIZE] == current_pattern:
                match_count += 1
                if scan_history[i + PATTERN_SIZE] == "BIG":
                    big_next += 1
        
        if match_count < 3: return 0.0
        
        big_prob = big_next / match_count
        if big_prob > 0.75: return 1.0     # Strong BIG Pattern
        if big_prob < 0.25: return -1.0    # Strong SMALL Pattern
        return 0.0

# =============================================================================
# SECTION 2: THE ENGINES
# =============================================================================

class Engines:
    @staticmethod
    def trend_engine(outcomes: List[str]) -> float:
        if len(outcomes) < 5: return 0.0
        last_4 = outcomes[-4:]
        if last_4.count("BIG") == 4: return 1.0     
        if last_4.count("SMALL") == 4: return -1.0  
        if last_4 == ['BIG', 'SMALL', 'BIG', 'SMALL']: return 1.0 
        if last_4 == ['SMALL', 'BIG', 'SMALL', 'BIG']: return -1.0 
        return 0.0

    @staticmethod
    def reversion_engine(numbers: List[float]) -> float:
        if len(numbers) < 15: return 0.0
        z = MathLib.get_z_score(numbers[-20:])
        if z > 2.0: return -1.0
        elif z < -2.0: return 1.0
        return 0.0

    @staticmethod
    def neuren_engine(numbers: List[float]) -> float:
        if len(numbers) < 6: return 0.0
        jerk = MathLib.get_derivative(numbers, 3)
        if jerk > 5.0: return -1.0  
        if jerk < -5.0: return 1.0  
        return 0.0

    @staticmethod
    def qaum_engine(numbers: List[float]) -> float:
        if len(numbers) < 5: return 0.0
        recent = numbers[-4:]
        try:
            variance = statistics.pvariance(recent)
        except: return 0.0 # Handle rare math errors
        
        if variance < 0.8:
            return 1.0 if (sum(recent)/len(recent)) < 4.5 else -1.0        
        return 0.0

# =============================================================================
# SECTION 3: SUPERVISORS (Monitor & Manager)
# =============================================================================

class MarketMonitor:
    @staticmethod
    def check_volatility(numbers: List[float]) -> Tuple[bool, str]:
        if len(numbers) < 10: return False, "OK"
        
        # FIX 2: Relaxed Volatility.
        # We look at last 8 numbers. We only skip if it is PURE CHAOS.
        recent = numbers[-8:]
        
        # 1. Ping Pong Check (0,1,0,1,0...)
        binary = [0 if x <= 4 else 1 for x in recent]
        switches = sum(1 for i in range(len(binary)-1) if binary[i] != binary[i+1])
        
        # Max possible switches in 8 items is 7. We only stop if it switches EVERY time.
        if switches >= 7: return True, "EXTREME_CHOP"

        return False, "SAFE"

class EngineManager:
    @staticmethod
    def update_performance(last_result_str: str):
        """Updates win/loss for each engine and bans/unbans them."""
        actual = None
        if last_result_str == "BIG": actual = TradeDecision.BIG
        elif last_result_str == "SMALL": actual = TradeDecision.SMALL
        else: return # Ignore 0/5 (Green/Violet) or Errors
        
        for name, engine in state.engines.items():
            if engine.last_vote is None or engine.last_vote == TradeDecision.SKIP:
                continue
            
            if engine.last_vote == actual:
                # WON
                if engine.consecutive_losses > 0: engine.consecutive_losses -= 1
                # Recovery: Unban quickly if it gets 1 right
                if not engine.is_active:
                    print(f"[MANAGER] {name.upper()} Recovered. UNBANNED.")
                    engine.is_active = True
            else:
                # LOST
                engine.consecutive_losses += 1
                if engine.is_active and engine.consecutive_losses >= config.stop_loss_streak:
                    print(f"[MANAGER] {name.upper()} Failed {config.stop_loss_streak}x. BANNED.")
                    engine.is_active = False

# =============================================================================
# SECTION 4: VOTING COUNCIL
# =============================================================================

class VotingCouncil:
    def cast_votes(self, numbers: List[float], outcomes: List[str]) -> Tuple[TradeDecision, float, List[str]]:
        score = 0.0
        reasons = []
        
        # 1. Collect All Votes
        raw_votes = {
            'trend': Engines.trend_engine(outcomes),
            'reversion': Engines.reversion_engine(numbers),
            'neuren': Engines.neuren_engine(numbers),
            'qaum': Engines.qaum_engine(numbers),
            'fractal': FractalLib.search_pattern(outcomes, outcomes) # Pass full history
        }
        
        # 2. Weigh Votes (Only Active Engines)
        active_weight_sum = 0.0
        
        for name, val in raw_votes.items():
            eng = state.engines[name]
            
            # Record vote for Manager check next round
            if val > 0: eng.last_vote = TradeDecision.BIG
            elif val < 0: eng.last_vote = TradeDecision.SMALL
            else: eng.last_vote = TradeDecision.SKIP
            
            if not eng.is_active:
                reasons.append(f"{name}(BANNED)")
                continue
                
            active_weight_sum += eng.weight
            score += val * eng.weight
            if val != 0: reasons.append(f"{name}({val:+.1f})")

        # 3. Final Decision
        if active_weight_sum == 0:
            return TradeDecision.SKIP, 0.0, ["ALL_ENGINES_DEAD"]
            
        # Normalize Score (Scale roughly to -1..1 range)
        normalized_score = score / 2.5 
        
        decision = TradeDecision.SKIP
        conf = 0.0
        
        # FIX 3: LOWERED THRESHOLD (1.0 -> 0.6)
        # This allows trades when only moderate consensus is found.
        THRESHOLD = 0.6
        
        if normalized_score >= THRESHOLD:
            decision = TradeDecision.BIG
            conf = min(0.6 + (normalized_score/5), 0.98)
        elif normalized_score <= -THRESHOLD:
            decision = TradeDecision.SMALL
            conf = min(0.6 + (abs(normalized_score)/5), 0.98)
            
        return decision, conf, reasons

# =============================================================================
# MAIN EXPORT: ULTRA AI PREDICT
# =============================================================================

def ultraAIPredict(history: List[Dict], currentbankroll: float, lastresult: Optional[str] = None) -> Dict:
    
    # --- PHASE 0: INITIALIZATION ---
    if state.session_start_bankroll == 0:
        state.session_start_bankroll = currentbankroll
    state.current_profit = currentbankroll - state.session_start_bankroll

    # --- PHASE 1: DATA CLEANING ---
    clean_nums = []
    clean_outcomes = []
    for item in reversed(history):
        v = MathLib.safe_float(item.get('actual_number'))
        if v is not None:
            clean_nums.append(v)
            if 0 <= int(v) <= 4: clean_outcomes.append("SMALL")
            elif 5 <= int(v) <= 9: clean_outcomes.append("BIG")

    # --- PHASE 2: FEEDBACK LOOP (Manager & Ghost) ---
    if lastresult and state.last_prediction and state.last_prediction != TradeDecision.SKIP:
        # 2a. Update Engines
        EngineManager.update_performance(lastresult)
        
        # 2b. Check Ghost Protocol
        real_res = None
        if lastresult == "BIG": real_res = TradeDecision.BIG
        elif lastresult == "SMALL": real_res = TradeDecision.SMALL
        
        if real_res:
            did_win = (state.last_prediction == real_res)
            
            if did_win:
                state.loss_streak = 0
                if state.inversion_mode:
                    print("[GHOST] Inversion Logic WON. Keeping Ghost Mode Active.")
                else:
                    state.consecutive_fails = 0 
            else:
                state.loss_streak += 1
                if not state.inversion_mode:
                    state.consecutive_fails += 1
                    if state.consecutive_fails >= 3:
                        state.inversion_mode = True
                        print("[GHOST] ⚠️ 3x Fail. ACTIVATING INVERSION PROTOCOL.")
                else:
                    # Lost in Ghost Mode -> Chaos -> Reset
                    state.inversion_mode = False
                    state.consecutive_fails = 0
                    print("[GHOST] Inversion Failed. Resetting to Standard.")

    # --- PHASE 3: SAFETY CHECKS ---
    if state.cooling_off_counter > 0:
        state.cooling_off_counter -= 1
        # Removing the "Return Skip" allows aggressive recovery
        # but for safety, we just reduce the count and let it try to trade if strong enough.
                
    is_unsafe, vol_reason = MarketMonitor.check_volatility(clean_nums)
    if is_unsafe:
        state.cooling_off_counter = 1
        return {'finalDecision': "SKIP", 'confidence': 0, 'positionsize': 0, 
                'level': "DANGER", 'reason': f"Volatile: {vol_reason}", 'topsignals': ["MARKET_UNSAFE"]}

    # --- PHASE 4: PREDICTION ---
    council = VotingCouncil()
    decision, confidence, signals = council.cast_votes(clean_nums, clean_outcomes)
    
    # --- PHASE 5: GHOST PROTOCOL (THE FLIP) ---
    final_decision_str = decision.value
    meta_status = "STANDARD"
    
    if decision != TradeDecision.SKIP and state.inversion_mode:
        meta_status = "GHOST-ACTIVE"
        if decision == TradeDecision.BIG:
            final_decision_str = "SMALL" # FLIP
            signals.append("INVERTED(BIG->SMALL)")
        elif decision == TradeDecision.SMALL:
            final_decision_str = "BIG"   # FLIP
            signals.append("INVERTED(SMALL->BIG)")

    # --- PHASE 6: DYNAMIC STAKE ---
    if final_decision_str == "SKIP":
        state.last_prediction = TradeDecision.SKIP
        return {'finalDecision': "SKIP", 'confidence': 0, 'positionsize': 0, 
                'level': "WAIT", 'reason': "Low Conf / Banned", 'topsignals': signals}

    # House Money Logic
    base_stake = max(currentbankroll * config.base_risk_percent, config.min_bet_amount)
    
    if state.current_profit > 100:
        base_stake *= 1.5
        meta_status += "|HOUSE_MONEY"
    elif state.current_profit < -150:
        base_stake *= 0.5
        meta_status += "|DEFENSIVE"
        
    # Martingale
    stake = base_stake * (config.martingale_multiplier ** state.loss_streak)
    
    # Hard Limit (20% of bankroll max)
    if stake > (currentbankroll * 0.20): 
        stake = currentbankroll * 0.20
        meta_status += "|MAX_CLAMP"

    # Save State
    state.last_prediction = TradeDecision(final_decision_str)
    state.last_confidence = confidence

    return {
        'finalDecision': final_decision_str,
        'confidence': round(confidence, 4),
        'positionsize': int(stake),
        'level': f"L{state.loss_streak} ({meta_status})",
        'reason': " | ".join(signals),
        'topsignals': signals
    }

if __name__ == "__main__":
    print("TITAN V700 SOVEREIGN (AGGRESSIVE ACCURACY) LOADED.")
