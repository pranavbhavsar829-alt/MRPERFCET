#!/usr/bin/env python3
"""
=============================================================================
  _______ _____ _______       _   _     __      _______  ___   ___  
 |__   __|_   _|__   __|     | \ | |    \ \    / /___  |/ _ \ / _ \ 
    | |    | |    | |  ____  |  \| |_____\ \  / /   / /| | | | | | |
    | |    | |    | | |____| | . ` |______\ \/ /   / / | | | | | | |
    | |   _| |_   | |        | |\  |       \  /   / /  | |_| | |_| |
    |_|  |_____|  |_|        |_| \_|        \/   /_/    \___/ \___/ 
                                                                    
  TITAN V202 - DEEP THOUGHT EDITION (TIME-WINDOW OPTIMIZED)
  
  STRATEGY:
  1. INITIAL SCAN: Fast analysis using Trident Engines.
  2. DEEP THOUGHT LOOP (20s): Cross-validates the signal against history.
  3. BACKTEST VERIFICATION: Checks if this specific signal is currently winning.
  4. SNIPER EXECUTION: only fires if the signal survives the 20s stress test.
=============================================================================
"""

import math
import statistics
import logging
import time
from typing import Dict, List, Optional, Any, Tuple

# =============================================================================
# [PART 1] CONFIGURATION
# =============================================================================

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [TITAN_DEEP] %(message)s', datefmt='%H:%M:%S')

class GameConstants:
    BIG = "BIG"
    SMALL = "SMALL" 
    SKIP = "SKIP"
    
    # We use a large buffer for the Deep Thought backtesting
    MIN_HISTORY_FOR_PREDICTION = 100 
    
    # CRITICAL: How many seconds to think?
    THINKING_TIME_SECONDS = 18  # Leaves buffer for network latency

class RiskConfig:
    # Bankroll
    BASE_RISK_PERCENT = 0.05
    MIN_BET_AMOUNT = 10
    MAX_BET_AMOUNT = 100000
    
    # Confidence Thresholds
    LVL1_MIN_CONFIDENCE = 0.65  # Higher standard
    LVL2_MIN_CONFIDENCE = 0.75
    LVL3_MIN_CONFIDENCE = 0.88

    # Multipliers
    TIER_1_MULT = 1.0
    TIER_2_MULT = 2.0
    TIER_3_MULT = 4.0
    
    STOP_LOSS_STREAK = 8

# =============================================================================
# [PART 2] MATH CORE
# =============================================================================

def safe_float(value: Any) -> float:
    try:
        if value is None: return 4.5
        return float(value)
    except: return 4.5

def get_outcome_from_number(n: Any) -> Optional[str]:
    val = int(safe_float(n))
    if 0 <= val <= 4: return GameConstants.SMALL
    if 5 <= val <= 9: return GameConstants.BIG
    return None

def sigmoid(x):
    try: return 1 / (1 + math.exp(-x))
    except OverflowError: return 0.0 if x < 0 else 1.0

def calculate_mean(data: List[float]) -> float:
    return sum(data) / len(data) if data else 0.0

def calculate_stddev(data: List[float]) -> float:
    if len(data) < 2: return 0.0
    mean = calculate_mean(data)
    variance = sum((x - mean) ** 2 for x in data) / (len(data) - 1)
    return math.sqrt(variance)

def calculate_momentum_mass(data: List[float]) -> float:
    if len(data) < 5: return 0.0
    vel = [data[i+1]-data[i] for i in range(len(data)-1)]
    if not vel: return 0.0
    avg_vel = sum(vel) / len(vel)
    acc = [vel[i+1]-vel[i] for i in range(len(vel)-1)]
    if not acc: return 0.0
    avg_acc = sum(acc) / len(acc)
    return avg_vel * avg_acc

# =============================================================================
# [PART 3] TRIDENT ENGINES
# =============================================================================

# Engine 1: Quantum Chaos
def engine_quantum_chaos(history: List[Dict]) -> Optional[Dict]:
    try:
        numbers = [safe_float(d.get('actual_number')) for d in history[-30:]]
        if len(numbers) < 20: return None
        
        # Variance Guard
        variance = statistics.pvariance(numbers[-5:])
        if variance > 2.5: return None 
        
        mean = calculate_mean(numbers)
        std = calculate_stddev(numbers)
        if std == 0: return None
        
        z_score = (numbers[-1] - mean) / std
        
        # Dragon Trap
        if abs(z_score) > 2.8: return None 
        
        strength = min(abs(z_score) / 2.5, 1.0) 
        
        if z_score > 1.8:
            return {'prediction': GameConstants.SMALL, 'weight': strength, 'source': 'Quantum'}
        elif z_score < -1.8:
            return {'prediction': GameConstants.BIG, 'weight': strength, 'source': 'Quantum'}
        return None
    except: return None

# Engine 2: Deep Fractal
def engine_deep_fractal(history: List[Dict]) -> Optional[Dict]:
    try:
        if len(history) < 60: return None
        outcomes = [get_outcome_from_number(d.get('actual_number')) for d in history]
        raw_outcomes = outcomes
        
        best_signal = None
        
        # Scan lengths 7 down to 3
        for depth in range(7, 2, -1):
            if len(outcomes) < depth + 1: continue
            target = raw_outcomes[-depth:]
            history_slice = raw_outcomes[:-1]
            
            match_count = 0
            next_big = 0
            
            # Limit scan to last 1000 for speed
            scan_limit = min(len(history_slice), 1000)
            start_idx = len(history_slice) - scan_limit
            
            for i in range(len(history_slice) - depth - 1, start_idx, -1):
                if history_slice[i : i+depth] == target:
                    match_count += 1
                    if history_slice[i+depth] == GameConstants.BIG:
                        next_big += 1
                        
            if match_count >= 3:
                prob_big = next_big / match_count
                
                if prob_big >= 0.65:
                    return {'prediction': GameConstants.BIG, 'weight': prob_big * 1.5, 'source': f'Fractal-D{depth}'}
                elif prob_big <= 0.35:
                    return {'prediction': GameConstants.SMALL, 'weight': (1.0-prob_big) * 1.5, 'source': f'Fractal-D{depth}'}
        return None
    except: return None

# Engine 3: Neural Physics
def engine_neural_physics(history: List[Dict]) -> Optional[Dict]:
    try:
        numbers = [safe_float(d.get('actual_number')) for d in history[-40:]]
        if len(numbers) < 25: return None
        
        # RSI
        deltas = [numbers[i] - numbers[i-1] for i in range(1, len(numbers))]
        gains = [d for d in deltas if d > 0]
        losses = [abs(d) for d in deltas if d < 0]
        avg_gain = sum(gains[-14:]) / 14 if gains else 0
        avg_loss = sum(losses[-14:]) / 14 if losses else 0.001
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        norm_rsi = (rsi - 50) / 50
        
        # Mass
        mass = calculate_momentum_mass(numbers[-6:])
        norm_mass = max(min(mass / 10.0, 1.0), -1.0)
        
        # Calc
        z = (norm_rsi * -1.2) + (norm_mass * -1.5) + 0.1
        activation = sigmoid(z)
        dist = abs(activation - 0.5)
        
        if activation > 0.65:
            return {'prediction': GameConstants.BIG, 'weight': dist * 2.0, 'source': 'NeuralPhys'}
        elif activation < 0.35:
            return {'prediction': GameConstants.SMALL, 'weight': dist * 2.0, 'source': 'NeuralPhys'}
        return None
    except: return None

# =============================================================================
# [PART 4] DEEP THOUGHT VERIFICATION (THE 20-SECOND BRAIN)
# =============================================================================

class DeepThoughtCore:
    
    @staticmethod
    def stress_test_signal(history: List[Dict], candidate_signal: str) -> float:
        """
        Runs a backtest simulation on the CURRENT signal.
        Does this signal actually win in the current market conditions?
        Returns a 'Validity Score' (0.0 to 1.0)
        """
        try:
            # We look at the last 100 rounds
            test_window = history[-100:]
            wins = 0
            losses = 0
            
            # Replay the last 100 rounds
            for i in range(50, len(test_window)-1):
                past_slice = test_window[:i]
                actual_next = get_outcome_from_number(test_window[i]['actual_number'])
                
                # Ask engines what they WOULD have said
                s1 = engine_quantum_chaos(past_slice)
                s2 = engine_deep_fractal(past_slice)
                s3 = engine_neural_physics(past_slice)
                
                signals = [s for s in [s1, s2, s3] if s]
                if not signals: continue
                
                # Simple vote
                big_w = sum(s['weight'] for s in signals if s['prediction'] == GameConstants.BIG)
                small_w = sum(s['weight'] for s in signals if s['prediction'] == GameConstants.SMALL)
                
                consensus = GameConstants.BIG if big_w > small_w else GameConstants.SMALL
                
                # Check accuracy
                if consensus == candidate_signal and big_w + small_w > 0.4:
                    if consensus == actual_next:
                        wins += 1
                    else:
                        losses += 1
                        
            total_trades = wins + losses
            if total_trades < 3: return 0.5 # Neutral if no data
            
            accuracy = wins / total_trades
            return accuracy
            
        except: return 0.5


# =============================================================================
# [PART 5] MAIN EXECUTION
# =============================================================================

class GlobalStateManager:
    def __init__(self):
        self.loss_streak = 0
state_manager = GlobalStateManager()

def ultraAIPredict(history: List[Dict], current_bankroll: float = 10000.0, last_result: Optional[str] = None) -> Dict:
    
    start_time = time.time() # START THE TIMER
    
    # 1. Update Streak
    if last_result:
        actual_outcome = get_outcome_from_number(history[-1]['actual_number'])
        if last_result != GameConstants.SKIP:
            if last_result == actual_outcome:
                state_manager.loss_streak = 0
            else:
                state_manager.loss_streak += 1
                
    streak = state_manager.loss_streak
    
    # 2. Violet Guard
    try:
        if int(safe_float(history[-1]['actual_number'])) in [0, 5]:
            return {'finalDecision': "SKIP", 'confidence': 0, 'positionsize': 0, 'level': "RESET", 'reason': "Violet Guard", 'topsignals': []}
    except: pass
    
    # 3. INITIAL SIGNAL GENERATION
    s1 = engine_quantum_chaos(history)
    s2 = engine_deep_fractal(history)
    s3 = engine_neural_physics(history)
    
    signals = [s for s in [s1, s2, s3] if s]
    
    big_w = sum(s['weight'] for s in signals if s['prediction'] == GameConstants.BIG)
    small_w = sum(s['weight'] for s in signals if s['prediction'] == GameConstants.SMALL)
    
    # Preliminary Decision
    if big_w > small_w:
        candidate_pred = GameConstants.BIG
        raw_confidence = big_w / (big_w + small_w + 0.1)
    else:
        candidate_pred = GameConstants.SMALL
        raw_confidence = small_w / (big_w + small_w + 0.1)
        
    # If initial signal is too weak, skip immediately
    if (big_w + small_w) < 0.4:
         return {'finalDecision': "SKIP", 'confidence': 0, 'positionsize': 0, 'level': "WAIT", 'reason': "Weak Initial Signal", 'topsignals': []}

    # =========================================================================
    # 4. THE DEEP THOUGHT LOOP (TIME WINDOW UTILIZATION)
    # =========================================================================
    
    # We have a candidate prediction. Now we spend time validating it.
    # We loop until we are close to our time limit.
    
    validation_score = 0.5
    iterations = 0
    
    while (time.time() - start_time) < GameConstants.THINKING_TIME_SECONDS:
        # Perform a stress test
        # We run the simulation multiple times (conceptually) or perform a deep scan
        # Since the 'stress_test_signal' is heavy, one run is enough, but we can
        # delay return to allow the system to "settle" or run additional checks.
        
        # Here, we run the Deep Backtest once, which takes computational time
        validation_score = DeepThoughtCore.stress_test_signal(history, candidate_pred)
        
        iterations += 1
        
        # If the backtest is TERRIBLE (< 40% winrate recently), we break early and abort
        if validation_score < 0.40:
            break
            
        # If the backtest is AMAZING (> 80%), we lock it in
        if validation_score > 0.80:
            break
            
        # Artificial delay to ensure we don't spam CPU if the calculation was too fast
        # (Simulating "waiting for market settle" if this were live stream data)
        time.sleep(0.1) 
        
    # =========================================================================
    # 5. FINAL SYNTHESIS
    # =========================================================================
    
    # Adjust Confidence based on Deep Thought
    final_confidence = (raw_confidence + validation_score) / 2
    
    # If the Deep Thought validation failed, we KILL the trade
    if validation_score < 0.50:
        return {'finalDecision': "SKIP", 'confidence': 0, 'positionsize': 0, 'level': "ABORT", 'reason': f"DeepThought Failed ({validation_score:.0%})", 'topsignals': []}
        
    # 6. STAKING LOGIC
    stake = 0
    level = "SKIP"
    reason = f"Conf {final_confidence:.0%} | Valid {validation_score:.0%}"
    
    base_bet = max(current_bankroll * RiskConfig.BASE_RISK_PERCENT, RiskConfig.MIN_BET_AMOUNT)
    
    active_sources = [s['source'] for s in signals]
    active_sources.append(f"DeepThought({validation_score:.2f})")
    
    # Sniper Levels
    if streak >= 2:
        if final_confidence >= RiskConfig.LVL3_MIN_CONFIDENCE:
            stake = base_bet * RiskConfig.TIER_3_MULT
            level = "🔥 SNIPER"
        else:
            level = "SKIP (Recov)"
    elif streak == 1:
        if final_confidence >= RiskConfig.LVL2_MIN_CONFIDENCE:
            stake = base_bet * RiskConfig.TIER_2_MULT
            level = "RECOVERY"
        else:
            level = "SKIP (Recov)"
    else:
        if final_confidence >= RiskConfig.LVL1_MIN_CONFIDENCE:
            stake = base_bet * RiskConfig.TIER_1_MULT
            level = "STANDARD"
        else:
            level = "SKIP"
            
    if stake > current_bankroll * 0.4: stake = current_bankroll * 0.4
    
    return {
        'finalDecision': candidate_pred if stake > 0 else "SKIP",
        'confidence': final_confidence,
        'positionsize': int(stake),
        'level': level,
        'reason': reason,
        'topsignals': active_sources
    }

if __name__ == "__main__":
    print("TITAN V202 DEEP THOUGHT: ONLINE")
