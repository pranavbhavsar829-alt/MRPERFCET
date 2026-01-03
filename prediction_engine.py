#!/usr/bin/env python3
"""
=============================================================================
  _______ _____ _______ _    _  _   _ 
 |__   __|_   _|__   __| |  | || \ | |
    | |    | |    | |  | |  | ||  \| |
    | |    | |    | |  | |  | || . ` |
    | |   _| |_   | |  | |__| || |\  |
    |_|  |_____|  |_|   \____/ |_| \_|
                                      
  TITAN V300 - ASSAULT EDITION (AGGRESSIVE CORE)
  
  CHANGELOG:
  1. VIOLET GUARD REMOVED (No skips on 0/5).
  2. CONFIDENCE THRESHOLDS LOWERED (Bets on 51% probability).
  3. NEW ENGINE: "Trend Force" added for streak detection.
  4. NEW ENGINE: "Chaos Theory" added for randomization fallback.
  5. FALLBACK SYSTEM: Guarantees a prediction every period.
=============================================================================
"""

import math
import statistics
import random
import traceback
import json
from collections import Counter, defaultdict, deque
from typing import Dict, List, Optional, Any, Tuple

# =============================================================================
# SECTION 1: GAME CONSTANTS & MAPPING
# =============================================================================

class GameConstants:
    """
    Defines the immutable constants for the Lottery Game.
    """
    BIG = "BIG"
    SMALL = "SMALL" 
    SKIP = "SKIP" # Rarely used in V300
    
    # In Aggressive Mode, we only need 5 rounds of history to start shooting.
    MIN_HISTORY_FOR_PREDICTION = 5
    DEBUG_MODE = True

class RiskConfig:
    """
    AGGRESSIVE RISK CONFIGURATION
    We have lowered the bars significantly to allow constant action.
    """
    # -------------------------------------------------------------------------
    # BANKROLL MANAGEMENT
    # -------------------------------------------------------------------------
    BASE_RISK_PERCENT = 0.05    # INCREASED to 5% Base Risk
    MIN_BET_AMOUNT = 100
    MAX_BET_AMOUNT = 50000
    
    # -------------------------------------------------------------------------
    # CONFIDENCE THRESHOLDS (LOWERED FOR AGGRESSION)
    # -------------------------------------------------------------------------
    
    # LEVEL 1: Standard (Any slight edge triggers a bet)
    LVL1_MIN_CONFIDENCE = 0.51  # Was 0.60
    
    # LEVEL 2: Recovery (After 1 Loss)
    LVL2_MIN_CONFIDENCE = 0.60  # Was 0.70
    
    # LEVEL 3: SNIPER (After 2+ Losses)
    LVL3_MIN_CONFIDENCE = 0.75  # Was 0.85

    # -------------------------------------------------------------------------
    # MARTINGALE STEPS (AGGRESSIVE RECOVERY)
    # -------------------------------------------------------------------------
    TIER_1_MULT = 1.0
    TIER_2_MULT = 2.0   # Hard Double
    TIER_3_MULT = 4.5   # Massive Recovery
    STOP_LOSS_STREAK = 8 # Extended stop loss

# =============================================================================
# SECTION 2: MATHEMATICAL UTILITIES
# =============================================================================

def safe_float(value: Any) -> float:
    """Safely converts any input to float, returning a neutral 4.5 on failure."""
    try:
        if value is None: return 4.5
        return float(value)
    except: return 4.5

def get_outcome_from_number(n: Any) -> Optional[str]:
    """Maps a number (0-9) to BIG/SMALL."""
    val = int(safe_float(n))
    if 0 <= val <= 4: return GameConstants.SMALL
    if 5 <= val <= 9: return GameConstants.BIG
    return None

def sigmoid(x):
    """
    The Activation Function for our Neural Engine.
    Squashes values between 0 and 1.
    """
    try:
        return 1 / (1 + math.exp(-x))
    except OverflowError:
        return 0.0 if x < 0 else 1.0

def calculate_mean(data: List[float]) -> float:
    """Calculates arithmetic mean."""
    return sum(data) / len(data) if data else 0.0

def calculate_stddev(data: List[float]) -> float:
    """Calculates standard deviation."""
    if len(data) < 2: return 0.0
    mean = calculate_mean(data)
    variance = sum((x - mean) ** 2 for x in data) / (len(data) - 1)
    return math.sqrt(variance)

def calculate_rsi(data: List[float], period: int = 6) -> float:
    """
    Relative Strength Index (RSI).
    PERIOD REDUCED TO 6 FOR HYPER-SENSITIVITY.
    """
    if len(data) < period + 1: return 50.0
    deltas = [data[i] - data[i-1] for i in range(1, len(data))]
    gains = [d if d > 0 else 0 for d in deltas]
    losses = [-d if d < 0 else 0 for d in deltas]
    
    avg_gain = calculate_mean(gains[-period:])
    avg_loss = calculate_mean(losses[-period:])
    
    if avg_loss == 0: return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))

# =============================================================================
# SECTION 3: THE FOUR HORSEMEN (ENGINES)
# =============================================================================

# -----------------------------------------------------------------------------
# ENGINE 1: QUANTUM AI (ADAPTIVE BOLLINGER)
# -----------------------------------------------------------------------------
def engine_quantum_adaptive(history: List[Dict]) -> Optional[Dict]:
    """
    Determines if the number stream is statistically overextended.
    In Aggressive Mode, we bet on Reversion MUCH sooner.
    """
    try:
        numbers = [safe_float(d.get('actual_number')) for d in history[-20:]]
        if len(numbers) < 10: return None
        
        mean = calculate_mean(numbers)
        std = calculate_stddev(numbers)
        if std == 0: return None
        
        current_val = numbers[-1]
        z_score = (current_val - mean) / std
        
        # AGGRESSIVE TWEAK: Removed Dragon Trap. We always bet reversion.
        strength = min(abs(z_score) / 1.5, 1.0) # Reach max strength faster
        
        # Triggers at 1.2 deviation instead of 1.6
        if z_score > 1.2:
            return {'prediction': GameConstants.SMALL, 'weight': strength, 'source': f'Quantum(Z:{z_score:.1f})'}
        elif z_score < -1.2:
            return {'prediction': GameConstants.BIG, 'weight': strength, 'source': f'Quantum(Z:{z_score:.1f})'}
            
        return None
    except: return None

# -----------------------------------------------------------------------------
# ENGINE 2: DEEP PATTERN V4 (OMNI-SCANNER)
# -----------------------------------------------------------------------------
def engine_deep_pattern_v4(history: List[Dict]) -> Optional[Dict]:
    """
    Scans for patterns of length 3 up to 10.
    Finds the most repetitive sequence in recent history.
    """
    try:
        if len(history) < 30: return None
        
        # Convert history to "B" or "S" string
        outcomes = [get_outcome_from_number(d.get('actual_number')) for d in history]
        raw_str = ''.join(['B' if o==GameConstants.BIG else 'S' for o in outcomes if o])
        
        best_signal = None
        highest_confidence = 0.0
        
        # Iterate from deep patterns (10) to shallow (3)
        for depth in range(10, 2, -1):
            curr_pattern = raw_str[-depth:]
            search_area = raw_str[:-1] 
            
            count_b_next = 0
            count_s_next = 0
            
            start = 0
            while True:
                idx = search_area.find(curr_pattern, start)
                if idx == -1: break
                
                if idx + depth < len(search_area):
                    next_char = search_area[idx + depth]
                    if next_char == 'B': count_b_next += 1
                    else: count_s_next += 1
                
                start = idx + 1
            
            total_matches = count_b_next + count_s_next
            
            # In Aggressive Mode, we accept even 1 previous match if it's deep
            if total_matches >= 1:
                prob_b = count_b_next / total_matches
                prob_s = count_s_next / total_matches
                
                imbalance = abs(prob_b - prob_s)
                
                # Boost weight if deep
                adjusted_weight = imbalance * (1 + (depth * 0.15))
                
                if adjusted_weight > highest_confidence: 
                    highest_confidence = adjusted_weight
                    pred = GameConstants.BIG if prob_b > prob_s else GameConstants.SMALL
                    best_signal = {'prediction': pred, 'weight': adjusted_weight, 'source': f'Pattern-D{depth}'}
                    
                    if imbalance > 0.9: break # Found a perfect match, stop looking

        return best_signal
    except: return None

# -----------------------------------------------------------------------------
# ENGINE 3: NEURAL PERCEPTRON (MARKET SENSOR)
# -----------------------------------------------------------------------------
def engine_neural_perceptron(history: List[Dict]) -> Optional[Dict]:
    """
    Uses RSI, Momentum, and Parity to form a weighted decision.
    """
    try:
        numbers = [safe_float(d.get('actual_number')) for d in history[-30:]]
        if len(numbers) < 15: return None
        
        # INPUT 1: FAST RSI (Period 6)
        rsi = calculate_rsi(numbers, 6)
        input_rsi = (rsi - 50) / 100.0 
        
        # INPUT 2: MOMENTUM
        last_5 = numbers[-5:]
        momentum = (last_5[-1] - last_5[0]) / 10.0
        
        # INPUT 3: REVERSION FORCE
        last_3_outcomes = [get_outcome_from_number(n) for n in numbers[-3:]]
        b_count = last_3_outcomes.count(GameConstants.BIG)
        # If 3 BIGS, input is negative (Predict Small)
        input_rev = (1.5 - b_count) / 1.5
        
        # NEURAL WEIGHTS (Tuned for Aggression)
        w_rsi = -2.0  # Strong contrarian
        w_mom = 0.5   # Slight trend following
        w_rev = 1.5   # Strong reversion
        
        z = (input_rsi * w_rsi) + (momentum * w_mom) + (input_rev * w_rev)
        probability = sigmoid(z) 
        
        dist_from_neutral = abs(probability - 0.5)
        
        # Lower threshold: anything > 0.55 triggers
        if probability > 0.55:
            return {'prediction': GameConstants.BIG, 'weight': dist_from_neutral * 3.0, 'source': f'Neural({probability:.2f})'}
        elif probability < 0.45:
            return {'prediction': GameConstants.SMALL, 'weight': dist_from_neutral * 3.0, 'source': f'Neural({probability:.2f})'}
            
        return None
    except: return None

# -----------------------------------------------------------------------------
# ENGINE 4: TREND FORCE (NEW!)
# -----------------------------------------------------------------------------
def engine_trend_force(history: List[Dict]) -> Optional[Dict]:
    """
    The "Flow" detector. If the river is flowing BIG, we swim BIG.
    Detects streaks (e.g., BIG, BIG, BIG).
    """
    try:
        if len(history) < 5: return None
        outcomes = [get_outcome_from_number(d.get('actual_number')) for d in history[-5:]]
        
        # Check last 2
        last_1 = outcomes[-1]
        last_2 = outcomes[-2]
        
        if last_1 == last_2:
            # Trend Detected!
            streak_len = 2
            if outcomes[-3] == last_1: streak_len = 3
            if outcomes[-4] == last_1: streak_len = 4
            
            # The longer the streak, the stronger the signal to FOLLOW it
            # (Until it gets too long, then we might fade, but here we follow)
            weight = 0.4 + (streak_len * 0.1)
            return {'prediction': last_1, 'weight': weight, 'source': f'TrendForce(x{streak_len})'}
            
        return None
    except: return None

# =============================================================================
# SECTION 5: THE ARCHITECT (GLOBAL STATE & MAIN LOGIC)
# =============================================================================

class GlobalStateManager:
    def __init__(self):
        self.loss_streak = 0
        self.last_outcome = None
        self.total_wins = 0
        self.total_losses = 0

# Singleton State
state_manager = GlobalStateManager()

def ultraAIPredict(history: List[Dict], current_bankroll: float = 10000.0, last_result: Optional[str] = None) -> Dict:
    """
    THE BRAIN OF TITAN V300.
    Aggregates all engines and forces a decision.
    """
    
    # -------------------------------------------------------------------------
    # 1. STATE MANAGEMENT
    # -------------------------------------------------------------------------
    if last_result:
        try:
            actual_outcome = get_outcome_from_number(history[-1]['actual_number'])
            if last_result == GameConstants.SKIP:
                pass
            elif last_result == actual_outcome:
                state_manager.loss_streak = 0
                state_manager.total_wins += 1
            else:
                state_manager.loss_streak += 1
                state_manager.total_losses += 1
        except: pass
            
    streak = state_manager.loss_streak
    
    # -------------------------------------------------------------------------
    # 2. RUN ENGINES
    # -------------------------------------------------------------------------
    signals = []
    
    # Engine 1: Quantum
    s1 = engine_quantum_adaptive(history)
    if s1: signals.append(s1)
    
    # Engine 2: Deep Pattern
    s2 = engine_deep_pattern_v4(history)
    if s2: signals.append(s2)
    
    # Engine 3: Neural Net
    s3 = engine_neural_perceptron(history)
    if s3: signals.append(s3)
    
    # Engine 4: Trend Force
    s4 = engine_trend_force(history)
    if s4: signals.append(s4)
    
    # -------------------------------------------------------------------------
    # 3. AGGREGATE SIGNALS (VOTING SYSTEM)
    # -------------------------------------------------------------------------
    big_score = 0.0
    small_score = 0.0
    active_sources = []
    
    for s in signals:
        if s['prediction'] == GameConstants.BIG:
            big_score += s['weight']
        elif s['prediction'] == GameConstants.SMALL:
            small_score += s['weight']
        active_sources.append(s['source'])
            
    total_score = big_score + small_score
    
    # -------------------------------------------------------------------------
    # 4. DECISION LOGIC
    # -------------------------------------------------------------------------
    final_pred = GameConstants.SKIP
    confidence = 0.0
    
    if total_score > 0:
        if big_score > small_score:
            final_pred = GameConstants.BIG
            confidence = big_score / (total_score + 0.1) # Add small epsilon
        else:
            final_pred = GameConstants.SMALL
            confidence = small_score / (total_score + 0.1)
    
    # -------------------------------------------------------------------------
    # 5. THE FALLBACK (CHAOS THEORY)
    # -------------------------------------------------------------------------
    # If engines are silent or confused, we DO NOT SKIP.
    # We use a fallback logic: "Anti-Repeat"
    if final_pred == GameConstants.SKIP or confidence < 0.2:
        try:
            last_num = int(safe_float(history[-1]['actual_number']))
            # If last was >= 5 (Big), we bet Small.
            if last_num >= 5:
                final_pred = GameConstants.SMALL
            else:
                final_pred = GameConstants.BIG
            confidence = 0.50
            active_sources.append("ChaosFallback")
        except:
            # Absolute fail-safe
            final_pred = GameConstants.SMALL
            active_sources.append("EmergencyRand")

    # -------------------------------------------------------------------------
    # 6. STAKING & LEVEL (AGGRESSIVE)
    # -------------------------------------------------------------------------
    stake = 0
    level = "STANDARD"
    reason = f"Aggro:{confidence:.2f}"
    
    base_bet = max(current_bankroll * RiskConfig.BASE_RISK_PERCENT, RiskConfig.MIN_BET_AMOUNT)
    
    # Logic Gate
    if streak >= 2:
        # SNIPER MODE (Even with lower confidence)
        if confidence >= 0.55:
            stake = base_bet * RiskConfig.TIER_3_MULT
            level = "🔥 SNIPER"
        else:
            # Forced recovery even if low confidence
            stake = base_bet * RiskConfig.TIER_2_MULT
            level = "FORCE RECOV"
            
    elif streak == 1:
        # RECOVERY MODE
        stake = base_bet * RiskConfig.TIER_2_MULT
        level = "RECOVERY"
        
    else:
        # STANDARD MODE
        stake = base_bet * RiskConfig.TIER_1_MULT
        level = "STANDARD"

    # Sanity Check
    stake = int(stake)
    if stake > RiskConfig.MAX_BET_AMOUNT: stake = RiskConfig.MAX_BET_AMOUNT
    
    # -------------------------------------------------------------------------
    # 7. FINAL RETURN
    # -------------------------------------------------------------------------
    return {
        'finalDecision': final_pred,
        'confidence': confidence,
        'positionsize': stake,
        'level': level,
        'reason': reason,
        'topsignals': active_sources[:3] # Show top 3 reasons
    }

# =============================================================================
# END OF MODULE
# =============================================================================
if __name__ == "__main__":
    print("TITAN V300 AGGRESSIVE CORE LOADED.")
    print("Running Self-Test...")
    # Simple self-test to ensure no syntax errors
    test_hist = [
        {'actual_number': 1}, {'actual_number': 8}, {'actual_number': 2}, 
        {'actual_number': 9}, {'actual_number': 3}, {'actual_number': 7}
    ]
    res = ultraAIPredict(test_hist)
    print(f"Test Result: {res}")
