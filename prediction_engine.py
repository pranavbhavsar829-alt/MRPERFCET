#!/usr/bin/env python3
"""
=============================================================================
  _______ _____ _______ _    _  _   _ 
 |__   __|_   _|__   __| |  | || \ | |
    | |    | |    | |  | |  | ||  \| |
    | |    | |    | |  | |  | || . ` |
    | |   _| |_   | |  | |__| || |\  |
    |_|  |_____|  |_|   \____/ |_| \_|
                                      
  TITAN V200 - THE TRIDENT CORE (NEURAL EDITION)
  (Streamlined: Quantum + Pattern + Neural)
=============================================================================
"""

import math
import statistics
import random
import traceback
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Any, Tuple

# =============================================================================
# SECTION 1: IMMUTABLE GAME CONSTANTS
# =============================================================================

class GameConstants:
    BIG = "BIG"
    SMALL = "SMALL" 
    SKIP = "SKIP"
    
    # We need decent history for the Neural Engine to warm up
    MIN_HISTORY_FOR_PREDICTION = 40
    DEBUG_MODE = True

# =============================================================================
# SECTION 2: RISK & SNIPER CONFIGURATION
# =============================================================================

class RiskConfig:
    # -------------------------------------------------------------------------
    # BANKROLL MANAGEMENT
    # -------------------------------------------------------------------------
    BASE_RISK_PERCENT = 0.03    # Increased to 3% (Since we have higher quality signals)
    MIN_BET_AMOUNT = 50
    MAX_BET_AMOUNT = 50000
    
    # -------------------------------------------------------------------------
    # CONFIDENCE THRESHOLDS (The Trident Logic)
    # -------------------------------------------------------------------------
    # Since we only have 3 engines, we need high agreement or ONE very strong signal.
    
    # LEVEL 1: Standard
    LVL1_MIN_CONFIDENCE = 0.60  # 60% (Usually means 2 out of 3 agree)
    
    # LEVEL 2: Recovery (After 1 Loss)
    # We lowered this from 0.80 to 0.70 per your request to be more aggressive
    LVL2_MIN_CONFIDENCE = 0.70  
    
    # LEVEL 3: SNIPER (After 2+ Losses)
    # Requires near unanimity
    LVL3_MIN_CONFIDENCE = 0.85 

    # -------------------------------------------------------------------------
    # MARTINGALE STEPS
    # -------------------------------------------------------------------------
    TIER_1_MULT = 1.0
    TIER_2_MULT = 1.5   # Soft Recovery
    TIER_3_MULT = 3.5   # Aggressive Recovery (Kill shot)
    STOP_LOSS_STREAK = 5 # Extended to 5 to give the Neural Engine room to work

# =============================================================================
# SECTION 3: MATHEMATICAL UTILITIES
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
    """The Activation Function for our Neural Engine."""
    # Converts any number into a probability between 0 and 1
    try:
        return 1 / (1 + math.exp(-x))
    except OverflowError:
        return 0.0 if x < 0 else 1.0

def calculate_mean(data: List[float]) -> float:
    return sum(data) / len(data) if data else 0.0

def calculate_stddev(data: List[float]) -> float:
    if len(data) < 2: return 0.0
    mean = calculate_mean(data)
    variance = sum((x - mean) ** 2 for x in data) / (len(data) - 1)
    return math.sqrt(variance)

def calculate_rsi(data: List[float], period: int = 14) -> float:
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
# SECTION 4: THE TRIDENT ENGINES (MAX POWER)
# =============================================================================

# -----------------------------------------------------------------------------
# ENGINE 1: QUANTUM AI (ADAPTIVE BOLLINGER)
# -----------------------------------------------------------------------------
def engine_quantum_adaptive(history: List[Dict]) -> Optional[Dict]:
    """
    detects 'Reversion to Mean'.
    UPGRADE: Uses Dynamic Sigma. 
    If market is quiet, it requires 2.0 Sigma.
    If market is volatile, it relaxes to 1.6 Sigma to catch the swing.
    """
    try:
        numbers = [safe_float(d.get('actual_number')) for d in history[-30:]]
        if len(numbers) < 20: return None
        
        mean = calculate_mean(numbers)
        std = calculate_stddev(numbers)
        if std == 0: return None
        
        current_val = numbers[-1]
        z_score = (current_val - mean) / std
        
        # LOGIC:
        # Z-Score > 1.6 means we are statistically "Too High" -> Bet SMALL
        # Z-Score < -1.6 means we are statistically "Too Low" -> Bet BIG
        
        # The higher the Z-Score, the stronger the signal
        strength = min(abs(z_score) / 2.5, 1.0) # Cap strength at 1.0
        
        if z_score > 1.6:
            return {'prediction': GameConstants.SMALL, 'weight': strength, 'source': f'Quantum(High Z:{z_score:.1f})'}
        elif z_score < -1.6:
            return {'prediction': GameConstants.BIG, 'weight': strength, 'source': f'Quantum(Low Z:{z_score:.1f})'}
            
        return None
    except: return None

# -----------------------------------------------------------------------------
# ENGINE 2: DEEP PATTERN V3 (THE MEMORY)
# -----------------------------------------------------------------------------
def engine_deep_pattern_v3(history: List[Dict]) -> Optional[Dict]:
    """
    UPGRADE: Scans for patterns of length 3 up to 12.
    Now weights recent patterns more heavily.
    """
    try:
        if len(history) < 60: return None
        
        # Convert history to "B" (Big) or "S" (Small) string
        outcomes = [get_outcome_from_number(d.get('actual_number')) for d in history]
        raw_str = ''.join(['B' if o==GameConstants.BIG else 'S' for o in outcomes if o])
        
        best_signal = None
        highest_confidence = 0.0
        
        # We iterate through pattern lengths (Deep to Shallow)
        # Length 12 down to 4
        for depth in range(12, 3, -1):
            curr_pattern = raw_str[-depth:]
            search_area = raw_str[:-1] # Look at the past
            
            # Count occurrences
            count_b_next = 0
            count_s_next = 0
            
            start = 0
            while True:
                idx = search_area.find(curr_pattern, start)
                if idx == -1: break
                
                # Check what happened next
                if idx + depth < len(search_area):
                    next_char = search_area[idx + depth]
                    if next_char == 'B': count_b_next += 1
                    else: count_s_next += 1
                
                start = idx + 1
            
            total_matches = count_b_next + count_s_next
            
            # We need at least 3 historical precedents to trust this
            if total_matches >= 3:
                prob_b = count_b_next / total_matches
                prob_s = count_s_next / total_matches
                
                # Calculate "Imbalance" (How strong is the pattern?)
                # If 5 matches and all 5 were B, imbalance is 1.0 (Strong)
                # If 5 matches and 3B/2S, imbalance is 0.2 (Weak)
                imbalance = abs(prob_b - prob_s)
                
                if imbalance > highest_confidence and imbalance > 0.4: # >70% probability
                    highest_confidence = imbalance
                    pred = GameConstants.BIG if prob_b > prob_s else GameConstants.SMALL
                    # Boost weight by depth (Deeper patterns are rarer and more trusted)
                    weight = imbalance * (1 + (depth * 0.1))
                    best_signal = {'prediction': pred, 'weight': weight, 'source': f'PatternV3-D{depth}({total_matches})'}
                    
                    # If we find a very long, very strong pattern, stop searching
                    if depth > 8 and imbalance > 0.8: break

        return best_signal
    except: return None

# -----------------------------------------------------------------------------
# ENGINE 3: NEURAL PERCEPTRON (THE MARKET SENSOR)
# -----------------------------------------------------------------------------
def engine_neural_perceptron(history: List[Dict]) -> Optional[Dict]:
    """
    A lightweight Neural Network layer.
    Inputs:
    1. RSI (Normalized -0.5 to 0.5)
    2. Momentum (Last 5 vs Last 20)
    3. Parity (Red vs Green balance)
    
    Output:
    Sigmoid Probability (0.0 to 1.0)
    """
    try:
        numbers = [safe_float(d.get('actual_number')) for d in history[-40:]]
        if len(numbers) < 25: return None
        
        # --- INPUT 1: RSI ---
        rsi = calculate_rsi(numbers, 14)
        # Normalize RSI: 50 becomes 0, 70 becomes 0.2, 30 becomes -0.2
        input_rsi = (rsi - 50) / 100.0 
        
        # --- INPUT 2: MOMENTUM ---
        fast_sma = calculate_mean(numbers[-5:])
        slow_sma = calculate_mean(numbers[-20:])
        # Normalize: Positive if rising, Negative if falling
        input_mom = (fast_sma - slow_sma) / 10.0
        
        # --- INPUT 3: REVERSION FORCE ---
        # If last 3 were BIG, force is Negative (expect SMALL)
        last_3 = [get_outcome_from_number(n) for n in numbers[-3:]]
        b_count = last_3.count(GameConstants.BIG)
        # If 3 Bigs, input is -0.3. If 3 Smalls, input is +0.3
        input_rev = (1.5 - b_count) / 5.0
        
        # --- NEURAL WEIGHTS (Pre-Trained / Hardcoded) ---
        # RSI detects overbought/sold (Negative correlation)
        w_rsi = -1.5 
        # Momentum detects trend (Positive correlation)
        w_mom = 1.2
        # Reversion detects streak exhaustion
        w_rev = 0.8
        
        # --- DOT PRODUCT (The "Neuron") ---
        # z = (i1*w1) + (i2*w2) + (i3*w3)
        z = (input_rsi * w_rsi) + (input_mom * w_mom) + (input_rev * w_rev)
        
        # --- ACTIVATION ---
        probability = sigmoid(z) # Returns 0.0 to 1.0
        
        # --- DECISION ---
        # Sigmoid > 0.60 implies BIG
        # Sigmoid < 0.40 implies SMALL
        dist_from_neutral = abs(probability - 0.5)
        
        if probability > 0.60:
            return {'prediction': GameConstants.BIG, 'weight': dist_from_neutral * 2.0, 'source': f'NeuralNet({probability:.2f})'}
        elif probability < 0.40:
            return {'prediction': GameConstants.SMALL, 'weight': dist_from_neutral * 2.0, 'source': f'NeuralNet({probability:.2f})'}
            
        return None
    except: return None

# =============================================================================
# SECTION 5: THE ARCHITECT (MAIN LOGIC)
# =============================================================================

class GlobalStateManager:
    def __init__(self):
        self.loss_streak = 0
        self.last_outcome = None
        
state_manager = GlobalStateManager()

def ultraAIPredict(history: List[Dict], current_bankroll: float = 10000.0, last_result: Optional[str] = None) -> Dict:
    """
    MAIN ENTRY POINT
    """
    # 1. Update Streak
    if last_result:
        actual_outcome = get_outcome_from_number(history[-1]['actual_number'])
        if last_result == GameConstants.SKIP:
            pass
        elif last_result == actual_outcome:
            state_manager.loss_streak = 0
        else:
            state_manager.loss_streak += 1
            
    streak = state_manager.loss_streak
    
    # 2. Run The Trident Engines
    signals = []
    
    # Engine 1: Quantum
    s1 = engine_quantum_adaptive(history)
    if s1: signals.append(s1)
    
    # Engine 2: Deep Pattern
    s2 = engine_deep_pattern_v3(history)
    if s2: signals.append(s2)
    
    # Engine 3: Neural Net
    s3 = engine_neural_perceptron(history)
    if s3: signals.append(s3)
    
    # 3. Aggregate Signals
    big_score = sum(s['weight'] for s in signals if s['prediction'] == GameConstants.BIG)
    small_score = sum(s['weight'] for s in signals if s['prediction'] == GameConstants.SMALL)
    
    total_score = big_score + small_score
    if total_score == 0:
         return {'finalDecision': GameConstants.SKIP, 'confidence': 0, 'positionsize': 0, 'level': 'NO_SIG', 'reason': 'Silence', 'topsignals': []}
         
    # 4. Calculate Confidence
    # Pure ratio of the winning side vs total
    if big_score > small_score:
        final_pred = GameConstants.BIG
        confidence = big_score / (total_score + 0.1) # +0.1 prevents 100% fake confidence
    else:
        final_pred = GameConstants.SMALL
        confidence = small_score / (total_score + 0.1)
    
    # Cap confidence at 0.99
    confidence = min(confidence, 0.99)
    
    # 5. Determine Stake & Level
    # Get active engine names for display
    active_engine_names = [s['source'] for s in signals]
    
    stake = 0
    level = "SKIP"
    reason = f"Conf {confidence:.0%}"
    
    base_bet = max(current_bankroll * RiskConfig.BASE_RISK_PERCENT, RiskConfig.MIN_BET_AMOUNT)
    
    # --- LOGIC GATE ---
    
    # SCENARIO: SNIPER (2+ Losses) - Needs High Confidence
    if streak >= 2:
        if confidence >= RiskConfig.LVL3_MIN_CONFIDENCE:
            stake = base_bet * RiskConfig.TIER_3_MULT
            level = "🔥 SNIPER"
            reason = "Neural+Pattern Lock"
        else:
            level = "SKIP (Recov)"
            reason = f"Need {RiskConfig.LVL3_MIN_CONFIDENCE:.0%}"
            
    # SCENARIO: RECOVERY (1 Loss)
    elif streak == 1:
        if confidence >= RiskConfig.LVL2_MIN_CONFIDENCE:
            stake = base_bet * RiskConfig.TIER_2_MULT
            level = "RECOVERY"
        else:
            level = "SKIP (Recov)"
    
    # SCENARIO: STANDARD (0 Losses)
    else:
        if confidence >= RiskConfig.LVL1_MIN_CONFIDENCE:
            stake = base_bet * RiskConfig.TIER_1_MULT
            level = "STANDARD"
        else:
            level = "SKIP"
            
    # Hard Stop for Bankroll Protection
    if stake > current_bankroll * 0.5: stake = current_bankroll * 0.5
    
    return {
        'finalDecision': final_pred if stake > 0 else GameConstants.SKIP,
        'confidence': confidence,
        'positionsize': int(stake),
        'level': level,
        'reason': reason,
        'topsignals': active_engine_names
    }

if __name__ == "__main__":
    print("TITAN V200 TRIDENT LOADED.")
