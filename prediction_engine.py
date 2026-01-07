#!/usr/bin/env python3
"""
=============================================================================
  _______ _____ _______ _    _  _   _ 
 |__   __|_   _|__   __| |  | || \ | |
    | |    | |    | |  | |  | ||  \| |
    | |    | |    | |  | |  | || . ` |
    | |   _| |_   | |  | |__| || |\  |
    |_|  |_____|  |_|   \____/ |_| \_|
                                      
  TITAN V300 - THE OMNI-CORE (FULL PATTERN EDITION)
  (Quantum + Deep Memory + Neural + Static Signatures)
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
    BASE_RISK_PERCENT = 0.03    # 3% Base Bet
    MIN_BET_AMOUNT = 50
    MAX_BET_AMOUNT = 50000
    
    # -------------------------------------------------------------------------
    # CONFIDENCE THRESHOLDS (The 4-Engine Logic)
    # -------------------------------------------------------------------------
    
    # LEVEL 1: Standard (Any 2 engines agree, or 1 strong pattern)
    LVL1_MIN_CONFIDENCE = 0.60  
    
    # LEVEL 2: Recovery (Aggressive recovery)
    LVL2_MIN_CONFIDENCE = 0.70  
    
    # LEVEL 3: SNIPER (Kill Shot - Requires Pattern Lock)
    LVL3_MIN_CONFIDENCE = 0.85 

    # -------------------------------------------------------------------------
    # MARTINGALE STEPS
    # -------------------------------------------------------------------------
    TIER_1_MULT = 1.0
    TIER_2_MULT = 1.5   # Soft Recovery
    TIER_3_MULT = 3.5   # Aggressive Recovery
    STOP_LOSS_STREAK = 5 

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
# SECTION 4: THE TITAN ENGINES
# =============================================================================

# -----------------------------------------------------------------------------
# ENGINE 1: QUANTUM AI (ADAPTIVE BOLLINGER)
# -----------------------------------------------------------------------------
def engine_quantum_adaptive(history: List[Dict]) -> Optional[Dict]:
    """
    Detects 'Reversion to Mean' using Z-Scores.
    """
    try:
        numbers = [safe_float(d.get('actual_number')) for d in history[-30:]]
        if len(numbers) < 20: return None
        
        mean = calculate_mean(numbers)
        std = calculate_stddev(numbers)
        if std == 0: return None
        
        current_val = numbers[-1]
        z_score = (current_val - mean) / std
        
        strength = min(abs(z_score) / 2.5, 1.0)
        
        if z_score > 1.6:
            return {'prediction': GameConstants.SMALL, 'weight': strength, 'source': f'Quantum(High Z:{z_score:.1f})'}
        elif z_score < -1.6:
            return {'prediction': GameConstants.BIG, 'weight': strength, 'source': f'Quantum(Low Z:{z_score:.1f})'}
            
        return None
    except: return None

# -----------------------------------------------------------------------------
# ENGINE 2: DEEP PATTERN V3 (DYNAMIC MEMORY)
# -----------------------------------------------------------------------------
def engine_deep_pattern_v3(history: List[Dict]) -> Optional[Dict]:
    """
    Scans for mathematical repeats of length 4 to 12.
    """
    try:
        if len(history) < 60: return None
        outcomes = [get_outcome_from_number(d.get('actual_number')) for d in history]
        raw_str = ''.join(['B' if o==GameConstants.BIG else 'S' for o in outcomes if o])
        
        best_signal = None
        highest_confidence = 0.0
        
        for depth in range(12, 3, -1):
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
            if total_matches >= 3:
                prob_b = count_b_next / total_matches
                prob_s = count_s_next / total_matches
                imbalance = abs(prob_b - prob_s)
                
                if imbalance > highest_confidence and imbalance > 0.4:
                    highest_confidence = imbalance
                    pred = GameConstants.BIG if prob_b > prob_s else GameConstants.SMALL
                    weight = imbalance * (1 + (depth * 0.1))
                    best_signal = {'prediction': pred, 'weight': weight, 'source': f'PatternV3-D{depth}'}
                    if depth > 8 and imbalance > 0.8: break

        return best_signal
    except: return None

# -----------------------------------------------------------------------------
# ENGINE 3: NEURAL PERCEPTRON (MARKET SENSOR)
# -----------------------------------------------------------------------------
def engine_neural_perceptron(history: List[Dict]) -> Optional[Dict]:
    """
    Input: RSI, Momentum, Reversion Force.
    Output: Sigmoid Probability.
    """
    try:
        numbers = [safe_float(d.get('actual_number')) for d in history[-40:]]
        if len(numbers) < 25: return None
        
        # Inputs
        rsi = calculate_rsi(numbers, 14)
        input_rsi = (rsi - 50) / 100.0 
        
        fast_sma = calculate_mean(numbers[-5:])
        slow_sma = calculate_mean(numbers[-20:])
        input_mom = (fast_sma - slow_sma) / 10.0
        
        last_3 = [get_outcome_from_number(n) for n in numbers[-3:]]
        b_count = last_3.count(GameConstants.BIG)
        input_rev = (1.5 - b_count) / 5.0
        
        # Weights
        w_rsi, w_mom, w_rev = -1.5, 1.2, 0.8
        
        # Calculation
        z = (input_rsi * w_rsi) + (input_mom * w_mom) + (input_rev * w_rev)
        probability = sigmoid(z)
        dist_from_neutral = abs(probability - 0.5)
        
        if probability > 0.60:
            return {'prediction': GameConstants.BIG, 'weight': dist_from_neutral * 2.0, 'source': f'Neural({probability:.2f})'}
        elif probability < 0.40:
            return {'prediction': GameConstants.SMALL, 'weight': dist_from_neutral * 2.0, 'source': f'Neural({probability:.2f})'}
            
        return None
    except: return None

# -----------------------------------------------------------------------------
# ENGINE 4: TITAN STATIC PATTERNS (THE 35+ SIGNATURES)
# -----------------------------------------------------------------------------
def engine_static_patterns(history: List[Dict]) -> Optional[Dict]:
    """
    Scans for specific named psychological patterns (Dragons, ZigZags, Ratios).
    Logic: Determines if we should 'FOLLOW' the trend or 'FLIP' based on pattern type.
    """
    try:
        if len(history) < 20: return None

        # 1. Convert history to string
        outcomes = [get_outcome_from_number(d.get('actual_number')) for d in history]
        raw_str = ''.join(['B' if o==GameConstants.BIG else 'S' for o in outcomes if o])
        
        # 2. Define Pattern Library (35+ Patterns)
        # Format: 'Name': (['List of Shapes'], 'Action')
        # Action 'flip': Bet Opposite of last char
        # Action 'follow': Bet Same as last char
        
        library = {
            # --- BASIC TRENDS (12) ---
            '1v1_PingPong':   (['BSBSBS', 'SBSBSB'], 'flip'),
            '2v2_Double':     (['SSBBSSBB', 'BBSSBBSS'], 'flip'), # End of sequence usually flips
            '3v3_Triple':     (['BBBSSSBBB', 'SSSBBBSSS'], 'flip'),
            '4v4_Block':      (['SSSSBBBBSSSSBBBB', 'BBBBSSSSBBBBSSSS'], 'flip'),
            '3v1_Break':      (['BBBSBBB', 'SSSBSSS'], 'flip'), # Usually returns to 3
            '2v1_Switch':     (['SSBSSB', 'BBSBBS'], 'flip'),
            '3v2_Offset':     (['BBBSSBBB', 'SSSBBSSS'], 'flip'),
            '4v1_LongBreak':  (['SSSSBSSSS', 'BBBBSBBBB'], 'flip'),
            '4v2_LongOff':    (['BBBBSSBBBB', 'SSSSBBSSSS'], 'flip'),
            'Dragon_B':       (['BBBBBB', 'BBBBBBB', 'BBBBBBBB'], 'follow'), # Always follow Dragon
            'Dragon_S':       (['SSSSSS', 'SSSSSSS', 'SSSSSSSS'], 'follow'), # Always follow Dragon
            
            # --- RATIO METHODS (11) ---
            '1A1B_Tight':     (['BSBS', 'SBSB'], 'flip'),
            '2A2B_Tight':     (['BBSSBBSS', 'SSBBSSBB'], 'flip'),
            '3A3B_Tight':     (['BBBSSSBBB'], 'flip'),
            '4A4B_Tight':     (['BBBBSSSSBBBB'], 'flip'),
            '4A1B_Ratio':     (['BBBB SBB', 'SSSS BSS'], 'follow'), # Return to trend
            '4A2B_Ratio':     (['BBBBSSBB', 'SSSSBBSS'], 'flip'),
            '1A2B_Ratio':     (['BSSBSS', 'SBB SBB'], 'flip'),
            '1A3B_Ratio':     (['BSSS', 'SBBB'], 'flip'),
            
            # --- ADVANCED SEQUENCES (12+) ---
            'Stairs_Up':      (['BSBBSSBBBSSS'], 'follow'), # 1,2,3... expect 4
            'Decay_Rev':      (['BBBBBBBBBBS', 'SSSSSSSSSSB'], 'follow'), # Trend broken, follow reversal
            'Mirror_Point':   (['BBBBSSBSSBBBB'], 'flip'), # Mirror complete
            'Cut_Pattern':    (['BSBBBBBBSBBBB'], 'follow'),
            'Stabilizer':     (['BSSBSSBSS'], 'flip'),
            'Jump_Pattern':   (['SSBSSSB'], 'flip'),
            'ZigZag_Road':    (['BSBSBSBSB'], 'flip'),
            'Fib_Streak':     (['BSBBSSSBBBBB'], 'follow'),
            'Wave_Motion':    (['BBSBBBSS'], 'flip'),
            'Chop_Zone':      (['BSBSBSBSBSBS'], 'flip'),
            'Overdue_B':      (['SSSSSSSS'], 'flip'), # Gambler's Fallacy: Expect B
            'Overdue_S':      (['BBBBBBBBB'], 'flip') # Gambler's Fallacy: Expect S
        }

        best_match = None
        max_len = 0
        
        # Scan Library
        for name, (patterns, action) in library.items():
            for pat in patterns:
                # Remove spaces from user list if any
                clean_pat = pat.replace(" ", "")
                
                if raw_str.endswith(clean_pat):
                    if len(clean_pat) >= max_len:
                        max_len = len(clean_pat)
                        
                        last_char = clean_pat[-1]
                        
                        # Determine Prediction
                        if action == 'flip':
                            pred = GameConstants.BIG if last_char == 'S' else GameConstants.SMALL
                        else: # follow
                            pred = GameConstants.BIG if last_char == 'B' else GameConstants.SMALL
                            
                        # Weight Calculation
                        # Named patterns are high confidence (0.85+)
                        # Complex patterns (len > 8) get huge boost
                        weight = 0.85
                        if len(clean_pat) >= 8: weight = 0.95
                        
                        best_match = {
                            'prediction': pred,
                            'weight': weight,
                            'source': f'StaticPat({name})'
                        }
        
        return best_match

    except Exception as e:
        return None

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
    MAIN ENTRY POINT: Aggregates all 4 engines.
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
    
    # 2. Run The Trident + Static Engines
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

    # Engine 4: Static Patterns (NEW)
    s4 = engine_static_patterns(history)
    if s4: signals.append(s4)
    
    # 3. Aggregate Signals
    big_score = sum(s['weight'] for s in signals if s['prediction'] == GameConstants.BIG)
    small_score = sum(s['weight'] for s in signals if s['prediction'] == GameConstants.SMALL)
    
    total_score = big_score + small_score
    if total_score == 0:
         return {'finalDecision': GameConstants.SKIP, 'confidence': 0, 'positionsize': 0, 'level': 'NO_SIG', 'reason': 'Silence', 'topsignals': []}
         
    # 4. Calculate Confidence
    if big_score > small_score:
        final_pred = GameConstants.BIG
        confidence = big_score / (total_score + 0.1) 
    else:
        final_pred = GameConstants.SMALL
        confidence = small_score / (total_score + 0.1)
    
    # Cap confidence
    confidence = min(confidence, 0.99)
    
    # 5. Determine Stake & Level
    active_engine_names = [s['source'] for s in signals]
    
    stake = 0
    level = "SKIP"
    reason = f"Conf {confidence:.0%}"
    
    base_bet = max(current_bankroll * RiskConfig.BASE_RISK_PERCENT, RiskConfig.MIN_BET_AMOUNT)
    
    # --- LOGIC GATE ---
    
    # SCENARIO: SNIPER (2+ Losses)
    if streak >= 2:
        if confidence >= RiskConfig.LVL3_MIN_CONFIDENCE:
            stake = base_bet * RiskConfig.TIER_3_MULT
            level = "🔥 SNIPER"
            reason = "Pattern Lock"
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
            
    # Hard Stop
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
    print("TITAN V300 OMNI-CORE LOADED.")
