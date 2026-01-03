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
    
    # Reduced history requirement to get into the game faster
    MIN_HISTORY_FOR_PREDICTION = 15 
    DEBUG_MODE = True

# =============================================================================
# SECTION 2: RISK & SNIPER CONFIGURATION (AGGRESSIVE TUNING)
# =============================================================================

class RiskConfig:
    # -------------------------------------------------------------------------
    # BANKROLL MANAGEMENT
    # -------------------------------------------------------------------------
    BASE_RISK_PERCENT = 0.03    # 3% Base Risk
    MIN_BET_AMOUNT = 50
    MAX_BET_AMOUNT = 50000
    
    # -------------------------------------------------------------------------
    # CONFIDENCE THRESHOLDS (TUNED FOR ACTION)
    # -------------------------------------------------------------------------
    
    # LEVEL 1: Standard - ENTRY BARRIER LOWERED
    # We now take the bet if we are 55% sure (Edge > Chance)
    LVL1_MIN_CONFIDENCE = 0.55
    
    # LEVEL 2: Recovery (After 1 Loss)
    LVL2_MIN_CONFIDENCE = 0.65 
    
    # LEVEL 3: SNIPER (After 2+ Losses) - MUST BE ACCURATE
    LVL3_MIN_CONFIDENCE = 0.80 

    # -------------------------------------------------------------------------
    # MARTINGALE STEPS (3 LEVEL CAP)
    # -------------------------------------------------------------------------
    TIER_1_MULT = 1.0
    TIER_2_MULT = 2.0   # Standard Recovery
    TIER_3_MULT = 4.5   # The "Make or Break" Shot
    STOP_LOSS_STREAK = 3 # HARD STOP after Level 3 to prevent drain

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
# SECTION 4: THE TRIDENT ENGINES (SENSITIVITY BOOSTED)
# =============================================================================

# -----------------------------------------------------------------------------
# ENGINE 1: QUANTUM AI (ADAPTIVE BOLLINGER + DRAGON TRAP)
# -----------------------------------------------------------------------------
def engine_quantum_adaptive(history: List[Dict]) -> Optional[Dict]:
    """
    Sensitivity Tuned: Now reacts to Z-Score > 1.2 (Standard Deviation).
    """
    try:
        numbers = [safe_float(d.get('actual_number')) for d in history[-30:]]
        if len(numbers) < 15: return None
        
        mean = calculate_mean(numbers)
        std = calculate_stddev(numbers)
        if std == 0: return None
        
        current_val = numbers[-1]
        z_score = (current_val - mean) / std
        
        # DRAGON TRAP: Still protects against massive runs (> 2.8),
        # but allows more aggressive reversion bets.
        if abs(z_score) > 2.8:
            return None 
        
        # Boosted Strength Calculation
        strength = min(abs(z_score) / 2.0, 1.0) 
        
        # LOWERED THRESHOLD: 1.6 -> 1.2
        if z_score > 1.2:
            # Statistical High -> Bet Small
            return {'prediction': GameConstants.SMALL, 'weight': strength, 'source': f'Quantum(Z:{z_score:.1f})'}
        elif z_score < -1.2:
            # Statistical Low -> Bet Big
            return {'prediction': GameConstants.BIG, 'weight': strength, 'source': f'Quantum(Z:{z_score:.1f})'}
            
        return None
    except: return None

# -----------------------------------------------------------------------------
# ENGINE 2: DEEP PATTERN V3 (THE MEMORY)
# -----------------------------------------------------------------------------
def engine_deep_pattern_v3(history: List[Dict]) -> Optional[Dict]:
    try:
        if len(history) < 30: return None
        
        outcomes = [get_outcome_from_number(d.get('actual_number')) for d in history]
        raw_str = ''.join(['B' if o==GameConstants.BIG else 'S' for o in outcomes if o])
        
        best_signal = None
        highest_confidence = 0.0
        
        # Reduced max depth search to speed up and catch shorter, fresher trends
        for depth in range(8, 2, -1):
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
            
            if total_matches >= 2: # Lowered match requirement
                prob_b = count_b_next / total_matches
                prob_s = count_s_next / total_matches
                
                imbalance = abs(prob_b - prob_s)
                
                # Lowered imbalance requirement: 0.4 -> 0.3
                if imbalance > highest_confidence and imbalance > 0.3: 
                    highest_confidence = imbalance
                    pred = GameConstants.BIG if prob_b > prob_s else GameConstants.SMALL
                    # WEIGHT BOOST
                    weight = imbalance * 1.5 
                    best_signal = {'prediction': pred, 'weight': weight, 'source': f'PatternV3-D{depth}'}
                    
                    if depth > 5 and imbalance > 0.7: break

        return best_signal
    except: return None

# -----------------------------------------------------------------------------
# ENGINE 3: NEURAL PERCEPTRON (THE MARKET SENSOR)
# -----------------------------------------------------------------------------
def engine_neural_perceptron(history: List[Dict]) -> Optional[Dict]:
    try:
        numbers = [safe_float(d.get('actual_number')) for d in history[-40:]]
        if len(numbers) < 20: return None
        
        # --- INPUTS ---
        rsi = calculate_rsi(numbers, 14)
        input_rsi = (rsi - 50) / 100.0 
        
        fast_sma = calculate_mean(numbers[-5:])
        slow_sma = calculate_mean(numbers[-20:])
        input_mom = (fast_sma - slow_sma) / 10.0
        
        last_3 = [get_outcome_from_number(n) for n in numbers[-3:]]
        b_count = last_3.count(GameConstants.BIG)
        input_rev = (1.5 - b_count) / 5.0
        
        # --- TUNED NEURAL WEIGHTS (More Aggressive) ---
        w_rsi = -1.8  # Increased impact of RSI
        w_mom = 1.4
        w_rev = 1.0
        
        # --- COMPUTATION ---
        z = (input_rsi * w_rsi) + (input_mom * w_mom) + (input_rev * w_rev)
        probability = sigmoid(z) 
        
        dist_from_neutral = abs(probability - 0.5)
        
        # Reduced Threshold: 0.60 -> 0.55 (More signals)
        if probability > 0.55:
            # Multiplier x3.0 to make Neural clearer
            return {'prediction': GameConstants.BIG, 'weight': dist_from_neutral * 3.0, 'source': f'NeuralNet({probability:.2f})'}
        elif probability < 0.45:
            return {'prediction': GameConstants.SMALL, 'weight': dist_from_neutral * 3.0, 'source': f'NeuralNet({probability:.2f})'}
            
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
    
    # --- DELETED VIOLET GUARD ---
    # We no longer skip on 0 or 5. We play through it.
    
    # 2. Run Engines
    signals = []
    
    s1 = engine_quantum_adaptive(history)
    if s1: signals.append(s1)
    
    s2 = engine_deep_pattern_v3(history)
    if s2: signals.append(s2)
    
    s3 = engine_neural_perceptron(history)
    if s3: signals.append(s3)
    
    # 3. Aggregate Signals
    big_score = sum(s['weight'] for s in signals if s['prediction'] == GameConstants.BIG)
    small_score = sum(s['weight'] for s in signals if s['prediction'] == GameConstants.SMALL)
    
    total_score = big_score + small_score

    # LOWERED FAKE CONFIDENCE CHECK
    # We allow weaker consensus now (0.35 -> 0.15)
    if total_score < 0.15:
         return {
             'finalDecision': GameConstants.SKIP, 
             'confidence': 0, 
             'positionsize': 0, 
             'level': 'NO_SIG', 
             'reason': 'Silence', 
             'topsignals': []
         }
         
    # 4. Calculate Confidence (PURE RATIO)
    # Removed the "+ 0.1" which was suppressing confidence values
    if big_score > small_score:
        final_pred = GameConstants.BIG
        confidence = big_score / total_score 
    else:
        final_pred = GameConstants.SMALL
        confidence = small_score / total_score
    
    confidence = min(confidence, 0.99)
    
    # 5. Determine Stake & Level
    active_engine_names = [s['source'] for s in signals]
    stake = 0
    level = "SKIP"
    reason = f"Conf {confidence:.0%}"
    
    base_bet = max(current_bankroll * RiskConfig.BASE_RISK_PERCENT, RiskConfig.MIN_BET_AMOUNT)
    
    # --- LOGIC GATE (STAKING STRATEGY) ---
    
    # SCENARIO: SNIPER (2+ Losses) - Needs High Confidence
    if streak >= 2:
        if confidence >= RiskConfig.LVL3_MIN_CONFIDENCE:
            stake = base_bet * RiskConfig.TIER_3_MULT
            level = "🔥 SNIPER"
            reason = "Max Aggression"
        else:
            # Fallback: If we are deep in loss but confidence is decent (70%), take a defensive shot
            if confidence >= 0.70:
                 stake = base_bet * RiskConfig.TIER_2_MULT
                 level = "DEFENSIVE"
                 reason = "Soft Recovery"
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
        # Very Low barrier for entry on Level 1 to keep action moving
        if confidence >= RiskConfig.LVL1_MIN_CONFIDENCE:
            stake = base_bet * RiskConfig.TIER_1_MULT
            level = "STANDARD"
        else:
            level = "SKIP"
            
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
    print("TITAN V201 UNCHAINED LOADED.")
