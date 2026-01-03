
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
    BASE_RISK_PERCENT = 0.03    # 3% Base Risk
    MIN_BET_AMOUNT = 50
    MAX_BET_AMOUNT = 50000
    
    # -------------------------------------------------------------------------
    # CONFIDENCE THRESHOLDS (The Trident Logic)
    # -------------------------------------------------------------------------
    
    # LEVEL 1: Standard
    LVL1_MIN_CONFIDENCE = 0.60  # 60%
    
    # LEVEL 2: Recovery (After 1 Loss)
    LVL2_MIN_CONFIDENCE = 0.70  # 70%
    
    # LEVEL 3: SNIPER (After 2+ Losses)
    LVL3_MIN_CONFIDENCE = 0.85  # 85%

    # -------------------------------------------------------------------------
    # MARTINGALE STEPS
    # -------------------------------------------------------------------------
    TIER_1_MULT = 1.0
    TIER_2_MULT = 1.5   # Soft Recovery
    TIER_3_MULT = 3.5   # Aggressive Recovery (Sniper Shot)
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
# SECTION 4: THE TRIDENT ENGINES (PATCHED)
# =============================================================================

# -----------------------------------------------------------------------------
# ENGINE 1: QUANTUM AI (ADAPTIVE BOLLINGER + DRAGON TRAP)
# -----------------------------------------------------------------------------
def engine_quantum_adaptive(history: List[Dict]) -> Optional[Dict]:
    """
    Detects 'Reversion to Mean'.
    FIX: Now includes 'Dragon Trap'. If Z-Score > 2.5, it stays SILENT 
    to avoid betting against a massive trend.
    """
    try:
        numbers = [safe_float(d.get('actual_number')) for d in history[-30:]]
        if len(numbers) < 20: return None
        
        mean = calculate_mean(numbers)
        std = calculate_stddev(numbers)
        if std == 0: return None
        
        current_val = numbers[-1]
        z_score = (current_val - mean) / std
        
        # --- THE DRAGON TRAP FIX ---
        # A Z-Score > 2.5 means the market is running away (Trend).
        # We do NOT bet Reversion here. We wait.
        if abs(z_score) > 2.5:
            return None 
        
        # The higher the Z-Score, the stronger the signal (capped at 1.0)
        strength = min(abs(z_score) / 2.5, 1.0) 
        
        if z_score > 1.6:
            # Statistical High -> Bet Small
            return {'prediction': GameConstants.SMALL, 'weight': strength, 'source': f'Quantum(Z:{z_score:.1f})'}
        elif z_score < -1.6:
            # Statistical Low -> Bet Big
            return {'prediction': GameConstants.BIG, 'weight': strength, 'source': f'Quantum(Z:{z_score:.1f})'}
            
        return None
    except: return None

# -----------------------------------------------------------------------------
# ENGINE 2: DEEP PATTERN V3 (THE MEMORY)
# -----------------------------------------------------------------------------
def engine_deep_pattern_v3(history: List[Dict]) -> Optional[Dict]:
    """
    Scans for patterns of length 3 up to 12.
    """
    try:
        if len(history) < 60: return None
        
        # Convert history to "B" or "S" string
        outcomes = [get_outcome_from_number(d.get('actual_number')) for d in history]
        raw_str = ''.join(['B' if o==GameConstants.BIG else 'S' for o in outcomes if o])
        
        best_signal = None
        highest_confidence = 0.0
        
        # Iterate through pattern lengths (Deep to Shallow)
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
                    # Boost weight by depth (Deeper patterns are rarer and more trusted)
                    weight = imbalance * (1 + (depth * 0.1))
                    best_signal = {'prediction': pred, 'weight': weight, 'source': f'PatternV3-D{depth}({total_matches})'}
                    
                    if depth > 8 and imbalance > 0.8: break

        return best_signal
    except: return None

# -----------------------------------------------------------------------------
# ENGINE 3: NEURAL PERCEPTRON (THE MARKET SENSOR)
# -----------------------------------------------------------------------------
def engine_neural_perceptron(history: List[Dict]) -> Optional[Dict]:
    """
    A lightweight Neural Network layer using RSI, Momentum, and Parity.
    """
    try:
        numbers = [safe_float(d.get('actual_number')) for d in history[-40:]]
        if len(numbers) < 25: return None
        
        # --- INPUT 1: RSI ---
        rsi = calculate_rsi(numbers, 14)
        input_rsi = (rsi - 50) / 100.0 
        
        # --- INPUT 2: MOMENTUM ---
        fast_sma = calculate_mean(numbers[-5:])
        slow_sma = calculate_mean(numbers[-20:])
        input_mom = (fast_sma - slow_sma) / 10.0
        
        # --- INPUT 3: REVERSION FORCE ---
        last_3 = [get_outcome_from_number(n) for n in numbers[-3:]]
        b_count = last_3.count(GameConstants.BIG)
        input_rev = (1.5 - b_count) / 5.0
        
        # --- NEURAL WEIGHTS ---
        w_rsi = -1.5 
        w_mom = 1.2
        w_rev = 0.8
        
        # --- COMPUTATION ---
        z = (input_rsi * w_rsi) + (input_mom * w_mom) + (input_rev * w_rev)
        probability = sigmoid(z) 
        
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
    # 1. Update Streak based on result
    if last_result:
        actual_outcome = get_outcome_from_number(history[-1]['actual_number'])
        if last_result == GameConstants.SKIP:
            pass
        elif last_result == actual_outcome:
            state_manager.loss_streak = 0
        else:
            state_manager.loss_streak += 1
            
    streak = state_manager.loss_streak
    
    # -------------------------------------------------------------------------
    # FIX: THE VIOLET GUARD (0 & 5 DETECTOR)
    # -------------------------------------------------------------------------
    # If the last number was 0 or 5, the algorithm usually resets the seed.
    # We SKIP this turn to avoid betting into chaos.
    try:
        last_num = int(safe_float(history[-1]['actual_number']))
        if last_num in [0, 5]:
            return {
                'finalDecision': GameConstants.SKIP,
                'confidence': 0,
                'positionsize': 0,
                'level': 'VIOLET_GUARD',
                'reason': f'Violet ({last_num}) Reset',
                'topsignals': []
            }
    except Exception:
        pass # If logic fails, proceed as normal
    
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

    # -------------------------------------------------------------------------
    # FIX: FAKE CONFIDENCE CHECK (MINIMUM VOTING QUORUM)
    # -------------------------------------------------------------------------
    # If total score is too low, it means only one weak engine is speaking.
    # We require significant agreement or strength.
    if total_score < 0.35:
         return {
             'finalDecision': GameConstants.SKIP, 
             'confidence': 0, 
             'positionsize': 0, 
             'level': 'NO_SIG', 
             'reason': 'Weak Signal', 
             'topsignals': []
         }
         
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
    
    # --- LOGIC GATE (STAKING STRATEGY) ---
    
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
    print("TITAN V201 PATCHED CORE LOADED.")
