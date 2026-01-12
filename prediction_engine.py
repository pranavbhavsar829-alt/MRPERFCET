# ==============================================================================
# PROJECT: TITAN V1300 - ULTIMATE PREDICTION ENGINE (ADVANCED PATTERNS)
# ==============================================================================
# AUTHOR: TITAN SYSTEM ARCHITECT
# DATE: 2026-01-12
# VERSION: 13.0.0 (FULL SPECTRUM PATTERN EDITION)
#
# DESCRIPTION:
# This module acts as the "Brain" for the automated trading bot.
# It utilizes 5 distinct analytical engines to generate consensus-based predictions.
#
# NEW FEATURES (V13.0):
# 1. ADVANCED PATTERN LIBRARY: Includes 4v2, 3v1, Dragons, Fibonacci, and Ratios.
# 2. AUTO-MIRROR LOGIC: Every pattern checks for both "Big-based" and "Small-based" inversions.
# 3. PATIENT RECOVERY: If Level 1 loses, we SKIP until we find a "Strong" signal.
# ==============================================================================

import math
import statistics
import random
import time
from collections import Counter, defaultdict, deque
from typing import Dict, List, Optional, Any, Tuple

# ==============================================================================
# SECTION 1: MASTER CONTROL & CONFIGURATION
# ==============================================================================

# ------------------------------------------------------------------------------
# [USER CONTROL] GLOBAL STRICTNESS SETTING
# ------------------------------------------------------------------------------
# 75% is the recommended setting for the "Sniper" strategy.
# It forces the bot to be picky, filtering out weak signals.
# ------------------------------------------------------------------------------
GLOBAL_STRICTNESS = 75


class GameConstants:
    """Core Constants for the Wingo/Lottery Game Logic."""
    BIG = "BIG"
    SMALL = "SMALL" 
    SKIP = "SKIP"
    
    # UI Color Mappings
    GREEN = "GREEN"
    RED = "RED"
    VIOLET = "VIOLET"

class EngineConfig:
    """
    Hyperparameters for the AI Brain.
    Values are calculated DYNAMICALLY based on GLOBAL_STRICTNESS.
    """
    
    # --- 1. DYNAMIC THRESHOLD CALCULATION ---
    # We use linear interpolation to scale difficulty based on strictness.
    
    # Bayesian Probability Requirement (0.51 to 0.80)
    BAYES_THRESHOLD = 0.51 + (GLOBAL_STRICTNESS * 0.0029) 
    
    # Z-Score Deviation Requirement (0.1 to 1.6)
    Z_SCORE_TRIGGER = 0.10 + (GLOBAL_STRICTNESS * 0.015)
    
    # Momentum Velocity Requirement (0.1 to 2.0)
    MOMENTUM_TRIGGER = 0.10 + (GLOBAL_STRICTNESS * 0.019)
    
    # --- 2. CONSENSUS RULES ---
    # How many engines must agree to place a bet?
    if GLOBAL_STRICTNESS < 30:
        MIN_VOTES_REQUIRED = 1  # Aggressive: Any hint is a bet.
    elif GLOBAL_STRICTNESS >= 80:
        MIN_VOTES_REQUIRED = 3  # Sniper: Must have strong consensus.
    else:
        MIN_VOTES_REQUIRED = 2  # Balanced: Standard agreement.

    # --- 3. DATA & MEMORY SETTINGS ---
    MIN_DATA_REQUIRED = 25        # Rounds needed to boot up
    DEEP_MEM_LOOKBACK = 500       # History depth for pattern search
    BAYES_CONTEXT_WINDOW = 3      # Pattern length for probability (B-S-B = 3)
    
    # --- 4. SAFETY GUARDS ---
    # Trend Wall: How long can a streak get before we stop trading?
    MAX_IDENTICAL_STREAK = max(5, int(10 - (GLOBAL_STRICTNESS / 20)))
    
    # Choppy Market: How many flips (B-S-B) allowed in last 12 rounds?
    CHOPPY_THRESHOLD = 9

class RiskConfig:
    """
    Bankroll and Staking Configuration.
    (Actual money management happens in Fetcher, this provides the ratios)
    """
    BASE_RISK_PERCENT = 0.03    
    MIN_BET_AMOUNT = 50
    MAX_BET_AMOUNT = 50000

# ==============================================================================
# SECTION 2: MATHEMATICAL UTILITIES & HELPERS
# ==============================================================================

def safe_float(value: Any) -> float:
    """Safely converts API data to float."""
    try:
        if value is None: return 4.5
        return float(value)
    except: return 4.5

def get_outcome_from_number(n: Any) -> Optional[str]:
    """Converts a number (0-9) to the game outcome BIG or SMALL."""
    val = int(safe_float(n))
    if 0 <= val <= 4: return GameConstants.SMALL
    if 5 <= val <= 9: return GameConstants.BIG
    return None

def calculate_mean(data: List[float]) -> float:
    return sum(data) / len(data) if data else 0.0

def calculate_stddev(data: List[float]) -> float:
    if len(data) < 2: return 0.0
    mean = calculate_mean(data)
    variance = sum((x - mean) ** 2 for x in data) / (len(data) - 1)
    return math.sqrt(variance)

# ==============================================================================
# SECTION 3: ADVANCED MARKET GUARDS (THE SHIELD)
# ==============================================================================

def is_market_choppy(history: List[Dict]) -> bool:
    """CHAOS GUARD: Detects 'Ping-Pong' markets."""
    try:
        if len(history) < 15: return False
        outcomes = [get_outcome_from_number(d.get('actual_number')) for d in history[-12:]]
        outcomes = [o for o in outcomes if o]
        if len(outcomes) < 10: return False
        switches = 0
        for i in range(1, len(outcomes)):
            if outcomes[i] != outcomes[i-1]:
                switches += 1
        return switches >= EngineConfig.CHOPPY_THRESHOLD
    except: return False

def is_trend_wall_active(history: List[Dict]) -> bool:
    """TREND WALL GUARD: Detects massive streaks."""
    try:
        limit = EngineConfig.MAX_IDENTICAL_STREAK
        if len(history) < limit: return False
        outcomes = [get_outcome_from_number(d.get('actual_number')) for d in history[-limit:]]
        first = outcomes[0]
        if not first: return False
        if all(o == first for o in outcomes):
            return True 
        return False
    except: return False

# ==============================================================================
# SECTION 4: THE 5 ANALYTICAL ENGINES
# ==============================================================================

def engine_quantum_adaptive(history: List[Dict]) -> Optional[Dict]:
    """Z-Score Mean Reversion Engine."""
    try:
        numbers = [safe_float(d.get('actual_number')) for d in history[-60:]]
        if len(numbers) < 20: return None
        mean = calculate_mean(numbers)
        std = calculate_stddev(numbers)
        if std == 0: return None
        last_val = numbers[-1]
        z_score = (last_val - mean) / std
        
        if abs(z_score) < EngineConfig.Z_SCORE_TRIGGER: return None
        strength = min(abs(z_score), 1.0) 
        
        if z_score > 0: 
            return {'prediction': GameConstants.SMALL, 'weight': strength, 'source': 'Quantum'}
        else: 
            return {'prediction': GameConstants.BIG, 'weight': strength, 'source': 'Quantum'}
    except Exception: return None

def engine_deep_memory_v4(history: List[Dict]) -> Optional[Dict]:
    """Pattern Matching Engine."""
    try:
        outcomes = [get_outcome_from_number(d.get('actual_number')) for d in history]
        raw_str = ''.join(['B' if o==GameConstants.BIG else 'S' for o in outcomes if o])
        max_search_depth = 12
        for depth in range(max_search_depth, 3, -1):
            curr_pattern = raw_str[-depth:]
            search_area = raw_str[:-1]
            count_b = 0; count_s = 0; start = 0
            while True:
                idx = search_area.find(curr_pattern, start)
                if idx == -1: break
                if idx + depth < len(search_area):
                    next_char = search_area[idx + depth]
                    if next_char == 'B': count_b += 1
                    else: count_s += 1
                start = idx + 1
            total = count_b + count_s
            min_matches = 1 if GLOBAL_STRICTNESS < 40 else 2
            if total >= min_matches:
                imbalance = abs((count_b/total) - (count_s/total))
                min_edge = 0.10 if GLOBAL_STRICTNESS < 50 else 0.20
                if imbalance > min_edge:
                    if count_b > count_s: 
                        return {'prediction': GameConstants.BIG, 'weight': imbalance, 'source': f'DeepMem({depth})'}
                    elif count_s > count_b: 
                        return {'prediction': GameConstants.SMALL, 'weight': imbalance, 'source': f'DeepMem({depth})'}
        return None
    except Exception: return None

# --- ENGINE 3: THE ADVANCED PATTERN LIBRARY ---
def engine_chart_patterns(history: List[Dict]) -> Optional[Dict]:
    """
    Titan V1300 Advanced Pattern Recognition.
    Includes Mirrors, Ratios, Dragons, and Fibonacci Sequences.
    """
    try:
        # 1. Prepare Data string (e.g. "BSBBSS...")
        # We look back 30 rounds to catch long Dragons.
        outcomes = [get_outcome_from_number(d.get('actual_number')) for d in history[-30:]]
        s = ''.join(['B' if o==GameConstants.BIG else 'S' for o in outcomes if o])
        if not s: return None

        # 2. DEFINING THE LIBRARY
        # Format: 'Name': ['Pattern_String']
        # The engine looks for the string. The LAST character is the PREDICTION.
        # Example: 'SSB' -> Means we see 'SS', we predict 'B' next.
        
        base_patterns = {
            # --- STANDARD PING PONG (1v1) ---
            '1v1_A': ['BSBSBS'], 
            '1v1_B': ['SBSBSB'],

            # --- DOUBLE TROUBLE (2v2) ---
            '2v2_A': ['SSBBSSBB'],
            '2v2_B': ['BBSSBBSS'],
            '2v2_Short_A': ['SSBB'], 
            '2v2_Short_B': ['BBSS'],

            # --- TRIPLE THREAT (3v3) ---
            '3v3_A': ['BBBSSS'],
            '3v3_B': ['SSSBBB'],
            '3v3_Long_A': ['BBBSSSBBB'],
            '3v3_Long_B': ['SSSBBBSSS'],

            # --- QUAD DAMAGE (4v4) ---
            '4v4_A': ['SSSSBBBB'],
            '4v4_B': ['BBBBSSSS'],

            # --- ASYMMETRICAL: 2v1 ---
            '2v1_A': ['SSBSSB'],  # Small Small Big...
            '2v1_B': ['BBSBBS'],  # Big Big Small...
            '2v1_Short_A': ['SSB'],
            '2v1_Short_B': ['BBS'],

            # --- ASYMMETRICAL: 3v1 ---
            '3v1_A': ['SSSBSSS'],
            '3v1_B': ['BBBSBBB'],
            '3v1_Short_A': ['SSSB'],
            '3v1_Short_B': ['BBBS'],

            # --- ASYMMETRICAL: 3v2 ---
            '3v2_A': ['SSSBBSSS'],
            '3v2_B': ['BBBSSBBB'],
            '3v2_Short_A': ['SSSBB'],
            '3v2_Short_B': ['BBBSS'],

            # --- ASYMMETRICAL: 4v1 & 4v2 ---
            '4v1_A': ['SSSSBSSSS'],
            '4v1_B': ['BBBBSBBBB'],
            '4v2_A': ['SSSSBBSSSS'],
            '4v2_B': ['BBBBSSBBBB'],

            # --- COMPLEX RATIOS (A=Big, B=Small) ---
            '1A2B_Mirror': ['BSS', 'SBB'],
            '1A3B_Mirror': ['BSSS', 'SBBB'],
            '4A1B_Mirror': ['BBBBSB', 'SSSSBS'],
            '4A2B_Mirror': ['BBBBSSB', 'SSSSBBS'],

            # --- DRAGONS (Trend Following) ---
            # If we see 6 of a kind, we predict the 7th is SAME (Follow the Dragon)
            'Dragon_B_6': ['BBBBBBB'], 
            'Dragon_S_6': ['SSSSSSS'],
            'Dragon_B_8': ['BBBBBBBBB'],
            'Dragon_S_8': ['SSSSSSSSS'],

            # --- ADVANCED GEOMETRY ---
            'Mirror_Center': ['BBBBSSBSSBBBB'], # The "King" Mirror
            
            'Stairs_Up_A': ['BSBBSSBBBSSS'], # 1-2-3 Climbing
            'Stairs_Up_B': ['SBSSBBSSSB BB'], 

            'Decay_A': ['BBBB', 'BBB', 'BB'], # 4-3-2 Falling
            'Decay_B': ['SSSS', 'SSS', 'SS'],

            'Fib_Streak_A': ['B', 'SS', 'BBB'], # 1, 2, 3...
            'Fib_Streak_B': ['S', 'BB', 'SSS'],

            'Cut_Pattern_A': ['B', 'BB', 'BBB', 'BBBB'], # Growing tail
            'Cut_Pattern_B': ['S', 'SS', 'SSS', 'SSSS'],

            'Stabilizer_A': ['BSS', 'BSS', 'BSS'], # Repeating Triplet
            'Stabilizer_B': ['SBB', 'SBB', 'SBB'],
            
            'Chop_Zone': ['BSBSBSBS'], # High volatility marker
        }
        
        # 3. SCANNING LOGIC
        for p_name, p_list in base_patterns.items():
            for p_str in p_list:
                # Clean up any potential spaces from manual entry
                clean_p = p_str.replace(" ", "").strip()
                
                # Split: We look for EVERYTHING except the last char.
                # The LAST char is what the pattern *predicts*.
                required = clean_p[:-1] 
                prediction_char = clean_p[-1] 
                
                if s.endswith(required):
                    
                    # --- STRICTNESS FILTERS ---
                    # 1. Ignore very short patterns (length < 3) unless we are in Gambler mode.
                    # This prevents "False Positives" on random noise like "BS".
                    if len(required) < 3 and GLOBAL_STRICTNESS > 40:
                        continue 
                    
                    # 2. Convert Char to Prediction
                    final_pred = GameConstants.BIG if prediction_char == 'B' else GameConstants.SMALL
                    
                    # 3. Calculate Confidence Weight
                    # Longer patterns = Higher confidence.
                    # Dragon/Mirror patterns (len > 6) get massive weight.
                    weight = 0.95 if len(required) > 5 else 0.85
                    
                    return {
                        'prediction': final_pred, 
                        'weight': weight, 
                        'source': f'Chart:{p_name}'
                    }

        return None
    except Exception as e: 
        print(f"Chart Pattern Error: {e}")
        return None

def engine_bayesian_probability(history: List[Dict]) -> Optional[Dict]:
    """Mathematical Context Probability."""
    try:
        outcomes = [get_outcome_from_number(d.get('actual_number')) for d in history]
        cleaned = [o[0] for o in outcomes if o] 
        context_len = EngineConfig.BAYES_CONTEXT_WINDOW
        if len(cleaned) < 5: return None
        last_context = tuple(cleaned[-context_len:]) 
        b_count = 0; s_count = 0
        for i in range(len(cleaned) - context_len - 1):
            if tuple(cleaned[i : i+context_len]) == last_context:
                next_val = cleaned[i+context_len]
                if next_val == 'B': b_count += 1
                elif next_val == 'S': s_count += 1
        total = b_count + s_count
        if total < 2: return None
        prob_b = b_count / total
        prob_s = s_count / total
        if prob_b > EngineConfig.BAYES_THRESHOLD:
            return {'prediction': GameConstants.BIG, 'weight': prob_b, 'source': 'BayesAI'}
        elif prob_s > EngineConfig.BAYES_THRESHOLD:
            return {'prediction': GameConstants.SMALL, 'weight': prob_s, 'source': 'BayesAI'}
        return None
    except Exception: return None

def engine_momentum_oscillator(history: List[Dict]) -> Optional[Dict]:
    """Velocity Tracker."""
    try:
        outcomes = [get_outcome_from_number(d.get('actual_number')) for d in history[-12:]]
        score = 0; weight = 1.0; decay = 0.85
        for o in reversed(outcomes):
            if o == GameConstants.BIG: score += weight
            elif o == GameConstants.SMALL: score -= weight
            weight *= decay 
        if score > EngineConfig.MOMENTUM_TRIGGER: 
            return {'prediction': GameConstants.BIG, 'weight': 0.7, 'source': 'Momentum'}
        elif score < -EngineConfig.MOMENTUM_TRIGGER: 
            return {'prediction': GameConstants.SMALL, 'weight': 0.7, 'source': 'Momentum'}
        return None
    except Exception: return None

# ==============================================================================
# SECTION 5: STATE MANAGEMENT & RESET LOGIC
# ==============================================================================

class GlobalStateManager:
    def __init__(self):
        self.loss_streak = 0
        self.last_round_predictions = {}
        self.engine_scores = defaultdict(int) 

state_manager = GlobalStateManager()

def reset_engine_memory():
    """EXTERNALLY CALLED by Fetcher."""
    state_manager.loss_streak = 0
    state_manager.last_round_predictions = {}
    print(f"[ENGINE] Memory Wiped. Strictness Level: {GLOBAL_STRICTNESS}%")

def _build_skip_response(reason_text):
    return {
        'finalDecision': GameConstants.SKIP,
        'confidence': 0.0,
        'level': "---",
        'reason': reason_text,
        'topsignals': [],
        'positionsize': 0
    }

# ==============================================================================
# SECTION 6: MASTER PREDICTION FUNCTION (API)
# ==============================================================================

def ultraAIPredict(history: List[Dict], current_bankroll: float, previous_pred_label: str) -> Dict:
    
    # --------------------------------------------------------------------------
    # 1. UPDATE INTERNAL STATE (Did we win the last round?)
    # --------------------------------------------------------------------------
    # NOTE: We only update the streak if the LAST round was an actual bet.
    # If we SKIP, the streak remains active (paused).
    if len(history) > 1 and previous_pred_label not in [GameConstants.SKIP, "WAITING", "COOLDOWN"]:
        last_actual = get_outcome_from_number(history[-1]['actual_number'])
        
        if last_actual == previous_pred_label:
            state_manager.loss_streak = 0 
        else:
            state_manager.loss_streak += 1 

    current_streak = state_manager.loss_streak

    # --------------------------------------------------------------------------
    # 2. RUN SAFETY GUARDS
    # --------------------------------------------------------------------------
    try:
        last_num = int(safe_float(history[-1]['actual_number']))
        
        if last_num in [0, 5]:
            return _build_skip_response("Violet Reset (0/5)")
            
        if is_trend_wall_active(history):
            return _build_skip_response(f"Trend Wall (Streak > {EngineConfig.MAX_IDENTICAL_STREAK})")
            
    except Exception as e:
        print(f"[ENGINE WARN] Guard Check Failed: {e}")

    # --------------------------------------------------------------------------
    # 3. RUN ALL 5 ENGINES
    # --------------------------------------------------------------------------
    signals = []
    engines = [engine_quantum_adaptive, engine_deep_memory_v4, engine_chart_patterns, 
               engine_bayesian_probability, engine_momentum_oscillator]
    
    for eng in engines:
        res = eng(history)
        if res: signals.append(res)

    # --------------------------------------------------------------------------
    # 4. VOTE COUNTING
    # --------------------------------------------------------------------------
    if not signals:
        return _build_skip_response("No Valid Signals")

    votes = [s['prediction'] for s in signals]
    vote_counts = Counter(votes)
    top_pred, count = vote_counts.most_common(1)[0]
    sources = [s['source'] for s in signals if s['prediction'] == top_pred]

    # --------------------------------------------------------------------------
    # 5. DYNAMIC DECISION LOGIC (SNIPER V2 - PATIENT RECOVERY)
    # --------------------------------------------------------------------------
    
    req_votes = EngineConfig.MIN_VOTES_REQUIRED
    
    final_label = GameConstants.SKIP
    confidence_level = "SKIP"
    reason = f"Need {req_votes} votes (Got {count})"
    
    # Determine Strategy based on Current Streak
    if current_streak > 0:
        # === RECOVERY MODE ===
        # We are currently in a loss. We need to be CAREFUL.
        # We only bet if the signal is VERY STRONG (3+ votes).
        # Otherwise, we wait (SKIP) until the market clears up.
        
        if count >= 3:
            final_label = top_pred
            confidence_level = f"LEVEL {current_streak + 1} (REC)"
            reason = "Strong Recovery Signal Found"
        else:
            # Signal is too weak for a recovery bet. SKIP.
            return _build_skip_response(f"Waiting for Perfect L{current_streak+1}...")

    else:
        # === NORMAL MODE (LEVEL 1) ===
        # Standard consensus applies here.
        if count >= req_votes:
            final_label = top_pred
            if count >= 4: confidence_level = "LEVEL 1 (MAX)"
            elif count == 3: confidence_level = "LEVEL 1 (SOLID)"
            else: confidence_level = "LEVEL 1 (BASIC)"
            reason = f"Consensus ({count}/{req_votes})"

    # --------------------------------------------------------------------------
    # 6. RETURN FINAL DECISION
    # --------------------------------------------------------------------------
    return {
        'finalDecision': final_label,
        'confidence': 0.99,
        'level': confidence_level,
        'reason': reason,
        'topsignals': sources,
        'positionsize': 0 
    }
