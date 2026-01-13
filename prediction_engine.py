# ==============================================================================
# FILE: prediction_engine.py
# PROJECT: TITAN V15.0 - AGGRESSIVE HYBRID (HIGH FREQUENCY TUNED)
# ==============================================================================
# This module provides the core logic for the Titan prediction system.
# UPDATES:
# - Lowered thresholds for higher participation (30-40% frequency).
# - Enabled "Solo Signal" betting (no need for 2 engines to agree if 1 is strong).
# - Added short-term momentum patterns.
# ==============================================================================

import math
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Any

# ==============================================================================
# SECTION 1: CONFIGURATION (AGGRESSIVE SETTINGS)
# ==============================================================================

class SniperConfig:
    """
    Hybrid Settings: Aggressive on trends, Safe on chaos.
    """
    # 1. BAYESIAN ENGINE
    # Lowered from 0.70 to 0.58 to catch more probabilistic edges.
    BAYES_THRESHOLD = 0.58  
    
    # 2. DEEP MEMORY ENGINE
    # We need at least 2 exact historical matches.
    MIN_MEMORY_MATCHES = 2 
    # Lowered edge requirement (12% edge is enough to bet).
    MEMORY_EDGE_REQ = 0.12 

    # 3. SAFETY GUARDS
    MAX_IDENTICAL_STREAK = 12  # Relaxed: Only stop if streak hits 12 (Dragon friendly).
    CHOPPY_THRESHOLD = 12      # Relaxed: Allow a bit more chop before stopping.
    SKIP_ON_0_5 = False        # Keep Violet avoidance off for maximum frequency.

class GameConstants:
    """Core constants used across the system."""
    BIG = "BIG"
    SMALL = "SMALL" 
    SKIP = "SKIP"

# ==============================================================================
# SECTION 2: UTILITIES & STATE MANAGEMENT
# ==============================================================================

def safe_float(value: Any) -> float:
    """Safely converts input to float."""
    try:
        if value is None: return 4.5
        return float(value)
    except: return 4.5

def get_outcome_from_number(n: Any) -> Optional[str]:
    """
    Converts number 0-9 to BIG/SMALL.
    CRITICAL: This function name MUST match fetcher.py imports exactly.
    """
    val = int(safe_float(n))
    if 0 <= val <= 4: return GameConstants.SMALL
    if 5 <= val <= 9: return GameConstants.BIG
    return None

class GlobalStateManager:
    """Tracks the bot's short-term memory."""
    def __init__(self):
        self.loss_streak = 0
        self.last_round_predictions = {}

state_manager = GlobalStateManager()

def reset_engine_memory():
    """
    Externally called by fetcher.py to wipe memory on session reset.
    """
    state_manager.loss_streak = 0
    state_manager.last_round_predictions = {}
    print("[ENGINE] Memory Wiped via External Reset.")

# ==============================================================================
# SECTION 3: SAFETY GUARDS (THE SHIELD)
# ==============================================================================

def is_market_choppy(history: List[Dict]) -> bool:
    """
    Detects 'Ping-Pong' markets (e.g., B-S-B-S-B).
    If market is too chaotic, we force a SKIP.
    """
    try:
        if len(history) < 15: return False
        
        # Get last 12 outcomes
        outcomes = [get_outcome_from_number(d.get('actual_number')) for d in history[-12:]]
        outcomes = [o for o in outcomes if o] # Filter Nones
        
        if len(outcomes) < 10: return False
        
        # Count how many times the color switched vs stayed same
        switches = 0
        for i in range(1, len(outcomes)):
            if outcomes[i] != outcomes[i-1]:
                switches += 1
        
        # If it switches almost every time (e.g. 10/12 times), it's too risky.
        return switches >= SniperConfig.CHOPPY_THRESHOLD
    except: return False

def is_trend_wall_active(history: List[Dict]) -> bool:
    """
    Detects massive streaks (Dragon).
    If we see 12+ results of the same color, we STOP predicting to avoid 'catching a falling knife'.
    """
    try:
        limit = SniperConfig.MAX_IDENTICAL_STREAK
        if len(history) < limit: return False
        
        # Get last 'limit' outcomes
        outcomes = [get_outcome_from_number(d.get('actual_number')) for d in history[-limit:]]
        first = outcomes[0]
        
        if not first: return False
        
        if all(o == first for o in outcomes):
            return True 
        return False
    except: return False

# ==============================================================================
# SECTION 4: THE 3 ENGINES (TUNED FOR FREQUENCY)
# ==============================================================================

# ------------------------------------------------------------------------------
# ENGINE A: DEEP MEMORY (The Historian)
# ------------------------------------------------------------------------------
def engine_deep_memory(history: List[Dict]) -> Optional[Dict]:
    """
    Looks for the current pattern in the last 500 rounds.
    """
    try:
        if len(history) < 50: return None
        outcomes = [get_outcome_from_number(d.get('actual_number')) for d in history]
        raw_str = ''.join(['B' if o==GameConstants.BIG else 'S' for o in outcomes if o])
        
        best_signal = None
        highest_weight = 0

        # Look for patterns of length 3 to 6 (Including shorter patterns now)
        for depth in range(6, 2, -1):
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
            
            if total >= SniperConfig.MIN_MEMORY_MATCHES:
                prob_b = count_b / total
                prob_s = count_s / total
                
                edge = abs(prob_b - prob_s)
                # RELAXED: If we have a 12% edge, we consider it.
                if edge >= SniperConfig.MEMORY_EDGE_REQ:
                    weight = edge * (depth / 10) 
                    if weight > highest_weight:
                        highest_weight = weight
                        pred = GameConstants.BIG if count_b > count_s else GameConstants.SMALL
                        best_signal = {'prediction': pred, 'source': f'DeepMem({depth})', 'weight': weight}

        return best_signal
    except: return None


# ------------------------------------------------------------------------------
# ENGINE B: BAYESIAN PROBABILITY (The Mathematician)
# ------------------------------------------------------------------------------
def engine_bayesian(history: List[Dict]) -> Optional[Dict]:
    """
    Calculates probability based on context (Trigrams).
    """
    try:
        outcomes = [get_outcome_from_number(d.get('actual_number')) for d in history]
        cleaned = [o[0] for o in outcomes if o] 
        
        context_len = 3
        if len(cleaned) < 10: return None
        
        # Try Trigram (Last 3)
        last_context = tuple(cleaned[-context_len:])
        b_count = 0; s_count = 0
        
        for i in range(len(cleaned) - context_len - 1):
            if tuple(cleaned[i : i+context_len]) == last_context:
                next_val = cleaned[i+context_len]
                if next_val == 'B': b_count += 1
                elif next_val == 'S': s_count += 1
        
        total = b_count + s_count
        
        # Fallback to Bigram (Last 2) if Trigram has no data
        if total == 0:
            context_len = 2
            last_context = tuple(cleaned[-context_len:])
            for i in range(len(cleaned) - context_len - 1):
                if tuple(cleaned[i : i+context_len]) == last_context:
                    next_val = cleaned[i+context_len]
                    if next_val == 'B': b_count += 1
                    elif next_val == 'S': s_count += 1
            total = b_count + s_count

        if total == 0: return None
        
        prob_b = b_count / total
        prob_s = s_count / total
        
        # RELAXED: Return signal if > 58% certainty
        if prob_b >= SniperConfig.BAYES_THRESHOLD:
            return {'prediction': GameConstants.BIG, 'source': 'Bayes', 'weight': prob_b}
        elif prob_s >= SniperConfig.BAYES_THRESHOLD:
            return {'prediction': GameConstants.SMALL, 'source': 'Bayes', 'weight': prob_s}
            
        return None
    except: return None


# ------------------------------------------------------------------------------
# ENGINE C: TITAN TREND (The Shape Scanner)
# ------------------------------------------------------------------------------
def engine_trend_patterns(history: List[Dict]) -> Optional[Dict]:
    """
    Standard Chart Patterns.
    EXPANDED: Now includes ZigZags and simple repeats.
    """
    try:
        outcomes = [get_outcome_from_number(d.get('actual_number')) for d in history[-20:]]
        s = ''.join(['B' if o==GameConstants.BIG else 'S' for o in outcomes if o])
        if not s: return None

        patterns = {
            # 1. DRAGON (Strong Momentum)
            'Dragon_B': ['BBBB'], 
            'Dragon_S': ['SSSS'],
            
            # 2. PING PONG (1v1)
            'PingPong_B': ['SBSB'], # Expect B
            'PingPong_S': ['BSBS'], # Expect S
            
            # 3. DOUBLE STAIRS
            '2v2_B': ['SSBBSS'], 
            '2v2_S': ['BBSSBB'],
            
            # 4. ZIG ZAG (A-B-A) - NEW!
            'ZigZag_B': ['BSB'], # Expect S (Break) or B (Continue)? 
                                 # Usually, BSB -> S is chop. BSB -> B is difficult.
                                 # Let's stick to reversal patterns:
            'Flip_B':   ['SSB'], # AAB pattern, expecting B continuation
            'Flip_S':   ['BBS'], # AAB pattern, expecting S continuation
        }
        
        for p_name, p_list in patterns.items():
            for p_str in p_list:
                if s.endswith(p_str):
                    pred = GameConstants.BIG if '_B' in p_name else GameConstants.SMALL
                    return {'prediction': pred, 'source': f'Trend:{p_name}', 'weight': 0.85}
        return None
    except: return None

# ==============================================================================
# SECTION 5: MASTER PREDICTION LOGIC (THE API)
# ==============================================================================

def _skip(reason):
    return {
        'finalDecision': GameConstants.SKIP,
        'confidence': 0.0,
        'level': "---",
        'reason': reason,
        'topsignals': [],
        'positionsize': 0
    }

def _bet(decision, level, sources, conf=0.8):
    return {
        'finalDecision': decision,
        'confidence': conf,
        'level': level,
        'reason': "+".join(sources),
        'topsignals': sources,
        'positionsize': 0
    }

def ultraAIPredict(history: List[Dict], current_bankroll: float, previous_pred_label: str) -> Dict:
    """
    THE AGGRESSIVE HYBRID BRAIN.
    """
    
    # --- 1. UPDATE INTERNAL STATE ---
    if len(history) > 1 and previous_pred_label not in [GameConstants.SKIP, "WAITING"]:
        last_actual = get_outcome_from_number(history[-1]['actual_number'])
        if last_actual == previous_pred_label:
            state_manager.loss_streak = 0 
        else:
            state_manager.loss_streak += 1

    # --- 2. SAFETY GUARDS ---
    # We still check guards, but they are slightly relaxed in config.
    try:
        last_num = int(safe_float(history[-1]['actual_number']))
        
        if SniperConfig.SKIP_ON_0_5 and last_num in [0, 5]:
             return _skip("Violet Protection")
             
        if is_trend_wall_active(history):
            return _skip("Trend Wall Limit")
            
        if is_market_choppy(history):
            return _skip("Choppy Market")
            
    except Exception as e:
        print(f"[WARN] Guard Error: {e}")

    # --- 3. RUN ENGINES ---
    signals = []
    
    # Engine A: Memory
    res_mem = engine_deep_memory(history)
    if res_mem: signals.append(res_mem)
    
    # Engine B: Bayes
    res_bayes = engine_bayesian(history)
    if res_bayes: signals.append(res_bayes)
    
    # Engine C: Trend
    res_trend = engine_trend_patterns(history)
    if res_trend: signals.append(res_trend)
    
    # --- 4. DECISION LOGIC (AGGRESSIVE) ---
    if not signals:
        return _skip("No Pattern Found")
        
    votes = [s['prediction'] for s in signals]
    counts = Counter(votes)
    top_pred, count = counts.most_common(1)[0]
    
    sources = [s['source'] for s in signals if s['prediction'] == top_pred]
    
    # LOGIC A: CONSENSUS (Best)
    # If 2 or more engines agree -> HIGH CONFIDENCE
    if count >= 2:
        return _bet(top_pred, "TITAN CONFIRM", sources, conf=0.95)

    # LOGIC B: STRONG SOLO (Aggressive)
    # If only 1 engine agrees, but it's a strong signal, we TAKE IT.
    single_signal = [s for s in signals if s['prediction'] == top_pred][0]
    
    # Case 1: Trend Engine is usually reliable for simple patterns
    if "Trend" in single_signal['source']:
        return _bet(top_pred, "Trend Play", sources, conf=0.80)
        
    # Case 2: Deep Memory found a very strong match (>15% edge)
    if "DeepMem" in single_signal['source'] and single_signal['weight'] > 0.15:
         return _bet(top_pred, "Hist Pattern", sources, conf=0.75)
         
    # Case 3: Bayes is very sure (>65%)
    if "Bayes" in single_signal['source'] and single_signal['weight'] > 0.65:
        return _bet(top_pred, "Math Prob", sources, conf=0.75)

    # LOGIC C: WEAK SOLO (Skip)
    # If we only have 1 weak signal (e.g. Bayes 58%), we still skip to avoid total garbage.
    return _skip(f"Weak {sources[0]}")

if __name__ == "__main__":
    print("="*60)
    print(f" TITAN V15.0 AGGRESSIVE LOADED")
    print(f" FREQUENCY: HIGH (30-50%)")
    print("="*60)
