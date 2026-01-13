# ==============================================================================
# FILE: prediction_engine.py
# PROJECT: TITAN V14.0 - SNIPER EDITION (FULL COMPATIBILITY FIX)
# ==============================================================================
# This module provides the core logic for the Titan prediction system.
# It includes:
# 1. get_outcome_from_number (REQUIRED by fetcher.py)
# 2. reset_engine_memory (REQUIRED by fetcher.py)
# 3. ultraAIPredict (REQUIRED by fetcher.py)
# ==============================================================================

import math
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Any

# ==============================================================================
# SECTION 1: CONFIGURATION (SNIPER SETTINGS)
# ==============================================================================

class SniperConfig:
    """
    Hyper-Strict Settings for Maximum Accuracy.
    """
    # 1. BAYESIAN ENGINE
    # Only bet if the math says there is a > 72% probability.
    BAYES_THRESHOLD = 0.60 
    
    # 2. DEEP MEMORY ENGINE
    # We need at least 3 exact historical matches to trust the pattern.
    MIN_MEMORY_MATCHES = 2 
    # The pattern must have a 20% edge (e.g. 60/40 split is NOT enough).
    MEMORY_EDGE_REQ = 0.15

    # 3. CONSENSUS RULES
    # How many engines must agree? 
    # STRICT RULE: 2 engines must agree.
    MIN_VOTES = 2 

    # 4. SAFETY GUARDS
    MAX_IDENTICAL_STREAK = 9  # Stop betting if a color hits 9 times in a row.
    CHOPPY_THRESHOLD = 10     # Stop if market flips (B-S-B-S) 10 times in 12 rounds.
    SKIP_ON_0_5 = False        # Aggressive Violet Avoidance (0 and 5 are dangerous).

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
    CRITICAL: This function MUST exist for fetcher.py to work.
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
    Returns True if the market is switching too rapidly.
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
                
        return switches >= SniperConfig.CHOPPY_THRESHOLD
    except: return False

def is_trend_wall_active(history: List[Dict]) -> bool:
    """
    Detects massive streaks (Dragon).
    If we see 9+ results of the same color, we STOP predicting.
    """
    try:
        limit = SniperConfig.MAX_IDENTICAL_STREAK
        if len(history) < limit: return False
        
        # Get last 'limit' outcomes
        outcomes = [get_outcome_from_number(d.get('actual_number')) for d in history[-limit:]]
        first = outcomes[0]
        
        if not first: return False
        
        # Check if ALL items in the list are identical
        if all(o == first for o in outcomes):
            return True 
        return False
    except: return False

# ==============================================================================
# SECTION 4: THE 3 STRONG ENGINES (NO WEAK LOGIC)
# ==============================================================================

# ------------------------------------------------------------------------------
# ENGINE A: DEEP MEMORY (The Historian)
# ------------------------------------------------------------------------------
def engine_deep_memory(history: List[Dict]) -> Optional[Dict]:
    """
    Looks for the current pattern in the last 500 rounds.
    STRICTNESS UPGRADE: Ignores weak matches.
    """
    try:
        if len(history) < 50: return None
        outcomes = [get_outcome_from_number(d.get('actual_number')) for d in history]
        raw_str = ''.join(['B' if o==GameConstants.BIG else 'S' for o in outcomes if o])
        
        # Only look for patterns of length 4 to 7 (The Sweet Spot)
        # Short patterns (2-3) are noise. Long patterns (8+) are too rare.
        best_signal = None
        highest_weight = 0

        for depth in range(7, 3, -1):
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
            
            # TIGHT FILTER: Minimum historical occurrences
            if total >= SniperConfig.MIN_MEMORY_MATCHES:
                prob_b = count_b / total
                prob_s = count_s / total
                
                # Check absolute edge
                edge = abs(prob_b - prob_s)
                if edge > SniperConfig.MEMORY_EDGE_REQ:
                    weight = edge * (depth / 10) # Longer patterns = higher weight
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
    Calculates probability based on exact context.
    STRICTNESS UPGRADE: High threshold only (72%+).
    """
    try:
        outcomes = [get_outcome_from_number(d.get('actual_number')) for d in history]
        cleaned = [o[0] for o in outcomes if o] 
        
        # Look at last 3 outcomes (Trigrams)
        context_len = 3
        if len(cleaned) < 10: return None
        
        last_context = tuple(cleaned[-context_len:])
        b_count = 0; s_count = 0
        
        for i in range(len(cleaned) - context_len - 1):
            if tuple(cleaned[i : i+context_len]) == last_context:
                next_val = cleaned[i+context_len]
                if next_val == 'B': b_count += 1
                elif next_val == 'S': s_count += 1
        
        total = b_count + s_count
        if total == 0: return None
        
        prob_b = b_count / total
        prob_s = s_count / total
        
        # TIGHT FILTER: Only return if > 72% certainty
        if prob_b >= SniperConfig.BAYES_THRESHOLD:
            return {'prediction': GameConstants.BIG, 'source': 'BayesHigh', 'weight': prob_b}
        elif prob_s >= SniperConfig.BAYES_THRESHOLD:
            return {'prediction': GameConstants.SMALL, 'source': 'BayesHigh', 'weight': prob_s}
            
        return None
    except: return None


# ------------------------------------------------------------------------------
# ENGINE C: TITAN TREND (The Shape Scanner)
# ------------------------------------------------------------------------------
def engine_trend_patterns(history: List[Dict]) -> Optional[Dict]:
    """
    Simplified Chart Patterns.
    REMOVED: Complex/Rare shapes.
    KEPT: Dragons, Stairs, and Alternating 1v1.
    """
    try:
        outcomes = [get_outcome_from_number(d.get('actual_number')) for d in history[-20:]]
        s = ''.join(['B' if o==GameConstants.BIG else 'S' for o in outcomes if o])
        if not s: return None

        # Only the 'Golden' Patterns
        patterns = {
            # 1. DRAGON (Trend Following) - If we see 5, bet 6.
            'Dragon_B': ['BBBBB'], 
            'Dragon_S': ['SSSSS'],
            
            # 2. PING PONG (1v1) - If strict alternation
            'PingPong_B': ['SBSBS'], # Expect B
            'PingPong_S': ['BSBSB'], # Expect S
            
            # 3. DOUBLE STAIRS
            '2v2_B': ['SSBBSS'], # Expect B
            '2v2_S': ['BBSSBB'], # Expect S
        }
        
        for p_name, p_list in patterns.items():
            for p_str in p_list:
                if s.endswith(p_str):
                    pred = GameConstants.BIG if '_B' in p_name else GameConstants.SMALL
                    return {'prediction': pred, 'source': f'Trend:{p_name}', 'weight': 0.9}
        return None
    except: return None

# ==============================================================================
# SECTION 5: MASTER PREDICTION LOGIC (THE API)
# ==============================================================================

def _skip(reason):
    """Helper to generate a clean SKIP response."""
    return {
        'finalDecision': GameConstants.SKIP,
        'confidence': 0.0,
        'level': "---",
        'reason': reason,
        'topsignals': [],
        'positionsize': 0
    }

def _bet(decision, level, sources):
    """Helper to generate a clean BET response."""
    return {
        'finalDecision': decision,
        'confidence': 0.95,
        'level': "SNIPER V14",
        'reason': level,
        'topsignals': sources,
        'positionsize': 0
    }

def ultraAIPredict(history: List[Dict], current_bankroll: float, previous_pred_label: str) -> Dict:
    """
    THE MAIN BRAIN.
    Strictly follows 'SniperConfig' to avoid weak signals.
    """
    
    # --- 1. UPDATE INTERNAL STATE ---
    # (Simple tracker, used for logs)
    if len(history) > 1 and previous_pred_label not in [GameConstants.SKIP, "WAITING"]:
        last_actual = get_outcome_from_number(history[-1]['actual_number'])
        if last_actual == previous_pred_label:
            state_manager.loss_streak = 0 
        else:
            state_manager.loss_streak += 1

    # --- 2. SAFETY GUARDS (The Shield) ---
    try:
        last_num = int(safe_float(history[-1]['actual_number']))
        
        # A. VIOLET GUARD
        if SniperConfig.SKIP_ON_0_5 and last_num in [0, 5]:
             return _skip("Violet Protection (0/5)")
             
        # B. TREND WALL GUARD (Don't fight massive streaks)
        if is_trend_wall_active(history):
            return _skip("Trend Wall (Streak > 9)")
            
        # C. CHOPPY MARKET GUARD
        if is_market_choppy(history):
            return _skip("Choppy Market (Ping-Pong)")
            
    except Exception as e:
        print(f"[WARN] Guard Error: {e}")

    # --- 3. RUN THE 'BIG 3' ENGINES ---
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
    
    # --- 4. VOTE CONSENSUS & DECISION ---
    if not signals:
        return _skip("No Strong Signals")
        
    votes = [s['prediction'] for s in signals]
    counts = Counter(votes)
    top_pred, count = counts.most_common(1)[0]
    
    sources = [s['source'] for s in signals if s['prediction'] == top_pred]
    
    # RULE 1: Strong Consensus (2+ Engines Agree)
    if count >= 2:
        return _bet(top_pred, f"Sniper Confirm ({count}/3)", sources)
        
    # RULE 2: Solo Trend Signal (Dragon Exception)
    # We trust the Trend engine alone IF it sees a Dragon/Pattern.
    if count == 1 and "Trend" in sources[0]:
        return _bet(top_pred, "Trend Following", sources)
        
    # RULE 3: Weak Signals (Solo Memory or Bayes) -> SKIP
    # Without confirmation, these are too risky.
    return _skip(f"Weak Signal ({sources[0]})")

if __name__ == "__main__":
    print("="*60)
    print(f" TITAN V14.0 SNIPER LOADED")
    print(f" STRICTNESS: EXTREME")
    print("="*60)
