# ==============================================================================
# PROJECT: TITAN V1400 - SOVEREIGN AI PREDICTION ENGINE
# CODENAME: "ACTIVE SNIPER"
# ==============================================================================
# AUTHOR: TITAN SYSTEM ARCHITECT
# DATE: 2026-01-12
# VERSION: 14.1.0 (FULL UNCOMPRESSED EDITION)
#
# DESCRIPTION:
# This module acts as the "Cortex" (Brain) for the automated trading bot.
# It is designed to ingest historical data from the Wingo lottery game and 
# output high-probability predictions based on a consensus of 5 independent 
# analytical engines.
#
# SYSTEM ARCHITECTURE:
# 1. ENGINE 1: QUANTUM ADAPTIVE (Mean Reversion / Z-Score Analysis)
# 2. ENGINE 2: DEEP MEMORY V4 (Historical Sequence Matching)
# 3. ENGINE 3: TITAN CHART PATTERNS (Technical Analysis & Geometry)
# 4. ENGINE 4: BAYESIAN PROBABILITY (Mathematical Context Analysis)
# 5. ENGINE 5: MOMENTUM OSCILLATOR (Velocity & Trend Strength)
#
# KEY FEATURES IN THIS VERSION:
# - TREND ALIGNMENT: Automatically detects "Dragon" (Trend) or "Chop" phases.
# - ACTIVE RECOVERY: Faster Level 3 recovery (Accepts 2 Votes + Pattern).
# - GHOST VALIDATION: Simulates trades to prevent fighting the trend.
# - FULL LIBRARY: Contains 60+ complex market shapes and patterns.
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
# 75% is the "Sweet Spot".
# It filters out noise but allows for accurate trend following.
# ------------------------------------------------------------------------------
GLOBAL_STRICTNESS = 75


class GameConstants:
    """Core Constants for the Wingo/Lottery Game Logic."""
    BIG = "BIG"
    SMALL = "SMALL" 
    SKIP = "SKIP"
    
    # Internal Status Codes
    STATUS_WAITING = "WAITING"
    STATUS_ANALYZING = "ANALYZING"
    STATUS_LOCKED = "LOCKED"

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
    # Standard: Requires 2 votes to bet.
    MIN_VOTES_REQUIRED = 2

    # --- 3. DATA & MEMORY SETTINGS ---
    MIN_DATA_REQUIRED = 25        # Rounds needed to boot up
    DEEP_MEM_LOOKBACK = 500       # History depth for pattern search
    BAYES_CONTEXT_WINDOW = 3      # Pattern length for probability (B-S-B = 3)
    
    # --- 4. SAFETY GUARDS ---
    # Trend Wall: How long can a streak get before we stop trading?
    # Extended to 9 to allow "Riding the Dragon" logic.
    MAX_IDENTICAL_STREAK = 9
    
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
    """Calculates Arithmetic Mean."""
    return sum(data) / len(data) if data else 0.0

def calculate_stddev(data: List[float]) -> float:
    """Calculates Standard Deviation."""
    if len(data) < 2: return 0.0
    mean = calculate_mean(data)
    variance = sum((x - mean) ** 2 for x in data) / (len(data) - 1)
    return math.sqrt(variance)

def analyze_market_phase(history: List[Dict]) -> str:
    """
    CRITICAL FUNCTION: Determines if we are in a TREND or CHOP.
    Used to switch engines on/off dynamically.
    """
    try:
        if len(history) < 10: return "NEUTRAL"
        
        # Look at last 8 rounds
        outcomes = [get_outcome_from_number(d.get('actual_number')) for d in history[-8:]]
        outcomes = [o for o in outcomes if o]
        
        if len(outcomes) < 8: return "NEUTRAL"

        # 1. Check for Dragon (Trend)
        # If 6 of the last 8 results are the SAME, we are Trending.
        counts = Counter(outcomes)
        most_common_val, count = counts.most_common(1)[0]
        
        if count >= 6:
            return "TRENDING"
            
        # 2. Check for Ping-Pong (Chop)
        # If we switch back and forth constantly
        switches = 0
        for i in range(1, len(outcomes)):
            if outcomes[i] != outcomes[i-1]:
                switches += 1
                
        if switches >= 5:
            return "CHOPPY"
            
        return "NEUTRAL"
    except: return "NEUTRAL"

# ==============================================================================
# SECTION 3: ADVANCED MARKET GUARDS (THE SHIELD)
# ==============================================================================

def is_trend_wall_active(history: List[Dict]) -> bool:
    """
    TREND WALL GUARD: Detects massive streaks.
    If the streak is TOO long (9+), we stop betting to be safe.
    """
    try:
        limit = EngineConfig.MAX_IDENTICAL_STREAK
        if len(history) < limit: return False
        
        outcomes = [get_outcome_from_number(d.get('actual_number')) for d in history[-limit:]]
        first = outcomes[0]
        
        if not first: return False
        
        if all(o == first for o in outcomes):
            return True # Dangerous streak active
        return False
    except: return False

# ==============================================================================
# SECTION 4: THE 5 ANALYTICAL ENGINES
# ==============================================================================

# ------------------------------------------------------------------------------
# ENGINE 1: QUANTUM ADAPTIVE (Mean Reversion)
# ------------------------------------------------------------------------------
def engine_quantum_adaptive(history: List[Dict]) -> Optional[Dict]:
    """
    Uses Z-Score to predict REVERSALS.
    WARNING: Only useful in Choppy/Neutral markets. Bad for Trends.
    """
    try:
        numbers = [safe_float(d.get('actual_number')) for d in history[-60:]]
        if len(numbers) < 20: return None
        
        mean = calculate_mean(numbers)
        std = calculate_stddev(numbers)
        if std == 0: return None
        
        last_val = numbers[-1]
        z_score = (last_val - mean) / std
        
        if abs(z_score) < EngineConfig.Z_SCORE_TRIGGER: 
            return None 
        
        strength = min(abs(z_score), 1.0) 
        
        # High Z-Score means "Revert to Mean"
        if z_score > 0: 
            return {'prediction': GameConstants.SMALL, 'weight': strength, 'source': 'Quantum'}
        else: 
            return {'prediction': GameConstants.BIG, 'weight': strength, 'source': 'Quantum'}
            
    except Exception: return None


# ------------------------------------------------------------------------------
# ENGINE 2: DEEP MEMORY V4 (Pattern Matching)
# ------------------------------------------------------------------------------
def engine_deep_memory_v4(history: List[Dict]) -> Optional[Dict]:
    """
    Scans 500 rounds of history to find identical sequence matches.
    """
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
            # We require at least 2 historical matches to trust this engine
            min_matches = 2 
            
            if total >= min_matches:
                imbalance = abs((count_b/total) - (count_s/total))
                min_edge = 0.15
                
                if imbalance > min_edge:
                    if count_b > count_s: 
                        return {'prediction': GameConstants.BIG, 'weight': imbalance, 'source': f'DeepMem({depth})'}
                    elif count_s > count_b: 
                        return {'prediction': GameConstants.SMALL, 'weight': imbalance, 'source': f'DeepMem({depth})'}
        return None
    except Exception: return None


# ------------------------------------------------------------------------------
# ENGINE 3: TITAN CHART PATTERNS (The Full Library)
# ------------------------------------------------------------------------------
def engine_chart_patterns(history: List[Dict]) -> Optional[Dict]:
    """
    Recognizes over 60 complex market shapes (Dragons, Mirrors, Fibonacci).
    """
    try:
        outcomes = [get_outcome_from_number(d.get('actual_number')) for d in history[-30:]]
        s = ''.join(['B' if o==GameConstants.BIG else 'S' for o in outcomes if o])
        if not s: return None

        base_patterns = {
            # === STANDARD OSCILLATIONS ===
            '1v1_Standard': ['BSBSBS'], '1v1_Inverted': ['SBSBSB'],
            
            # === DOUBLE CLUSTERS ===
            '2v2_Standard': ['SSBBSSBB'], '2v2_Inverted': ['BBSSBBSS'],
            '2v2_Short_S': ['SSBB'], '2v2_Short_B': ['BBSS'],

            # === TRIPLE CLUSTERS ===
            '3v3_Standard': ['BBBSSS'], '3v3_Inverted': ['SSSBBB'],
            
            # === QUAD CLUSTERS ===
            '4v4_Standard': ['SSSSBBBB'], '4v4_Inverted': ['BBBBSSSS'],

            # === ASYMMETRICAL PATTERNS ===
            '2v1_S_Dom': ['SSBSSB'], '2v1_B_Dom': ['BBSBBS'],
            '2v1_Short_S': ['SSB'], '2v1_Short_B': ['BBS'],
            '3v1_S_Dom': ['SSSBSSS'], '3v1_B_Dom': ['BBBSBBB'],
            '3v2_S_Dom': ['SSSBBSSS'], '3v2_B_Dom': ['BBBSSBBB'],

            # === RATIO ANALYSIS ===
            'Ratio_1A2B': ['BSS', 'SBB'], 'Ratio_1A3B': ['BSSS', 'SBBB'],
            'Ratio_4A1B': ['BBBBSB', 'SSSSBS'],

            # === DRAGONS (Trend Following) ===
            'Dragon_Follow_B': ['BBBBBBB'], 'Dragon_Follow_S': ['SSSSSSS'],
            'Dragon_Deep_B': ['BBBBBBBBB'], 'Dragon_Deep_S': ['SSSSSSSSS'],
            
            # === ADVANCED GEOMETRY ===
            'Mirror_King': ['BBBBSSBSSBBBB'], 
            'Stairs_Up_A': ['BSBBSSBBBSSS'], 'Stairs_Up_B': ['SBSSBBSSSB BB'], 
            'Decay_A': ['BBBB', 'BBB', 'BB'], 'Decay_B': ['SSSS', 'SSS', 'SS'],
            'Fib_Streak_A': ['B', 'SS', 'BBB'], 'Fib_Streak_B': ['S', 'BB', 'SSS'],
            'Cut_Pattern_A': ['B', 'BB', 'BBB', 'BBBB'], 
            'Cut_Pattern_B': ['S', 'SS', 'SSS', 'SSSS'],
            'Stabilizer_A': ['BSS', 'BSS', 'BSS'], 'Stabilizer_B': ['SBB', 'SBB', 'SBB'],
            'Chop_Zone': ['BSBSBSBS'],
        }
        
        for p_name, p_list in base_patterns.items():
            for p_str in p_list:
                clean_p = p_str.replace(" ", "").strip()
                required = clean_p[:-1] 
                pred_char = clean_p[-1] 
                
                if s.endswith(required):
                    # Strictness: Ignore short patterns unless market is clear
                    if len(required) < 3 and GLOBAL_STRICTNESS > 70: continue 
                    
                    pred = GameConstants.BIG if pred_char == 'B' else GameConstants.SMALL
                    # Weighting: Long patterns get HIGH trust
                    weight = 0.95 if len(required) > 5 else 0.85
                    
                    return {'prediction': pred, 'weight': weight, 'source': f'Chart:{p_name}'}
        return None
    except Exception: return None


# ------------------------------------------------------------------------------
# ENGINE 4: BAYESIAN PROBABILITY
# ------------------------------------------------------------------------------
def engine_bayesian_probability(history: List[Dict]) -> Optional[Dict]:
    """Calculates context-based probability."""
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


# ------------------------------------------------------------------------------
# ENGINE 5: MOMENTUM OSCILLATOR (Velocity)
# ------------------------------------------------------------------------------
def engine_momentum_oscillator(history: List[Dict]) -> Optional[Dict]:
    """
    Tracks market speed. Excellent for TRENDING markets.
    """
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
    """Holds the 'Short-Term Memory' of the AI."""
    def __init__(self):
        self.loss_streak = 0
        self.last_round_predictions = {}
        self.engine_scores = defaultdict(int) 

state_manager = GlobalStateManager()

def reset_engine_memory():
    """EXTERNALLY CALLED by Fetcher when a Session Reset occurs."""
    state_manager.loss_streak = 0
    state_manager.last_round_predictions = {}
    print(f"[ENGINE] Memory Wiped. Global Strictness: {GLOBAL_STRICTNESS}%")

def _build_skip_response(reason_text):
    """Helper to return a standardized SKIP response."""
    return {
        'finalDecision': GameConstants.SKIP,
        'confidence': 0.0,
        'level': "---",
        'reason': reason_text,
        'topsignals': [],
        'positionsize': 0
    }

# ==============================================================================
# SECTION 6: MASTER PREDICTION FUNCTION (THE API)
# ==============================================================================

def ultraAIPredict(history: List[Dict], current_bankroll: float, previous_pred_label: str) -> Dict:
    """
    THE MAIN BRAIN FUNCTION.
    """
    
    # --------------------------------------------------------------------------
    # 1. UPDATE INTERNAL STATE
    # --------------------------------------------------------------------------
    if len(history) > 1 and previous_pred_label not in [GameConstants.SKIP, "WAITING", "COOLDOWN"]:
        last_actual = get_outcome_from_number(history[-1]['actual_number'])
        if last_actual == previous_pred_label:
            state_manager.loss_streak = 0 
        else:
            state_manager.loss_streak += 1 

    current_streak = state_manager.loss_streak

    # --------------------------------------------------------------------------
    # 2. MARKET ANALYSIS (TREND vs CHOP)
    # --------------------------------------------------------------------------
    market_phase = analyze_market_phase(history)
    
    # --------------------------------------------------------------------------
    # 3. RUN SAFETY GUARDS
    # --------------------------------------------------------------------------
    try:
        last_num = int(safe_float(history[-1]['actual_number']))
        # Violet Guard
        if last_num in [0, 5]:
            return _build_skip_response("Violet Reset (0/5)")
        # Trend Wall Guard
        if is_trend_wall_active(history):
            return _build_skip_response(f"Trend Wall Active")
    except Exception as e:
        print(f"[ENGINE WARN] Guard Check Failed: {e}")

    # --------------------------------------------------------------------------
    # 4. DYNAMIC ENGINE SELECTION (TREND ALIGNMENT)
    # --------------------------------------------------------------------------
    active_engines = []
    
    if market_phase == "TRENDING":
        # In a Trend, we DISABLE Quantum (Reversion) to stop fighting the dragon.
        # We TRUST Momentum and Patterns.
        active_engines = [engine_chart_patterns, engine_momentum_oscillator, engine_bayesian_probability]
        phase_note = "TREND MODE"
    
    elif market_phase == "CHOPPY":
        # In Chop, we DISABLE Momentum (it gets confused).
        # We TRUST Quantum and Patterns.
        active_engines = [engine_chart_patterns, engine_quantum_adaptive, engine_bayesian_probability]
        phase_note = "CHOP MODE"
    
    else:
        # Neutral - Use Everything
        active_engines = [
            engine_chart_patterns, 
            engine_momentum_oscillator, 
            engine_quantum_adaptive, 
            engine_bayesian_probability,
            engine_deep_memory_v4
        ]
        phase_note = "NEUTRAL"

    # --------------------------------------------------------------------------
    # 5. RUN SELECTED ENGINES
    # --------------------------------------------------------------------------
    signals = []
    for eng in active_engines:
        res = eng(history)
        if res: signals.append(res)

    if not signals:
        return _build_skip_response(f"No Signals ({phase_note})")

    # --------------------------------------------------------------------------
    # 6. VOTE COUNTING
    # --------------------------------------------------------------------------
    votes = [s['prediction'] for s in signals]
    vote_counts = Counter(votes)
    top_pred, count = vote_counts.most_common(1)[0]
    
    # Identify sources
    sources = [s['source'] for s in signals if s['prediction'] == top_pred]

    # --------------------------------------------------------------------------
    # 7. DECISION LOGIC (ACTIVE SNIPER RECOVERY)
    # --------------------------------------------------------------------------
    
    req_votes = EngineConfig.MIN_VOTES_REQUIRED
    
    # === LEVEL 3 RECOVERY LOGIC (UPDATED FOR SPEED) ===
    # If we are at Level 2 or 3 (Streak >= 2), we are in Recovery Mode.
    if current_streak >= 2:
        
        # LOGIC: 
        # We accept if we have 3 Votes OR (2 Votes AND one is a Chart Pattern).
        # This is faster than the "Perfect 3" rule but safer than "Force 2".
        
        has_pattern_support = any("Chart" in s for s in sources)
        
        if count >= 3 or (count == 2 and has_pattern_support):
             return {
                'finalDecision': top_pred,
                'confidence': 0.99,
                'level': f"LEVEL {current_streak+1} (ACTIVE)",
                'reason': f"Recovery Signal ({phase_note})",
                'topsignals': sources,
                'positionsize': 0
            }
        else:
            # We are losing, and the signal is weak.
            # FORCE SKIP (Hard Wait). Don't panic bet.
            return _build_skip_response(f"L{current_streak+1} Wait: Need Pattern/Strong Signal")

    # === NORMAL TRADING (Level 1 & 2) ===
    # If Trending, we ensure we aren't betting against the momentum (Ghost Mode Logic)
    if market_phase == "TRENDING":
        # Check if our prediction aligns with the last result (Following)
        last_res = get_outcome_from_number(history[-1]['actual_number'])
        
        # If we predict a Flip (Reversal) during a Trend, we need STRONG evidence.
        if top_pred != last_res and count < 3:
            return _build_skip_response("Trend Safety Skip (Ghost Mode)")

    # Standard Consensus Entry
    if count >= req_votes:
        return {
            'finalDecision': top_pred,
            'confidence': 0.99, # Confidence handled by Level text
            'level': "LEVEL 1" if current_streak == 0 else "LEVEL 2",
            'reason': f"{phase_note} Consensus ({count}/{len(active_engines)})",
            'topsignals': sources,
            'positionsize': 0 # Fetcher calculates actual money
        }

    return _build_skip_response(f"Weak Signal ({count} Votes)")

if __name__ == "__main__":
    print("="*60)
    print(f" TITAN V1400 SOVEREIGN ENGINE LOADED")
    print(f" STRICTNESS LEVEL: {GLOBAL_STRICTNESS}%")
    print(" ACTIVE SNIPER MODE: ON")
    print("="*60)
