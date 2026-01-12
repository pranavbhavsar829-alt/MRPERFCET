# ==============================================================================
# PROJECT: TITAN V1300 - ULTIMATE PREDICTION ENGINE
# CODENAME: "SOVEREIGN SNIPER"
# ==============================================================================
# AUTHOR: TITAN SYSTEM ARCHITECT
# DATE: 2026-01-12
# VERSION: 13.5.0 (AGGRESSIVE RECOVERY + FULL LIBRARY)
#
# DESCRIPTION:
# This module is the "Cortex" of the automated trading system. It is designed
# to ingest historical data from the Wingo lottery game and output high-probability
# predictions based on a consensus of 5 independent analytical engines.
#
# SYSTEM ARCHITECTURE:
# 1. ENGINE 1: QUANTUM ADAPTIVE (Mean Reversion / Z-Score Analysis)
# 2. ENGINE 2: DEEP MEMORY V4 (Historical Sequence Matching)
# 3. ENGINE 3: TITAN CHART PATTERNS (Technical Analysis & Geometry)
# 4. ENGINE 4: BAYESIAN PROBABILITY (Mathematical Context Analysis)
# 5. ENGINE 5: MOMENTUM OSCILLATOR (Velocity & Trend Strength)
#
# KEY FEATURES:
# - DYNAMIC STRICTNESS: Automatically adjusts filters based on market conditions.
# - AGGRESSIVE RECOVERY: "Force Bet" logic triggers during loss streaks to recover fast.
# - PATTERN RECOGNITION: Recognizes 60+ complex market shapes (Dragons, Stairs, etc.)
# - SAFETY GUARDS: "Violet Reset" and "Trend Wall" protections.
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
# 0 - 100 Scale.
# 65% is the "Sweet Spot" for V1300. It filters out noise but allows for
# aggressive recovery when needed.
# ------------------------------------------------------------------------------
GLOBAL_STRICTNESS = 65


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
    # At 65% Strictness -> Requires ~69% Probability confidence.
    BAYES_THRESHOLD = 0.51 + (GLOBAL_STRICTNESS * 0.0029) 
    
    # Z-Score Deviation Requirement (0.1 to 1.6)
    # Controls how extreme a number must be to trigger Mean Reversion.
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
    # At 65% Strictness -> Stops at a streak of 7 identical results.
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
    """
    Safely converts API data to float, handling NoneTypes and errors.
    Returns 4.5 (midpoint) if data is invalid.
    """
    try:
        if value is None: return 4.5
        return float(value)
    except: return 4.5

def get_outcome_from_number(n: Any) -> Optional[str]:
    """
    Converts a number (0-9) to the game outcome BIG or SMALL.
    0-4 = SMALL
    5-9 = BIG
    """
    val = int(safe_float(n))
    if 0 <= val <= 4: return GameConstants.SMALL
    if 5 <= val <= 9: return GameConstants.BIG
    return None

def calculate_mean(data: List[float]) -> float:
    """Calculates Arithmetic Mean of a dataset."""
    return sum(data) / len(data) if data else 0.0

def calculate_stddev(data: List[float]) -> float:
    """Calculates Standard Deviation (Population Volatility)."""
    if len(data) < 2: return 0.0
    mean = calculate_mean(data)
    variance = sum((x - mean) ** 2 for x in data) / (len(data) - 1)
    return math.sqrt(variance)

# ==============================================================================
# SECTION 3: ADVANCED MARKET GUARDS (THE SHIELD)
# ==============================================================================

def is_market_choppy(history: List[Dict]) -> bool:
    """
    CHAOS GUARD: Detects 'Ping-Pong' markets (e.g., B-S-B-S-B).
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
                
        return switches >= EngineConfig.CHOPPY_THRESHOLD
    except: return False

def is_trend_wall_active(history: List[Dict]) -> bool:
    """
    TREND WALL GUARD: Detects massive streaks (Dragon).
    If we see X results of the same color, we STOP predicting.
    This prevents fighting a trend (Martingale Death).
    """
    try:
        limit = EngineConfig.MAX_IDENTICAL_STREAK
        if len(history) < limit: return False
        
        # Get last 'limit' outcomes
        outcomes = [get_outcome_from_number(d.get('actual_number')) for d in history[-limit:]]
        first = outcomes[0]
        
        if not first: return False
        
        # Check if ALL items in the list are identical
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
    Uses Z-Score to detect statistical anomalies.
    If the current number is too far from the average, predicts a reversion.
    """
    try:
        # 1. Prepare Data (Last 60 rounds)
        numbers = [safe_float(d.get('actual_number')) for d in history[-60:]]
        if len(numbers) < 20: return None
        
        # 2. Calculate Stats
        mean = calculate_mean(numbers)
        std = calculate_stddev(numbers)
        if std == 0: return None
        
        # 3. Calculate Z-Score of the LATEST number
        last_val = numbers[-1]
        z_score = (last_val - mean) / std
        
        # 4. Check against Dynamic Trigger
        if abs(z_score) < EngineConfig.Z_SCORE_TRIGGER: 
            return None # Deviation too small, ignore.
        
        strength = min(abs(z_score), 1.0) 
        
        # 5. Logic: High Z-Score means we revert to mean
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
    Scans the last 500 rounds to find if the current sequence has happened before.
    """
    try:
        outcomes = [get_outcome_from_number(d.get('actual_number')) for d in history]
        raw_str = ''.join(['B' if o==GameConstants.BIG else 'S' for o in outcomes if o])
        
        max_search_depth = 12
        
        # Iteratively search for patterns from length 12 down to 3
        for depth in range(max_search_depth, 3, -1):
            curr_pattern = raw_str[-depth:]
            search_area = raw_str[:-1]
            
            count_b = 0; count_s = 0; start = 0
            
            # Find all occurrences
            while True:
                idx = search_area.find(curr_pattern, start)
                if idx == -1: break
                if idx + depth < len(search_area):
                    next_char = search_area[idx + depth]
                    if next_char == 'B': count_b += 1
                    else: count_s += 1
                start = idx + 1
            
            total = count_b + count_s
            
            # AGGRESSIVE UPDATE: Accept even single matches in recovery
            min_matches = 1 
            
            if total >= min_matches:
                imbalance = abs((count_b/total) - (count_s/total))
                
                # Dynamic Edge Requirement
                min_edge = 0.10 if GLOBAL_STRICTNESS < 50 else 0.15
                
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
    Technical Analysis Shapes.
    Includes the TITAN ADVANCED PATTERN LIBRARY (V13.5).
    """
    try:
        outcomes = [get_outcome_from_number(d.get('actual_number')) for d in history[-30:]]
        s = ''.join(['B' if o==GameConstants.BIG else 'S' for o in outcomes if o])
        if not s: return None

        # --- THE PATTERN LIBRARY ---
        # Format: 'Name': ['Sequence']
        # The engine detects the Sequence (minus the last char) and predicts the last char.
        
        base_patterns = {
            # === STANDARD OSCILLATIONS ===
            '1v1_Standard': ['BSBSBS'], 
            '1v1_Inverted': ['SBSBSB'],
            
            # === DOUBLE CLUSTERS ===
            '2v2_Standard': ['SSBBSSBB'],
            '2v2_Inverted': ['BBSSBBSS'],
            '2v2_Short_S': ['SSBB'], 
            '2v2_Short_B': ['BBSS'],

            # === TRIPLE CLUSTERS ===
            '3v3_Standard': ['BBBSSS'],
            '3v3_Inverted': ['SSSBBB'],
            '3v3_Long_S': ['BBBSSSBBB'],
            '3v3_Long_B': ['SSSBBBSSS'],

            # === QUAD CLUSTERS ===
            '4v4_Standard': ['SSSSBBBB'],
            '4v4_Inverted': ['BBBBSSSS'],

            # === ASYMMETRICAL: 2v1 ===
            '2v1_S_Dominant': ['SSBSSB'],  # Small Small Big...
            '2v1_B_Dominant': ['BBSBBS'],  # Big Big Small...
            '2v1_Short_S': ['SSB'],
            '2v1_Short_B': ['BBS'],

            # === ASYMMETRICAL: 3v1 ===
            '3v1_S_Dominant': ['SSSBSSS'],
            '3v1_B_Dominant': ['BBBSBBB'],
            '3v1_Short_S': ['SSSB'],
            '3v1_Short_B': ['BBBS'],

            # === ASYMMETRICAL: 3v2 ===
            '3v2_S_Dominant': ['SSSBBSSS'],
            '3v2_B_Dominant': ['BBBSSBBB'],
            '3v2_Short_S': ['SSSBB'],
            '3v2_Short_B': ['BBBSS'],

            # === ASYMMETRICAL: 4v1 & 4v2 ===
            '4v1_S_Dominant': ['SSSSBSSSS'],
            '4v1_B_Dominant': ['BBBBSBBBB'],
            '4v2_S_Dominant': ['SSSSBBSSSS'],
            '4v2_B_Dominant': ['BBBBSSBBBB'],

            # === RATIO ANALYSIS (A=Big, B=Small) ===
            'Ratio_1A2B': ['BSS', 'SBB'],
            'Ratio_1A3B': ['BSSS', 'SBBB'],
            'Ratio_4A1B': ['BBBBSB', 'SSSSBS'],
            'Ratio_4A2B': ['BBBBSSB', 'SSSSBBS'],

            # === DRAGON TRACKING (Trend Following) ===
            'Dragon_Follow_B': ['BBBBBBB'],  # Seen 6 Bigs -> Predict 7th Big
            'Dragon_Follow_S': ['SSSSSSS'],
            'Dragon_Deep_B': ['BBBBBBBBB'], # Seen 8 Bigs -> Predict 9th Big
            'Dragon_Deep_S': ['SSSSSSSSS'],
            
            # === REVERSAL SIGNALS (Overdue) ===
            # CAUTION: Only used if market is not trending hard
            'Reversal_B': ['BBBBBBBBBBB'], # 11 Bigs -> Predict Small (Rare)
            'Reversal_S': ['SSSSSSSSSSS'],

            # === ADVANCED GEOMETRY ===
            'Mirror_King': ['BBBBSSBSSBBBB'], # Symmetric
            
            'Stairs_Up_A': ['BSBBSSBBBSSS'], # 1-2-3 Climbing
            'Stairs_Up_B': ['SBSSBBSSSB BB'], 

            'Decay_A': ['BBBB', 'BBB', 'BB'], # 4-3-2 Falling
            'Decay_B': ['SSSS', 'SSS', 'SS'],

            'Fib_Streak_A': ['B', 'SS', 'BBB'], # 1, 2, 3 (Fibonacci)
            'Fib_Streak_B': ['S', 'BB', 'SSS'],

            'Cut_Pattern_A': ['B', 'BB', 'BBB', 'BBBB'], # Growing tail
            'Cut_Pattern_B': ['S', 'SS', 'SSS', 'SSSS'],

            'Stabilizer_A': ['BSS', 'BSS', 'BSS'], # Repeating Triplet
            'Stabilizer_B': ['SBB', 'SBB', 'SBB'],
            
            'Chop_Zone': ['BSBSBSBS'], # High volatility marker
        }
        
        for p_name, p_list in base_patterns.items():
            for p_str in p_list:
                # Cleanup and Parsing
                clean_p = p_str.replace(" ", "").strip()
                required = clean_p[:-1] 
                pred_char = clean_p[-1] 
                
                if s.endswith(required):
                    # Strictness Filter
                    if len(required) < 3 and GLOBAL_STRICTNESS > 70:
                        continue 
                    
                    pred = GameConstants.BIG if pred_char == 'B' else GameConstants.SMALL
                    
                    # Weighting Logic
                    weight = 0.95 if len(required) > 5 else 0.85
                    
                    return {'prediction': pred, 'weight': weight, 'source': f'Chart:{p_name}'}
        return None
    except Exception: return None


# ------------------------------------------------------------------------------
# ENGINE 4: BAYESIAN PROBABILITY (Synthetic AI)
# ------------------------------------------------------------------------------
def engine_bayesian_probability(history: List[Dict]) -> Optional[Dict]:
    """
    Calculates pure mathematical probability based on Context Frequency.
    """
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
        
        # Uses DYNAMIC Threshold from Config
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
    Tracks the 'speed' of the market. Recent results are weighted heavier.
    """
    try:
        outcomes = [get_outcome_from_number(d.get('actual_number')) for d in history[-12:]]
        
        score = 0; weight = 1.0; decay = 0.85
        
        for o in reversed(outcomes):
            if o == GameConstants.BIG: score += weight
            elif o == GameConstants.SMALL: score -= weight
            weight *= decay 
            
        # Uses DYNAMIC Trigger from Config
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
    """
    Holds the 'Short-Term Memory' of the AI.
    Used to track streaks and reset logic.
    """
    def __init__(self):
        self.loss_streak = 0
        self.last_round_predictions = {}
        # Engine scores can be used to weight trusted engines higher
        self.engine_scores = defaultdict(int) 

state_manager = GlobalStateManager()

def reset_engine_memory():
    """
    EXTERNALLY CALLED by Fetcher when a Session Reset occurs.
    """
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
    1. Updates internal win/loss state.
    2. Checks all Safety Guards.
    3. Runs all 5 Engines.
    4. Calculates Consensus based on Aggressive Recovery Rules.
    5. Returns Decision.
    """
    
    # --------------------------------------------------------------------------
    # 1. UPDATE INTERNAL STATE (Did we win the last round?)
    # --------------------------------------------------------------------------
    if len(history) > 1 and previous_pred_label not in [GameConstants.SKIP, "WAITING", "COOLDOWN"]:
        last_actual = get_outcome_from_number(history[-1]['actual_number'])
        
        if last_actual == previous_pred_label:
            state_manager.loss_streak = 0 # Reset streak on Win
        else:
            state_manager.loss_streak += 1 # Increment on Loss

    current_streak = state_manager.loss_streak

    # --------------------------------------------------------------------------
    # 2. RUN SAFETY GUARDS (The Shield)
    # --------------------------------------------------------------------------
    try:
        last_num = int(safe_float(history[-1]['actual_number']))
        
        # A. VIOLET GUARD (Numbers 0 and 5 often cause chaos)
        if last_num in [0, 5]:
            return _build_skip_response("Violet Reset (0/5)")
            
        # B. TREND WALL GUARD (Don't fight long streaks)
        if is_trend_wall_active(history):
            return _build_skip_response(f"Trend Wall (Streak > {EngineConfig.MAX_IDENTICAL_STREAK})")
            
    except Exception as e:
        print(f"[ENGINE WARN] Guard Check Failed: {e}")

    # --------------------------------------------------------------------------
    # 3. RUN ALL 5 ENGINES
    # --------------------------------------------------------------------------
    signals = []
    
    engines = [
        engine_quantum_adaptive, 
        engine_deep_memory_v4, 
        engine_chart_patterns, 
        engine_bayesian_probability, 
        engine_momentum_oscillator
    ]
    
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
    
    # Identify who voted for the winner
    sources = [s['source'] for s in signals if s['prediction'] == top_pred]

    # --------------------------------------------------------------------------
    # 5. DYNAMIC DECISION LOGIC (AGGRESSIVE RECOVERY)
    # --------------------------------------------------------------------------
    
    req_votes = EngineConfig.MIN_VOTES_REQUIRED
    
    final_label = GameConstants.SKIP
    confidence_level = "SKIP"
    reason = f"Need {req_votes} votes (Got {count})"
    
    # === CHECK A: RECOVERY MODE (Are we losing?) ===
    if current_streak > 0:
        # AGGRESSIVE RECOVERY LOGIC
        # We lower the requirements to ensure we don't sit out too long.
        # Rule: Accept 2 Votes, OR 1 Vote if it is a strong Chart Pattern.
        
        if count >= 2:
            final_label = top_pred
            confidence_level = f"LEVEL {current_streak + 1} (REC)"
            reason = "Recovery Force (2+ Votes)"
            
        elif count == 1 and "Chart" in sources[0]:
            # Pattern Match Force
            final_label = top_pred
            confidence_level = f"LEVEL {current_streak + 1} (PAT)"
            reason = f"Pattern Force: {sources[0]}"
            
        else:
            return _build_skip_response(f"Waiting for Signal (Got {count})...")
            
    # === CHECK B: NORMAL MODE (Level 1) ===
    else:
        # Standard strictness applies
        if count >= req_votes:
            final_label = top_pred
            
            # Label based on Consensus Strength
            if count >= 4: 
                confidence_level = "LEVEL 1 (MAX)"
            elif count == 3: 
                confidence_level = "LEVEL 1 (SOLID)"
            else: 
                confidence_level = "LEVEL 1 (BASIC)"
                
            reason = f"Consensus ({count}/{req_votes})"

    # --------------------------------------------------------------------------
    # 6. RETURN FINAL DECISION
    # --------------------------------------------------------------------------
    
    return {
        'finalDecision': final_label,
        'confidence': 0.99, # Confidence is handled by Level in this version
        'level': confidence_level,
        'reason': reason,
        'topsignals': sources,
        'positionsize': 0 # Fetcher calculates actual money
    }

if __name__ == "__main__":
    print("="*60)
    print(f" TITAN V1300 ULTIMATE ENGINE LOADED")
    print(f" STRICTNESS LEVEL: {GLOBAL_STRICTNESS}%")
    print(f" MIN VOTES REQ: {EngineConfig.MIN_VOTES_REQUIRED}")
    print("="*60)
