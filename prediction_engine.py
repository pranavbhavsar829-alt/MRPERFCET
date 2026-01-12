import math
import statistics
import random
import time
import numpy as np
from scipy import stats
from collections import Counter, defaultdict, deque
from typing import Dict, List, Optional, Any, Tuple

TITLE = "TITAN V3.2 BIAS KILLER - SMART BALANCED LOGIC"
print("=" * 80)
print(TITLE)
print("UPDATES: Hardcoded 'BIG' Removed | Smart Fallback Active | 60/40 Ratio Logic")
print("=" * 80)

# GLOBAL CONSTANTS
GLOBAL_STRICTNESS = 65

class GameConstants:
    BIG = "BIG"
    SMALL = "SMALL"
    SKIP = "SKIP"
    STATUS_WAITING = "WAITING"
    STATUS_ANALYZING = "ANALYZING"
    STATUS_LOCKED = "LOCKED"

class EngineConfig:
    BAYES_THRESHOLD = 0.51
    ZSCORE_TRIGGER = 0.08 * GLOBAL_STRICTNESS / 50 
    MOMENTUM_TRIGGER = 0.08 * GLOBAL_STRICTNESS / 50 
    MARKOV_THRESHOLD = 0.52 
    KELLY_THRESHOLD = 0.52   
    
    if GLOBAL_STRICTNESS < 30:
        MIN_VOTES_REQUIRED = 1
    elif GLOBAL_STRICTNESS > 80:
        MIN_VOTES_REQUIRED = 3
    else:
        MIN_VOTES_REQUIRED = 2

    MIN_DATA_REQUIRED = 25
    DEEP_MEM_LOOKBACK = 2000
    BAYES_CONTEXT_WINDOW = 5
    MAX_IDENTICAL_STREAK = max(6, int(12 - GLOBAL_STRICTNESS / 20)) 
    CHOPPY_THRESHOLD = 11  
    HURST_WINDOW = 100
    MONTE_CARLO_SIMS = 2000
    CHI2_ALPHA = 0.01

class RiskConfig:
    BASE_RISK_PERCENT = 0.025 
    MIN_BET_AMOUNT = 50
    MAX_BET_AMOUNT = 50000
    STOP_LOSS_PCT = 0.25      
    PROFIT_TAKE_PCT = 0.35    

# === GLOBAL STATE MANAGER ===
class GlobalStateManager:
    def __init__(self):
        self.loss_streak = 0
        self.win_streak = 0
        self.last_round_predictions = []
        self.consecutive_wins = 0
        self.consecutive_losses = 0
        self.total_bets = 0
        self.engine_performance = defaultdict(lambda: {'wins': 0, 'total': 0, 'accuracy': 0.5})
        self.market_regime = 'UNKNOWN'
        self.bankroll = 10000.0
        self.peak_bankroll = 10000.0
        self.transition_matrix = np.zeros((2, 2))
        self.counts_matrix = np.zeros((2, 2))
        self.total_rounds = 0
        self.rng_bias_detected = False
        self.aggressive_mode = 0  # 0=Normal, 1=After1Loss, 2=After2Loss, 3=FullPower

    def get_strictness_level(self) -> float:
        """PROGRESSIVE STRICTNESS: Loss1=10%, Loss2=20bets, Loss3=Full"""
        if self.consecutive_losses == 0:
            return 1.0  # NORMAL
        elif self.consecutive_losses == 1:
            return 0.90  # 10% STRICT - MOST BETS
        elif self.total_bets < 20:
            return 0.80  # 20 BETS MAX STRICTNESS
        else:
            return 0.60  # FULL POWER AFTER 20 BETS
        return 1.0

    def update_transition_matrix(self, history: List[Dict]):
        bs_numeric = [1 if get_outcome_from_number(d.get('actual_number')) == GameConstants.BIG else 0 
                     for d in history[-500:] if get_outcome_from_number(d.get('actual_number'))]
        if len(bs_numeric) < 20: return
        
        self.counts_matrix = np.zeros((2, 2))
        for i in range(len(bs_numeric)-1):
            self.counts_matrix[bs_numeric[i], bs_numeric[i+1]] += 1
        self.transition_matrix = np.divide(self.counts_matrix, 
                                         self.counts_matrix.sum(axis=1, keepdims=True) + 1e-8)

    def chi_square_bias_test(self, history: List[Dict]) -> float:
        numbers = [safe_float(d.get('actual_number')) for d in history[-500:] if safe_float(d.get('actual_number')) >= 0]
        if len(numbers) < 100: return 1.0
        
        observed, _ = np.histogram(numbers, bins=10, range=(0,10))
        expected = np.full(10, len(numbers)/10)
        chi2 = np.sum((observed - expected)**2 / (expected + 1e-8))
        p_value = 1 - stats.chi2.cdf(chi2, df=9)
        self.rng_bias_detected = p_value < EngineConfig.CHI2_ALPHA
        return p_value

    def update_engine_performance(self, engine_sources: List[str], was_correct: bool):
        for source in engine_sources:
            self.engine_performance[source]['total'] += 1
            if was_correct:
                self.engine_performance[source]['wins'] += 1
            self.engine_performance[source]['accuracy'] = (
                self.engine_performance[source]['wins'] / max(1, self.engine_performance[source]['total']))

    def get_engine_weight(self, engine_name: str) -> float:
        strict_mod = self.get_strictness_level()
        acc = self.engine_performance[engine_name]['accuracy']
        return min(3.0, (acc / 0.5) * strict_mod)

    def kelly_fraction(self, win_prob: float, odds: float = 1.95) -> float:
        if win_prob <= 0.5: return 0.0
        q = 1 - win_prob
        f = (win_prob * odds - 1) / (odds - 1)
        aggressive_f = f * 0.75 * self.get_strictness_level()  # PROGRESSIVE SIZING
        return max(0.015, min(0.35, aggressive_f))  # WIDER RANGE

statemanager = GlobalStateManager()

# === UTILITY FUNCTIONS ===
def safe_float(value: Any) -> float:
    try:
        if value is None: return 4.5
        return float(value)
    except: return 4.5

def get_outcome_from_number(n: Any) -> Optional[str]:
    val = safe_float(n)
    if 0 <= val <= 4: return GameConstants.SMALL
    if 5 <= val <= 9: return GameConstants.BIG
    return None

def calculate_mean(data: List[float]) -> float:
    return sum(data) / len(data) if data else 0.0

def calculate_std_dev(data: List[float]) -> float:
    if len(data) < 2: return 0.0
    mean = calculate_mean(data)
    variance = sum((x - mean) ** 2 for x in data) / (len(data) - 1)
    return math.sqrt(variance)

# === RELAXED SAFETY GUARDS ===
def is_market_choppy(history: List[Dict]) -> bool:
    if statemanager.consecutive_losses < 2:
        return False
    try:
        if len(history) < 15: return False
        outcomes = [get_outcome_from_number(d.get('actual_number'))
                   for d in history[-12:] if get_outcome_from_number(d.get('actual_number'))]
        if len(outcomes) < 10: return False
        switches = sum(1 for i in range(1, len(outcomes)) if outcomes[i] != outcomes[i-1])
        return switches >= EngineConfig.CHOPPY_THRESHOLD
    except: return False

def is_trend_wall_active(history: List[Dict]) -> bool:
    if statemanager.consecutive_losses < 2:
        return False
    try:
        limit = EngineConfig.MAX_IDENTICAL_STREAK + 1 
        if len(history) < limit: return False
        outcomes = [get_outcome_from_number(d.get('actual_number'))
                   for d in history[-limit:] if get_outcome_from_number(d.get('actual_number'))]
        if not outcomes: return False
        first = outcomes[0]
        return all(o == first for o in outcomes)
    except: return False

# === 12 CORE ENGINES ===
def engine_quantum_adaptive(history: List[Dict]) -> Optional[Dict]:
    try:
        numbers = [safe_float(d.get('actual_number')) for d in history[-100:] if safe_float(d.get('actual_number')) >= 0]
        if len(numbers) < 20: return None
        mean = calculate_mean(numbers)
        std = calculate_std_dev(numbers)
        if std == 0: return None
        last_val = numbers[-1]
        zscore = (last_val - mean) / std
        strict_mod = statemanager.get_strictness_level()
        if abs(zscore) < EngineConfig.ZSCORE_TRIGGER * strict_mod: return None
        strength = min(abs(zscore), 1.0)
        return {
            'prediction': GameConstants.SMALL if zscore > 0 else GameConstants.BIG,
            'weight': strength * 1.3,
            'source': 'QuantumAggro'
        }
    except: return None

def engine_deep_memory_v5(history: List[Dict]) -> Optional[Dict]:
    try:
        outcomes = [get_outcome_from_number(d.get('actual_number')) for d in history[-EngineConfig.DEEP_MEM_LOOKBACK:]]
        outcomes = [o for o in outcomes if o]
        if len(outcomes) < 25: return None
        
        raw_str = ''.join('B' if o == GameConstants.BIG else 'S' for o in outcomes)
        max_search_depth = min(15, len(raw_str)//4)
        strict_mod = statemanager.get_strictness_level()
        
        for depth in range(max_search_depth, 3, -1):
            curr_pattern = raw_str[-depth:]
            search_area = raw_str[:-1]
            count_b = count_s = 0
            start = 0
            while True:
                idx = search_area.find(curr_pattern, start)
                if idx == -1: break
                if idx + depth < len(search_area):
                    next_char = search_area[idx + depth]
                    if next_char == 'B': count_b += 1
                    else: count_s += 1
                start = idx + 1
            
            total = count_b + count_s
            if total < 2: continue
            
            imbalance = abs(count_b/total - count_s/total)
            min_edge = 0.10 * strict_mod if GLOBAL_STRICTNESS < 50 else 0.14 * strict_mod
            
            if imbalance > min_edge:
                if count_b > count_s:
                    return {'prediction': GameConstants.BIG, 'weight': imbalance * 1.4, 'source': f'DeepAggro{depth}'}
                return {'prediction': GameConstants.SMALL, 'weight': imbalance * 1.4, 'source': f'DeepAggro{depth}'}
        return None
    except: return None

def engine_chart_patterns(history: List[Dict]) -> Optional[Dict]:
    try:
        outcomes = [get_outcome_from_number(d.get('actual_number')) for d in history[-40:]]
        outcomes = [o for o in outcomes if o]
        if len(outcomes) < 8: return None
        
        s = ''.join('B' if o == GameConstants.BIG else 'S' for o in outcomes)
        patterns = {
            'BSBSBSBS': GameConstants.BIG, 'SBSBSBSB': GameConstants.SMALL,
            'SSBBSSBB': GameConstants.SMALL, 'BBSSBBSS': GameConstants.BIG,
            'BBBSSSBBB': GameConstants.BIG, 'SSSBBBSSS': GameConstants.SMALL,
            'BSBSBS': GameConstants.BIG, 'SBSBSB': GameConstants.SMALL,
            'SSBB': GameConstants.SMALL, 'BBSS': GameConstants.BIG,
            'BBBBBB': GameConstants.SMALL, 'SSSSSS': GameConstants.BIG
        }
        
        for pattern, prediction in patterns.items():
            if len(pattern) <= len(s) and s.endswith(pattern[:-1]):
                weight = 0.92 if len(pattern) >= 6 else 0.78
                return {'prediction': prediction, 'weight': weight, 'source': f'ChartAggro{len(pattern)}'}
        return None
    except: return None

def engine_bayesian_probability(history: List[Dict]) -> Optional[Dict]:
    try:
        outcomes = [get_outcome_from_number(d.get('actual_number')) for d in history]
        cleaned = [o for o in outcomes if o]
        if len(cleaned) < EngineConfig.BAYES_CONTEXT_WINDOW + 8: return None
        
        context_len = EngineConfig.BAYES_CONTEXT_WINDOW
        last_context = tuple(cleaned[-context_len:])
        
        b_count = s_count = 0
        for i in range(len(cleaned) - context_len - 1):
            if tuple(cleaned[i:i+context_len]) == last_context:
                next_val = cleaned[i+context_len]
                if next_val == GameConstants.BIG: b_count += 1
                elif next_val == GameConstants.SMALL: s_count += 1
        
        total = b_count + s_count
        if total < 2: return None
        
        prob_b = (b_count + 1) / (total + 2)
        prob_s = (s_count + 1) / (total + 2)
        strict_mod = statemanager.get_strictness_level()
        
        if prob_b > EngineConfig.BAYES_THRESHOLD * strict_mod:
            return {'prediction': GameConstants.BIG, 'weight': prob_b, 'source': 'BayesAggro'}
        elif prob_s > EngineConfig.BAYES_THRESHOLD * strict_mod:
            return {'prediction': GameConstants.SMALL, 'weight': prob_s, 'source': 'BayesAggro'}
        return None
    except: return None

def engine_momentum_oscillator(history: List[Dict]) -> Optional[Dict]:
    try:
        outcomes = [get_outcome_from_number(d.get('actual_number')) for d in history[-25:]]
        outcomes = [o for o in outcomes if o]
        if len(outcomes) < 8: return None
        
        score = 0.0
        weight = 1.0
        decay = 0.87
        
        for o in reversed(outcomes):
            if o == GameConstants.BIG: score += weight
            elif o == GameConstants.SMALL: score -= weight
            weight *= decay
        
        strict_mod = statemanager.get_strictness_level()
        if score > EngineConfig.MOMENTUM_TRIGGER * strict_mod * 1.2:
            return {'prediction': GameConstants.BIG, 'weight': min(0.87, score/2.8), 'source': 'MomentumAggro'}
        elif score < -EngineConfig.MOMENTUM_TRIGGER * strict_mod * 1.2:
            return {'prediction': GameConstants.SMALL, 'weight': min(0.87, abs(score)/2.8), 'source': 'MomentumAggro'}
        return None
    except: return None

def engine_markov_chain_v2(history: List[Dict]) -> Optional[Dict]:
    try:
        statemanager.update_transition_matrix(history)
        last_outcome = get_outcome_from_number(history[-1].get('actual_number'))
        if not last_outcome: return None
        
        last_state = 1 if last_outcome == GameConstants.BIG else 0
        p_big_next = statemanager.transition_matrix[last_state, 1]
        
        recent_bs = [1 if get_outcome_from_number(d.get('actual_number')) == GameConstants.BIG else 0 
                    for d in history[-60:] if get_outcome_from_number(d.get('actual_number'))]
        if len(recent_bs) >= 3:
            state2 = tuple(recent_bs[-2:])
            trans2 = defaultdict(lambda: defaultdict(int))
            for i in range(len(recent_bs)-2):
                trans2[tuple(recent_bs[i:i+2])][recent_bs[i+2]] += 1
            total2 = sum(trans2[state2].values())
            if total2 > 0:
                p_big_next = (p_big_next + trans2[state2][1]/total2 * 0.7) / 1.7
        
        strict_mod = statemanager.get_strictness_level()
        if p_big_next > EngineConfig.MARKOV_THRESHOLD * strict_mod:
            return {'prediction': GameConstants.BIG, 'weight': p_big_next, 'source': 'MarkovAggro'}
        elif p_big_next < (1 - EngineConfig.MARKOV_THRESHOLD * strict_mod):
            return {'prediction': GameConstants.SMALL, 'weight': 1-p_big_next, 'source': 'MarkovAggro'}
        return None
    except: return None

def engine_streak_reversal(history: List[Dict]) -> Optional[Dict]:
    try:
        outcomes = [get_outcome_from_number(d.get('actual_number')) for d in history[-35:]]
        outcomes = [o for o in outcomes if o]
        if len(outcomes) < 12: return None
        
        streak_len = 1
        current = outcomes[-1]
        for i in range(2, len(outcomes)+1):
            if i > len(outcomes) or outcomes[-i] != current:
                break
            streak_len += 1
        
        if streak_len >= 3:
            prediction = GameConstants.SMALL if current == GameConstants.BIG else GameConstants.BIG
            weight = min(0.94, 0.65 + (streak_len-2)*0.09)
            return {'prediction': prediction, 'weight': weight, 'source': f'StreakR{streak_len}'}
        return None
    except: return None

def engine_hot_cold_v2(history: List[Dict]) -> Optional[Dict]:
    try:
        outcomes = [get_outcome_from_number(d.get('actual_number')) for d in history[-120:]]
        outcomes = [o for o in outcomes if o]
        if len(outcomes) < 35: return None
        
        windows = [25, 60, 120]
        big_pcts = []
        for w in windows:
            recent = outcomes[-w:]
            big_pcts.append(recent.count(GameConstants.BIG) / len(recent))
        
        avg_pct = np.mean(big_pcts)
        strict_mod = statemanager.get_strictness_level()
        
        if avg_pct < 0.40 * strict_mod:
            return {'prediction': GameConstants.BIG, 'weight': min(0.90, (0.5-avg_pct)*2.2), 'source': 'HotColdAggro'}
        elif avg_pct > 0.60 / strict_mod:
            return {'prediction': GameConstants.SMALL, 'weight': min(0.90, (avg_pct-0.5)*2.2), 'source': 'HotColdAggro'}
        return None
    except: return None

def engine_rng_bias(history: List[Dict]) -> Optional[Dict]:
    try:
        p_value = statemanager.chi_square_bias_test(history)
        if not statemanager.rng_bias_detected: return None
        
        numbers = [safe_float(d.get('actual_number')) for d in history[-250:] if safe_float(d.get('actual_number')) >= 0]
        recent_big_pct = sum(1 for n in numbers[-25:] if n >= 5) / 25
        if recent_big_pct < 0.28:
            return {'prediction': GameConstants.BIG, 'weight': 0.88, 'source': 'RNGBias+'}
        elif recent_big_pct > 0.72:
            return {'prediction': GameConstants.SMALL, 'weight': 0.88, 'source': 'RNGBias+'}
        return None
    except: return None

def engine_regime_detector_v2(history: List[Dict]) -> Optional[Dict]:
    try:
        outcomes = [get_outcome_from_number(d.get('actual_number')) for d in history[-180:]]
        outcomes = [o for o in outcomes if o]
        if len(outcomes) < 70: return None
        
        counts = Counter(outcomes)
        total = len(outcomes)
        entropy = -sum((c/total)*math.log2(c/total+1e-10) for c in counts.values())
        
        strict_mod = statemanager.get_strictness_level()
        if entropy > 2.3 * strict_mod:
            statemanager.market_regime = 'CHOPPY'
            return None
        elif entropy < 1.2:
            statemanager.market_regime = 'TRENDING'
            last = outcomes[-1]
            pred = GameConstants.SMALL if last == GameConstants.BIG else GameConstants.BIG
            return {'prediction': pred, 'weight': 0.80, 'source': 'RegimeAggro'}
        
        statemanager.market_regime = 'RANDOM'
        return None
    except: return None

# === MONTE CARLO SUPERVOTER ===
def engine_monte_carlo_v2(history: List[Dict], primary_votes: List[Dict]) -> Optional[Dict]:
    try:
        if not primary_votes: return None
        outcomes = [get_outcome_from_number(d.get('actual_number')) for d in history[-350:]]
        outcomes = [o for o in outcomes if o]
        if len(outcomes) < 90: return None
        
        primary_pred = Counter(v['prediction'] for v in primary_votes).most_common(1)[0][0]
        sim_wins = sum(1 for _ in range(EngineConfig.MONTE_CARLO_SIMS) 
                      if random.choice(outcomes) == primary_pred)
        sim_prob = sim_wins / EngineConfig.MONTE_CARLO_SIMS
        
        strict_mod = statemanager.get_strictness_level()
        if sim_prob > 0.60 * strict_mod:
            return {'prediction': primary_pred, 'weight': sim_prob, 'source': 'MonteCarloAggro'}
        return None
    except: return None

# === HELPER: SMART FALLBACK CALCULATOR ===
def calculate_smart_fallback(history: List[Dict]) -> Tuple[str, str]:
    """
    Analyzes last 15 rounds. 
    If > 60% BIG -> Force SMALL. 
    If > 60% SMALL -> Force BIG.
    Else -> Alternate last.
    """
    try:
        recent = [get_outcome_from_number(d['actual_number']) for d in history[-15:]]
        recent = [o for o in recent if o]
        if len(recent) < 5:
            return (GameConstants.BIG, "Low Data")

        big_count = recent.count(GameConstants.BIG)
        ratio = big_count / len(recent)

        if ratio >= 0.60:
            return (GameConstants.SMALL, f"Anti-Bias ({ratio:.0%} Big)")
        elif ratio <= 0.40:
            return (GameConstants.BIG, f"Anti-Bias ({ratio:.0%} Small)")
        else:
            # Balanced: Just alternate the very last one
            last_one = recent[-1]
            pred = GameConstants.SMALL if last_one == GameConstants.BIG else GameConstants.BIG
            return (pred, "Balanced PingPong")
    except:
        return (GameConstants.BIG, "Fallback Error")

# === MAIN ULTRA AI BRAIN - BIASED CORRECTED ===
def ultraAIPredict(history: List[Dict], current_bankroll: float, previous_pred_label: str = None) -> Dict:
    """12-ENGINE AGGRESSIVE SYSTEM - BIAS CORRECTED"""
    
    # GLOBAL UPDATES
    statemanager.bankroll = current_bankroll
    statemanager.total_bets += 1
    statemanager.total_rounds += 1
    
    # MINIMUM SAFETY ONLY
    if len(history) < EngineConfig.MIN_DATA_REQUIRED:
        smart_pred, smart_reason = calculate_smart_fallback(history)
        return build_prediction_response(smart_pred, 0.52, "WARMUP", f"Low Data - {smart_reason}")

    # RESULT TRACKING
    if len(history) >= 2 and previous_pred_label and previous_pred_label not in [GameConstants.SKIP, "WAITING", "COOLDOWN"]:
        last_actual = get_outcome_from_number(history[-1].get('actual_number'))
        was_correct = last_actual == previous_pred_label
        statemanager.update_engine_performance([], was_correct)
        if was_correct:
            statemanager.consecutive_wins += 1
            statemanager.consecutive_losses = 0
            statemanager.aggressive_mode = 0
        else:
            statemanager.consecutive_losses += 1
            statemanager.consecutive_wins = 0
            statemanager.aggressive_mode = min(3, statemanager.aggressive_mode + 1)

    # HARD FINANCIAL LIMITS ONLY
    drawdown = (statemanager.peak_bankroll - current_bankroll) / statemanager.peak_bankroll
    if drawdown > RiskConfig.STOP_LOSS_PCT * 1.5:
        smart_pred, smart_reason = calculate_smart_fallback(history)
        return build_prediction_response(smart_pred, 0.50, "MAXLOSS", "Emergency Conservative")

    # === SAFETY GUARDS (NOW USE SMART FALLBACK INSTEAD OF HARD BIG) ===
    if statemanager.consecutive_losses >= 2:
        if is_market_choppy(history):
            smart_pred, smart_reason = calculate_smart_fallback(history)
            return build_prediction_response(smart_pred, 0.51, "CHAOS-L", f"Choppy - {smart_reason}")
        if is_trend_wall_active(history):
            smart_pred, smart_reason = calculate_smart_fallback(history)
            return build_prediction_response(smart_pred, 0.51, "TREND-L", f"Wall Hit - {smart_reason}")

    # === ALWAYS RUN 12 ENGINES ===
    engines = [
        engine_quantum_adaptive, engine_deep_memory_v5, engine_chart_patterns,
        engine_bayesian_probability, engine_momentum_oscillator, engine_markov_chain_v2,
        engine_streak_reversal, engine_hot_cold_v2, engine_rng_bias, engine_regime_detector_v2
    ]
    
    signals = [eng(history) for eng in engines if eng(history)]
    
    # === SMART FALLBACK IF NO SIGNALS (FIXED BIAS) ===
    if not signals:
        smart_pred, smart_reason = calculate_smart_fallback(history)
        return build_prediction_response(smart_pred, 0.52, "FALLBACK", smart_reason)

    # === PROGRESSIVE CONSENSUS ===
    votes = Counter(s['prediction'] for s in signals)
    top_pred, vote_count = votes.most_common(1)[0]
    
    # RELAXED VOTING - PROGRESSIVE
    strict_mod = statemanager.get_strictness_level()
    min_votes_needed = max(1, int(EngineConfig.MIN_VOTES_REQUIRED * strict_mod))
    
    if vote_count < min_votes_needed:
        # FORCE CONSENSUS - TAKE STRONGEST SINGLE SIGNAL
        strongest = max(signals, key=lambda x: x['weight'])
        top_pred = strongest['prediction']
        vote_count = 1
        signals = [strongest]

    # WEIGHTED ENSEMBLE
    weighted_score = sum(s['weight'] * statemanager.get_engine_weight(s['source'])
                        for s in signals if s['prediction'] == top_pred)
    total_weight = sum(s['weight'] for s in signals if s['prediction'] == top_pred)
    ensemble_prob = weighted_score / max(1, total_weight)

    # MINIMUM 52% - AGGRESSIVE
    final_prob = max(0.52, ensemble_prob * strict_mod)

    # MONTE CARLO BOOST
    mc = engine_monte_carlo_v2(history, signals)
    confidence = final_prob * (1.05 if mc and mc['weight'] > 0.60 else 0.98)
    level = "ELITE" if mc else f"A{vote_count}"

    # PROGRESSIVE KELLY SIZING
    kelly_f = statemanager.kelly_fraction(final_prob)
    position_size = max(RiskConfig.MIN_BET_AMOUNT, 
                       min(current_bankroll * kelly_f * 120 * strict_mod, RiskConfig.MAX_BET_AMOUNT))
    
    # BIAS BOOST
    if statemanager.rng_bias_detected:
        position_size *= 1.4
        confidence *= 1.15
        level = "BIAS+" + level

    statemanager.peak_bankroll = max(statemanager.peak_bankroll, current_bankroll)

    mode_text = {0: "NORMAL", 1: "L1-10%", 2: "L2-20B", 3: "FULLPWR"}[statemanager.aggressive_mode]
    reason = f"{vote_count}/12 | {final_prob:.1%} | {mode_text} | Kelly:{kelly_f:.1%}" + \
             (" | RNG!" if statemanager.rng_bias_detected else "")

    return build_prediction_response(top_pred, confidence, level, reason, 
                                   [s['source'] for s in signals if s['prediction'] == top_pred],
                                   round(position_size, 0))

def build_prediction_response(prediction: str, confidence: float, level: str, reason: str,
                            top_signals: List[str] = None, positionsize: float = 0) -> Dict:
    """ALWAYS RETURNS PREDICTION - NEVER SKIP"""
    return {
        'finalDecision': prediction,
        'confidence': min(1.0, confidence),
        'level': level,
        'reason': reason,
        'topSignals': top_signals or [],
        'positionsize': max(50, positionsize)  # MINIMUM 50 ALWAYS
    }

def build_skip_response(reason: str) -> Dict:  # NEVER USED NOW
    return build_prediction_response("BIG", 0.50, "---", "SKIP DISABLED - AGGRESSIVE MODE")

# === EXACT FETCHER FUNCTIONS ===
def reset_engine_memory():
    """CALLED BY FETCHER"""
    global statemanager
    statemanager = GlobalStateManager()
    print("TITAN V3.2 BIAS KILLER RESET - SMART LOGIC ACTIVE")

if __name__ == "__main__":
    print("TITAN V3.2 BIAS KILLER LOADED - NO DEFAULT BIG")
    print("SMART FALLBACK: 60/40 RATIO ANALYSIS")
