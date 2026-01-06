#!/usr/bin/env python3

"""
=============================================================================

TITAN LITE V3 – SPECIAL FORCES + META ENGINES + REGIME BRAIN (FULL EXPANDED)

PHILOSOPHY: "Speed, Precision & Depth"

ACTIVE ENGINES:

1. PATTERN SNIPER (Visual)
   - Detects ZigZags, Mirrors, Double Pairs, and Dragons.
   - Explicit rules for both BIG and SMALL.

2. MIRROR_PATTERN (Meta)
   - Digit-level mirror symmetry analysis on recent window.

3. CLUSTER_DOMINANCE (Meta)
   - Detects dominant value / range clusters and entropy.

4. FREQUENCY_REVERSION (Meta)
   - Long-term vs short-term frequency imbalance & z-score.

5. SEQUENCE_TREND (Meta)
   - Short-term directional trend on raw numbers.

6. MOMENTUM_ANALYSIS (Meta)
   - Strength of recent BIG/SMALL momentum and streak.

7. PERIOD_PARITY (Meta)
   - Even/odd bias, modular cycles, and micro-periodicity.

8. VOLATILITY_REGIME (Meta)
   - Classifies regime: CALM / NORMAL / EXPLOSIVE from volatility.

9. STREAK_STRUCTURE (Meta)
   - Compares current streak to typical streak distribution.

10. LOCAL_GRAMMAR (Meta)
    - Token-level transitions on BB / SS / BS / SB.

11. ENTROPY_GUARD (Meta)
    - Shannon entropy of recent patterns; reduces bets when too random.

12. CONSENSUS_INDEX (Meta)
    - Aggregates all engines into a single signed index in [-1,1].

LOGIC FLOW (SUMMARY):

- Pattern Sniper is the only "base" directional engine.
- All other engines vote with weights; a consensus index combines them.
- Volatility + Entropy adjust how aggressive we bet.
- If consensus weak or entropy too high → SKIP.
- If consensus strong and regime favorable → FIRE with scaled size.

STATUS: FULLY UNCOMPRESSED CODE

=============================================================================
"""

import math
import statistics
import logging
import time
import math as _math
from typing import Dict, List, Optional, Any

# =============================================================================
# [PART 1] CONFIGURATION & RISK SETTINGS
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [TITAN_LITE] %(message)s',
    datefmt='%H:%M:%S'
)


class GameConstants:
    """Core Game Definitions"""

    BIG = "BIG"
    SMALL = "SMALL"
    SKIP = "SKIP"

    # NOTE: Violet guard removed – no auto-skip on 0 / 5.
    VIOLET_NUMBERS: List[int] = []  # kept only for compatibility

    # Minimum rounds of history needed before we start predicting.
    MIN_HISTORY = 30


class RiskConfig:
    """Money Management Strategy"""

    # CONFIDENCE THRESHOLD
    REQ_CONFIDENCE = 0.80  # base

    # BETTING LIMITS
    BASE_RISK_PERCENT = 0.08  # 8% of Bankroll
    MIN_BET_AMOUNT = 10
    MAX_BET_AMOUNT = 50000

    # RECOVERY SYSTEM
    LEVEL_1_MULT = 1.0
    LEVEL_2_MULT = 2.2
    LEVEL_3_MULT = 5.0

    # STOP LOSS
    STOP_LOSS_STREAK = 3


# =============================================================================
# [PART 2] HELPER UTILITIES
# =============================================================================

def safe_float(value: Any) -> float:
    """Converts API data to a safe float number."""
    try:
        if value is None:
            return 4.5
        return float(value)
    except Exception:
        return 4.5


def get_outcome_from_number(n: Any) -> Optional[str]:
    """
    Decodes the number:
    0-4 -> SMALL
    5-9 -> BIG
    """
    val = int(safe_float(n))
    if 0 <= val <= 4:
        return GameConstants.SMALL
    if 5 <= val <= 9:
        return GameConstants.BIG
    return None


def get_history_string(history: List[Dict], length: int = 12) -> str:
    """
    Converts the last 'length' rounds into a string like 'BBSSB'.
    """
    out = ""
    for item in history[-length:]:
        res = get_outcome_from_number(item['actual_number'])
        if res == GameConstants.BIG:
            out += "B"
        elif res == GameConstants.SMALL:
            out += "S"
    return out


def extract_numbers(history: List[Dict], window: int) -> List[float]:
    """Safely extract last `window` numeric values."""
    return [safe_float(d.get('actual_number')) for d in history[-window:]]


def extract_binary_sequence(history: List[Dict], window: int) -> List[int]:
    """
    Convert to +1 (BIG) / -1 (SMALL) for the last `window` items.
    Used for trend / momentum style calculations.
    """
    seq: List[int] = []
    for item in history[-window:]:
        label = get_outcome_from_number(item['actual_number'])
        if label == GameConstants.BIG:
            seq.append(1)
        elif label == GameConstants.SMALL:
            seq.append(-1)
    return seq


def shannon_entropy(probs: List[float]) -> float:
    """Compute Shannon entropy (base 2) from list of probabilities."""
    s = 0.0
    for p in probs:
        if p > 0:
            s -= p * math.log2(p)
    return s


# =============================================================================
# [PART 3] ENGINE 1: PATTERN SNIPER (THE VISUAL BRAIN)
# =============================================================================

class PatternEngine:

    @staticmethod
    def scan(history: List[Dict]) -> Optional[Dict]:
        """
        Scans for specific visual patterns in BIG/SMALL.
        """
        full_seq = get_history_string(history, 24)
        if len(full_seq) < 10:
            return None

        # BIG patterns
        if full_seq.endswith("BSBSBS"):
            return {'prediction': GameConstants.BIG, 'weight': 1.2, 'source': 'Sniper-ZigZag'}

        if full_seq.endswith("BBSS"):
            return {'prediction': GameConstants.BIG, 'weight': 1.2, 'source': 'Sniper-2A2B'}

        if full_seq.endswith("SSBSS"):
            return {'prediction': GameConstants.BIG, 'weight': 1.3, 'source': 'Sniper-Mirror'}

        if full_seq.endswith("BBBB"):
            return {'prediction': GameConstants.BIG, 'weight': 1.4, 'source': 'Sniper-Dragon'}

        if full_seq.endswith("BBBSSS"):
            return {'prediction': GameConstants.BIG, 'weight': 1.1, 'source': 'Sniper-3-3-Block'}

        if full_seq.endswith("SSB"):
            return {'prediction': GameConstants.BIG, 'weight': 1.0, 'source': 'Sniper-Pair-Fix'}

        # SMALL patterns
        if full_seq.endswith("SBSBSB"):
            return {'prediction': GameConstants.SMALL, 'weight': 1.2, 'source': 'Sniper-ZigZag'}

        if full_seq.endswith("SSBB"):
            return {'prediction': GameConstants.SMALL, 'weight': 1.2, 'source': 'Sniper-2A2B'}

        if full_seq.endswith("BBSBB"):
            return {'prediction': GameConstants.SMALL, 'weight': 1.3, 'source': 'Sniper-Mirror'}

        if full_seq.endswith("SSSS"):
            return {'prediction': GameConstants.SMALL, 'weight': 1.4, 'source': 'Sniper-Dragon'}

        if full_seq.endswith("SSSBBB"):
            return {'prediction': GameConstants.SMALL, 'weight': 1.1, 'source': 'Sniper-3-3-Block'}

        if full_seq.endswith("BBS"):
            return {'prediction': GameConstants.SMALL, 'weight': 1.0, 'source': 'Sniper-Pair-Fix'}

        return None


# =============================================================================
# [PART 4] META ENGINES (YOUR 6 ORIGINAL NEW LOGICS)
# =============================================================================

# 4.1 MIRROR_PATTERN – Mirror number analysis
def engine_mirror_pattern(history: List[Dict]) -> Optional[Dict]:
    """
    Digit-level mirror symmetry feature on last 20 points.
    """
    numbers = extract_numbers(history, 20)
    if len(numbers) < 10:
        return None

    mirror_map = {
        0: 0, 1: 1, 2: 5, 5: 2, 6: 9, 9: 6, 8: 8,
        3: None, 4: None, 7: None
    }

    seq = [int(x) for x in numbers]
    mirrored: List[Optional[int]] = []
    for x in reversed(seq):
        d = x % 10
        md = mirror_map.get(d, None)
        if md is None:
            mirrored.append(None)
        else:
            mirrored.append(md)

    matches = 0
    valid = 0
    for a, b in zip(seq, mirrored):
        if b is None:
            continue
        valid += 1
        if a == b:
            matches += 1

    if valid == 0:
        return None

    score = matches / valid  # 0..1

    if score > 0.6:
        last_label = get_outcome_from_number(seq[-1])
        if last_label == GameConstants.BIG:
            pred = GameConstants.SMALL
        else:
            pred = GameConstants.BIG
        return {
            'prediction': pred,
            'weight': 0.7,
            'source': f'MirrorPattern({score:.2f})'
        }

    return None


# 4.2 CLUSTER_DOMINANCE – Cluster size analysis
def engine_cluster_dominance(history: List[Dict]) -> Optional[Dict]:
    """
    Cluster dominance on last 40 numbers using low/mid/high buckets.
    """
    numbers = extract_numbers(history, 40)
    if len(numbers) < 20:
        return None

    low = sum(1 for x in numbers if 0 <= x <= 3)
    mid = sum(1 for x in numbers if 4 <= x <= 6)
    high = sum(1 for x in numbers if 7 <= x <= 9)
    total = len(numbers)

    counts = {'LOW': low, 'MID': mid, 'HIGH': high}
    dominant_bucket = max(counts, key=counts.get)
    dom_frac = counts[dominant_bucket] / total

    if dom_frac < 0.45:
        return None

    last = numbers[-1]
    last_label = get_outcome_from_number(last)

    if dominant_bucket == 'LOW':
        dom_label = GameConstants.SMALL
    elif dominant_bucket == 'HIGH':
        dom_label = GameConstants.BIG
    else:
        dom_label = last_label

    return {
        'prediction': dom_label,
        'weight': 0.6 + 0.4 * (dom_frac - 0.45) / 0.55,
        'source': f'ClusterDom({dominant_bucket}:{dom_frac:.2f})'
    }


# 4.3 FREQUENCY_REVERSION – Statistical reversion
def engine_frequency_reversion(history: List[Dict]) -> Optional[Dict]:
    """
    Long-term vs short-term frequency imbalance on BIG/SMALL.
    """
    if len(history) < 60:
        return None

    long_hist = history[-200:] if len(history) >= 200 else history
    short_hist = history[-50:]

    def ratio(seq: List[Dict], label: str) -> float:
        if not seq:
            return 0.0
        cnt = sum(1 for d in seq if get_outcome_from_number(d['actual_number']) == label)
        return cnt / len(seq)

    pL_big = ratio(long_hist, GameConstants.BIG)
    pS_big = ratio(short_hist, GameConstants.BIG)

    d_big = pS_big - pL_big

    if abs(d_big) < 0.10:
        return None

    long_len = len(long_hist)
    if pL_big in (0, 1) or long_len == 0:
        return None
    var = pL_big * (1 - pL_big) / max(1, len(short_hist))
    if var == 0:
        return None
    z = d_big / math.sqrt(var)

    if z > 0:
        pred = GameConstants.SMALL  # BIG overplayed
    else:
        pred = GameConstants.BIG    # BIG underplayed

    weight = min(0.9, 0.4 + 0.1 * abs(z))

    return {
        'prediction': pred,
        'weight': weight,
        'source': f'FreqRevert(z={z:.2f})'
    }


# 4.4 SEQUENCE_TREND – Pattern continuation
def engine_sequence_trend(history: List[Dict]) -> Optional[Dict]:
    """
    Simple linear trend on last 20 numbers.
    """
    numbers = extract_numbers(history, 20)
    if len(numbers) < 10:
        return None

    n = len(numbers)
    xs = list(range(1, n + 1))
    mean_x = (n + 1) / 2.0
    mean_y = sum(numbers) / n

    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, numbers))
    den = sum((x - mean_x) ** 2 for x in xs)
    if den == 0:
        return None
    slope = num / den

    std_y = statistics.pstdev(numbers) if n > 1 else 0.0
    if std_y == 0:
        return None

    norm_slope = slope / max(1e-6, std_y)

    if abs(norm_slope) < 0.15:
        return None

    if norm_slope > 0:
        pred = GameConstants.BIG
    else:
        pred = GameConstants.SMALL

    weight = min(0.8, 0.4 + 0.2 * abs(norm_slope))

    return {
        'prediction': pred,
        'weight': weight,
        'source': f'SeqTrend({norm_slope:.2f})'
    }


# 4.5 MOMENTUM_ANALYSIS – Directional momentum
def engine_momentum(history: List[Dict]) -> Optional[Dict]:
    """
    Momentum using binary +1/-1 sequence and streak.
    """
    seq = extract_binary_sequence(history, 20)
    if len(seq) < 8:
        return None

    momentum = sum(seq[-8:])
    streak = 0
    last = None
    for s in reversed(seq):
        if last is None:
            last = s
            streak = 1
        elif s == last:
            streak += 1
        else:
            break

    score = momentum + streak * (1 if last is not None else 0)
    if score == 0:
        return None

    if score > 0:
        pred = GameConstants.BIG
    else:
        pred = GameConstants.SMALL

    weight = min(0.9, 0.3 + 0.1 * abs(score))

    return {
        'prediction': pred,
        'weight': weight,
        'source': f'Momentum(s={score})'
    }


# 4.6 PERIOD_PARITY – Mathematical properties
def engine_period_parity(history: List[Dict]) -> Optional[Dict]:
    """
    Combines parity bias, residue cycle, and small-period repetition.
    """
    numbers = extract_numbers(history, 40)
    if len(numbers) < 20:
        return None

    evens = sum(1 for x in numbers if int(x) % 2 == 0)
    odds = len(numbers) - evens
    parity_bias = (evens - odds) / len(numbers)

    if parity_bias > 0:
        bias_label = GameConstants.SMALL
    else:
        bias_label = GameConstants.BIG

    k = 3
    residues = [int(x) % k for x in numbers]
    counts = [residues.count(r) for r in range(k)]
    dom_r = max(range(k), key=lambda r: counts[r])
    dom_frac = counts[dom_r] / len(residues)

    best_P = None
    best_match = 0.0
    for P in range(2, 6 + 1):
        matches = 0
        total = 0
        for i in range(P, len(numbers)):
            total += 1
            if int(numbers[i]) == int(numbers[i - P]):
                matches += 1
        if total > 0:
            rate = matches / total
            if rate > best_match:
                best_match = rate
                best_P = P

    base_weight = 0.4 + 0.3 * abs(parity_bias)
    if dom_frac > 0.4:
        base_weight += 0.1
    if best_match > 0.4:
        base_weight += 0.1

    base_weight = min(0.85, base_weight)

    if base_weight < 0.5:
        return None

    return {
        'prediction': bias_label,
        'weight': base_weight,
        'source': f'PeriodParity(P={best_P},bias={parity_bias:.2f})'
    }


# =============================================================================
# [PART 5] NEW META ENGINES (5 EXTRA)
# =============================================================================

# 5.1 VOLATILITY_REGIME – calm / normal / explosive
def engine_volatility_regime(history: List[Dict]) -> Optional[Dict]:
    """
    Detect volatility regime using std over short vs long windows.
    """
    long_nums = extract_numbers(history, 100)
    short_nums = extract_numbers(history, 20)

    if len(long_nums) < 40 or len(short_nums) < 10:
        return None

    long_std = statistics.pstdev(long_nums) if len(long_nums) > 1 else 0.0
    short_std = statistics.pstdev(short_nums) if len(short_nums) > 1 else 0.0

    if long_std <= 0:
        return None

    ratio = short_std / max(long_std, 1e-6)

    if ratio < 0.7:
        regime = "CALM"
        weight = 0.8
    elif ratio > 1.5:
        regime = "EXPLOSIVE"
        weight = 0.9
    else:
        regime = "NORMAL"
        weight = 0.6

    return {
        'prediction': None,  # regime only; no BIG/SMALL
        'weight': weight,
        'regime': regime,
        'source': f'VolReg({regime},r={ratio:.2f})'
    }


# 5.2 STREAK_STRUCTURE – typical vs current streak
def engine_streak_structure(history: List[Dict]) -> Optional[Dict]:
    """
    Compare current streak length with typical streak distribution.
    """
    labels: List[str] = []
    for d in history:
        lab = get_outcome_from_number(d['actual_number'])
        if lab:
            labels.append(lab)

    if len(labels) < 40:
        return None

    # compute distribution of streak lengths up to some cap
    streak_lens: List[int] = []
    cur = labels[0]
    run = 1
    for lab in labels[1:]:
        if lab == cur:
            run += 1
        else:
            streak_lens.append(run)
            cur = lab
            run = 1
    streak_lens.append(run)

    if not streak_lens:
        return None

    avg_streak = sum(streak_lens) / len(streak_lens)
    max_streak = max(streak_lens)

    # current streak
    last_lab = labels[-1]
    cur_run = 1
    for lab in reversed(labels[:-1]):
        if lab == last_lab:
            cur_run += 1
        else:
            break

    # If current streak much longer than typical, fade the streak.
    # If shorter or around avg, follow it (continuation).
    if cur_run >= max(3, avg_streak + 1.5):
        # fade
        if last_lab == GameConstants.BIG:
            pred = GameConstants.SMALL
        else:
            pred = GameConstants.BIG
        mode = "FADE"
    else:
        # follow
        pred = last_lab
        mode = "FOLLOW"

    # Weight scales with how extreme the streak is
    extremeness = abs(cur_run - avg_streak) / max(1.0, avg_streak)
    weight = min(0.9, 0.4 + 0.3 * extremeness)

    return {
        'prediction': pred,
        'weight': weight,
        'source': f'StreakStruct({mode},cur={cur_run},avg={avg_streak:.1f})'
    }


# 5.3 LOCAL_GRAMMAR – token transitions BB/SS/BS/SB
def engine_local_grammar(history: List[Dict]) -> Optional[Dict]:
    """
    Build simple token transitions over pairs of BIG/SMALL.
    """
    seq = get_history_string(history, 40)
    if len(seq) < 10:
        return None

    tokens: List[str] = []
    for i in range(len(seq) - 1):
        tokens.append(seq[i:i+2])

    if len(tokens) < 6:
        return None

    # build transitions: token -> next token
    trans_counts: Dict[str, Dict[str, int]] = {}
    for i in range(len(tokens) - 1):
        t = tokens[i]
        nxt = tokens[i+1]
        if t not in trans_counts:
            trans_counts[t] = {}
        trans_counts[t][nxt] = trans_counts[t].get(nxt, 0) + 1

    last_token = tokens[-1]
    if last_token not in trans_counts:
        return None

    next_map = trans_counts[last_token]
    total = sum(next_map.values())
    if total == 0:
        return None

    # choose the most likely next token
    best_next = max(next_map, key=next_map.get)
    prob = next_map[best_next] / total

    # from next token, infer next single label (second char of token)
    next_char = best_next[-1]
    if next_char == "B":
        pred = GameConstants.BIG
    elif next_char == "S":
        pred = GameConstants.SMALL
    else:
        return None

    if prob < 0.55:
        return None

    weight = min(0.85, 0.4 + 0.4 * (prob - 0.55) / 0.45)

    return {
        'prediction': pred,
        'weight': weight,
        'source': f'LocalGrammar({last_token}->{best_next},p={prob:.2f})'
    }


# 5.4 ENTROPY_GUARD – randomness detector
def engine_entropy_guard(history: List[Dict]) -> Optional[Dict]:
    """
    Estimate randomness using entropy of BIG/SMALL sequence.
    Returns a guard signal with a 'risk_factor' telling how much to scale risk.
    """
    seq = get_history_string(history, 40)
    if len(seq) < 10:
        return None

    # probabilities of B and S
    total = len(seq)
    pB = seq.count("B") / total
    pS = seq.count("S") / total
    ent = shannon_entropy([pB, pS])  # max = 1 when pB=pS=0.5

    # treat entropy in [0,1]
    # low entropy (<0.6) -> structured, high risk allowed
    # medium (0.6-0.9) -> normal
    # high (>0.9) -> random, lower risk
    if ent < 0.6:
        risk_factor = 1.0
        mode = "STRUCTURED"
    elif ent > 0.9:
        risk_factor = 0.4
        mode = "CHAOS"
    else:
        risk_factor = 0.7
        mode = "MIXED"

    return {
        'prediction': None,
        'weight': 1.0,
        'risk_factor': risk_factor,
        'entropy': ent,
        'mode': mode,
        'source': f'EntropyGuard({mode},H={ent:.2f})'
    }


# 5.5 CONSENSUS_INDEX – aggregated vote
def compute_consensus_index(signals: List[Dict[str, Any]], base_side: Optional[str]) -> float:
    """
    Compute a signed consensus index in [-1,1]:
    positive -> BIG, negative -> SMALL, magnitude = strength.
    """
    big_score = 0.0
    small_score = 0.0

    for s in signals:
        pred = s.get('prediction', None)
        w = float(s.get('weight', 0.5))
        if pred == GameConstants.BIG:
            big_score += w
        elif pred == GameConstants.SMALL:
            small_score += w

    total = big_score + small_score
    if total <= 0:
        return 0.0

    # index in [-1,1]
    ci = (big_score - small_score) / total

    # optional small bias toward base side
    if base_side == GameConstants.BIG:
        ci = min(1.0, ci + 0.05)
    elif base_side == GameConstants.SMALL:
        ci = max(-1.0, ci - 0.05)

    return ci


# =============================================================================
# [PART 6] STATE MANAGER (MEMORY)
# =============================================================================

class GlobalStateManager:
    """Keeps track of Wins and Losses across rounds."""
    def __init__(self):
        self.loss_streak = 0
        self.last_bet_result = "NONE"


state_manager = GlobalStateManager()


# =============================================================================
# [PART 7] MAIN EXECUTION CONTROLLER
# =============================================================================

def ultraAIPredict(
    history: List[Dict],
    current_bankroll: float = 10000.0,
    last_result: Optional[str] = None
) -> Dict:
    """
    Main decision brain.
    """

    if len(history) < GameConstants.MIN_HISTORY:
        return {
            'finalDecision': "SKIP",
            'confidence': 0,
            'positionsize': 0,
            'level': "WARMUP",
            'reason': "Not enough history",
            'topsignals': []
        }

    # --- STEP 1: UPDATE STREAK ---
    if last_result and last_result != "SKIP":
        try:
            actual = get_outcome_from_number(history[-1]['actual_number'])
            if last_result == actual:
                state_manager.loss_streak = 0
            else:
                state_manager.loss_streak += 1
        except Exception:
            pass

    streak = state_manager.loss_streak

    # --- STEP 2: RUN ENGINES ---

    # Base visual engine
    s_patt = PatternEngine.scan(history)

    # Original 6 meta engines
    meta_engines_core = [
        engine_mirror_pattern,
        engine_cluster_dominance,
        engine_frequency_reversion,
        engine_sequence_trend,
        engine_momentum,
        engine_period_parity,
    ]

    meta_signals: List[Dict[str, Any]] = []
    for eng in meta_engines_core:
        try:
            s = eng(history)
            if s:
                meta_signals.append(s)
        except Exception:
            continue

    # New regime/guard engines
    regime_signal = engine_volatility_regime(history)
    streak_signal = engine_streak_structure(history)
    grammar_signal = engine_local_grammar(history)
    entropy_signal = engine_entropy_guard(history)

    signals: List[Dict[str, Any]] = []
    if s_patt:
        signals.append({
            'prediction': s_patt['prediction'],
            'weight': s_patt.get('weight', 1.0),
            'source': s_patt['source']
        })

    signals.extend(meta_signals)

    # add new ones that have predictions
    if streak_signal:
        signals.append(streak_signal)
    if grammar_signal:
        signals.append(grammar_signal)

    # --- STEP 3: HANDLE NO-SIGNAL CASE ---
    if not signals:
        return {
            'finalDecision': "SKIP",
            'confidence': 0,
            'positionsize': 0,
            'level': "WAIT",
            'reason': "No clear signal from any engine",
            'topsignals': []
        }

    # --- STEP 4: BASE CANDIDATE (PATTERN OR META) ---
    base_candidate: Optional[str] = None
    base_conf = 0.0
    base_reason = ""

    if s_patt:
        base_candidate = s_patt['prediction']
        base_conf = 0.85
        base_reason = f"Visual: {s_patt['source']}"
    else:
        # meta-only candidate: majority vote among all meta engines
        vote_score: Dict[str, float] = {}
        for s in signals:
            pred = s.get('prediction', None)
            if pred in (GameConstants.BIG, GameConstants.SMALL):
                w = float(s.get('weight', 0.5))
                vote_score[pred] = vote_score.get(pred, 0.0) + w

        if not vote_score:
            return {
                'finalDecision': "SKIP",
                'confidence': 0,
                'positionsize': 0,
                'level': "WAIT",
                'reason': "Meta engines inconclusive",
                'topsignals': [s['source'] for s in signals]
            }

        base_candidate = max(vote_score, key=vote_score.get)
        base_conf = min(0.9, 0.6 + vote_score[base_candidate] / 10.0)
        base_reason = f"Meta Majority (votes={vote_score[base_candidate]:.2f})"

    # --- STEP 5: CONSENSUS INDEX + BOOST ---
    consensus_index = compute_consensus_index(signals, base_candidate)
    # consensus_index in [-1,1], magnitude = strength
    ci_mag = abs(consensus_index)

    # If consensus sign contradicts base_candidate strongly, SKIP (safety)
    if (consensus_index > 0 and base_candidate == GameConstants.SMALL and ci_mag > 0.6) or \
       (consensus_index < 0 and base_candidate == GameConstants.BIG and ci_mag > 0.6):
        return {
            'finalDecision': "SKIP",
            'confidence': 0,
            'positionsize': 0,
            'level': "CONFLICT",
            'reason': "Consensus strongly opposes base side",
            'topsignals': [s['source'] for s in signals]
        }

    # align direction with base_candidate; consensus only boosts confidence
    base_boost = 0.15 * ci_mag
    final_conf = max(0.0, min(0.99, base_conf + base_boost))

    reason_detail = base_reason + f" | CI={consensus_index:.2f}"

    # --- STEP 6: APPLY REGIME + ENTROPY GUARD TO CONFIDENCE & RISK ---

    risk_factor = 1.0
    regime_text = ""
    if regime_signal:
        regime = regime_signal.get('regime', 'NORMAL')
        regime_text = regime_signal['source']
        if regime == "CALM":
            risk_factor *= 1.0
            final_conf += 0.03
        elif regime == "NORMAL":
            risk_factor *= 0.9
        elif regime == "EXPLOSIVE":
            risk_factor *= 0.7
            final_conf -= 0.05

    entropy_text = ""
    if entropy_signal:
        risk_factor *= float(entropy_signal.get('risk_factor', 1.0))
        entropy_text = entropy_signal['source']

    final_conf = max(0.0, min(0.99, final_conf))

    if regime_text:
        reason_detail += f" | {regime_text}"
    if entropy_text:
        reason_detail += f" | {entropy_text}"

    # Hard floor: if consensus extremely weak and entropy chaotic, SKIP.
    if ci_mag < 0.15 and entropy_signal and entropy_signal.get('mode') == "CHAOS":
        return {
            'finalDecision': "SKIP",
            'confidence': final_conf,
            'positionsize': 0,
            'level': "WAIT",
            'reason': f"Consensus weak (|CI|={ci_mag:.2f}) in CHAOS regime",
            'topsignals': [s['source'] for s in signals]
        }

    # --- STEP 7: RISK MANAGEMENT & STOP LOSS ---

    req_conf = RiskConfig.REQ_CONFIDENCE
    if streak == 1:
        req_conf = 0.85
    elif streak == 2:
        req_conf = 0.92
    elif streak >= RiskConfig.STOP_LOSS_STREAK:
        return {
            'finalDecision': "SKIP",
            'confidence': 0,
            'positionsize': 0,
            'level': "STOP LOSS",
            'reason': "Max Streak Reached (Cooldown)",
            'topsignals': [s['source'] for s in signals]
        }

    active_sources = [s['source'] for s in signals]

    if final_conf >= req_conf:
        mult = RiskConfig.LEVEL_1_MULT
        if streak == 1:
            mult = RiskConfig.LEVEL_2_MULT
        elif streak == 2:
            mult = RiskConfig.LEVEL_3_MULT

        base_stake = max(current_bankroll * RiskConfig.BASE_RISK_PERCENT, RiskConfig.MIN_BET_AMOUNT)
        stake = base_stake * mult * risk_factor

        stake = min(stake, RiskConfig.MAX_BET_AMOUNT)
        if stake < RiskConfig.MIN_BET_AMOUNT * 0.5:
            return {
                'finalDecision': "SKIP",
                'confidence': final_conf,
                'positionsize': 0,
                'level': "WAIT",
                'reason': "Risk guard reduced stake below minimum",
                'topsignals': active_sources
            }

        return {
            'finalDecision': base_candidate,
            'confidence': final_conf,
            'positionsize': int(stake),
            'level': f"L{streak + 1}",
            'reason': reason_detail,
            'topsignals': active_sources
        }

    return {
        'finalDecision': "SKIP",
        'confidence': final_conf,
        'positionsize': 0,
        'level': "WAIT",
        'reason': f"Low Confidence ({final_conf:.2f} < {req_conf:.2f})",
        'topsignals': active_sources
    }


# =============================================================================
# [PART 8] SYSTEM BOOT
# =============================================================================

if __name__ == "__main__":
    print("TITAN LITE V3 (PATTERN + 11 META ENGINES) ONLINE.")
    print("Strategy: Visual + 6 Core Meta + Volatility + Streak + Grammar + Entropy + Consensus.")
