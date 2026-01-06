#!/usr/bin/env python3

"""
=============================================================================
TITAN V11 – ACTION EDITION (MINIMUM SKIP)
=============================================================================

PHILOSOPHY: "Better to have a small bet in the game than to watch and wait."

CHANGES:
1. THRESHOLD: Lowered to 0.40. almost any signal triggers a bet.
2. WEAK STATUS: If confidence is 40%-50%, we place a HALF BET (Safe Entry).
3. PENALTIES: Math penalties reduced by 50% so confidence stays higher.

STATUS TIERS:
- SKIP (< 40%)       -> Only for absolute chaos.
- WEAK (40% - 50%)   -> HALF BET (Action Mode).
- GOOD (50% - 65%)   -> NORMAL BET.
- HIGH (65% - 80%)   -> NORMAL BET.
- SURE (> 80%)       -> NORMAL BET.

=============================================================================
"""

import math
import statistics
import logging
from typing import Dict, List, Optional, Any, Tuple

# =============================================================================
# [PART 1] CONFIGURATION
# =============================================================================

logging.basicConfig(level=logging.INFO, format='[TITAN_V11] %(message)s')

class GameConstants:
    BIG = "BIG"
    SMALL = "SMALL"
    SKIP = "SKIP"
    MIN_HISTORY = 10

class RiskConfig:
    # AGGRESSIVE: We bet on almost anything above 40%
    REQ_CONFIDENCE = 0.40 
    
    BASE_RISK_PERCENT = 0.08 
    MIN_BET_AMOUNT = 10
    MAX_BET_AMOUNT = 50000
    
    STOP_LOSS_STREAK = 6 # Extended for high activity
    LEVEL_1_MULT = 1.0
    LEVEL_2_MULT = 2.2
    LEVEL_3_MULT = 5.0
    LEVEL_4_MULT = 11.0

# =============================================================================
# [PART 2] UTILITIES & STATUS
# =============================================================================

def safe_float(value: Any) -> float:
    try:
        return float(value) if value is not None else 4.5
    except:
        return 4.5

def get_outcome(n: Any) -> str:
    val = int(safe_float(n))
    return GameConstants.SMALL if 0 <= val <= 4 else GameConstants.BIG

def get_history_string(history: List[Dict], length: int) -> str:
    out = ""
    for item in history[-length:]:
        out += "B" if get_outcome(item['actual_number']) == GameConstants.BIG else "S"
    return out

def extract_numbers(history: List[Dict], window: int) -> List[float]:
    return [safe_float(d['actual_number']) for d in history[-window:]]

def get_status_label(conf: float) -> str:
    if conf < 0.40: return "SKIP" # Absolute Trash
    if conf < 0.50: return "WEAK" # Action Mode
    if conf < 0.65: return "GOOD"
    if conf < 0.80: return "HIGH"
    return "SURE"

# =============================================================================
# [PART 3] MATH CORE (TUNED FOR ACTION)
# =============================================================================

class MathBrain:
    @staticmethod
    def calculate_volatility(history: List[Dict], window: int = 20) -> float:
        if len(history) < window: return 0.0
        seq = [1 if get_outcome(x['actual_number']) == GameConstants.BIG else 0 for x in history[-window:]]
        if len(seq) < 2: return 0.0
        return statistics.pstdev(seq)

    @staticmethod
    def detect_trap_pattern(history: List[Dict], window: int = 12) -> Tuple[bool, float]:
        seq = get_history_string(history, window)
        if len(seq) < 6: return False, 0.0
        alternates = 0
        for i in range(len(seq) - 1):
            if seq[i] != seq[i+1]: alternates += 1
        trap_index = alternates / max(1, (len(seq) - 1))
        return (trap_index > 0.85), trap_index # Loosened trap definition

    @staticmethod
    def calculate_mirror_score(history: List[Dict], length: int = 6) -> float:
        seq = get_history_string(history, length * 2)
        if len(seq) < length * 2: return 0.0
        recent = seq[-length:]
        older = seq[-length*2:-length]
        matches = sum(1 for x, y in zip(recent, reversed(older)) if x == y)
        return matches / length

    @staticmethod
    def shannon_entropy(history: List[Dict], window: int = 40) -> float:
        seq = get_history_string(history, window)
        if len(seq) < 10: return 0.0
        pB = seq.count("B") / len(seq)
        pS = seq.count("S") / len(seq)
        s = 0.0
        for p in [pB, pS]:
            if p > 0: s -= p * math.log2(p)
        return s

    @staticmethod
    def penalty_for_disagreement(confidences: List[float]) -> float:
        if not confidences or len(confidences) < 2: return 0.0
        sd = statistics.pstdev(confidences)
        # REDUCED PENALTY: 0.4 -> 0.2
        # We don't punish disagreement as hard anymore.
        return max(0.0, sd * 0.2) 

# =============================================================================
# [PART 4] THE ENGINES
# =============================================================================

def engine_guard_sanity(history: List[Dict]) -> Dict:
    ent = MathBrain.shannon_entropy(history)
    vol = MathBrain.calculate_volatility(history)
    
    # Only veto if entropy is practically 1.0 (Pure Randomness)
    if ent > 0.998: return {'status': 'VETO', 'risk': 0.0, 'reason': 'MAX_CHAOS'}
    
    risk = 1.0
    if vol > 0.49: risk *= 0.8 # Less punishment
    if ent > 0.98: risk *= 0.8 # Less punishment
    return {'status': 'PASS', 'risk': risk, 'details': f'E:{ent:.2f}'}

class GeneralPatternSniper:
    @staticmethod
    def scan(history: List[Dict]) -> Optional[Dict]:
        seq = get_history_string(history, 20)
        # High Conf
        if seq.endswith("BBSS"): return {'pred': GameConstants.BIG, 'conf': 0.85, 'name': '2A2B'}
        if seq.endswith("SSBB"): return {'pred': GameConstants.SMALL, 'conf': 0.85, 'name': '2A2B'}
        if seq.endswith("BSBSBS"): return {'pred': GameConstants.BIG, 'conf': 0.90, 'name': 'ZigZag'}
        # Med Conf
        if seq.endswith("BBB"): return {'pred': GameConstants.BIG, 'conf': 0.75, 'name': 'Dragon'}
        if seq.endswith("SSS"): return {'pred': GameConstants.SMALL, 'conf': 0.75, 'name': 'Dragon'}
        # Low Conf (High Frequency)
        if seq.endswith("BB"): return {'pred': GameConstants.BIG, 'conf': 0.65, 'name': 'Trend2'}
        if seq.endswith("SS"): return {'pred': GameConstants.SMALL, 'conf': 0.65, 'name': 'Trend2'}
        if seq.endswith("BSB"): return {'pred': GameConstants.SMALL, 'conf': 0.60, 'name': 'PingPong'}
        if seq.endswith("SBS"): return {'pred': GameConstants.BIG, 'conf': 0.60, 'name': 'PingPong'}
        return None

def general_momentum(history: List[Dict]) -> Optional[Dict]:
    nums = extract_numbers(history, 15)
    if not nums: return None
    force = 0
    for n in nums: force += 1 if get_outcome(n) == GameConstants.BIG else -1
    # ACTION TWEAK: Even a force of 1 is enough to bet now
    if abs(force) < 1: return None
    pred = GameConstants.BIG if force > 0 else GameConstants.SMALL
    return {'pred': pred, 'conf': 0.60 + (abs(force)/20)}

def advisor_micro_trend(history: List[Dict]) -> Optional[Dict]:
    """Last 4 rounds."""
    seq = get_history_string(history, 4)
    if not seq: return None
    if seq.count("B") >= 3: return {'pred': GameConstants.BIG, 'conf': 0.60}
    if seq.count("S") >= 3: return {'pred': GameConstants.SMALL, 'conf': 0.60}
    # ACTION TWEAK: Even 2 in a row is a micro trend
    if seq.endswith("BB"): return {'pred': GameConstants.BIG, 'conf': 0.55}
    if seq.endswith("SS"): return {'pred': GameConstants.SMALL, 'conf': 0.55}
    return None

def advisor_reversion(history: List[Dict]) -> Optional[Dict]:
    seq = get_history_string(history, 40)
    if not seq: return None
    b_rate = seq.count("B") / len(seq)
    if b_rate > 0.70: return {'pred': GameConstants.SMALL, 'conf': 0.70}
    if b_rate < 0.30: return {'pred': GameConstants.BIG, 'conf': 0.70}
    return None

def advisor_cluster(history: List[Dict]) -> Optional[Dict]:
    nums = extract_numbers(history, 20)
    if not nums: return None
    low = sum(1 for x in nums if 0<=x<=3)
    high = sum(1 for x in nums if 7<=x<=9)
    if high > low + 3: return {'pred': GameConstants.BIG, 'conf': 0.6}
    if low > high + 3: return {'pred': GameConstants.SMALL, 'conf': 0.6}
    return None

# =============================================================================
# [PART 5] STATE
# =============================================================================

class GlobalStateManager:
    def __init__(self):
        self.loss_streak = 0
        self.engine_trust = {
            'Sniper': 1.0, 'Momentum': 1.0, 'Micro': 1.0,
            'Reversion': 1.0, 'Cluster': 1.0
        }
        self.last_round_predictions = {}

    def update_trust(self, actual_outcome: str):
        for name, pred in self.last_round_predictions.items():
            if not pred: continue
            if pred == actual_outcome:
                self.engine_trust[name] = min(2.0, self.engine_trust[name] + 0.1)
            else:
                self.engine_trust[name] = max(0.5, self.engine_trust[name] - 0.1) # Less punishment
        self.last_round_predictions = {}

state_manager = GlobalStateManager()

# =============================================================================
# [PART 6] LOGIC
# =============================================================================

def run_deep_scan(history: List[Dict]) -> Dict:
    scores = {GameConstants.BIG: 0.0, GameConstants.SMALL: 0.0}
    total_weight = 0.0
    log_details = []
    current_preds = {}
    engine_confidences = []

    # 1. GUARDS
    guard = engine_guard_sanity(history)
    if guard['status'] == 'VETO':
        return {'decision': 'SKIP', 'reason': guard['reason'], 'confidence': 0, 'details': []}

    # 2. ENGINES (Boosted Weights for Action)
    council = [
        ('Sniper', GeneralPatternSniper.scan, 2.5),
        ('Momentum', general_momentum, 2.0),
        ('Micro', advisor_micro_trend, 1.5),
        ('Reversion', advisor_reversion, 1.0),
        ('Cluster', advisor_cluster, 1.0)
    ]

    for name, func, base_w in council:
        try:
            res = func(history)
            if res:
                trust = state_manager.engine_trust.get(name, 1.0)
                final_w = base_w * trust
                scores[res['pred']] += final_w * res['conf']
                total_weight += final_w
                current_preds[name] = res['pred']
                engine_confidences.append(res['conf'])
                log_details.append(f"{name[:3]}[{res['pred'][0]}]")
            else:
                current_preds[name] = None
        except:
            current_preds[name] = None

    state_manager.last_round_predictions = current_preds

    # 3. CONSENSUS
    if total_weight == 0:
        return {'decision': 'SKIP', 'reason': 'No Signal', 'confidence': 0, 'details': []}
        
    big_s = scores[GameConstants.BIG]
    small_s = scores[GameConstants.SMALL]
    
    if big_s > small_s:
        final_decision = GameConstants.BIG
        winner_score = big_s
        loser_score = small_s
    else:
        final_decision = GameConstants.SMALL
        winner_score = small_s
        loser_score = big_s
        
    raw_conf = (winner_score - loser_score) / total_weight
    
    # 4. PENALTIES (Weakened)
    disagreement_penalty = MathBrain.penalty_for_disagreement(engine_confidences)
    final_conf = raw_conf - disagreement_penalty
    final_conf *= guard['risk']
    
    # 5. TRAP LOGIC
    is_trap, trap_idx = MathBrain.detect_trap_pattern(history)
    reason_extra = ""
    if is_trap and final_conf < 0.8:
        final_decision = GameConstants.SMALL if final_decision == GameConstants.BIG else GameConstants.BIG
        reason_extra = "[TRAP-INVERT]"
    
    return {
        'decision': final_decision,
        'confidence': max(0.0, final_conf),
        'details': log_details,
        'risk_mult': guard['risk'],
        'extra': reason_extra
    }

def ultraAIPredict(history: List[Dict], current_bankroll: float, last_result: Optional[str] = None) -> Dict:
    
    # 1. Update State
    if last_result and last_result != "SKIP":
        try:
            actual = get_outcome(history[-1]['actual_number'])
            state_manager.update_trust(actual)
            if last_result == actual:
                state_manager.loss_streak = 0
            else:
                state_manager.loss_streak += 1
        except:
            pass
            
    streak = state_manager.loss_streak
    
    # 2. EMERGENCY CORRECTOR (2-LOSS)
    if streak >= 2 and streak < RiskConfig.STOP_LOSS_STREAK:
        last_out = get_outcome(history[-1]['actual_number'])
        emerg_pred = GameConstants.SMALL if last_out == GameConstants.BIG else GameConstants.BIG
        mult = RiskConfig.LEVEL_3_MULT if streak == 2 else RiskConfig.LEVEL_4_MULT
        stake = RiskConfig.BASE_RISK_PERCENT * current_bankroll * mult
        stake = max(RiskConfig.MIN_BET_AMOUNT, min(stake, RiskConfig.MAX_BET_AMOUNT))
        
        return {
            'finalDecision': emerg_pred,
            'confidence': 0.99,
            'positionsize': int(stake),
            'level': "EMERGENCY",
            'reason': "STATUS: SURE (Emergency)",
            'topsignals': ["FORCE_ANTI"]
        }

    # 3. Stop Loss
    if streak >= RiskConfig.STOP_LOSS_STREAK:
        return {'finalDecision': "SKIP", 'confidence': 0, 'positionsize': 0, 
                'level': "STOP_LOSS", 'reason': "STATUS: COOLING DOWN", 'topsignals': []}

    # 4. Deep Scan
    scan = run_deep_scan(history)
    conf = scan['confidence']
    status_label = get_status_label(conf)
    
    # --- VISUAL DEBUGGER ---
    recents = get_history_string(history, 15)
    print("\n" + "="*50)
    print(f" TITAN V11 ACTION | HISTORY: ...{recents}") 
    print(f" ENGINES : {scan['details']}")
    print(f" STATUS  : [{status_label}] ({conf:.2f}) {scan.get('extra','')}")
    print("="*50 + "\n")
    
    if scan['decision'] == 'SKIP' or status_label == "SKIP":
         return {
            'finalDecision': "SKIP",
            'confidence': conf,
            'positionsize': 0,
            'level': "WAIT",
            'reason': f"STATUS: {status_label}",
            'topsignals': scan.get('details', [])
        }
    
    # 5. EXECUTE BET (Logic for WEAK vs GOOD)
    mult = RiskConfig.LEVEL_1_MULT
    if streak == 1: mult = RiskConfig.LEVEL_2_MULT
    
    stake = RiskConfig.BASE_RISK_PERCENT * current_bankroll * mult
    stake = stake * scan['risk_mult']
    
    # If Status is WEAK, we cut the bet size in half for safety
    if status_label == "WEAK":
        stake = stake * 0.5
        
    stake = max(RiskConfig.MIN_BET_AMOUNT, min(stake, RiskConfig.MAX_BET_AMOUNT))
    
    return {
        'finalDecision': scan['decision'],
        'confidence': conf,
        'positionsize': int(stake),
        'level': f"L{streak+1}",
        'reason': f"STATUS: {status_label}",
        'topsignals': scan['details']
    }

if __name__ == "__main__":
    print("TITAN V11 (ACTION EDITION) LOADED.")
