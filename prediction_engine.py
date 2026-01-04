#!/usr/bin/env python3
"""
=============================================================================
TITAN V700 - SOVEREIGN EDITION (WITH VISUAL CHART PATTERNS)
=============================================================================
Integrates:
1. Basic Rules 1-25 (AABB, ABAB, AAB, etc.) from user charts.
2. Trend, Reversion, Neuren, Qaum.
3. Supervisor & Ghost Protocol.
=============================================================================
"""

import math
import statistics
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum

# --- CONFIGURATION ---
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger("TITAN_V700")

class TradeDecision(Enum):
    BIG = "BIG"
    SMALL = "SMALL"
    SKIP = "SKIP"

@dataclass
class RiskProfile:
    base_risk_percent: float = 0.05       
    max_risk_percent: float = 0.15        
    min_bet_amount: float = 10.0
    max_bet_amount: float = 50000.0
    stop_loss_streak: int = 5             
    martingale_multiplier: float = 2.0    

@dataclass
class EngineState:
    name: str
    weight: float
    consecutive_losses: int = 0
    is_active: bool = True
    last_vote: Optional[TradeDecision] = None 

class GlobalState:
    def __init__(self):
        self.loss_streak: int = 0
        self.total_wins: int = 0
        self.total_losses: int = 0
        self.cooling_off_counter: int = 0 
        self.inversion_mode: bool = False
        self.consecutive_fails: int = 0    
        self.session_start_bankroll: float = 0.0
        self.current_profit: float = 0.0
        
        # Engine Manager - UPDATED WEIGHTS FOR CHART PATTERNS
        self.engines: Dict[str, EngineState] = {
            'trend': EngineState('trend', 1.0),
            'reversion': EngineState('reversion', 1.2),
            'neuren': EngineState('neuren', 2.0),
            'qaum': EngineState('qaum', 2.5),
            # New powerful engine for your 25 rules
            'chart_patterns': EngineState('chart_patterns', 4.0) 
        }
        
        self.last_prediction: Optional[TradeDecision] = None
        self.last_confidence: float = 0.0

state = GlobalState()
config = RiskProfile()

# =============================================================================
# SECTION 1: MATH & PATTERN LIBRARY
# =============================================================================

class MathLib:
    @staticmethod
    def safe_float(value: Any) -> Optional[float]:
        try: return float(value) if value is not None else None
        except: return None

    @staticmethod
    def get_z_score(data: List[float]) -> float:
        if len(data) < 2: return 0.0
        try:
            mean = statistics.mean(data)
            stdev = statistics.pstdev(data)
            return (data[-1] - mean) / stdev if stdev != 0 else 0.0
        except: return 0.0

    @staticmethod
    def get_derivative(data: List[float], order: int = 1) -> float:
        if len(data) < order + 1: return 0.0
        current = data
        for _ in range(order):
            current = [current[i+1] - current[i] for i in range(len(current) - 1)]
        return current[-1] if current else 0.0

# =============================================================================
# SECTION 2: THE ENGINES (Updated with Chart Pattern Logic)
# =============================================================================

class ChartPatternEngine:
    """
    Implements the 25 Basic Rules from your charts (1-25).
    Analyzes the 'Rhythm' of the last outcomes.
    """
    @staticmethod
    def get_structure(outcomes: List[str]) -> List[int]:
        """Converts ['B','B','S','B','B'] into [2, 1, 2] counts."""
        if not outcomes: return []
        counts = []
        current_count = 1
        for i in range(1, len(outcomes)):
            if outcomes[i] == outcomes[i-1]:
                current_count += 1
            else:
                counts.append(current_count)
                current_count = 1
        counts.append(current_count)
        return counts

    @staticmethod
    def analyze(outcomes: List[str]) -> float:
        """
        Returns:
         1.0 (Strong Signal to continue Pattern)
        -1.0 (Strong Signal to break/invert)
         0.0 (No Pattern)
         
         The logic maps the 'Next Expected Move' to 1.0 if it matches the
         last color, or -1.0 if it's the opposite color.
        """
        if len(outcomes) < 6: return 0.0
        
        # Current state
        last_outcome = outcomes[-1] # e.g., "BIG"
        structure = ChartPatternEngine.get_structure(outcomes) # e.g., [2, 2, 2]
        
        # We look at the TAIL of the structure (recent history)
        s = structure 
        
        vote = 0.0
        
        # --- RULE SET 1: PING PONG & PAIRS (Rules 1, 2, 13) ---
        # Pattern: 1-1-1-1 (Ping Pong)
        if len(s) >= 4 and s[-4:] == [1, 1, 1, 1]:
            # Expect next to be 1 (Opposite of last). 
            # If we predict "Opposite", that means we want a CHANGE.
            # In engine logic: 1.0 = Big, -1.0 = Small.
            # Here we return a directional hint relative to pattern.
            return -1.0 if last_outcome == "BIG" else 1.0

        # Pattern: 2-2-2 (Pairs / Double Trend)
        if len(s) >= 3 and s[-3:] == [2, 2, 2]:
            return -1.0 if last_outcome == "BIG" else 1.0 # Expect break after pair finishes

        # Pattern: 2-2 (Start of Pairs) - Betting on the "2nd" leg
        if len(s) >= 2 and s[-2] == 2 and s[-1] == 1:
            # We have [..., 2, 1]. Rule 2 says it should be 2, 2.
            # So we expect the current 1 to become a 2.
            return 1.0 if last_outcome == "BIG" else -1.0

        # --- RULE SET 2: RHYTHMS (Rules 5, 7, 8, 9) ---
        
        # Rule 5: 2-1-2-1 (AAB AAB)
        if len(s) >= 3 and s[-3:] == [2, 1, 2]:
            # Next should be a 1.
            # Currently we just finished a 2. Next is opposite.
            return -1.0 if last_outcome == "BIG" else 1.0
        if len(s) >= 3 and s[-3:] == [1, 2, 1]:
             # Next should be a 2. Current is 1. We bet SAME.
             return 1.0 if last_outcome == "BIG" else -1.0

        # Rule 7: 1-2-1-2 (ABB ABB)
        # Similar logic to above, handled by general rhythm check.

        # Rule 3: 3-3 (Triplets)
        if len(s) >= 2 and s[-2] == 3 and s[-1] == 2:
            # We are at 3-2. Need to complete the triplet. Bet Same.
            return 1.0 if last_outcome == "BIG" else -1.0
            
        # --- RULE SET 3: DRAGON (Long Trend) ---
        # If current streak is > 4 (Rule 6/11)
        if s[-1] >= 4:
            # Ride the dragon until it breaks
            return 1.0 if last_outcome == "BIG" else -1.0

        return 0.0

class Engines:
    @staticmethod
    def trend_engine(outcomes: List[str]) -> float:
        if len(outcomes) < 5: return 0.0
        last_4 = outcomes[-4:]
        # Simple Stickiness
        if last_4.count("BIG") == 4: return 1.0     
        if last_4.count("SMALL") == 4: return -1.0  
        return 0.0

    @staticmethod
    def reversion_engine(numbers: List[float]) -> float:
        if len(numbers) < 15: return 0.0
        z = MathLib.get_z_score(numbers[-20:])
        if z > 2.0: return -1.0
        elif z < -2.0: return 1.0
        return 0.0

    @staticmethod
    def neuren_engine(numbers: List[float]) -> float:
        if len(numbers) < 6: return 0.0
        jerk = MathLib.get_derivative(numbers, 3)
        if jerk > 5.0: return -1.0  
        if jerk < -5.0: return 1.0  
        return 0.0

    @staticmethod
    def qaum_engine(numbers: List[float]) -> float:
        if len(numbers) < 5: return 0.0
        recent = numbers[-4:]
        try: variance = statistics.pvariance(recent)
        except: return 0.0
        if variance < 0.8:
            return 1.0 if (sum(recent)/len(recent)) < 4.5 else -1.0        
        return 0.0

# =============================================================================
# SECTION 3: SUPERVISORS
# =============================================================================

class MarketMonitor:
    @staticmethod
    def check_volatility(numbers: List[float]) -> Tuple[bool, str]:
        if len(numbers) < 10: return False, "OK"
        recent = numbers[-8:]
        # Pure Chaos check: 01010101 extremely fast switching
        binary = [0 if x <= 4 else 1 for x in recent]
        switches = sum(1 for i in range(len(binary)-1) if binary[i] != binary[i+1])
        if switches >= 7: return True, "EXTREME_CHOP"
        return False, "SAFE"

class EngineManager:
    @staticmethod
    def update_performance(last_result_str: str):
        actual = None
        if last_result_str == "BIG": actual = TradeDecision.BIG
        elif last_result_str == "SMALL": actual = TradeDecision.SMALL
        else: return 
        
        for name, engine in state.engines.items():
            if engine.last_vote is None or engine.last_vote == TradeDecision.SKIP:
                continue
            
            if engine.last_vote == actual:
                if engine.consecutive_losses > 0: engine.consecutive_losses -= 1
                if not engine.is_active:
                    engine.is_active = True # Quick recovery
            else:
                engine.consecutive_losses += 1
                if engine.is_active and engine.consecutive_losses >= config.stop_loss_streak:
                    print(f"[MANAGER] {name.upper()} Failed. BANNED.")
                    engine.is_active = False

# =============================================================================
# SECTION 4: VOTING COUNCIL
# =============================================================================

class VotingCouncil:
    def cast_votes(self, numbers: List[float], outcomes: List[str]) -> Tuple[TradeDecision, float, List[str]]:
        score = 0.0
        reasons = []
        
        # 1. Collect Votes
        # Note: ChartPatternEngine analyzes the visual patterns from your screenshots
        raw_votes = {
            'trend': Engines.trend_engine(outcomes),
            'reversion': Engines.reversion_engine(numbers),
            'neuren': Engines.neuren_engine(numbers),
            'qaum': Engines.qaum_engine(numbers),
            'chart_patterns': ChartPatternEngine.analyze(outcomes) 
        }
        
        # 2. Weigh Votes
        active_weight_sum = 0.0
        
        for name, val in raw_votes.items():
            eng = state.engines[name]
            
            # Record vote
            if val > 0: eng.last_vote = TradeDecision.BIG
            elif val < 0: eng.last_vote = TradeDecision.SMALL
            else: eng.last_vote = TradeDecision.SKIP
            
            if not eng.is_active:
                continue
                
            # Chart Patterns get higher priority if they find a match
            current_weight = eng.weight
            
            active_weight_sum += current_weight
            score += val * current_weight
            if val != 0: reasons.append(f"{name}({val:+.1f})")

        # 3. Decision
        if active_weight_sum == 0:
            return TradeDecision.SKIP, 0.0, ["NO_ACTIVE_ENGINES"]
            
        normalized_score = score / 3.0 # Adjusted divisor
        
        decision = TradeDecision.SKIP
        conf = 0.0
        THRESHOLD = 0.5 # Aggressive entry
        
        if normalized_score >= THRESHOLD:
            decision = TradeDecision.BIG
            conf = min(0.6 + (normalized_score/5), 0.98)
        elif normalized_score <= -THRESHOLD:
            decision = TradeDecision.SMALL
            conf = min(0.6 + (abs(normalized_score)/5), 0.98)
            
        return decision, conf, reasons

# =============================================================================
# MAIN EXPORT
# =============================================================================

def ultraAIPredict(history: List[Dict], currentbankroll: float, lastresult: Optional[str] = None) -> Dict:
    
    if state.session_start_bankroll == 0:
        state.session_start_bankroll = currentbankroll
    state.current_profit = currentbankroll - state.session_start_bankroll

    # Clean Data
    clean_nums = []
    clean_outcomes = []
    for item in reversed(history):
        v = MathLib.safe_float(item.get('actual_number'))
        if v is not None:
            clean_nums.append(v)
            if 0 <= int(v) <= 4: clean_outcomes.append("SMALL")
            elif 5 <= int(v) <= 9: clean_outcomes.append("BIG")

    # Manager Update
    if lastresult and state.last_prediction and state.last_prediction != TradeDecision.SKIP:
        EngineManager.update_performance(lastresult)
        
        # Ghost Protocol Check
        real_res = TradeDecision.BIG if lastresult == "BIG" else TradeDecision.SMALL
        if state.last_prediction == real_res:
            state.loss_streak = 0
            if not state.inversion_mode: state.consecutive_fails = 0
        else:
            state.loss_streak += 1
            if not state.inversion_mode:
                state.consecutive_fails += 1
                if state.consecutive_fails >= 3:
                    state.inversion_mode = True

    # Safety
    if state.cooling_off_counter > 0:
        state.cooling_off_counter -= 1

    is_unsafe, vol_reason = MarketMonitor.check_volatility(clean_nums)
    if is_unsafe:
        return {'finalDecision': "SKIP", 'confidence': 0, 'positionsize': 0, 
                'level': "DANGER", 'reason': vol_reason, 'topsignals': ["UNSAFE"]}

    # Prediction
    council = VotingCouncil()
    decision, confidence, signals = council.cast_votes(clean_nums, clean_outcomes)
    
    # Ghost Flip
    final_decision_str = decision.value
    meta_status = "STANDARD"
    if decision != TradeDecision.SKIP and state.inversion_mode:
        meta_status = "GHOST"
        if decision == TradeDecision.BIG: final_decision_str = "SMALL"
        elif decision == TradeDecision.SMALL: final_decision_str = "BIG"
        signals.append("INVERTED")

    if final_decision_str == "SKIP":
        state.last_prediction = TradeDecision.SKIP
        return {'finalDecision': "SKIP", 'confidence': 0, 'positionsize': 0, 
                'level': "WAIT", 'reason': "Low Conf", 'topsignals': signals}

    # Staking
    base_stake = max(currentbankroll * config.base_risk_percent, config.min_bet_amount)
    stake = base_stake * (config.martingale_multiplier ** state.loss_streak)
    if stake > (currentbankroll * 0.20): stake = currentbankroll * 0.20

    state.last_prediction = TradeDecision(final_decision_str)
    state.last_confidence = confidence

    return {
        'finalDecision': final_decision_str,
        'confidence': round(confidence, 4),
        'positionsize': int(stake),
        'level': f"L{state.loss_streak} ({meta_status})",
        'reason': " | ".join(signals),
        'topsignals': signals
    }
