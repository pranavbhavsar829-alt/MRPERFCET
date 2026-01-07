#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
================================================================================
  _______ _____ _______ _    _  _   _ 
 |__   __|_   _|__   __| |  | || \ | |
    | |    | |    | |  | |  | ||  \| |
    | |    | |    | |  | |  | || . ` |
    | |   _| |_   | |  | |__| || |\  |
    |_|  |_____|  |_|   \____/ |_| \_|
                                      
  TITAN V2000 - THE SINGULARITY EDITION (MERGED ARCHITECTURE)
  ------------------------------------------------------------------------------
  COMBINED SYSTEMS: 
    - Titan V302 (Neural, Entropy, Regime Analysis)
    - Titan V1000 (Hyper-Flux, 4-Trend Dragon, True Memory)
  
  LOGIC:
  This engine runs 5 simultaneous strategies (The Penta-Core) and uses 
  V1000's Resonance logic to filter the V302 Neural signals.
  
  Risk Management: Aggressive Profit-Capture Martingale (1-3-7-15-35)
================================================================================
"""

import math
import statistics
import random
import traceback
import sys
import time
from collections import deque, Counter
from typing import Dict, List, Optional, Any, Tuple, Union

# ==============================================================================
# SECTION 1: GLOBAL CONFIGURATION & CONSTANTS
# ==============================================================================

class GameConstants:
    """Immutable rules of the game."""
    BIG = "BIG"
    SMALL = "SMALL" 
    SKIP = "SKIP"
    
    # Minimum history to start engines (Fast Start)
    MIN_HISTORY_REQUIRED = 15

class RiskConfig:
    """
    Unified Risk Protocol.
    Combines V302's Aggression with V1000's Structure.
    """
    # --------------------------------------------------------------------------
    # BANKROLL MANAGEMENT
    # --------------------------------------------------------------------------
    BASE_RISK_PERCENT = 0.01   # 1% of bankroll as base unit
    MIN_BET_AMOUNT = 10.0
    MAX_BET_AMOUNT = 50000.0
    MAX_BANKROLL_USAGE = 0.60  # Never use more than 60% of wallet in one bet
    
    # --------------------------------------------------------------------------
    # MARTINGALE PROFIT LADDER (From V1000)
    # Designed to recover previous losses AND profit.
    # Level 0: 1x
    # Level 1: 3x (Recovers 1 + Profit)
    # Level 2: 7x (Recovers 4 + Profit)
    # Level 3: 15x (Recovers 11 + Profit)
    # Level 4: 35x (Recovers 26 + Profit)
    # Level 5: 75x (Maximum Aggression)
    # --------------------------------------------------------------------------
    MARTINGALE_MULTIPLIERS = [1.0, 3.0, 7.0, 15.0, 35.0, 75.0]
    
    # Safety Stop
    MAX_CONSECUTIVE_LOSSES = 6
    
    # --------------------------------------------------------------------------
    # CONFIDENCE GATES
    # --------------------------------------------------------------------------
    # Entry level for standard trades
    CONFIDENCE_LEVEL_1 = 0.58  # 58% (Aggressive Entry)
    
    # Entry level for recovery trades
    CONFIDENCE_LEVEL_2 = 0.68  # 68% (Solid Signal)
    
    # Entry level for Sniper/Max trades
    CONFIDENCE_LEVEL_3 = 0.85  # 85% (Resonance Lock)

    # --------------------------------------------------------------------------
    # MARKET FILTERS
    # --------------------------------------------------------------------------
    # If the market flips more than 80% of the time, we pause (Chaos Mode)
    MAX_CHAOS_INDEX = 0.80

# ==============================================================================
# SECTION 2: ADVANCED MATHEMATICS & UTILITIES
# ==============================================================================

class MathUtils:
    """
    The Mathematical Core (Imported from V302).
    """
    @staticmethod
    def safe_float(value: Any) -> float:
        try:
            if value is None: return 4.5
            return float(value)
        except (ValueError, TypeError):
            return 4.5

    @staticmethod
    def get_outcome(n: Any) -> str:
        val = int(MathUtils.safe_float(n))
        if 0 <= val <= 4: return GameConstants.SMALL
        if 5 <= val <= 9: return GameConstants.BIG
        return GameConstants.SKIP

    @staticmethod
    def sigmoid(x: float) -> float:
        """Neural Activation"""
        try:
            return 1.0 / (1.0 + math.exp(-x))
        except OverflowError:
            return 0.0 if x < 0 else 1.0

    @staticmethod
    def calculate_mean(data: List[float]) -> float:
        return sum(data) / len(data) if data else 0.0

    @staticmethod
    def calculate_stddev(data: List[float]) -> float:
        if len(data) < 2: return 0.0
        return statistics.stdev(data)

    @staticmethod
    def calculate_rsi(data: List[float], period: int = 14) -> float:
        if len(data) < period + 1: return 50.0
        deltas = [data[i] - data[i-1] for i in range(1, len(data))]
        gains = [d if d > 0 else 0 for d in deltas]
        losses = [-d if d < 0 else 0 for d in deltas]
        avg_gain = sum(gains[-period:]) / period
        avg_loss = sum(losses[-period:]) / period
        if avg_loss == 0: return 100.0
        rs = avg_gain / avg_loss
        return 100.0 - (100.0 / (1.0 + rs))

    @staticmethod
    def calculate_entropy(outcomes: List[str]) -> float:
        """Shannon Entropy (0=Order, 1=Chaos)"""
        if not outcomes: return 1.0
        counts = Counter(outcomes)
        total = len(outcomes)
        entropy = 0.0
        for count in counts.values():
            p = count / total
            if p > 0:
                entropy -= p * math.log2(p)
        return entropy

# --- Compatibility Wrappers for Fetcher ---
def safe_float(value): return MathUtils.safe_float(value)
def get_outcome_from_number(n): return MathUtils.get_outcome(n)


# ==============================================================================
# SECTION 3: TITAN TRUE MEMORY (FROM V1000)
# ==============================================================================

class TitanMemory:
    """
    Persistently tracks wins/losses in RAM.
    This prevents the bot from resetting its Martingale strategy if the API lags.
    """
    def __init__(self):
        self.last_predicted_issue: Optional[str] = None
        self.last_predicted_label: Optional[str] = None
        self.loss_streak: int = 0
        self.wins: int = 0
        self.losses: int = 0
        self.history_log: deque = deque(maxlen=200)

    def update_streak(self, latest_issue: str, latest_outcome: str):
        """Verifies if the last bet won or lost."""
        if str(latest_issue) == str(self.last_predicted_issue):
            if self.last_predicted_label and self.last_predicted_label != GameConstants.SKIP:
                if self.last_predicted_label == latest_outcome:
                    # WIN
                    self.loss_streak = 0
                    self.wins += 1
                    self.history_log.append("W")
                else:
                    # LOSS
                    self.loss_streak += 1
                    self.losses += 1
                    self.history_log.append("L")
            
            # Reset pending
            self.last_predicted_issue = None
            self.last_predicted_label = None

    def register_bet(self, target_issue: str, label: str):
        """Locks in a bet for the next round."""
        self.last_predicted_issue = target_issue
        self.last_predicted_label = label

# Initialize Single Global Memory
titan_memory = TitanMemory()


# ==============================================================================
# SECTION 4: MARKET REGIME ANALYZER (FROM V302)
# ==============================================================================

class MarketRegime:
    """Analyzes the 'Texture' of the market (Trend vs Chop)."""
    TRENDING = "TRENDING"
    CHOPPY = "CHOPPY"
    NEUTRAL = "NEUTRAL"
    
    @staticmethod
    def analyze(history: List[Dict]) -> Dict[str, Any]:
        if len(history) < 20:
            return {"status": MarketRegime.NEUTRAL, "chop_index": 0.5}
            
        outcomes = [MathUtils.get_outcome(d['actual_number']) for d in history[-20:]]
        outcomes = [o for o in outcomes if o != GameConstants.SKIP]
        
        if len(outcomes) < 2: return {"status": MarketRegime.NEUTRAL, "chop_index": 0.5}

        # Calculate Alternations (Flips)
        alternations = 0
        for i in range(1, len(outcomes)):
            if outcomes[i] != outcomes[i-1]:
                alternations += 1
                
        max_alts = len(outcomes) - 1
        chop_index = alternations / max_alts if max_alts > 0 else 0
        
        status = MarketRegime.NEUTRAL
        if chop_index > 0.60:
            status = MarketRegime.CHOPPY
        elif chop_index < 0.40:
            status = MarketRegime.TRENDING
            
        return {"status": status, "chop_index": chop_index}


# ==============================================================================
# SECTION 5: THE PENTA-CORE ENGINE SYSTEM
# ==============================================================================

class EngineResult:
    def __init__(self, prediction: str, weight: float, source: str):
        self.prediction = prediction
        self.weight = weight
        self.source = source

# ------------------------------------------------------------------------------
# CORE 1: QUANTUM VOLATILITY (V302)
# ------------------------------------------------------------------------------
def engine_quantum(history: List[Dict]) -> Optional[EngineResult]:
    """Uses Bollinger Band logic to predict reversion."""
    try:
        numbers = [MathUtils.safe_float(d.get('actual_number')) for d in history[-30:]]
        if len(numbers) < 20: return None
        
        mean = MathUtils.calculate_mean(numbers)
        std = MathUtils.calculate_stddev(numbers)
        if std == 0: return None
        
        current = numbers[-1]
        z_score = (current - mean) / std
        
        # Aggressive Threshold (1.4 SD)
        if z_score > 1.4:
            return EngineResult(GameConstants.SMALL, 0.70, f"Quantum(High Z:{z_score:.1f})")
        elif z_score < -1.4:
            return EngineResult(GameConstants.BIG, 0.70, f"Quantum(Low Z:{z_score:.1f})")
        return None
    except: return None

# ------------------------------------------------------------------------------
# CORE 2: FRACTAL PATTERN V4 (V302)
# ------------------------------------------------------------------------------
def engine_pattern(history: List[Dict]) -> Optional[EngineResult]:
    """Scans history for repeating sequences."""
    try:
        if len(history) < 40: return None
        outcomes = [MathUtils.get_outcome(d.get('actual_number')) for d in history]
        raw = ''.join(['B' if o==GameConstants.BIG else 'S' for o in outcomes if o!=GameConstants.SKIP])
        
        best_prob = 0.0
        best_pred = None
        best_depth = 0
        
        # Scan depths 3 to 6
        for depth in range(6, 2, -1):
            if len(raw) < depth + 1: continue
            curr = raw[-depth:]
            search = raw[:-1]
            
            # Find history
            b_count = 0
            s_count = 0
            start = 0
            while True:
                idx = search.find(curr, start)
                if idx == -1: break
                if idx + depth < len(search):
                    nxt = search[idx + depth]
                    if nxt == 'B': b_count += 1
                    else: s_count += 1
                start = idx + 1
            
            total = b_count + s_count
            if total >= 3:
                if b_count > s_count:
                    prob = b_count/total
                    pred = GameConstants.BIG
                else:
                    prob = s_count/total
                    pred = GameConstants.SMALL
                
                if prob > best_prob and prob > 0.60:
                    best_prob = prob
                    best_pred = pred
                    best_depth = depth
                    
        if best_pred:
            weight = best_prob * (1.1 + (best_depth * 0.1))
            return EngineResult(best_pred, weight, f"Pattern-D{best_depth}({best_prob:.0%})")
        return None
    except: return None

# ------------------------------------------------------------------------------
# CORE 3: NEURAL PERCEPTRON (V302)
# ------------------------------------------------------------------------------
def engine_neural(history: List[Dict], regime: str) -> Optional[EngineResult]:
    """Lightweight Neural Net using RSI and Momentum."""
    try:
        numbers = [MathUtils.safe_float(d.get('actual_number')) for d in history[-45:]]
        if len(numbers) < 20: return None
        
        # Inputs
        rsi = MathUtils.calculate_rsi(numbers, 14)
        norm_rsi = (rsi - 50) / 100.0
        
        sma_fast = MathUtils.calculate_mean(numbers[-5:])
        sma_slow = MathUtils.calculate_mean(numbers[-20:])
        momentum = (sma_fast - sma_slow) / 10.0
        
        # Weights (Adaptive)
        if regime == MarketRegime.TRENDING:
            w_rsi, w_mom = -0.5, 2.5
        else:
            w_rsi, w_mom = -1.5, 1.0
            
        z = (norm_rsi * w_rsi) + (momentum * w_mom)
        prob = MathUtils.sigmoid(z)
        
        if prob > 0.55:
            return EngineResult(GameConstants.BIG, abs(prob-0.5)*2.5, f"Neural(B:{prob:.2f})")
        elif prob < 0.45:
            return EngineResult(GameConstants.SMALL, abs(prob-0.5)*2.5, f"Neural(S:{prob:.2f})")
        return None
    except: return None

# ------------------------------------------------------------------------------
# CORE 4: MICRO-STRUCTURE (V1000)
# ------------------------------------------------------------------------------
def engine_micro(history: List[Dict]) -> List[EngineResult]:
    """Speedster engine for last 3 results."""
    results = []
    outcomes = [MathUtils.get_outcome(d['actual_number']) for d in history[-10:]]
    h = "".join(["B" if x == GameConstants.BIG else "S" for x in outcomes if x != GameConstants.SKIP])
    
    if len(h) < 3: return []
    
    # 3-Streak Continuation
    if h.endswith("BBB"):
        results.append(EngineResult(GameConstants.BIG, 0.75, "Micro:Streak3"))
    elif h.endswith("SSS"):
        results.append(EngineResult(GameConstants.SMALL, 0.75, "Micro:Streak3"))
        
    # Chop Continuation (BSB -> S)
    if h.endswith("BSB"):
        results.append(EngineResult(GameConstants.SMALL, 0.70, "Micro:Chop"))
    elif h.endswith("SBS"):
        results.append(EngineResult(GameConstants.BIG, 0.70, "Micro:Chop"))
        
    return results

# ------------------------------------------------------------------------------
# CORE 5: MACRO-WAVES DRAGON (V1000)
# ------------------------------------------------------------------------------
def engine_macro(history: List[Dict]) -> List[EngineResult]:
    """Detects 4-Trend Dragons and Ping Pong."""
    results = []
    outcomes = [MathUtils.get_outcome(d['actual_number']) for d in history[-20:]]
    h = "".join(["B" if x == GameConstants.BIG else "S" for x in outcomes if x != GameConstants.SKIP])
    
    if len(h) < 5: return []
    
    # Dragon 4 (Fast Dragon)
    if h.endswith("BBBB"):
        results.append(EngineResult(GameConstants.BIG, 0.90, "Dragon-B(4)"))
    elif h.endswith("SSSS"):
        results.append(EngineResult(GameConstants.SMALL, 0.90, "Dragon-S(4)"))
        
    # Ping Pong 4 (BSBS -> B)
    if h.endswith("BSBS"):
        results.append(EngineResult(GameConstants.BIG, 0.85, "PingPong(4)"))
    elif h.endswith("SBSB"):
        results.append(EngineResult(GameConstants.SMALL, 0.85, "PingPong(4)"))
        
    return results


# ==============================================================================
# SECTION 6: THE ARCHITECT (MAIN PREDICTION LOGIC)
# ==============================================================================

def ultraAIPredict(history: List[Dict], current_bankroll: float, last_result: Optional[str] = None) -> Dict:
    """
    MASTER FUNCTION.
    """
    # --------------------------------------------------------------------------
    # 1. UPDATE TITAN MEMORY (Handling Win/Loss)
    # --------------------------------------------------------------------------
    if not history:
        return _build_response("SKIP", 0, 0, "BOOT", "Waiting for data", [])
        
    latest_record = history[-1]
    titan_memory.update_streak(
        str(latest_record['issue']), 
        MathUtils.get_outcome(latest_record['actual_number'])
    )
    
    streak = titan_memory.loss_streak
    
    # --------------------------------------------------------------------------
    # 2. ANALYZE REGIME & CHAOS
    # --------------------------------------------------------------------------
    regime_data = MarketRegime.analyze(history)
    current_regime = regime_data['status']
    
    # Chaos Guard (V1000 style)
    if regime_data['chop_index'] > RiskConfig.MAX_CHAOS_INDEX and streak == 0:
        return _build_response("SKIP", 0, 0, "🛑 CHAOS", f"Flip Rate {regime_data['chop_index']:.0%}", [])

    # --------------------------------------------------------------------------
    # 3. EXECUTE PENTA-CORE ENGINES
    # --------------------------------------------------------------------------
    signals: List[EngineResult] = []
    
    # V302 Engines
    s_quant = engine_quantum(history)
    if s_quant: signals.append(s_quant)
    
    s_patt = engine_pattern(history)
    if s_patt: signals.append(s_patt)
    
    s_neur = engine_neural(history, current_regime)
    if s_neur: signals.append(s_neur)
    
    # V1000 Engines (List returns)
    signals.extend(engine_micro(history))
    signals.extend(engine_macro(history))
    
    if not signals:
        return _build_response("SKIP", 0, 0, "WAITING", "No Pattern Found", [])

    # --------------------------------------------------------------------------
    # 4. SIGNAL AGGREGATION & RESONANCE
    # --------------------------------------------------------------------------
    big_weight = sum(s.weight for s in signals if s.prediction == GameConstants.BIG)
    small_weight = sum(s.weight for s in signals if s.prediction == GameConstants.SMALL)
    
    final_decision = GameConstants.SKIP
    confidence = 0.0
    
    if big_weight > small_weight:
        final_decision = GameConstants.BIG
        confidence = big_weight / (len(signals) * 0.6 + 1) # Normalize
    elif small_weight > big_weight:
        final_decision = GameConstants.SMALL
        confidence = small_weight / (len(signals) * 0.6 + 1)
        
    # --- HYPER-FLUX RESONANCE CHECK (The V1000 Magic) ---
    # Does Micro (Speed) agree with Macro (Dragon)?
    has_micro = any(s.source.startswith("Micro") and s.prediction == final_decision for s in signals)
    has_macro = any(s.source.startswith("Dragon") and s.prediction == final_decision for s in signals)
    
    is_resonant = False
    if has_micro and has_macro:
        confidence += 0.20 # Massive Boost
        is_resonant = True

    # Cap Confidence
    confidence = min(confidence, 0.99)

    # --------------------------------------------------------------------------
    # 5. RISK MANAGEMENT (MARTINGALE LOGIC)
    # --------------------------------------------------------------------------
    stake = 0.0
    level_label = "SKIP"
    reason = f"Signal {confidence:.1%}"
    
    # Base Bet
    base_bet = max(current_bankroll * RiskConfig.BASE_RISK_PERCENT, RiskConfig.MIN_BET_AMOUNT)
    if base_bet > 3000: base_bet = 3000

    should_bet = False
    
    # LEVEL 1: STANDARD (Streak 0)
    if streak == 0:
        if confidence >= RiskConfig.CONFIDENCE_LEVEL_1:
            should_bet = True
            level_label = "✅ ACTIVE"
            reason = f"Base Entry ({confidence:.0%})"
            
    # LEVEL 2: RECOVERY (Streak 1-2)
    elif 1 <= streak <= 2:
        # V302 Aggression: If we are resonant, we attack hard
        req_conf = RiskConfig.CONFIDENCE_LEVEL_2
        if is_resonant: req_conf -= 0.10 # Lower threshold if signals align perfectly
        
        if confidence >= req_conf:
            should_bet = True
            level_label = f"⚠️ RECOVER L{streak}"
            reason = "Martingale Active"
            
    # LEVEL 3: SNIPER (Streak 3+)
    elif streak >= 3:
        if streak >= RiskConfig.MAX_CONSECUTIVE_LOSSES:
            # STOP LOSS
            should_bet = False
            level_label = "🛑 STOP LOSS"
            reason = "Max Streak Hit"
        elif confidence >= RiskConfig.CONFIDENCE_LEVEL_3 or is_resonant:
            should_bet = True
            level_label = f"🔥 TITAN SNIPE L{streak}"
            reason = "High Probability"
        else:
            level_label = f"🛡️ HOLD L{streak}"
            reason = "Waiting for Dragon"

    # Calculate Stake
    if should_bet:
        m_idx = min(streak, len(RiskConfig.MARTINGALE_MULTIPLIERS) - 1)
        mult = RiskConfig.MARTINGALE_MULTIPLIERS[m_idx]
        stake = base_bet * mult
        
        # Safety Clamps
        stake = min(stake, RiskConfig.MAX_BET_AMOUNT)
        if stake > current_bankroll * RiskConfig.MAX_BANKROLL_USAGE:
            stake = current_bankroll * RiskConfig.MAX_BANKROLL_USAGE
            
        # Register the bet in memory for next round verification
        next_issue = str(int(latest_record['issue']) + 1)
        titan_memory.register_bet(next_issue, final_decision)

    # --------------------------------------------------------------------------
    # 6. RETURN PAYLOAD
    # --------------------------------------------------------------------------
    # Format top signals for display
    sorted_sigs = sorted(signals, key=lambda x: x.weight, reverse=True)[:3]
    top_sig_names = [f"{s.source}={s.prediction}" for s in sorted_sigs]

    return _build_response(
        final_decision if should_bet else GameConstants.SKIP,
        confidence,
        int(stake),
        level_label,
        reason,
        top_sig_names
    )

def _build_response(decision, conf, stake, level, reason, sources):
    """Standardized Dictionary Return."""
    return {
        'finalDecision': decision,
        'confidence': conf,
        'positionsize': stake,
        'level': level,
        'reason': reason,
        'topsignals': sources
    }

# ==============================================================================
# SELF-TEST
# ==============================================================================
if __name__ == "__main__":
    print("TITAN V2000 (SINGULARITY) INITIALIZED.")
    print("Testing Penta-Core Engines...")
    
    # Generate Mock Data (Dragon Pattern: B B B B)
    mock = []
    start_iss = 20240101000
    for i in range(50):
        # Create a pattern of 4 BIGs at the end
        if i >= 46: val = 8 # BIG
        elif i % 2 == 0: val = 2 # SMALL
        else: val = 7 # BIG
        
        mock.append({'issue': str(start_iss + i), 'actual_number': val})
        
    print("\n[Input]: Last 4 results are BIG (Dragon Pattern)")
    res = ultraAIPredict(mock, 10000.0)
    
    print(f"Decision: {res['finalDecision']}")
    print(f"Level:    {res['level']}")
    print(f"Reason:   {res['reason']}")
    print(f"Signals:  {res['topsignals']}")
    
    if "Dragon-B(4)=BIG" in str(res['topsignals']):
        print("\n>> SUCCESS: Dragon Detected.")
    else:
        print("\n>> WARNING: Dragon Not Detected.")
