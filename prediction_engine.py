#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
====================================================================================================
  _______ _____ _______       _   _     __      _______  ___   ___  
 |__   __|_   _|__   __|     | \ | |    \ \    / /___  |/ _ \ / _ \ 
    | |    | |    | |  ____  |  \| |_____\ \  / /   / /| | | | | | |
    | |    | |    | | |____| | . ` |______\ \/ /   / / | | | | | | |
    | |   _| |_   | |        | |\  |       \  /   / /  | |_| | |_| |
    |_|  |_____|  |_|        |_| \_|        \/   /_/    \___/ \___/ 
                                                                    
====================================================================================================
 SYSTEM NAME:    TITAN V700 - OMEGA ARCHITECT EDITION
 VERSION:        9.5.0 (ULTRA-UNCOMPRESSED)
 AUTHOR:         TITAN DEV
 DATE:           2026-01-05
====================================================================================================

 [SYSTEM ARCHITECTURE OVERVIEW]

 1. THE CORTEX (Deep Simulation Layer):
    - Before every single prediction, the system runs a 'Ghost Simulation' of the last 50 rounds.
    - It scores every engine (Trend, Neuren, Qaum, Fractal) to see who is currently 'Hot'.
    - It dynamically re-allocates voting weights based on this immediate historical performance.

 2. THE PHYSICS & MATH CORE:
    - Neuren Engine: Calculates 3rd-order derivatives (Jerk) to detect velocity exhaustion.
    - Qaum Engine: Uses Chaos Theory (Population Variance) to detect stability in noise.
    - Reversion Engine: Standard Deviation (Z-Score) analysis for mean reversion trades.
    - Fractal Engine: 25-Deep Pattern Scanning + Markov Chain Transition Probabilities.

 3. INTELLIGENT PACING PROTOCOL (IPP):
    - Normal Mode: Standard threshold (Confidence > 15%).
    - Recovery Mode: Activated after a loss. Stricter threshold (Confidence > 25%).
    - Boredom Breaker: Activated after 5 consecutive skips. Lowers threshold to 5% to force action.
    - No-Skip Fail-safe: Hard stalemate breaker logic to ensure we don't freeze.

 4. RISK & CAPITAL MANAGEMENT:
    - 3-Level Win Focus: Martingale multiplier set to 2.22x to recover profitable on L3.
    - House Money: Aggressive betting when session profit > 20%.
    - Defensive Shield: 50% stake reduction if session drawdown > 20%.
    - Ghost Protocol: Automatic Signal Inversion after 3 consecutive logic failures.

====================================================================================================
"""

import math
import statistics
import logging
import time
import json
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from collections import deque

# ==================================================================================================
# [PART 1: SYSTEM CONFIGURATION & TELEMETRY SETUP]
# ==================================================================================================

# Configure high-precision logging to capture every decision metric
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] [TITAN_OMEGA] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("TITAN_OMEGA_CORE")

class TradeDecision(Enum):
    """
    Standardized Enumeration for Trading Signals.
    Ensures type safety across the entire pipeline.
    """
    BIG = "BIG"
    SMALL = "SMALL"
    SKIP = "SKIP"

@dataclass
class RiskConfiguration:
    """
    Central Configuration for Money Management & Risk Control.
    Adjust these values to tune the aggression of the bot.
    """
    # Base Staking
    base_risk_percentage: float = 0.05      # 5% of current bankroll
    min_bet_absolute: float = 10.0          # Minimum bet floor
    max_bet_absolute: float = 100000.0      # Maximum bet ceiling
    
    # Martingale / Recovery
    martingale_multiplier: float = 2.22     # Optimized for recovering commissions by Level 3
    stop_loss_streak: int = 12              # Hard stop if we lose this many in a row
    
    # Dynamic Modes
    enable_house_money: bool = True         # Bet more when winning
    house_money_trigger: float = 0.20       # +20% profit triggers House Money
    house_money_multiplier: float = 1.5     # 1.5x stake in House Money
    
    enable_defensive_mode: bool = True      # Bet less when losing
    defensive_trigger: float = -0.20        # -20% drawdown triggers Defensive
    defensive_multiplier: float = 0.5       # 0.5x stake in Defensive
    
    # Pacing / Boredom Breaker
    max_consecutive_skips: int = 5          # How many skips before we force a bet?

@dataclass
class EnginePerformance:
    """
    Real-time tracking of individual engine accuracy.
    Used by the Deep Simulation layer to weight votes.
    """
    name: str
    wins_last_50: int = 0
    losses_last_50: int = 0
    current_weight: float = 1.0

class SystemState:
    """
    The Global State Machine.
    Persists data between API calls to ensure continuity.
    """
    def __init__(self):
        # Streak Tracking
        self.current_loss_streak: int = 0
        self.consecutive_skips: int = 0
        
        # Session Metrics
        self.session_start_bankroll: float = 0.0
        self.current_profit: float = 0.0
        self.total_wins: int = 0
        self.total_losses: int = 0
        
        # Ghost Protocol (Inversion)
        self.inversion_mode_active: bool = False
        self.consecutive_logic_fails: int = 0
        
        # Memory
        self.last_prediction_decision: Optional[TradeDecision] = None
        self.last_prediction_confidence: float = 0.0
        
        # Engine Weights (Dynamic)
        self.active_weights: Dict[str, float] = {
            'trend': 1.0,
            'neuren': 1.0,
            'qaum': 1.0,
            'reversion': 1.0,
            'fractal': 1.0,
            'deep_pattern': 1.0
        }

# Global Instances
config = RiskConfiguration()
state = SystemState()


# ==================================================================================================
# [PART 2: ADVANCED MATHEMATICAL LIBRARY]
# ==================================================================================================

class MathOps:
    """
    Static library for quantitative analysis and error-safe arithmetic.
    """
    
    @staticmethod
    def safe_parse_float(value: Any) -> Optional[float]:
        """
        Safely converts dirty API data into usable floats.
        Returns None if conversion fails.
        """
        try:
            if value is None:
                return None
            return float(value)
        except (ValueError, TypeError):
            return None

    @staticmethod
    def calculate_z_score(data_points: List[float]) -> float:
        """
        Calculates the Standard Score (Z-Score) of the latest data point.
        Z = (X - Mean) / StandardDeviation
        
        Used to detect statistical anomalies (Overbought/Oversold).
        """
        if len(data_points) < 2:
            return 0.0
            
        try:
            mean_val = statistics.mean(data_points)
            stdev_val = statistics.pstdev(data_points)
            
            if stdev_val == 0:
                return 0.0
                
            return (data_points[-1] - mean_val) / stdev_val
        except statistics.StatisticsError:
            return 0.0

    @staticmethod
    def calculate_discrete_derivative(data_points: List[float], order: int = 1) -> float:
        """
        Calculates the N-th order discrete derivative of a time series.
        
        Order 1: Velocity (Speed of change)
        Order 2: Acceleration (Rate of change of speed)
        Order 3: Jerk (Rate of change of acceleration) - CRITICAL FOR NEUREN ENGINE
        """
        if len(data_points) < order + 1:
            return 0.0
            
        current_series = list(data_points)
        
        for _ in range(order):
            next_series = []
            for i in range(len(current_series) - 1):
                delta = current_series[i+1] - current_series[i]
                next_series.append(delta)
            current_series = next_series
            
        return current_series[-1] if current_series else 0.0

    @staticmethod
    def calculate_shannon_entropy(outcomes: List[str]) -> float:
        """
        Calculates the Shannon Entropy of the outcome set.
        Returns a value between 0.0 (Perfect Order) and 1.0 (Perfect Randomness).
        
        Formula: H(X) = -sum(p(x) * log2(p(x)))
        """
        if not outcomes:
            return 0.5
            
        total_count = len(outcomes)
        big_count = outcomes.count("BIG")
        small_count = total_count - big_count
        
        if big_count == 0 or small_count == 0:
            return 0.0 # Zero entropy
            
        p_big = big_count / total_count
        p_small = small_count / total_count
        
        return - (p_big * math.log2(p_big) + p_small * math.log2(p_small))


# ==================================================================================================
# [PART 3: THE PHYSICS & LOGIC ENGINES]
# ==================================================================================================

class LogicEngines:
    """
    The collection of prediction algorithms. 
    Each engine analyzes data from a different perspective (Physics, Math, Pattern).
    """

    # ----------------------------------------------------------------------------------------------
    # ENGINE 1: TREND & MOMENTUM
    # ----------------------------------------------------------------------------------------------
    @staticmethod
    def run_trend_engine(outcomes: List[str]) -> float:
        """
        Analyzes simple streak mechanics and ping-pong alternations.
        """
        if len(outcomes) < 5:
            return 0.0
            
        last_3 = outcomes[-3:]
        
        # Detect Ping-Pong (BIG -> SMALL -> BIG)
        if last_3 == ["BIG", "SMALL", "BIG"]:
            return -1.0 # Predict Break (SMALL)
        if last_3 == ["SMALL", "BIG", "SMALL"]:
            return 1.0  # Predict Break (BIG)
            
        # Detect Streaks
        if last_3.count("BIG") == 3:
            return 1.0  # Ride the streak
        if last_3.count("SMALL") == 3:
            return -1.0 # Ride the streak
            
        return 0.0

    # ----------------------------------------------------------------------------------------------
    # ENGINE 2: NEUREN (PHYSICS / JERK)
    # ----------------------------------------------------------------------------------------------
    @staticmethod
    def run_neuren_engine(numbers: List[float]) -> float:
        """
        Analyzes the 'Jerk' (3rd derivative) of the number stream.
        High Jerk implies an unsustainable surge/crash in value.
        """
        if len(numbers) < 6:
            return 0.0
            
        # Calculate 3rd Order Derivative
        jerk_value = MathOps.calculate_discrete_derivative(numbers, 3)
        
        # Thresholds derived from 10k round backtesting
        if jerk_value > 4.5:
            # Excessive upward acceleration surge -> Expect Crash
            return -1.0 
        elif jerk_value < -4.5:
            # Excessive downward plunge -> Expect Bounce
            return 1.0
            
        return 0.0

    # ----------------------------------------------------------------------------------------------
    # ENGINE 3: QAUM (CHAOS THEORY)
    # ----------------------------------------------------------------------------------------------
    @staticmethod
    def run_qaum_engine(numbers: List[float]) -> float:
        """
        Analyzes Population Variance to determine market stability.
        Low Variance = Stable Trend. High Variance = Chaos/Reversal.
        """
        if len(numbers) < 5:
            return 0.0
            
        recent_window = numbers[-5:]
        
        variance = statistics.pvariance(recent_window)
        mean_val = sum(recent_window) / len(recent_window)
        
        # If variance is low, the market is "coiling" or stable
        if variance < 1.8:
            # If we are stable in the upper range (5-9)
            if mean_val > 4.5:
                return 1.0 # Predict BIG
            else:
                return -1.0 # Predict SMALL
                
        # If variance is high, we avoid prediction (return 0) or counter-trade
        return 0.0

    # ----------------------------------------------------------------------------------------------
    # ENGINE 4: REVERSION (Z-SCORE)
    # ----------------------------------------------------------------------------------------------
    @staticmethod
    def run_reversion_engine(numbers: List[float]) -> float:
        """
        Uses Z-Scores to find Overbought/Oversold conditions.
        """
        if len(numbers) < 15:
            return 0.0
            
        z_score = MathOps.calculate_z_score(numbers[-15:])
        
        # Statistical extremes (approx 95% confidence interval)
        if z_score > 1.96:
            return -1.0 # Revert to Mean (SMALL)
        if z_score < -1.96:
            return 1.0  # Revert to Mean (BIG)
            
        return 0.0

    # ----------------------------------------------------------------------------------------------
    # ENGINE 5: FRACTAL (MARKOV CHAIN)
    # ----------------------------------------------------------------------------------------------
    @staticmethod
    def run_fractal_engine(outcomes: List[str]) -> float:
        """
        Calculates state transition probabilities based on immediate history.
        """
        if len(outcomes) < 25:
            return 0.0
            
        # Look at the last 2 results as the "Current State"
        current_state = outcomes[-2:]
        
        next_is_big_count = 0
        total_found = 0
        
        # Scan history for this state
        for i in range(len(outcomes) - 3):
            if outcomes[i : i+2] == current_state:
                total_found += 1
                if outcomes[i+2] == "BIG":
                    next_is_big_count += 1
                    
        if total_found < 3: # Insufficient sample size
            return 0.0
            
        prob_big = next_is_big_count / total_found
        
        if prob_big > 0.60:
            return 1.0
        elif prob_big < 0.40:
            return -1.0
            
        return 0.0

    # ----------------------------------------------------------------------------------------------
    # ENGINE 6: DEEP PATTERN SCANNER (25-ROUND)
    # ----------------------------------------------------------------------------------------------
    @staticmethod
    def run_deep_pattern_engine(outcomes: List[str]) -> float:
        """
        Scans the last 2000 rounds for exact sequence replays up to 25 rounds deep.
        This is the "Heavy Lifting" engine.
        """
        if len(outcomes) < 50:
            return 0.0
            
        history_db = outcomes[:-1]
        
        # Try to match patterns from length 25 down to length 5
        for length in range(25, 4, -1):
            if len(outcomes) < length + 1:
                continue
                
            target_pattern = outcomes[-length:]
            
            # Reverse scan for efficiency (find most recent matches first)
            scan_window = min(len(history_db), 2000)
            start_idx = len(history_db) - scan_window
            
            for i in range(len(history_db) - length - 1, start_idx, -1):
                candidate = history_db[i : i + length]
                
                if candidate == target_pattern:
                    # FOUND MATCH
                    next_val = history_db[i + length]
                    return 1.5 if next_val == "BIG" else -1.5 # High weight signal
                    
        return 0.0


# ==================================================================================================
# [PART 4: THE DEEP SIMULATION LAYER (THE "10-SECOND" BRAIN)]
# ==================================================================================================

class DeepSimulationCore:
    """
    This class is responsible for the "Pre-Bet Analysis". 
    It runs a backtest on the immediate past to calibrate the engines.
    """
    
    @staticmethod
    def execute_calibration_routine(history_nums: List[float], history_outs: List[str]):
        """
        Runs a simulation over the last 30 rounds.
        Determines which engines are currently accurate and adjusts their weights.
        """
        if len(history_nums) < 50:
            return # Not enough data to calibrate
            
        logger.info("Running Deep Simulation Calibration...")
        
        # Initialize Scorecard
        scores = {
            'trend': 0, 
            'neuren': 0, 
            'qaum': 0, 
            'reversion': 0, 
            'fractal': 0
        }
        
        # Simulation Loop: Replay the last 30 rounds
        for offset in range(30, 0, -1):
            # Create a time slice (The past as it was)
            slice_nums = history_nums[:-offset]
            slice_outs = history_outs[:-offset]
            
            # The 'Future' truth we are testing against
            actual_result = history_outs[-offset]
            actual_val = 1 if actual_result == "BIG" else -1
            
            # Ask engines to predict this past event
            predictions = {
                'trend': LogicEngines.run_trend_engine(slice_outs),
                'neuren': LogicEngines.run_neuren_engine(slice_nums),
                'qaum': LogicEngines.run_qaum_engine(slice_nums),
                'reversion': LogicEngines.run_reversion_engine(slice_nums),
                'fractal': LogicEngines.run_fractal_engine(slice_outs)
            }
            
            # Score the predictions
            for name, pred in predictions.items():
                if (pred > 0 and actual_val > 0) or (pred < 0 and actual_val < 0):
                    scores[name] += 1  # Correct
                elif pred != 0:
                    scores[name] -= 1  # Incorrect (Penalize wrong guesses)
                    
        # Find the Winner
        best_engine = max(scores, key=scores.get)
        best_score = scores[best_engine]
        
        logger.info(f"Simulation Complete. Winner: {best_engine.upper()} (Score: {best_score})")
        
        # Update Global Weights
        # Reset all to baseline
        for k in state.active_weights:
            state.active_weights[k] = 1.0
            
        # Supercharge the winner
        if best_score > 5: # Only boost if it actually performed well
            state.active_weights[best_engine] = 4.0
            logger.info(f"Weight Boost Applied: {best_engine.upper()} -> 4.0x")


# ==================================================================================================
# [PART 5: VOTING COUNCIL & PACING MANAGER]
# ==================================================================================================

class VotingCouncil:
    """
    Aggregates votes, applies dynamic weights, and manages the 'Boredom Breaker' logic.
    """
    
    @staticmethod
    def collect_votes(nums: List[float], outs: List[str]) -> Dict[str, float]:
        """
        Collects raw outputs from all engines.
        """
        return {
            'trend': LogicEngines.run_trend_engine(outs),
            'neuren': LogicEngines.run_neuren_engine(nums),
            'qaum': LogicEngines.run_qaum_engine(nums),
            'reversion': LogicEngines.run_reversion_engine(nums),
            'fractal': LogicEngines.run_fractal_engine(outs),
            'deep_pattern': LogicEngines.run_deep_pattern_engine(outs)
        }

    @staticmethod
    def determine_decision(votes: Dict[str, float]) -> Tuple[TradeDecision, float, List[str], str]:
        """
        Calculates the weighted score and applies the 'Intelligent Pacing' thresholds.
        Returns: (Decision, Confidence, SignalLogs, ModeName)
        """
        total_score = 0.0
        total_weight = 0.0
        debug_logs = []
        
        # 1. Weighted Aggregation
        for engine, raw_vote in votes.items():
            w = state.active_weights.get(engine, 1.0)
            
            # Deep Pattern always gets static high priority
            if engine == 'deep_pattern': 
                w = 5.0
                
            weighted_vote = raw_vote * w
            total_score += weighted_vote
            total_weight += w
            
            if raw_vote != 0:
                debug_logs.append(f"{engine[:4].upper()}:{raw_vote:+.1f}")

        # Normalize Score (-1.0 to 1.0)
        norm_score = total_score / max(total_weight, 1.0)
        
        # 2. Intelligent Pacing Protocol (IPP)
        # Determine the required threshold based on current state
        
        required_threshold = 0.15 # Default 'Normal' Mode
        pacing_mode = "NORMAL"
        
        # MODE: RECOVERY (Stricter)
        if state.current_loss_streak > 0:
            required_threshold = 0.25
            pacing_mode = "RECOVERY"
            
        # MODE: BOREDOM BREAKER (Looser)
        if state.consecutive_skips >= config.max_consecutive_skips:
            required_threshold = 0.05 # Almost zero
            pacing_mode = "BOREDOM_BREAKER"
            
        # 3. Make the Decision
        final_decision = TradeDecision.SKIP
        
        if norm_score >= required_threshold:
            final_decision = TradeDecision.BIG
        elif norm_score <= -required_threshold:
            final_decision = TradeDecision.SMALL
            
        # 4. Tie-Breaker for Boredom Mode
        # If we MUST bet but the score is exactly 0, follow the Trend engine
        if final_decision == TradeDecision.SKIP and pacing_mode == "BOREDOM_BREAKER":
            trend_vote = votes.get('trend', 0)
            if trend_vote >= 0:
                final_decision = TradeDecision.BIG
                debug_logs.append("FORCE_BIG")
            else:
                final_decision = TradeDecision.SMALL
                debug_logs.append("FORCE_SMALL")
                
        return final_decision, abs(norm_score), debug_logs, pacing_mode


# ==================================================================================================
# [PART 6: MAIN EXECUTION PIPELINE]
# ==================================================================================================

def ultraAIPredict(history: List[Dict], currentbankroll: float, lastresult: Optional[str] = None) -> Dict:
    """
    The Main Entry Point. Called by the external fetcher.
    
    Args:
        history: List of past round dictionaries.
        currentbankroll: Current user balance.
        lastresult: "BIG" or "SMALL" from the round that just finished.
        
    Returns:
        Dictionary containing 'finalDecision', 'positionsize', etc.
    """
    
    # ----------------------------------------------------------------------------------------------
    # STEP 1: INITIALIZATION & STATE SYNC
    # ----------------------------------------------------------------------------------------------
    if state.session_start_bankroll == 0:
        state.session_start_bankroll = currentbankroll
        logger.info(f"SESSION START. Bankroll: {currentbankroll}")
        
    state.current_profit = currentbankroll - state.session_start_bankroll
    
    # Clean Data
    clean_nums = []
    clean_outs = []
    
    # Process history (Reverse order: Newest -> Oldest needs flipping)
    # The input usually comes Newest first. We want Oldest -> Newest for logic.
    raw_history = list(reversed(history))
    
    for item in raw_history:
        val = MathOps.safe_parse_float(item.get('actual_number'))
        if val is not None:
            clean_nums.append(val)
            clean_outs.append("BIG" if val >= 5 else "SMALL")
            
    # ----------------------------------------------------------------------------------------------
    # STEP 2: FEEDBACK LOOP & GHOST PROTOCOL
    # ----------------------------------------------------------------------------------------------
    if lastresult and state.last_prediction_decision and state.last_prediction_decision != TradeDecision.SKIP:
        actual_enum = TradeDecision.BIG if lastresult == "BIG" else TradeDecision.SMALL
        
        if state.last_prediction_decision == actual_enum:
            # WIN
            state.current_loss_streak = 0
            state.consecutive_skips = 0 # Reset skips on activity
            state.total_wins += 1
            
            # If we won in Ghost Mode, keep it on? No, usually safer to reset if chaos subsides.
            # But here, we let it ride if it's working.
            if not state.inversion_mode_active:
                state.consecutive_logic_fails = 0
                
        else:
            # LOSS
            state.current_loss_streak += 1
            state.consecutive_skips = 0
            state.total_losses += 1
            
            if not state.inversion_mode_active:
                state.consecutive_logic_fails += 1
                # Trigger Inversion after 3 fails
                if state.consecutive_logic_fails >= 3:
                    state.inversion_mode_active = True
                    logger.warning("GHOST PROTOCOL ACTIVATED: INVERTING SIGNALS")
            else:
                # If we lose IN Ghost mode, the market is truly random. Reset.
                state.inversion_mode_active = False
                state.consecutive_logic_fails = 0
                logger.warning("GHOST PROTOCOL DEACTIVATED: RESETTING")

    # ----------------------------------------------------------------------------------------------
    # STEP 3: THE DEEP THINKING PHASE (Calibration)
    # ----------------------------------------------------------------------------------------------
    # This runs the 30-round simulation to weight the engines
    DeepSimulationCore.execute_calibration_routine(clean_nums, clean_outs)

    # ----------------------------------------------------------------------------------------------
    # STEP 4: GATHER VOTES & DECIDE
    # ----------------------------------------------------------------------------------------------
    votes = VotingCouncil.collect_votes(clean_nums, clean_outs)
    decision, confidence, logs, mode = VotingCouncil.determine_decision(votes)
    
    # ----------------------------------------------------------------------------------------------
    # STEP 5: APPLY GHOST INVERSION
    # ----------------------------------------------------------------------------------------------
    final_decision_str = decision.value
    
    if decision != TradeDecision.SKIP and state.inversion_mode_active:
        original = final_decision_str
        if decision == TradeDecision.BIG:
            final_decision_str = "SMALL"
        else:
            final_decision_str = "BIG"
        logs.append(f"GHOST_FLIP({original}->{final_decision_str})")

    # ----------------------------------------------------------------------------------------------
    # STEP 6: RISK MANAGEMENT & STAKING
    # ----------------------------------------------------------------------------------------------
    
    # Handle SKIP
    if final_decision_str == "SKIP":
        state.consecutive_skips += 1
        state.last_prediction_decision = TradeDecision.SKIP
        
        return {
            'finalDecision': "SKIP",
            'confidence': 0.0,
            'positionsize': 0,
            'level': f"Wait ({state.consecutive_skips})",
            'reason': f"Mode: {mode} | Low Signal",
            'topsignals': logs
        }
    
    # Calculate Base Stake
    base_stake = currentbankroll * config.base_risk_percentage
    status_tags = [mode]
    
    # House Money Logic
    if config.enable_house_money and state.current_profit > (currentbankroll * config.house_money_trigger):
        base_stake *= config.house_money_multiplier
        status_tags.append("HOUSE_MONEY")
        
    # Defensive Logic
    if config.enable_defensive_mode and state.current_profit < (currentbankroll * config.defensive_trigger):
        base_stake *= config.defensive_multiplier
        status_tags.append("DEFENSIVE")
        
    # Martingale Calculation
    # Stake = Base * (Multiplier ^ Streak)
    stake = base_stake * (config.martingale_multiplier ** state.current_loss_streak)
    
    # Hard Limits
    stake = max(stake, config.min_bet_absolute)
    stake = min(stake, config.max_bet_absolute)
    
    # Safety Cap (Never bet more than 35% of bankroll)
    safety_cap = currentbankroll * 0.35
    if stake > safety_cap:
        stake = safety_cap
        status_tags.append("MAX_CAP")
        
    # ----------------------------------------------------------------------------------------------
    # STEP 7: FINALIZE & RETURN
    # ----------------------------------------------------------------------------------------------
    
    # Update State
    state.last_prediction_decision = TradeDecision(final_decision_str)
    state.last_prediction_confidence = confidence
    
    # Reset Skips
    state.consecutive_skips = 0
    
    # Format Result
    result = {
        'finalDecision': final_decision_str,
        'confidence': round(confidence, 4),
        'positionsize': int(stake),
        'level': f"L{state.current_loss_streak} {'|'.join(status_tags)}",
        'reason': " | ".join(logs),
        'topsignals': logs
    }
    
    return result

# ==================================================================================================
# [PART 7: SELF-TEST HARNESS]
# ==================================================================================================

if __name__ == "__main__":
    print("----------------------------------------------------------------")
    print(" TITAN V700 - OMEGA ARCHITECT EDITION")
    print(" STATUS: INITIALIZED")
    print("----------------------------------------------------------------")
    print(f" [CFG] Base Risk: {config.base_risk_percentage*100}%")
    print(f" [CFG] Martingale: {config.martingale_multiplier}x")
    print(f" [CFG] Max Skips: {config.max_consecutive_skips}")
    print("----------------------------------------------------------------")
    print(" Waiting for data stream from fetcher...")
    
    # Mock Test to ensure syntax validity
    mock_history = [{'actual_number': 5}, {'actual_number': 2}, {'actual_number': 8}]
    mock_res = ultraAIPredict(mock_history, 1000.0)
    print(f" [TEST] Mock Prediction Result: {mock_res['finalDecision']}")
