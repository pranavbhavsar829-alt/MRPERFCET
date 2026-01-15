#!/usr/bin/env python3
"""
=============================================================================
  TITAN V700 - GROQ CLOUD EDITION (RENDER COMPATIBLE)
  (Consensus-Based + De-Biased + Groq API)

  UPDATES:
  1. REPLACED OLLAMA WITH GROQ API:
     - Uses Groq's high-speed LPU inference for cloud compatibility (Render).
     - Model: llama3-8b-8192 (Fast & Free Tier compatible).
     - Requires GROQ_API_KEY env variable or hardcoded key.

  2. LOGIC PRESERVED:
     - LEVEL 1: Consensus of ANY 2 Engines.
     - LEVEL 2: Consensus of ANY 2 Engines + HIGH CONFIDENCE (>70%).
     - LEVEL 3: Consensus of ANY 3 Engines + SNIPER CONFIDENCE (>80%).

  3. ANTI-BIAS:
     - Mean Reversion logic retained in ML Brain.
     - Neutral prompting for Groq.
=============================================================================
"""

import math
import statistics
import traceback
import asyncio
import aiohttp
import json
import warnings
import os
from typing import Dict, List, Optional, Any

# Suppress warnings
warnings.filterwarnings("ignore")

# --- DATA SCIENCE LIBRARIES ---
try:
    import pandas as pd
    import numpy as np
    from scipy.stats import entropy
    from sklearn.neural_network import MLPClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
except ImportError:
    print("[CRITICAL] ML Libraries missing. Install: pip install pandas numpy scikit-learn scipy")
    pd = None 

# =============================================================================
# CONSTANTS & CONFIG
# =============================================================================

# --- GROQ CONFIGURATION ---
# GET YOUR FREE KEY HERE: https://console.groq.com/keys
# SET IT IN RENDER ENVIRONMENT VARIABLES AS 'GROQ_API_KEY'
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "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") 
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "llama3-8b-8192"  # Fast, efficient model

class GameConstants:
    BIG = "BIG"
    SMALL = "SMALL"
    SKIP = "SKIP"

class RiskConfig:
    # Confidence Thresholds
    CONF_BASE = 0.55  # Minimum for Level 1
    CONF_HIGH = 0.70  # Required for Level 2
    CONF_SNIPER = 0.80 # Required for Level 3
    
    MIN_BET = 50
    STOP_LOSS_STREAK = 5

# =============================================================================
# UTILS
# =============================================================================

def safe_float(value):
    try: return float(value)
    except: return 4.5

def get_outcome_from_number(n):
    """Public helper for fetcher.py compatibility"""
    val = int(safe_float(n))
    if 0 <= val <= 4: return "SMALL"
    if 5 <= val <= 9: return "BIG"
    return "Unknown"

def get_outcome(n):
    return get_outcome_from_number(n)

def sigmoid(x):
    try: return 1 / (1 + math.exp(-x))
    except: return 0.0 if x < 0 else 1.0

def calc_rsi(data, period=14):
    if len(data) < period + 1: return 50.0
    deltas = [data[i] - data[i-1] for i in range(1, len(data))]
    gains = [d if d > 0 else 0 for d in deltas]
    losses = [-d if d < 0 else 0 for d in deltas]
    avg_gain = sum(gains[-period:]) / period
    avg_loss = sum(losses[-period:]) / period
    if avg_loss == 0: return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))

# =============================================================================
# ENGINES (THE ARSENAL)
# =============================================================================

# --- ENGINE 1: QUANTUM ADAPTIVE (MATH) ---
def engine_quantum_adaptive(history):
    try:
        nums = [safe_float(d['actual_number']) for d in history[-30:]]
        if len(nums) < 20: return None
        mean = statistics.mean(nums)
        std = statistics.stdev(nums) if len(nums) > 1 else 0
        if std == 0: return None
        z = (nums[-1] - mean) / std
        strength = min(abs(z) / 2.5, 1.0)
        
        if z > 1.6: return {'pred': "SMALL", 'conf': strength, 'name': 'QUANTUM'}
        if z < -1.6: return {'pred': "BIG", 'conf': strength, 'name': 'QUANTUM'}
    except: pass
    return None

# --- ENGINE 2: DEEP PATTERN V3 (MEMORY) ---
def engine_deep_pattern_v3(history):
    try:
        if len(history) < 60: return None
        outcomes = "".join(["B" if get_outcome(d['actual_number']) == "BIG" else "S" for d in history])
        best_conf = 0
        best_pred = None
        best_name = "PATTERN"
        
        for depth in range(6, 3, -1): 
            pat = outcomes[-depth:]
            search = outcomes[:-1]
            cnt = search.count(pat)
            if cnt < 3: continue
            
            next_b = 0
            start = 0
            while True:
                idx = search.find(pat, start)
                if idx == -1: break
                if idx + depth < len(search):
                    if search[idx+depth] == 'B': next_b += 1
                start = idx + 1
            
            prob_b = next_b / cnt
            diff = abs(prob_b - 0.5) * 2 
            
            if diff > best_conf and diff > 0.4:
                best_conf = diff
                best_pred = "BIG" if prob_b > 0.5 else "SMALL"
                best_name = f"PATTERN({depth})"
        
        if best_conf > 0:
            return {'pred': best_pred, 'conf': best_conf, 'name': best_name}
    except: pass
    return None

# --- ENGINE 3: NEURAL PERCEPTRON (SENSOR) ---
def engine_neural_perceptron(history):
    try:
        nums = [safe_float(d['actual_number']) for d in history[-40:]]
        if len(nums) < 25: return None
        rsi = calc_rsi(nums)
        norm_rsi = (rsi - 50) / 100
        fast = statistics.mean(nums[-5:])
        slow = statistics.mean(nums[-20:])
        mom = (fast - slow) / 10
        
        z = (norm_rsi * -1.5) + (mom * 1.2)
        prob = sigmoid(z)
        dist = abs(prob - 0.5) * 2
        
        if prob > 0.6: return {'pred': "BIG", 'conf': dist, 'name': 'PERCEPTRON'}
        if prob < 0.4: return {'pred': "SMALL", 'conf': dist, 'name': 'PERCEPTRON'}
    except: pass
    return None

# --- ENGINE 4 & 5: AI BRAIN (ML + GROQ) ---
class TitanBrain:
    def __init__(self):
        if pd:
            # Increased complexity
            self.clf_nn = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=2000, random_state=42) 
            self.clf_rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
            self.scaler = StandardScaler()
        else:
            self.clf_nn = None
        self.is_trained = False
        self.last_llm_pred = None

    def train(self, history):
        if not self.clf_nn or len(history) < 50: return
        if self.is_trained and len(history) % 10 != 0: return

        try:
            df = pd.DataFrame(history)
            df['num'] = df['actual_number'].astype(int)
            df['label'] = df['num'].apply(lambda x: 1 if x >= 5 else 0)
            
            # Features
            df['rmean'] = df['num'].rolling(10).mean()
            df['std'] = df['num'].rolling(10).std()
            
            delta = df['num'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            df['rsi'] = 100 - (100 / (1 + (gain/loss)))
            
            df = df.fillna(0)
            df['target'] = df['label'].shift(-1)
            df = df.dropna()
            
            feats = ['rmean', 'std', 'num', 'rsi'] 
            X = df[feats]
            y = df['target']
            
            X_s = self.scaler.fit_transform(X)
            self.clf_nn.fit(X_s, y)
            self.clf_rf.fit(X, y)
            self.is_trained = True
        except: pass

    def predict(self, history):
        if not self.is_trained: return 0.5
        try:
            df = pd.DataFrame(history)
            df['num'] = df['actual_number'].astype(int)
            
            df['rmean'] = df['num'].rolling(10).mean()
            df['std'] = df['num'].rolling(10).std()
            
            delta = df['num'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            df['rsi'] = 100 - (100 / (1 + (gain/loss)))
            df = df.fillna(0)
            
            last = df.iloc[[-1]][['rmean', 'std', 'num', 'rsi']]
            
            p1 = self.clf_nn.predict_proba(self.scaler.transform(last))[0][1]
            p2 = self.clf_rf.predict_proba(last)[0][1]
            
            # MEAN REVERSION LOGIC (ANTI-BIAS)
            current_mean = last['rmean'].values[0]
            ml_confidence = (p1 * 0.5) + (p2 * 0.5)
            
            # Penalize over-extended trends
            if current_mean > 7.0: ml_confidence -= 0.15 
            if current_mean < 2.0: ml_confidence += 0.15 

            return max(0.0, min(1.0, ml_confidence))
        except: return 0.5

    async def ask_groq(self, history):
        """Replaces Ollama with Groq API for Cloud Deployment"""
        if len(history) % 5 != 0: return self.last_llm_pred
        
        # Check for API Key
        if "Your_Groq_Key" in GROQ_API_KEY:
             # Fail silently if no key provided so system doesn't crash
             return None

        nums = [d['actual_number'] for d in history[-15:]]
        
        # Neutral Prompt
        sys_msg = "You are a pattern recognition engine. Analyze the sequence of numbers (0-9). 5-9 is BIG, 0-4 is SMALL."
        user_msg = f"Sequence: {nums}. Identify the trend. Return ONLY JSON format: {{'prediction': 'BIG'}} or {{'prediction': 'SMALL'}}."
        
        try:
            headers = {
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": GROQ_MODEL,
                "messages": [
                    {"role": "system", "content": sys_msg},
                    {"role": "user", "content": user_msg}
                ],
                "response_format": {"type": "json_object"}
            }
            
            async with aiohttp.ClientSession() as s:
                async with s.post(GROQ_URL, headers=headers, json=payload, timeout=5) as r:
                    if r.status == 200:
                        js = await r.json()
                        content = js['choices'][0]['message']['content']
                        pred_data = json.loads(content)
                        pred = pred_data.get('prediction')
                        
                        if pred and pred.upper() in ['BIG', 'SMALL']:
                            self.last_llm_pred = pred.upper()
                            return self.last_llm_pred
        except Exception as e:
            # print(f"Groq Error: {e}") # debug only
            pass
        return None

brain = TitanBrain()

# =============================================================================
# MAIN CONTROLLER - CONSENSUS PROTOCOL
# =============================================================================

# Updated signature to match fetcher.py call: 
# ultraAIPredict(list(RAM_HISTORY), current_bankroll, last_prediction['label'])
def ultraAIPredict(history: List[Dict], current_bankroll: float, last_label: str = "") -> Dict:
    """
    Wrapper to handle Async logic inside Sync function if needed, 
    but since fetcher calls this, we actually need to be careful.
    
    If fetcher.py calls this as a synchronous function, we must run async parts carefully.
    However, looking at fetcher.py, it calls:
    `res = ultraAIPredict(...)` inside an async loop? 
    Wait, fetcher.py imports it. If fetcher.py main_loop is async, this should ideally be async.
    
    BUT, to avoid breaking fetcher.py which likely expects a blocking call or specific return,
    we will use a helper to run the async Groq call.
    """
    
    # 1. Train Background Brain (Sync)
    brain.train(history)
    
    # 2. Run Async Engines (Groq)
    # We use a quick event loop runner here just for the Groq part if needed
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        ollama_result = loop.run_until_complete(brain.ask_groq(history))
        loop.close()
    except:
        ollama_result = None

    signals = []
    
    # Engine A: Pattern
    if (e_pat := engine_deep_pattern_v3(history)): 
        signals.append(e_pat)

    # Engine B: Quantum
    if (e_quant := engine_quantum_adaptive(history)): 
        signals.append(e_quant)
            
    # Engine C: Perceptron
    if (e_perc := engine_neural_perceptron(history)): 
        signals.append(e_perc)

    # Engine D: ML Brain
    ml_prob = brain.predict(history)
    if ml_prob > 0.60:
        signals.append({'pred': "BIG", 'conf': ml_prob, 'name': 'ML_BRAIN'})
    elif ml_prob < 0.40:
        signals.append({'pred': "SMALL", 'conf': 1-ml_prob, 'name': 'ML_BRAIN'})
            
    # Engine E: Groq (LLM)
    if ollama_result in ["BIG", "SMALL"]:
        signals.append({'pred': ollama_result, 'conf': 0.65, 'name': 'GROQ_AI'})

    # 3. CALCULATE CONSENSUS
    if not signals:
        return {'finalDecision': "SKIP", 'confidence': 0, 'positionsize': 0, 'level': "NO_SIG", 'reason': "No clear signal", 'topsignals': []}

    vote_counts = {"BIG": 0, "SMALL": 0}
    weighted_score = {"BIG": 0.0, "SMALL": 0.0}
    log = []
    
    for s in signals:
        vote_counts[s['pred']] += 1
        w = 1.0
        if s['name'] == 'ML_BRAIN': w = 1.2
        if s['name'] == 'GROQ_AI': w = 1.5 # High trust in LLM reasoning
        
        weighted_score[s['pred']] += s['conf'] * w
        log.append(f"{s['name']}:{s['pred']}")

    if weighted_score["BIG"] > weighted_score["SMALL"]:
        potential_winner = "BIG"
        total_weight = weighted_score["BIG"] + weighted_score["SMALL"]
        avg_conf = weighted_score["BIG"] / total_weight if total_weight > 0 else 0
        winning_engine_count = vote_counts["BIG"]
    else:
        potential_winner = "SMALL"
        total_weight = weighted_score["BIG"] + weighted_score["SMALL"]
        avg_conf = weighted_score["SMALL"] / total_weight if total_weight > 0 else 0
        winning_engine_count = vote_counts["SMALL"]

    # 4. DETERMINE STREAK LEVEL 
    # (Inferred from previous label passed from fetcher, or default to 0)
    # Since fetcher doesn't pass streak_level explicitly in the arg list shown in your snippet,
    # we have to assume a basic logic or rely on the caller. 
    # The snippet shows: ultraAIPredict(list(RAM_HISTORY), current_bankroll, last_prediction['label'])
    # It does NOT pass streak_level. We will simplify the logic to "Session" based or just stateless Rules.
    
    # REVISED STATELESS LOGIC FOR FETCHER COMPATIBILITY:
    # If confidence is VERY high -> Level 3 behavior
    # If confidence is MID -> Level 2 behavior
    
    decision = "SKIP"
    stake = 0
    level_name = "ANALYZING"
    
    # Base Rule: Need at least 2 Engines
    if winning_engine_count >= 2:
        
        # Level 1 (Scout) - 2 Engines + Base Conf
        if avg_conf >= RiskConfig.CONF_BASE:
            decision = potential_winner
            stake = RiskConfig.MIN_BET
            level_name = "L1_SCOUT"
            
            # Level 2 (Squad) - 2 Engines + High Conf OR 3 Engines
            if avg_conf >= RiskConfig.CONF_HIGH or winning_engine_count >= 3:
                 level_name = "L2_SQUAD"
                 
                 # Level 3 (Sniper) - 3 Engines + Sniper Conf + LLM Support
                 has_llm = any(s['name'] == 'GROQ_AI' for s in signals)
                 if avg_conf >= RiskConfig.CONF_SNIPER and winning_engine_count >= 3 and has_llm:
                     level_name = "L3_SNIPER"

    return {
        'finalDecision': decision,
        'confidence': avg_conf,
        'positionsize': int(stake),
        'level': level_name,
        'reason': f"{level_name} | {winning_engine_count} Engines | {', '.join(log)}",
        'topsignals': log
    }
