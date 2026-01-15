#!/usr/bin/env python3
"""
=============================================================================
  TITAN V700 - STRICT HIERARCHY (RENDER COMPATIBLE)
  
  FIXES:
  1. Added 'reset_engine_memory' to fix the Import Error.
  2. Implements Strict Level 1/2/3 Logic.
  3. Uses Ollama (Local) or falls back gracefully.
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

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3"

class GameConstants:
    BIG = "BIG"
    SMALL = "SMALL"
    SKIP = "SKIP"

class RiskConfig:
    MIN_BET = 50
    STOP_LOSS_STREAK = 5

# =============================================================================
# UTILS
# =============================================================================

def safe_float(value):
    try: return float(value)
    except: return 4.5

def get_outcome_from_number(n):
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
# ENGINES
# =============================================================================

# --- ENGINE 1: QUANTUM ADAPTIVE ---
def engine_quantum_adaptive(history):
    try:
        nums = [safe_float(d['actual_number']) for d in history[-30:]]
        if len(nums) < 20: return None
        mean = statistics.mean(nums)
        std = statistics.stdev(nums) if len(nums) > 1 else 0
        if std == 0: return None
        z = (nums[-1] - mean) / std
        strength = min(abs(z) / 2.5, 1.0)
        
        if z > 1.4: return {'pred': "SMALL", 'conf': strength, 'name': 'QUANTUM'}
        if z < -1.4: return {'pred': "BIG", 'conf': strength, 'name': 'QUANTUM'}
    except: pass
    return None

# --- ENGINE 2: DEEP PATTERN V3 ---
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
            
            if diff > best_conf and diff > 0.3:
                best_conf = diff
                best_pred = "BIG" if prob_b > 0.5 else "SMALL"
                best_name = f"PATTERN({depth})"
        
        if best_conf > 0:
            return {'pred': best_pred, 'conf': best_conf, 'name': best_name}
    except: pass
    return None

# --- ENGINE 3: NEURAL PERCEPTRON ---
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
        
        if prob > 0.55: return {'pred': "BIG", 'conf': dist, 'name': 'PERCEPTRON'}
        if prob < 0.45: return {'pred': "SMALL", 'conf': dist, 'name': 'PERCEPTRON'}
    except: pass
    return None

# --- ENGINE 4 & 5: AI BRAIN (ML + OLLAMA) ---
class TitanBrain:
    def __init__(self):
        if pd:
            self.clf_nn = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=2000, random_state=42) 
            self.clf_rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
            self.scaler = StandardScaler()
        else:
            self.clf_nn = None
        self.is_trained = False
        self.last_ollama_pred = None

    def train(self, history):
        if not self.clf_nn or len(history) < 50: return
        if self.is_trained and len(history) % 10 != 0: return

        try:
            df = pd.DataFrame(history)
            df['num'] = df['actual_number'].astype(int)
            df['label'] = df['num'].apply(lambda x: 1 if x >= 5 else 0)
            
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
            
            current_mean = last['rmean'].values[0]
            ml_confidence = (p1 * 0.5) + (p2 * 0.5)
            
            if current_mean > 7.0: ml_confidence -= 0.15 
            if current_mean < 2.0: ml_confidence += 0.15 

            return max(0.0, min(1.0, ml_confidence))
        except: return 0.5

    async def ask_ollama(self, history):
        if len(history) % 5 != 0: return self.last_ollama_pred
        nums = [d['actual_number'] for d in history[-15:]]
        prompt = f"Data: {nums}. Analyze trend. Next likely? Reply JSON: {{'prediction': 'SMALL' or 'BIG'}}"
        try:
            async with aiohttp.ClientSession() as s:
                async with s.post(OLLAMA_URL, json={"model":OLLAMA_MODEL, "prompt":prompt, "stream":False, "format":"json"}) as r:
                    if r.status==200:
                        js = await r.json()
                        pred = json.loads(js['response']).get('prediction')
                        if pred and pred.upper() in ['BIG', 'SMALL']:
                            self.last_ollama_pred = pred.upper()
                            return self.last_ollama_pred
        except: pass
        return None

brain = TitanBrain()

# =============================================================================
# EXPORTED FUNCTION: RESET MEMORY (CRITICAL FIX)
# =============================================================================
def reset_engine_memory():
    """Clears the AI memory when a win/loss streak limit is hit."""
    global brain
    brain = TitanBrain()
    print("[ENGINE] Memory Wiped (Circuit Breaker Reset)")

# =============================================================================
# MAIN CONTROLLER - STRICT PRIORITY
# =============================================================================

def ultraAIPredict(history: List[Dict], current_bankroll: float, last_label: str = "") -> Dict:
    # 1. Train & Run Async Ollama
    brain.train(history)
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        ollama_result = loop.run_until_complete(brain.ask_ollama(history))
        loop.close()
    except: ollama_result = None

    signals = []
    
    # Gather Signals
    if (e_pat := engine_deep_pattern_v3(history)): signals.append(e_pat)
    if (e_quant := engine_quantum_adaptive(history)): signals.append(e_quant)
    if (e_perc := engine_neural_perceptron(history)): signals.append(e_perc)

    # ML Brain
    ml_prob = brain.predict(history)
    if ml_prob > 0.55: signals.append({'pred': "BIG", 'conf': ml_prob, 'name': 'ML_BRAIN'})
    elif ml_prob < 0.45: signals.append({'pred': "SMALL", 'conf': 1-ml_prob, 'name': 'ML_BRAIN'})
            
    # Ollama
    if ollama_result in ["BIG", "SMALL"]:
        signals.append({'pred': ollama_result, 'conf': 0.70, 'name': 'OLLAMA'})

    if not signals:
        return {'finalDecision': "SKIP", 'confidence': 0, 'positionsize': 0, 'level': "NO_SIG", 'reason': "Silence", 'topsignals': []}

    # Vote Counting
    votes_big = [s for s in signals if s['pred'] == "BIG"]
    votes_small = [s for s in signals if s['pred'] == "SMALL"]
    
    if len(votes_big) > len(votes_small):
        decision = "BIG"
        winning_team = votes_big
        opposing_team = votes_small
    else:
        decision = "SMALL"
        winning_team = votes_small
        opposing_team = votes_big

    # Calculate Confidence (Weighted)
    total_conf = sum(s['conf'] for s in winning_team)
    total_opp = sum(s['conf'] for s in opposing_team)
    avg_conf = total_conf / (total_conf + total_opp) if (total_conf + total_opp) > 0 else 0

    # --- STRICT HIERARCHY CHECKS ---
    supporting_names = [s['name'] for s in winning_team]
    count = len(winning_team)
    
    has_ml = "ML_BRAIN" in supporting_names
    has_ollama = "OLLAMA" in supporting_names
    
    stake = 0
    level_name = f"WAITING ({count} Votes)"
    
    # LEVEL 1: Any 2 Engines (Scouts)
    if count >= 2:
        level_name = "L1_SCOUT"
        stake = RiskConfig.MIN_BET
    
    # LEVEL 2: ML + OLLAMA (Strict Requirement)
    if has_ml and has_ollama:
        level_name = "L2_LEADER (ML+AI)"
        stake = RiskConfig.MIN_BET
    
    # LEVEL 3: ML + OLLAMA + 1 OTHER (Sniper)
    if has_ml and has_ollama and count >= 3:
        level_name = "L3_SNIPER"
        stake = RiskConfig.MIN_BET

    # FAILSAFE: If no level triggers, SKIP.
    if level_name.startswith("WAITING"):
        decision = "SKIP"
        stake = 0

    log_str = [f"{s['name']}:{s['pred']}" for s in signals]

    return {
        'finalDecision': decision,
        'confidence': avg_conf,
        'positionsize': int(stake),
        'level': level_name,
        'reason': f"{level_name} | Support: {', '.join(supporting_names)}",
        'topsignals': log_str
    }
