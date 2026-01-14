#!/usr/bin/env python3
"""
=============================================================================
  TITAN V500 - CLOUD EDITION (RENDER READY)
  
  CHANGES FOR DEPLOYMENT:
  1. Removed Ollama (Too heavy for Render).
  2. Added Groq Cloud API (Runs Llama 3 for free/fast).
  3. Fixed arguments to match fetcher.py.
  4. Added graceful error handling (Bot works even if AI fails).
=============================================================================
"""

import math
import statistics
import random
import traceback
import json
import warnings
import time
from collections import Counter
from typing import Dict, List, Optional, Any

# --- NETWORK LIBRARY ---
try:
    import requests
except ImportError:
    print("[CRITICAL] 'requests' library missing. Install: pip install requests")
    requests = None

# --- DATA SCIENCE LIBRARIES ---
try:
    import pandas as pd
    import numpy as np
    from scipy.stats import entropy
    from sklearn.neural_network import MLPClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
except ImportError:
    # On Render, if these fail, the bot will fall back to basic math engines
    print("[WARNING] ML Libraries missing. Bot will run in 'Lite Mode'.")
    pd = None 

# Suppress warnings
warnings.filterwarnings("ignore")

# =============================================================================
# CONFIGURATION
# =============================================================================

class GameConstants:
    BIG = "BIG"
    SMALL = "SMALL"
    SKIP = "SKIP"

class RiskConfig:
    CONF_STRONG = 0.70
    CONF_SNIPER = 0.85
    MIN_BET = 50
    STOP_LOSS_STREAK = 5

# --- CLOUD AI CONFIG (GROQ) ---
# 1. Get a FREE key here: https://console.groq.com/keys
# 2. Paste it below inside the quotes.
GROQ_API_KEY = "gsk_..."  # <--- PASTE YOUR KEY HERE
GROQ_MODEL = "llama3-8b-8192" 

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
    return None

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
# ENGINE SET A: CLASSIC LOGIC (V200)
# =============================================================================

def engine_quantum_adaptive(history):
    """Bollinger Band Mean Reversion"""
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

def engine_deep_pattern_v3(history):
    """Pattern Search"""
    try:
        if len(history) < 60: return None
        outcomes = "".join(["B" if get_outcome_from_number(d['actual_number']) == "BIG" else "S" for d in history])
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

def engine_neural_perceptron(history):
    """Simple Logic"""
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

# =============================================================================
# ENGINE SET B: AI BRAIN (V400)
# =============================================================================

class TitanBrain:
    def __init__(self):
        if pd:
            self.clf_nn = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=1000, random_state=42) 
            self.clf_rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
            self.scaler = StandardScaler()
        else:
            self.clf_nn = None
        self.is_trained = False
        self.last_ai_pred = None

    def train(self, history):
        if not self.clf_nn or len(history) < 100: return
        if self.is_trained and len(history) % 50 != 0: return

        try:
            df = pd.DataFrame(history)
            df['num'] = df['actual_number'].astype(int)
            df['label'] = df['num'].apply(lambda x: 1 if x >= 5 else 0)
            
            df['rmean'] = df['num'].rolling(10).mean()
            def roll_ent(s): return entropy(s.value_counts(), base=2)
            df['ent'] = df['num'].rolling(20).apply(roll_ent, raw=False)
            
            delta = df['num'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            df['rsi'] = 100 - (100 / (1 + (gain/loss)))
            
            df = df.fillna(0)
            df['target'] = df['label'].shift(-1)
            df = df.dropna()
            
            feats = ['rmean', 'ent', 'num', 'rsi'] 
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
            def roll_ent(s): return entropy(s.value_counts(), base=2)
            df['ent'] = df['num'].rolling(20).apply(roll_ent, raw=False)
            delta = df['num'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            df['rsi'] = 100 - (100 / (1 + (gain/loss)))
            df = df.fillna(0)
            
            last = df.iloc[[-1]][['rmean', 'ent', 'num', 'rsi']]
            
            p1 = self.clf_nn.predict_proba(self.scaler.transform(last))[0][1]
            p2 = self.clf_rf.predict_proba(last)[0][1]
            return (p1 * 0.6) + (p2 * 0.4)
        except: return 0.5

    def ask_cloud_ai(self, history):
        """Replaces Ollama with Groq (Cloud Llama 3)"""
        if not requests or "gsk_" not in GROQ_API_KEY: 
            return None
            
        # Limit API calls to every 5 rounds to avoid rate limits
        if len(history) % 5 != 0: return self.last_ai_pred
        
        nums = [d['actual_number'] for d in history[-15:]]
        prompt = f"Lottery Data: {nums}. The game is 'Big' (5-9) or 'Small' (0-4). Based on pattern, predict the ONE next outcome. Reply ONLY JSON: {{'prediction': 'BIG'}} or {{'prediction': 'SMALL'}}."
        
        try:
            headers = {
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": GROQ_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.5,
                "response_format": {"type": "json_object"}
            }
            
            resp = requests.post("https://api.groq.com/openai/v1/chat/completions", 
                                 headers=headers, 
                                 json=payload, 
                                 timeout=3)
            
            if resp.status_code == 200:
                js = resp.json()
                content = js['choices'][0]['message']['content']
                pred = json.loads(content).get('prediction')
                if pred in ["BIG", "SMALL"]:
                    self.last_ai_pred = pred
                    return pred
        except Exception as e: 
            print(f"[AI ERROR] {e}")
            pass
            
        return self.last_ai_pred

brain = TitanBrain()

# =============================================================================
# MAIN CONTROLLER
# =============================================================================

def ultraAIPredict(history: List[Dict], current_bankroll: float, last_outcome: str = "WAITING") -> Dict:
    """
    Main entry point called by fetcher.py
    """
    # 1. Train Brain (if data sufficient)
    brain.train(history)
    
    signals = []
    
    # 2. RUN CLASSIC ENGINES
    if (e1 := engine_quantum_adaptive(history)): signals.append(e1)
    if (e2 := engine_deep_pattern_v3(history)): signals.append(e2)
    if (e3 := engine_neural_perceptron(history)): signals.append(e3)
    
    # 3. RUN ML BRAIN (Scikit-Learn)
    ml_prob = brain.predict(history)
    if ml_prob > 0.60:
        signals.append({'pred': "BIG", 'conf': ml_prob, 'name': 'ML_SCI'})
    elif ml_prob < 0.40:
        signals.append({'pred': "SMALL", 'conf': 1-ml_prob, 'name': 'ML_SCI'})
        
    # 4. RUN CLOUD AI (Groq / Llama 3)
    ai_pred = brain.ask_cloud_ai(history)
    if ai_pred:
        signals.append({'pred': ai_pred, 'conf': 0.65, 'name': 'CLOUD_AI'})
        
    # 5. VOTE AGGREGATION
    votes = {"BIG": 0.0, "SMALL": 0.0}
    log_reasons = []
    
    for s in signals:
        w = 1.0
        if s['name'] == 'ML_SCI': w = 1.5
        if s['name'] == 'QUANTUM': w = 1.2
        if s['name'] == 'CLOUD_AI': w = 1.3
        
        votes[s['pred']] += s['conf'] * w
        log_reasons.append(f"{s['name']}")
        
    total = votes["BIG"] + votes["SMALL"]
    
    decision = "SKIP"
    conf = 0.0
    level = "---"
    reason_text = "Analyzing..."

    if total > 0:
        if votes["BIG"] > votes["SMALL"]:
            decision = "BIG"
            conf = votes["BIG"] / total
        else:
            decision = "SMALL"
            conf = votes["SMALL"] / total
        
        reason_text = f"Signals: {', '.join(list(set(log_reasons)))}"

    # 6. RISK MANAGEMENT
    stake = RiskConfig.MIN_BET
    level = "STD"
    
    if conf > RiskConfig.CONF_SNIPER:
        stake *= 2
        level = "SNIPER 🔥"
    elif conf < 0.55:
        decision = "SKIP"
        stake = 0
        level = "LOW_CONF"
        reason_text = "Low Confidence (<55%)"
        
    return {
        'finalDecision': decision,
        'confidence': conf,
        'positionsize': int(stake),
        'level': level,
        'reason': reason_text
    }
