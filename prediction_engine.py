#!/usr/bin/env python3
"""
=============================================================================
  TITAN V1000 - VISUAL THINKER (CLOUD EDITION)
  
  UPDATES:
  1. CLOUD BRAIN: Uses Groq (Llama 3.1) instead of Local Ollama.
  2. LIVE LOGGING: Prints every Cloud thought instantly to the console.
  3. VISIBILITY: You will see the 'Vote Count' building up in real-time.
=============================================================================
"""

import math
import statistics
import asyncio
import aiohttp
import json
import warnings
import time
import random
import os
from typing import Dict, List, Optional, Any

warnings.filterwarnings("ignore")

try:
    import pandas as pd
    import numpy as np
    from sklearn.neural_network import MLPClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
except ImportError:
    pd = None
    print("[TITAN] ML Libraries not found. Running in Lightweight Mode.")

# =============================================================================
# CONSTANTS & CONFIG
# =============================================================================

# GROQ CONFIGURATION
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "llama-3.1-8b-instant"

class GameConstants:
    BIG = "BIG"
    SMALL = "SMALL"
    SKIP = "SKIP"

class RiskConfig:
    MIN_BET = 50

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
# TRACKER
# =============================================================================

class EngineTracker:
    def __init__(self):
        initial_confidence = [1, 1, 1] 
        self.stats = {
            'QUANTUM': list(initial_confidence), 
            'PATTERN': list(initial_confidence), 
            'PERCEPTRON': list(initial_confidence), 
            'ML_ENSEMBLE': list(initial_confidence), 
            'OLLAMA_COUNCIL': list(initial_confidence) # Kept name for compatibility
        }

    def update(self, actual_outcome, engine_votes):
        for name, pred in engine_votes.items():
            if name in self.stats and pred in ["BIG", "SMALL"]:
                if len(self.stats[name]) >= 10: self.stats[name].pop(0)
                is_correct = 1 if pred == actual_outcome else 0
                self.stats[name].append(is_correct)

    def get_weight(self, name):
        history = self.stats.get(name, [])
        if not history: return 0.5 
        return sum(history) / len(history)
    
    def reset(self):
        self.__init__()

tracker = EngineTracker()

# =============================================================================
# EXPORTED FUNCTION (FIX FOR FETCHER ERROR)
# =============================================================================

def reset_engine_memory():
    """Called by fetcher.py to clear AI learning stats."""
    tracker.reset()
    print("[TITAN] Engine memory reset.")

# =============================================================================
# STANDARD ENGINES
# =============================================================================

def engine_quantum_adaptive(history):
    try:
        nums = [safe_float(d['actual_number']) for d in history[-30:]]
        if len(nums) < 20: return None
        mean = statistics.mean(nums)
        std = statistics.stdev(nums) if len(nums) > 1 else 0
        if std == 0: return None
        z = (nums[-1] - mean) / std
        strength = min(abs(z) / 2.5, 1.0)
        if z > 1.3: return {'pred': "SMALL", 'conf': strength, 'name': 'QUANTUM'}
        if z < -1.3: return {'pred': "BIG", 'conf': strength, 'name': 'QUANTUM'}
    except: pass
    return None

def engine_deep_pattern_v3(history):
    try:
        if len(history) < 60: return None
        outcomes = "".join(["B" if get_outcome(d['actual_number']) == "BIG" else "S" for d in history])
        best_conf = 0
        best_pred = None
        
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
            
            if diff > best_conf and diff > 0.25:
                best_conf = diff
                best_pred = "BIG" if prob_b > 0.5 else "SMALL"
        
        if best_conf > 0:
            return {'pred': best_pred, 'conf': best_conf, 'name': 'PATTERN'}
    except: pass
    return None

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
        
        if prob > 0.53: return {'pred': "BIG", 'conf': dist, 'name': 'PERCEPTRON'}
        if prob < 0.47: return {'pred': "SMALL", 'conf': dist, 'name': 'PERCEPTRON'}
    except: pass
    return None

# =============================================================================
# DEEP BRAIN (THE TIME-LOCKED ANALYST - NOW WITH GROQ)
# =============================================================================

class TitanDeepBrain:
    def __init__(self):
        self.scaler = StandardScaler() if pd else None
        self.models_ready = False
        self.m1 = None
        self.m2 = None
        # Different perspectives to ask Groq
        self.perspectives = [
            "Trend Analysis",
            "Reversal Check",
            "Noise Filtering",
            "Contrarian Logic",
            "Pattern Match"
        ]
        self.api_key = os.getenv("GROQ_API_KEY")

    def train_ensemble(self, history):
        if not pd or len(history) < 100: return
        try:
            df = pd.DataFrame(history)
            df['num'] = df['actual_number'].astype(int)
            df['label'] = df['num'].apply(lambda x: 1 if x >= 5 else 0)
            df['rmean'] = df['num'].rolling(10).mean()
            df['std'] = df['num'].rolling(10).std()
            df = df.fillna(0)
            df['target'] = df['label'].shift(-1)
            train_df = df.dropna()
            
            X = train_df[['rmean', 'std', 'num']]
            y = train_df['target']
            X_s = self.scaler.fit_transform(X)

            self.m1 = MLPClassifier(hidden_layer_sizes=(50,), max_iter=500).fit(X_s, y)
            self.m2 = RandomForestClassifier(n_estimators=50, max_depth=5).fit(X, y)
            self.models_ready = True
        except: pass

    async def forced_contemplation_loop(self, history, time_budget):
        """
        LOCKED LOOP: Keeps asking Groq until time expires.
        """
        if not self.api_key:
            return None, 0.0

        start_time = time.time()
        end_time = start_time + time_budget
        
        votes = []
        iteration = 0
        nums = [d['actual_number'] for d in history[-15:]]
        
        print(f"   [DEEP] Entering Cloud Time-Lock for {time_budget:.0f}s... (Watch the Thoughts)")
        
        while time.time() < end_time:
            iteration += 1
            
            # 1. Pick a perspective
            perspective_idx = iteration % len(self.perspectives)
            task_name = self.perspectives[perspective_idx]
            task_prompt = f"Analyze using {task_name}. Predict SMALL or BIG."
            
            # 2. Ask Groq
            prediction = await self._query_groq(nums, task_prompt)
            
            # 3. PRINT THE THOUGHT (VISUAL FEEDBACK)
            if prediction:
                votes.append(prediction)
                print(f"      -> [THOUGHT {iteration}] {task_name}: {prediction}")
            else:
                print(f"      -> [THOUGHT {iteration}] {task_name}: ...Thinking...")

            # 4. Pace the requests (1 second sleep)
            await asyncio.sleep(1.0)

        elapsed = time.time() - start_time
        print(f"   [DEEP] Analysis Finished. ({iteration} cloud simulations in {elapsed:.1f}s)")
        
        if not votes: return None, 0.0

        big_count = votes.count("BIG")
        small_count = votes.count("SMALL")
        total = big_count + small_count
        
        # Display Final Vote Count
        print(f"   [VOTE] BIG: {big_count} | SMALL: {small_count}")
        
        if total == 0: return None, 0.0
        
        if big_count > small_count:
            conf = big_count / total
            return "BIG", conf
        elif small_count > big_count:
            conf = small_count / total
            return "SMALL", conf
        
        return None, 0.0

    async def _query_groq(self, nums, task):
        nonce = random.randint(1000, 9999)
        prompt = f"Data: {nums}. Question: {task} [ID:{nonce}]. Reply JSON: {{'prediction': 'SMALL' or 'BIG'}}"
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": GROQ_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7, # Little variability for the 'Council' effect
            "response_format": {"type": "json_object"}
        }

        try:
            async with aiohttp.ClientSession() as s:
                async with s.post(GROQ_API_URL, json=payload, headers=headers, timeout=5) as r:
                    if r.status == 200:
                        js = await r.json()
                        content = js['choices'][0]['message']['content']
                        # Parse JSON response
                        try:
                            data = json.loads(content)
                            pred = data.get('prediction')
                            return pred.upper() if pred else None
                        except: return None
        except: return None
        return None

brain = TitanDeepBrain()

# =============================================================================
# MAIN CONTROLLER
# =============================================================================

def ultraAIPredict(history: List[Dict], current_bankroll: float, last_label: str = "", time_budget: int = 5) -> Dict:
    
    # 1. TRAIN ML
    brain.train_ensemble(history)
    
    # 2. RUN TIME-LOCKED ANALYSIS (Uses Groq)
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        ollama_pred, ollama_conf = loop.run_until_complete(brain.forced_contemplation_loop(history, time_budget))
        loop.close()
    except: 
        ollama_pred = None
        ollama_conf = 0.0

    # 3. GATHER RESULTS
    raw_signals = {}
    
    if (e := engine_deep_pattern_v3(history)): raw_signals['PATTERN'] = e
    if (e := engine_quantum_adaptive(history)): raw_signals['QUANTUM'] = e
    if (e := engine_neural_perceptron(history)): raw_signals['PERCEPTRON'] = e
    
    ml_prob = 0.5
    if brain.models_ready and pd:
         try:
            df = pd.DataFrame(history)
            df['num'] = df['actual_number'].astype(int)
            df['rmean'] = df['num'].rolling(10).mean()
            df['std'] = df['num'].rolling(10).std()
            df = df.fillna(0)
            last = df.iloc[[-1]][['rmean', 'std', 'num']]
            p1 = brain.m1.predict_proba(brain.scaler.transform(last))[0][1]
            p2 = brain.m2.predict_proba(last)[0][1]
            ml_prob = (p1 + p2) / 2
         except: pass

    if ml_prob > 0.55: raw_signals['ML_ENSEMBLE'] = {'pred': "BIG", 'conf': ml_prob, 'name': 'ML_ENSEMBLE'}
    elif ml_prob < 0.45: raw_signals['ML_ENSEMBLE'] = {'pred': "SMALL", 'conf': 1-ml_prob, 'name': 'ML_ENSEMBLE'}

    if ollama_pred:
         raw_signals['OLLAMA_COUNCIL'] = {'pred': ollama_pred, 'conf': ollama_conf, 'name': 'OLLAMA_COUNCIL'}

    # 4. FILTERING
    active_votes = []
    vote_map = {} 
    
    for name, sig in raw_signals.items():
        vote_map[name] = sig['pred']
        weight = tracker.get_weight(name)
        if weight < 0.15: continue 
        sig['weight'] = weight
        active_votes.append(sig)

    if not active_votes:
        return {'finalDecision': "SKIP", 'confidence': 0, 'positionsize': 0, 
                'level': "NO_SIG", 'reason': "Silence", 'raw_votes': {}, 'topsignals': []}

    big_team = [s for s in active_votes if s['pred'] == "BIG"]
    small_team = [s for s in active_votes if s['pred'] == "SMALL"]
    
    if len(big_team) > len(small_team):
        draft_decision = "BIG"
        primary = big_team
        opposing = small_team
    else:
        draft_decision = "SMALL"
        primary = small_team
        opposing = big_team

    # 5. VETO
    for opp in opposing:
        if opp['weight'] > 0.75:
             return {'finalDecision': "SKIP", 'confidence': 0, 'positionsize': 0, 
                'level': "VETOED", 'reason': f"Conflict: {opp['name']}", 'raw_votes': vote_map, 'topsignals': []}

    # 6. LEVEL ASSIGNMENT
    level = "WAITING"
    stake = 0
    supporting_names = [s['name'] for s in primary]
    
    if len(primary) >= 2:
        level = "L1_SCOUT"
        stake = RiskConfig.MIN_BET
    elif len(primary) == 1 and primary[0]['weight'] > 0.60:
        level = "L1_SCOUT (Solo)"
        stake = RiskConfig.MIN_BET

    if "ML_ENSEMBLE" in supporting_names and "OLLAMA_COUNCIL" in supporting_names:
        level = "L2_LEADER"
        stake = RiskConfig.MIN_BET 

    ml_conf = next((s['conf'] for s in primary if s['name'] == "ML_ENSEMBLE"), 0)
    if len(primary) >= 3 and "ML_ENSEMBLE" in supporting_names and ml_conf > 0.55:
        level = "L3_SNIPER"
        stake = RiskConfig.MIN_BET

    if level == "WAITING":
        return {'finalDecision': "SKIP", 'confidence': 0, 'positionsize': 0, 
                'level': "WEAK", 'reason': "Split", 'raw_votes': vote_map, 'topsignals': []}

    return {
        'finalDecision': draft_decision,
        'confidence': sum(s['weight'] for s in primary) / len(primary),
        'positionsize': stake,
        'level': level,
        'reason': f"{level} | {len(primary)} vs {len(opposing)}",
        'topsignals': [f"{s['name']}({s['weight']:.1f})" for s in primary],
        'raw_votes': vote_map
    }
