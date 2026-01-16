# ==============================================================================
# MODULE: PREDICTION_ENGINE.PY (V850 - CLOUD EDITION - FIXED)
# COMPATIBILITY: TITAN FETCHER V2026.8
# CLOUD LLM: GROQ API
# FIX: Added missing 'reset_engine_memory' function
# ==============================================================================

import math
import statistics
import asyncio
import aiohttp
import json
import warnings
import os
from typing import Dict, List

warnings.filterwarnings("ignore")

# --- OPTIONAL ML LIBRARIES ---
try:
    import pandas as pd
    import numpy as np
    from sklearn.neural_network import MLPClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    print("[TITAN] ML Libraries not found. Running in Lightweight Mode.")

# =============================================================================
# 1. CONFIGURATION & CONSTANTS
# =============================================================================

class GameConstants:
    BIG = "BIG"
    SMALL = "SMALL"
    SKIP = "SKIP"

class RiskConfig:
    MIN_BET = 50

# =============================================================================
# 2. UTILITY FUNCTIONS
# =============================================================================

def safe_float(value):
    try: return float(value)
    except: return 4.5

def get_outcome_from_number(n):
    """Maps 0-4 to SMALL, 5-9 to BIG."""
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
# 3. CLOUD LLM HANDLER (GROQ)
# =============================================================================

class CloudLLM:
    def __init__(self):
        self.api_key = os.getenv("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") 
        self.url = "https://api.groq.com/openai/v1/chat/completions"
        self.model = "llama-3.1-8b-instant" 
        self.last_pred = None

    async def analyze(self, history):
        if not self.api_key: return None 
        if len(history) % 3 != 0: return self.last_pred

        nums = [d['actual_number'] for d in history[-12:]]
        prompt = (
            f"Lottery Pattern Analysis. Sequence: {nums}. "
            "Numbers 0-4 are SMALL, 5-9 are BIG. "
            "Identify the trend. Return ONLY JSON: {\"prediction\": \"BIG\"} or {\"prediction\": \"SMALL\"}."
        )

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.5,
            "response_format": {"type": "json_object"}
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(self.url, headers=headers, json=payload, timeout=5) as response:
                    if response.status == 200:
                        data = await response.json()
                        content = data['choices'][0]['message']['content']
                        js = json.loads(content)
                        pred = js.get('prediction', '').upper()
                        if pred in ["BIG", "SMALL"]:
                            self.last_pred = pred
                            return pred
        except: pass
        return None

cloud_brain = CloudLLM()

# =============================================================================
# 4. ENGINE TRACKER (ADAPTIVE)
# =============================================================================

class EngineTracker:
    def __init__(self):
        self.initial_state = [1, 1, 0, 1.0] 
        self.stats = {
            'QUANTUM': list(self.initial_state), 
            'PATTERN': list(self.initial_state), 
            'PERCEPTRON': list(self.initial_state), 
            'ML_BRAIN': list(self.initial_state), 
            'CLOUD_LLM': list(self.initial_state)
        }
        self.pending_votes = {}

    def register_votes(self, issue, votes):
        self.pending_votes[issue] = votes
        keys = sorted(list(self.pending_votes.keys()))
        if len(keys) > 10:
            for k in keys[:-10]: del self.pending_votes[k]

    def process_feedback(self, actual_issue, actual_number):
        if str(actual_issue) not in self.pending_votes: return
        votes = self.pending_votes[str(actual_issue)]
        real_outcome = get_outcome_from_number(actual_number)

        for name, pred in votes.items():
            if name not in self.stats: self.stats[name] = list(self.initial_state)
            if pred == real_outcome:
                self.stats[name][0] += 1 
                self.stats[name][2] = 0  
            else:
                self.stats[name][2] += 1 
            self.stats[name][1] += 1 
            if self.stats[name][2] >= 4:
                self.stats[name][3] = 0.2 
            elif self.stats[name][2] == 0:
                self.stats[name][3] = 1.0 

    def get_weight(self, name):
        s = self.stats.get(name, self.initial_state)
        accuracy = s[0] / max(1, s[1])
        multiplier = s[3]
        return max(0.15, accuracy * multiplier)

    def reset(self):
        """Resets tracker stats to initial state."""
        self.__init__()

tracker = EngineTracker()

# =============================================================================
# 5. EXPORTED FUNCTIONS (INCLUDING FIX FOR IMPORT ERROR)
# =============================================================================

def reset_engine_memory():
    """
    Called by fetcher.py to clear AI learning stats.
    Added to fix ImportError.
    """
    global tracker
    tracker.reset()
    print("[TITAN] Engine memory reset.")

# =============================================================================
# 6. LOGIC ENGINES
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

class TitanBrain:
    def __init__(self):
        if ML_AVAILABLE:
            self.clf_nn = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42) 
            self.clf_rf = RandomForestClassifier(n_estimators=50, max_depth=8, random_state=42)
            self.scaler = StandardScaler()
        else:
            self.clf_nn = None
        self.is_trained = False

    def train(self, history):
        if not ML_AVAILABLE or len(history) < 50: return
        if self.is_trained and len(history) % 20 != 0: return
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
            return (p1 * 0.5) + (p2 * 0.5)
        except: return 0.5

ml_brain = TitanBrain()

# =============================================================================
# 7. MAIN CONTROLLER
# =============================================================================

def ultraAIPredict(history: List[Dict], current_bankroll: float, last_label: str = "") -> Dict:
    if not history:
        return {'finalDecision': "SKIP", 'confidence': 0, 'positionsize': 0, 'level': "NO_DATA", 'reason': "History Empty"}

    latest_record = history[-1]
    latest_issue = str(latest_record['issue'])
    latest_num = int(latest_record['actual_number'])
    
    tracker.process_feedback(latest_issue, latest_num)

    ml_brain.train(history)
    
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        llm_pred = loop.run_until_complete(cloud_brain.analyze(history))
        loop.close()
    except: llm_pred = None

    raw_signals = {}
    
    if (e := engine_deep_pattern_v3(history)): raw_signals['PATTERN'] = e
    if (e := engine_quantum_adaptive(history)): raw_signals['QUANTUM'] = e
    if (e := engine_neural_perceptron(history)): raw_signals['PERCEPTRON'] = e
    
    ml_prob = ml_brain.predict(history)
    if ml_prob > 0.55: raw_signals['ML_BRAIN'] = {'pred': "BIG", 'conf': ml_prob, 'name': 'ML_BRAIN'}
    elif ml_prob < 0.45: raw_signals['ML_BRAIN'] = {'pred': "SMALL", 'conf': 1-ml_prob, 'name': 'ML_BRAIN'}

    if llm_pred:
         raw_signals['CLOUD_LLM'] = {'pred': llm_pred, 'conf': 0.6, 'name': 'CLOUD_LLM'}

    active_votes = []
    vote_map_display = {}
    vote_map_tracking = {}
    
    for name, sig in raw_signals.items():
        vote_map_display[name] = sig['pred']
        vote_map_tracking[name] = sig['pred']
        weight = tracker.get_weight(name)
        
        if weight < 0.15: 
            vote_map_display[name] += " (MUTED)"
            continue
            
        sig['weight'] = weight
        active_votes.append(sig)

    try:
        next_issue_id = str(int(latest_issue) + 1)
        tracker.register_votes(next_issue_id, vote_map_tracking)
    except: pass

    if not active_votes:
        return {'finalDecision': "SKIP", 'confidence': 0, 'positionsize': 0, 
                'level': "NO_VALID_SIG", 'reason': "All engines low confidence", 'raw_votes': vote_map_display}

    big_team = [s for s in active_votes if s['pred'] == "BIG"]
    small_team = [s for s in active_votes if s['pred'] == "SMALL"]
    
    big_score = sum(s['weight'] for s in big_team)
    small_score = sum(s['weight'] for s in small_team)
    
    if big_score > small_score:
        draft_decision = "BIG"
        primary = big_team
        final_conf = big_score
    else:
        draft_decision = "SMALL"
        primary = small_team
        final_conf = small_score

    level = "WAITING"
    stake = 0
    supporting_names = [s['name'] for s in primary]
    
    if len(primary) >= 2:
        level = "L1_SCOUT"
        stake = RiskConfig.MIN_BET
    elif len(primary) == 1 and primary[0]['weight'] > 0.60:
        level = "L1_SCOUT (Solo)"
        stake = RiskConfig.MIN_BET

    if "ML_BRAIN" in supporting_names and "CLOUD_LLM" in supporting_names:
        level = "L2_LEADER"
        stake = RiskConfig.MIN_BET 

    ml_conf = next((s['conf'] for s in primary if s['name'] == "ML_BRAIN"), 0)
    if len(primary) >= 3 and "ML_BRAIN" in supporting_names and ml_conf > 0.55:
        level = "L3_SNIPER"
        stake = RiskConfig.MIN_BET

    if level == "WAITING":
        return {'finalDecision': "SKIP", 'confidence': 0, 'positionsize': 0, 
                'level': "WEAK_SIG", 'reason': "Split Vote", 'raw_votes': vote_map_display}

    return {
        'finalDecision': draft_decision,
        'confidence': final_conf,
        'positionsize': stake,
        'level': level,
        'reason': f"{level} | B:{len(big_team)} vs S:{len(small_team)}",
        'topsignals': [f"{s['name']}" for s in primary],
        'raw_votes': vote_map_display
    }
