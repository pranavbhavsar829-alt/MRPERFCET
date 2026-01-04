import aiohttp
import asyncio
import json
import time
import sys
import os
from collections import deque
from datetime import datetime

# --- IMPORT ENGINE ---
# Ensure prediction_engine.py is in the same folder
try:
    from prediction_engine import ultraAIPredict
    print("[INIT] Engine Loaded Successfully.")
except ImportError:
    print("[ERROR] prediction_engine.py missing!")

# --- CONFIGURATION ---
API_URL = "https://harshpredictor.site/api/api.php"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "application/json",
    "Connection": "keep-alive"
}

# --- RENDER STORAGE PATHS (MUST MATCH SERVER.PY) ---
if os.path.exists('/var/lib/data'):
    BASE_DIR = '/var/lib/data'
elif os.path.exists('/data'):
    BASE_DIR = '/data'
else:
    BASE_DIR = os.path.abspath(os.path.dirname(__file__))

DASHBOARD_FILE = os.path.join(BASE_DIR, 'dashboard_data.json')

# --- STATE VARIABLES ---
HISTORY_LIMIT = 2000
RAM_HISTORY = deque(maxlen=HISTORY_LIMIT)
currentbankroll = 10000.0
session_wins = 0
session_losses = 0

last_prediction = {
    "issue": None, 
    "label": "WAITING", 
    "stake": 0, 
    "conf": 0, 
    "strategy": "BOOTING",
    "reason": ""
}

def get_outcome_from_number(n):
    try:
        val = int(float(n))
        if 0 <= val <= 4: return "SMALL"
        if 5 <= val <= 9: return "BIG"
    except: pass
    return None

def update_dashboard(status_text="IDLE", timer_val=0):
    """Writes status to JSON file for Server to read."""
    total = session_wins + session_losses
    acc = f"{(session_wins/total)*100:.0f}%" if total > 0 else "0%"
    
    data = {
        "period": last_prediction['issue'] if last_prediction['issue'] else "---",
        "prediction": last_prediction['label'],
        "confidence": f"{last_prediction.get('conf', 0)*100:.1f}%",
        "strategy": last_prediction.get('strategy', 'STANDARD'),
        "status_text": status_text,
        "timer": timer_val,
        "stats": {"wins": session_wins, "losses": session_losses, "accuracy": acc},
        "timestamp": time.time()
    }
    
    # Atomic Write to prevent partial reads
    try:
        temp_file = DASHBOARD_FILE + ".tmp"
        with open(temp_file, "w") as f: 
            json.dump(data, f)
        os.replace(temp_file, DASHBOARD_FILE)
    except Exception as e: 
        print(f"[DASHBOARD ERROR] Write Failed: {e}")

async def fetch_data(session, limit=5):
    try:
        params = {'pageSize': limit, 'page': 1}
        async with session.get(API_URL, headers=HEADERS, params=params, timeout=10) as response:
            if response.status == 200:
                d = await response.json()
                return d.get('data', {}).get('list', [])
    except Exception as e:
        print(f"[FETCH ERROR] {e}")
    return None

async def main_loop():
    global last_prediction, session_wins, session_losses
    print("--- FETCHER STARTED ---")
    
    last_processed_issue = None
    
    async with aiohttp.ClientSession() as session:
        # 1. Startup: Load History
        print("[STARTUP] Loading history...")
        update_dashboard("SYNCING HISTORY...", 0)
        startup_data = await fetch_data(session, 1000)
        
        if startup_data:
            for item in reversed(startup_data):
                try:
                    curr_issue = str(item['issueNumber'])
                    curr_code = int(item['number'])
                    RAM_HISTORY.append({'issue': curr_issue, 'actual_number': curr_code})
                    last_processed_issue = curr_issue
                except: continue
        
        # 2. Main Loop
        while True:
            raw_list = await fetch_data(session, 5)
            
            if raw_list:
                latest = raw_list[0]
                curr_issue = str(latest['issueNumber'])
                curr_code = int(latest['number'])
                
                # Timer Logic
                now = datetime.now()
                seconds_left = 60 - now.second
                update_dashboard("LIVE MONITORING", seconds_left)
                
                # New Result Detected
                if curr_issue != last_processed_issue:
                    print(f"[NEW RESULT] {curr_issue}: {curr_code}")
                    
                    # Store Result
                    if not RAM_HISTORY or RAM_HISTORY[-1]['issue'] != curr_issue:
                        RAM_HISTORY.append({'issue': curr_issue, 'actual_number': curr_code})

                    # Check Win/Loss
                    if last_prediction['issue'] and last_prediction['label'] != "SKIP":
                        real = get_outcome_from_number(curr_code)
                        pred = last_prediction['label']
                        
                        # Violet Logic (5 is Green/Big, 0 is Red/Small)
                        is_win = (pred == real)
                        if pred == "RED" and curr_code in [0,2,4,6,8]: is_win = True
                        if pred == "GREEN" and curr_code in [1,3,5,7,9]: is_win = True
                        if curr_code == 5 and pred == "GREEN": is_win = True 
                        
                        if is_win: session_wins += 1
                        else: session_losses += 1

                    # Predict Next
                    next_issue = str(int(curr_issue) + 1)
                    
                    # "Thinking" Pause for Effect
                    for i in range(3, 0, -1):
                        update_dashboard(f"CALCULATING... {i}", seconds_left)
                        await asyncio.sleep(1)
                        seconds_left = 60 - datetime.now().second

                    # Execute Brain
                    try:
                        ai_result = ultraAIPredict(list(RAM_HISTORY), currentbankroll, get_outcome_from_number(curr_code))
                        last_prediction = {
                            "issue": next_issue,
                            "label": ai_result['finalDecision'],
                            "conf": ai_result['confidence'],
                            "strategy": ai_result.get('level', 'STD'),
                            "reason": ""
                        }
                    except Exception as e:
                        print(f"[ENGINE ERROR] {e}")
                        last_prediction['label'] = "SKIP"
                    
                    last_processed_issue = curr_issue

            await asyncio.sleep(1.0)

if __name__ == '__main__':
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main_loop())
