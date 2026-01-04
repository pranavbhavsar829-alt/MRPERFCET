import aiohttp
import asyncio
import json
import sqlite3
import time
import sys
import os
from collections import deque
from datetime import datetime

# --- IMPORT ENGINE ---
try:
    from prediction_engine import ultraAIPredict
    print("[INIT] TITAN V700 ENGINE LINKED SUCCESSFULLY.")
except ImportError:
    print("[ERROR] prediction_engine.py is missing! Predictions will fail.")

# --- CONFIGURATION ---
API_URL = "https://harshpredictor.site/api/api.php"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "application/json",
    "Connection": "keep-alive"
}

# --- CRITICAL FIX: FORCE RENDER DISK PATH ---
# We check if /var/lib/data exists (Render Disk). If yes, we MUST use it.
if os.path.exists('/var/lib/data'):
    BASE_DIR = '/var/lib/data'
    print("[SYSTEM] Using Render Persistent Disk: /var/lib/data")
else:
    BASE_DIR = os.path.abspath(os.path.dirname(__file__))
    print(f"[SYSTEM] Using Local Storage: {BASE_DIR}")

DASHBOARD_FILE = os.path.join(BASE_DIR, 'dashboard_data.json')

# --- SETTINGS ---
HISTORY_LIMIT = 2000       
INITIAL_FETCH_SIZE = 1000  
LIVE_FETCH_SIZE = 5        

# --- MEMORY STORAGE ---
RAM_HISTORY = deque(maxlen=HISTORY_LIMIT)
UI_HISTORY = deque(maxlen=50)

# --- GLOBAL STATE ---
currentbankroll = 10000.0 
last_prediction = {
    "issue": None, 
    "label": "WAITING", 
    "stake": 0, 
    "conf": 0, 
    "level": "---", 
    "reason": "System Initializing...", 
    "strategy": "BOOTING"
}
last_win_status = "NONE"
session_wins = 0
session_losses = 0

def get_outcome_from_number(n):
    try:
        val = int(float(n))
        if 0 <= val <= 4: return "SMALL"
        if 5 <= val <= 9: return "BIG"
    except: pass
    return None

def update_dashboard(status_text="IDLE", timer_val=0):
    total = session_wins + session_losses
    acc = f"{(session_wins/total)*100:.0f}%" if total > 0 else "0%"
    
    data = {
        "period": last_prediction['issue'] if last_prediction['issue'] else "---",
        "prediction": last_prediction['label'],
        "confidence": f"{last_prediction.get('conf', 0)*100:.1f}%",
        "stake": last_prediction['stake'],
        "level": last_prediction.get('level', '---'),
        "strategy": last_prediction.get('strategy', 'STANDARD'),
        "reason": last_prediction.get('reason', 'Initializing...'),
        "bankroll": currentbankroll,
        "lastresult_status": last_win_status,
        "status_text": status_text,
        "timer": timer_val,
        "data_size": len(RAM_HISTORY),
        "stats": {"wins": session_wins, "losses": session_losses, "accuracy": acc},
        "history": list(UI_HISTORY),
        "timestamp": time.time()
    }
    
    try:
        temp_file = DASHBOARD_FILE + ".tmp"
        with open(temp_file, "w") as f: 
            json.dump(data, f)
        os.replace(temp_file, DASHBOARD_FILE)
    except Exception as e: 
        print(f"[DASHBOARD ERROR] Could not write JSON: {e}")

async def fetch_data(session, limit=5):
    try:
        params = {'pageSize': limit, 'page': 1}
        async with session.get(API_URL, headers=HEADERS, params=params, timeout=10) as response:
            if response.status == 200:
                d = await response.json()
                return d.get('data', {}).get('list', [])
    except Exception as e:
        print(f"[FETCH ERROR] Connection failed: {e}")
    return None

async def main_loop():
    global currentbankroll, last_prediction, last_win_status, session_wins, session_losses
    print("--- TITAN V700 FETCHER STARTED ---")
    
    last_processed_issue = None
    
    async with aiohttp.ClientSession() as session:
        print(f"[STARTUP] Fetching history...")
        update_dashboard("LOADING HISTORY...", 0)
        
        startup_data = await fetch_data(session, INITIAL_FETCH_SIZE)
        
        if startup_data:
            for item in reversed(startup_data):
                try:
                    curr_issue = str(item['issueNumber'])
                    curr_code = int(item['number'])
                    new_record = {'issue': curr_issue, 'actual_number': curr_code}
                    RAM_HISTORY.append(new_record)
                except: continue
            
            print(f"[STARTUP] Loaded {len(RAM_HISTORY)} records.")
            if len(startup_data) > 0:
                last_processed_issue = str(startup_data[0]['issueNumber'])
        else:
            print("[STARTUP] Failed to load history.")

        print("[SYSTEM] Live Mode Active")
        
        while True:
            raw_list = await fetch_data(session, LIVE_FETCH_SIZE)
            
            if raw_list:
                latest = raw_list[0]
                curr_issue = str(latest['issueNumber'])
                curr_code = int(latest['number'])
                
                now = datetime.now()
                seconds_left = 60 - now.second
                update_dashboard("SYNCED", seconds_left)
                
                if curr_issue != last_processed_issue:
                    print(f"[NEW] {curr_issue}: {curr_code}")
                    
                    new_record = {'issue': curr_issue, 'actual_number': curr_code}
                    if not RAM_HISTORY or RAM_HISTORY[-1]['issue'] != curr_issue:
                        RAM_HISTORY.append(new_record)
                    
                    if last_prediction['issue'] and last_prediction['label'] != "SKIP":
                        real_outcome = get_outcome_from_number(curr_code)
                        predicted = last_prediction['label']
                        is_win = (predicted == real_outcome)
                        
                        if predicted in ["RED", "GREEN"]:
                            real_c = "RED" if curr_code in [0,2,4,6,8] else "GREEN"
                            if curr_code == 5: real_c = "GREEN"
                            if predicted == real_c: is_win = True

                        if is_win:
                            session_wins += 1
                            last_win_status = "WIN"
                            UI_HISTORY.appendleft({"period": last_prediction['issue'], "pred": predicted, "result": "WIN"})
                        else:
                            session_losses += 1
                            last_win_status = "LOSS"
                            UI_HISTORY.appendleft({"period": last_prediction['issue'], "pred": predicted, "result": "LOSS"})

                    next_issue = str(int(curr_issue) + 1)
                    
                    for i in range(5, 0, -1):
                        update_dashboard(f"ANALYZING... {i}", seconds_left)
                        await asyncio.sleep(1)
                        seconds_left = 60 - datetime.now().second

                    if len(RAM_HISTORY) > 10:
                        try:
                            ai_result = ultraAIPredict(list(RAM_HISTORY), currentbankroll, get_outcome_from_number(curr_code))
                            last_prediction = {
                                "issue": next_issue, 
                                "label": ai_result['finalDecision'], 
                                "stake": ai_result['positionsize'],
                                "conf": ai_result['confidence'],
                                "level": ai_result['level'],
                                "strategy": ai_result.get('strategy_status', 'STANDARD'),
                                "reason": ai_result.get('reason', 'Processing...')
                            }
                        except Exception as e:
                            print(f"[ERROR] Engine: {e}")
                            last_prediction['label'] = "ERROR"
                    
                    last_processed_issue = curr_issue

            await asyncio.sleep(1.0)

if __name__ == '__main__':
    if sys.platform == 'win32': 
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main_loop())
