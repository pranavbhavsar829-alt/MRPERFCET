import aiohttp
import asyncio
import json
import os
import time
import sys
from collections import deque
from datetime import datetime

# --- IMPORT ENGINE ---
try:
    from prediction_engine import ultraAIPredict
    print("[INIT] TITAN V700 ENGINE LINKED.")
except ImportError:
    print("[ERROR] prediction_engine.py is missing!")

# --- NEW CONFIGURATION ---
# Using the new API provided by the user
API_URL = "https://draw.ar-lottery01.com/WinGo/WinGo_1M/GetHistoryIssuePage.json"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "application/json"
}

# --- RENDER DISK PATH ---
if os.path.exists('/var/lib/data'):
    BASE_DIR = '/var/lib/data'
else:
    BASE_DIR = os.path.abspath(os.path.dirname(__file__))

DASHBOARD_FILE = os.path.join(BASE_DIR, 'dashboard_data.json')

# --- SETTINGS ---
HISTORY_LIMIT = 2000       
MIN_DATA_REQUIRED = 30  # START PREDICTING ONLY AFTER 30 RECORDS

# --- MEMORY STORAGE ---
RAM_HISTORY = deque(maxlen=HISTORY_LIMIT)
UI_HISTORY = deque(maxlen=50)

# --- GLOBAL STATE ---
currentbankroll = 10000.0 
last_prediction = {
    "issue": None, "label": "WAITING", "stake": 0, "conf": 0, 
    "level": "---", "reason": "Collecting Data...", "strategy": "BOOTING"
}
session_wins = 0
session_losses = 0
last_win_status = "NONE"

def get_outcome_from_number(n):
    try:
        val = int(float(n))
        return "SMALL" if 0 <= val <= 4 else "BIG"
    except: return None

def update_dashboard(status_text="IDLE", timer_val=0):
    total = session_wins + session_losses
    acc = f"{(session_wins/total)*100:.0f}%" if total > 0 else "0%"
    
    # Logic to show collection progress
    if len(RAM_HISTORY) < MIN_DATA_REQUIRED:
        status_text = f"COLLECTING DATA ({len(RAM_HISTORY)}/{MIN_DATA_REQUIRED})"
    
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
    with open(DASHBOARD_FILE, "w") as f: json.dump(data, f)

async def fetch_data(session):
    try:
        # Some APIs require POST, we try GET first as per user URL
        async with session.get(API_URL, headers=HEADERS, timeout=10) as response:
            if response.status == 200:
                d = await response.json()
                # Standard extraction for common WinGo JSON structures
                if isinstance(d, dict):
                    return d.get('data', {}).get('list', []) or d.get('list', [])
                return d
    except Exception as e:
        print(f"[FETCH ERROR] {e}")
    return None

async def main_loop():
    global currentbankroll, last_prediction, last_win_status, session_wins, session_losses
    last_processed_issue = None
    
    async with aiohttp.ClientSession() as session:
        print("[SYSTEM] Starting Titan V700 with New API...")
        
        while True:
            raw_list = await fetch_data(session)
            if raw_list:
                # API usually gives 10 records, we process all new ones
                for item in reversed(raw_list):
                    try:
                        # Extract issue and number (Supports multiple API formats)
                        issue = str(item.get('issueNumber') or item.get('issue'))
                        num = int(item.get('number') or item.get('result'))
                        
                        if not RAM_HISTORY or RAM_HISTORY[-1]['issue'] != issue:
                            RAM_HISTORY.append({'issue': issue, 'actual_number': num})
                            print(f"[STORED] {issue}: {num} (Total: {len(RAM_HISTORY)})")
                    except: continue

                latest = raw_list[0]
                curr_issue = str(latest.get('issueNumber') or latest.get('issue'))
                curr_code = int(latest.get('number') or latest.get('result'))
                
                now = datetime.now()
                seconds_left = 60 - now.second
                update_dashboard("SYNCED", seconds_left)

                if curr_issue != last_processed_issue:
                    # 1. Check Previous Result
                    if last_prediction['issue'] == curr_issue:
                        real = get_outcome_from_number(curr_code)
                        if last_prediction['label'] != "SKIP":
                            if last_prediction['label'] == real:
                                session_wins += 1
                                last_win_status = "WIN"
                                UI_HISTORY.appendleft({"period": curr_issue, "pred": last_prediction['label'], "result": "WIN"})
                            else:
                                session_losses += 1
                                last_win_status = "LOSS"
                                UI_HISTORY.appendleft({"period": curr_issue, "pred": last_prediction['label'], "result": "LOSS"})

                    # 2. Start Prediction Only if we have 30+ records
                    if len(RAM_HISTORY) >= MIN_DATA_REQUIRED:
                        try:
                            next_issue = str(int(curr_issue) + 1)
                            res = ultraAIPredict(list(RAM_HISTORY), currentbankroll, get_outcome_from_number(curr_code))
                            last_prediction = {
                                "issue": next_issue, 
                                "label": res['finalDecision'], 
                                "stake": res['positionsize'],
                                "conf": res['confidence'],
                                "level": res['level'],
                                "strategy": "SOVEREIGN",
                                "reason": res.get('reason', 'Pattern Match')
                            }
                        except Exception as e:
                            print(f"[ENGINE ERROR] {e}")
                    else:
                        last_prediction['label'] = "COLLECTING"
                        last_prediction['reason'] = f"Wait for 30 records (Current: {len(RAM_HISTORY)})"

                    last_processed_issue = curr_issue

            await asyncio.sleep(2)

if __name__ == '__main__':
    asyncio.run(main_loop())
