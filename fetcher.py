# ==============================================================================
# MODULE: FETCHER.PY (V2026.7 - SOVEREIGN INTEGRATED + DEEP HISTORY FETCH)
# ==============================================================================

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
    from prediction_engine import ultraAIPredict, get_outcome_from_number, GameConstants
    print("[INIT] TITAN V201 SOVEREIGN CORE LINKED.")
except ImportError as e:
    print(f"\n[CRITICAL ERROR] prediction_engine.py not found: {e}")
    sys.exit()

# --- API CONFIGURATION ---
# Using GET method as per your server configuration
API_URL = "https://api-wo4u.onrender.com/api/get_history"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
    "Origin": "https://draw.ar-lottery01.com",
    "Referer": "https://draw.ar-lottery01.com/"
}

# --- SETTINGS & STORAGE ---
HISTORY_LIMIT = 2000       
MIN_DATA_REQUIRED = 40  
DB_FILE = 'ar_lottery_history.db'
DASHBOARD_FILE = 'dashboard_data.json'

RAM_HISTORY = deque(maxlen=HISTORY_LIMIT)
UI_HISTORY = deque(maxlen=50)

# --- GLOBAL STATE ---
current_bankroll = 10000.0 
last_prediction = {
    "issue": None, "label": "WAITING", "stake": 0, "conf": 0, 
    "level": "---", "reason": "Collecting Data...", "strategy": "BOOTING"
}
session_wins = 0
session_losses = 0
last_win_status = "NONE"

# --- DATABASE HANDLER ---
def ensure_db_setup():
    """Initializes SQLite with the results table."""
    conn = sqlite3.connect(DB_FILE)
    conn.execute('CREATE TABLE IF NOT EXISTS results (issue TEXT PRIMARY KEY, code INTEGER, fetch_time TEXT)')
    conn.commit()
    conn.close()

async def save_to_db(issue, code):
    """Saves records and ensures they are committed to disk."""
    try:
        conn = sqlite3.connect(DB_FILE)
        # Use str(issue) to prevent scientific notation corruption
        conn.execute("INSERT OR IGNORE INTO results (issue, code, fetch_time) VALUES (?, ?, ?)", 
                       (str(issue), int(code), str(datetime.now())))
        conn.commit()
        conn.close()
    except: pass

async def load_db_to_ram():
    """Loads historical data from local storage into the active AI memory."""
    RAM_HISTORY.clear()
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute(f"SELECT issue, code FROM results ORDER BY issue DESC LIMIT {HISTORY_LIMIT}")
        rows = cursor.fetchall()
        conn.close()
        for r in reversed(rows):
            RAM_HISTORY.append({'issue': str(r[0]), 'actual_number': int(r[1])})
        return len(RAM_HISTORY)
    except: return 0

# --- UI DASHBOARD ---
def update_dashboard(status_text="IDLE", timer_val=0):
    total = session_wins + session_losses
    acc = f"{(session_wins/total)*100:.1f}%" if total > 0 else "0.0%"
    
    data = {
        "period": last_prediction['issue'] if last_prediction['issue'] else "---",
        "prediction": last_prediction['label'],
        "confidence": f"{last_prediction.get('conf', 0)*100:.1f}%",
        "stake": last_prediction['stake'],
        "level": last_prediction.get('level', '---'),
        "bankroll": current_bankroll,
        "lastresult_status": last_win_status,
        "status_text": status_text,
        "timer": timer_val,
        "data_size": len(RAM_HISTORY),
        "stats": {"wins": session_wins, "losses": session_losses, "accuracy": acc},
        "history": list(UI_HISTORY),
        "timestamp": time.time()
    }
    try:
        with open(DASHBOARD_FILE + ".tmp", "w") as f: json.dump(data, f)
        os.replace(DASHBOARD_FILE + ".tmp", DASHBOARD_FILE)
    except: pass

# --- API HANDLER ---
async def fetch_api_data(session, size_limit=20):
    """
    Fetches using GET to avoid 405 error.
    Uses query parameters to request more data (size/limit).
    """
    # Common query parameters for pagination in these APIs
    params = {
        "size": size_limit,
        "pageSize": size_limit, 
        "limit": size_limit,
        "count": size_limit,
        "pageNo": 1
    }
    
    try:
        async with session.get(API_URL, headers=HEADERS, params=params, timeout=15) as response:
            if response.status == 200:
                json_data = await response.json(content_type=None)
                # Map various possible JSON structures
                return json_data.get('data', {}).get('list', []) or json_data.get('list', [])
            else:
                print(f"[DEBUG] API Rejection: Status {response.status}")
    except Exception as e:
        print(f"[FETCH ERROR] {e}")
    return None

# --- MAIN LOOP ---
async def main_loop():
    global current_bankroll, last_prediction, last_win_status, session_wins, session_losses
    ensure_db_setup()
    last_processed_issue = None
    
    async with aiohttp.ClientSession() as session:
        print("\n" + "="*64)
        print("   TITAN V201 - SOVEREIGN ENGINE (HYBRID SYNC ACTIVE)")
        print("="*64)
        
        # ---------------------------------------------------------
        # 1. INITIAL DEEP SYNC (Fetch 500 records on boot)
        # ---------------------------------------------------------
        print("[BOOT] Fetching deep history (500 records)...")
        boot_data = await fetch_api_data(session, size_limit=500)
        
        if boot_data:
            count = 0
            # Reverse so we process oldest -> newest
            for item in reversed(boot_data):
                iss = item.get('issueNumber') or item.get('issue')
                num = item.get('number') or item.get('result')
                if iss and num is not None:
                    await save_to_db(iss, num)
                    count += 1
            print(f"[BOOT] Successfully synced {count} historical records.")
        else:
            print("[BOOT WARNING] API returned no data. Retrying with standard fetch...")
            # Fallback if deep fetch fails
            boot_data = await fetch_api_data(session, size_limit=20)
            if boot_data:
                for item in reversed(boot_data):
                    iss = item.get('issueNumber') or item.get('issue')
                    num = item.get('number') or item.get('result')
                    if iss and num is not None:
                        await save_to_db(iss, num)

        await load_db_to_ram()
        print(f"[BOOT] System Ready. Total Data Points: {len(RAM_HISTORY)}")

        # ---------------------------------------------------------
        # 2. REAL-TIME LOOP
        # ---------------------------------------------------------
        while True:
            # Regular fetch (lightweight, only 20 records needed for updates)
            raw_list = await fetch_api_data(session, size_limit=20)
            
            if raw_list:
                # A. Sync any new data found in the list
                for item in reversed(raw_list):
                    iss = str(item.get('issueNumber') or item.get('issue'))
                    num = int(item.get('number') or item.get('result'))
                    
                    # If this issue is not in our RAM, save it
                    if not any(d['issue'] == iss for d in RAM_HISTORY):
                        await save_to_db(iss, num)
                        RAM_HISTORY.append({'issue': iss, 'actual_number': num})
                        print(f"[SYNC] New Result Found: Period {iss} = {num}")

                # B. Identify latest state
                latest = raw_list[0]
                curr_issue = str(latest.get('issueNumber') or latest.get('issue'))
                curr_num = int(latest.get('number') or latest.get('result'))
                
                seconds_left = 60 - datetime.now().second
                update_dashboard("SYNCED", seconds_left)
                
                if curr_issue != last_processed_issue:
                    # C. Evaluate previous prediction result
                    if last_prediction['issue'] == curr_issue:
                        real_outcome = get_outcome_from_number(curr_num)
                        pred_label = last_prediction['label']
                        
                        if pred_label not in ["WAITING", "SKIP", GameConstants.SKIP]:
                            if pred_label == real_outcome:
                                profit = last_prediction['stake'] * 0.98
                                current_bankroll += profit
                                session_wins += 1
                                last_win_status = "WIN"
                                print(f"\n[RESULT] {curr_issue} was {real_outcome} - WIN (+{profit:.0f})")
                            else:
                                current_bankroll -= last_prediction['stake']
                                session_losses += 1
                                last_win_status = "LOSS"
                                print(f"\n[RESULT] {curr_issue} was {real_outcome} - LOSS (-{last_prediction['stake']:.0f})")
                            
                            UI_HISTORY.appendleft({
                                "period": curr_issue, 
                                "pred": pred_label, 
                                "result": last_win_status,
                                "bankroll": round(current_bankroll, 2)
                            })

                    # D. Generate next prediction
                    if len(RAM_HISTORY) >= MIN_DATA_REQUIRED:
                        try:
                            next_issue = str(int(curr_issue) + 1)
                            # Call the original Trident Engine
                            res = ultraAIPredict(list(RAM_HISTORY), current_bankroll, last_prediction['label'])
                            
                            last_prediction = {
                                "issue": next_issue, 
                                "label": res['finalDecision'], 
                                "stake": res['positionsize'],
                                "conf": res['confidence'], 
                                "level": res['level'], 
                                "reason": res.get('reason', 'Analyzing Pattern...')
                            }
                            print(f"[PRED] Target: {next_issue} | Decision: {last_prediction['label']} | Conf: {last_prediction['conf']:.0%}")
                        except Exception as e:
                            print(f"[ENGINE ERROR] {e}")
                    else:
                        print(f"[WARMUP] Data: {len(RAM_HISTORY)}/{MIN_DATA_REQUIRED}")
                    
                    last_processed_issue = curr_issue

            await asyncio.sleep(5)

if __name__ == '__main__':
    try:
        asyncio.run(main_loop())
    except KeyboardInterrupt:
        print("\n[EXIT] Titan Sovereign offline.")
