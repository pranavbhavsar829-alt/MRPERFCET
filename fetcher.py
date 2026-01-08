# ==============================================================================
# MODULE: FETCHER.PY (V2026.9 - GHOST PROTOCOL EDITION)
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

# --- GHOST PROTOCOL & SAFETY VARS ---
consecutive_wins = 0        # For Profit Lock
consecutive_real_losses = 0 # Triggers Ghost Mode
ghost_mode_active = False   # The Flag
virtual_win_streak = 0      # Counts wins inside Ghost Mode
cooldown_counter = 0        # For Profit Lock Pauses

# --- DATABASE HANDLER ---
def ensure_db_setup():
    """Initializes SQLite with the results table."""
    conn = sqlite3.connect(DB_FILE)
    conn.execute('CREATE TABLE IF NOT EXISTS results (issue TEXT PRIMARY KEY, code INTEGER, fetch_time TEXT)')
    conn.commit()
    conn.close()

async def save_to_db(issue, code):
    try:
        conn = sqlite3.connect(DB_FILE)
        conn.execute("INSERT OR IGNORE INTO results (issue, code, fetch_time) VALUES (?, ?, ?)", 
                       (str(issue), int(code), str(datetime.now())))
        conn.commit()
        conn.close()
    except: pass

async def load_db_to_ram():
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
    
    # DYNAMIC STATUS UPDATE
    if cooldown_counter > 0:
        status_text = f"PROFIT LOCK ({cooldown_counter})"
    elif ghost_mode_active:
        status_text = f"👻 GHOST MODE ({virtual_win_streak}/2)"
        
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
        "stats": {
            "wins": session_wins, 
            "losses": session_losses, 
            "accuracy": acc,
            "real_streak_l": consecutive_real_losses,
            "ghost_streak_w": virtual_win_streak
        },
        "history": list(UI_HISTORY),
        "timestamp": time.time()
    }
    try:
        with open(DASHBOARD_FILE + ".tmp", "w") as f: json.dump(data, f)
        os.replace(DASHBOARD_FILE + ".tmp", DASHBOARD_FILE)
    except: pass

# --- API HANDLER ---
async def fetch_api_data(session, size_limit=20):
    params = {"size": size_limit, "pageSize": size_limit, "limit": size_limit, "count": size_limit, "pageNo": 1}
    try:
        async with session.get(API_URL, headers=HEADERS, params=params, timeout=15) as response:
            if response.status == 200:
                json_data = await response.json(content_type=None)
                return json_data.get('data', {}).get('list', []) or json_data.get('list', [])
    except Exception as e:
        print(f"[FETCH ERROR] {e}")
    return None

# --- MAIN LOOP ---
async def main_loop():
    global current_bankroll, last_prediction, last_win_status, session_wins, session_losses
    global consecutive_wins, consecutive_real_losses, cooldown_counter
    global ghost_mode_active, virtual_win_streak
    
    ensure_db_setup()
    last_processed_issue = None
    
    async with aiohttp.ClientSession() as session:
        print("\n" + "="*64)
        print("   TITAN V201 - GHOST PROTOCOL ACTIVE")
        print("   Strategy: 2 Real Losses -> Enter Ghost Mode")
        print("   Recovery: 2 Virtual Wins -> Exit Ghost Mode")
        print("="*64)
        
        # 1. INITIAL DEEP SYNC
        print("[BOOT] Fetching deep history...")
        boot_data = await fetch_api_data(session, size_limit=500)
        
        if boot_data:
            for item in reversed(boot_data):
                iss = item.get('issueNumber') or item.get('issue')
                num = item.get('number') or item.get('result')
                if iss and num is not None:
                    await save_to_db(iss, num)
        await load_db_to_ram()
        print(f"[BOOT] System Ready. Data Points: {len(RAM_HISTORY)}")

        # 2. REAL-TIME LOOP
        while True:
            raw_list = await fetch_api_data(session, size_limit=20)
            
            if raw_list:
                # A. SYNC
                for item in reversed(raw_list):
                    iss = str(item.get('issueNumber') or item.get('issue'))
                    num = int(item.get('number') or item.get('result'))
                    if not any(d['issue'] == iss for d in RAM_HISTORY):
                        await save_to_db(iss, num)
                        RAM_HISTORY.append({'issue': iss, 'actual_number': num})
                        print(f"[SYNC] New: {iss} = {num}")

                # B. STATE UPDATE
                latest = raw_list[0]
                curr_issue = str(latest.get('issueNumber') or latest.get('issue'))
                curr_num = int(latest.get('number') or latest.get('result'))
                
                seconds_left = 60 - datetime.now().second
                update_dashboard("SYNCED", seconds_left)
                
                if curr_issue != last_processed_issue:
                    # =========================================================
                    # C. EVALUATE RESULT (REAL vs GHOST)
                    # =========================================================
                    if last_prediction['issue'] == curr_issue:
                        real_outcome = get_outcome_from_number(curr_num)
                        pred_label = last_prediction['label']
                        
                        if pred_label not in ["WAITING", "SKIP", "COOLDOWN", GameConstants.SKIP]:
                            is_win = (pred_label == real_outcome)
                            
                            # --- 1. HANDLING GHOST MODE ---
                            if ghost_mode_active:
                                if is_win:
                                    virtual_win_streak += 1
                                    print(f"\n[GHOST] {curr_issue} VIRTUAL WIN ({virtual_win_streak}/2)")
                                    last_win_status = "GHOST_WIN"
                                    
                                    # EXIT CONDITION: 2 Virtual Wins in a row
                                    if virtual_win_streak >= 2:
                                        ghost_mode_active = False
                                        consecutive_real_losses = 0 # Reset bad streak
                                        print(">>> [REVIVAL] ENGINES SYNCED. RESUMING REAL MONEY. <<<")
                                else:
                                    virtual_win_streak = 0
                                    print(f"\n[GHOST] {curr_issue} VIRTUAL LOSS (Streak Reset)")
                                    last_win_status = "GHOST_LOSS"

                            # --- 2. HANDLING REAL MODE ---
                            else:
                                if is_win:
                                    profit = last_prediction['stake'] * 0.98
                                    current_bankroll += profit
                                    session_wins += 1
                                    consecutive_wins += 1
                                    consecutive_real_losses = 0
                                    last_win_status = "WIN"
                                    print(f"\n[REAL] {curr_issue} WIN (+{profit:.0f}) | Streak: {consecutive_wins}")
                                else:
                                    current_bankroll -= last_prediction['stake']
                                    session_losses += 1
                                    consecutive_real_losses += 1
                                    consecutive_wins = 0
                                    last_win_status = "LOSS"
                                    print(f"\n[REAL] {curr_issue} LOSS (-{last_prediction['stake']:.0f}) | Loss Streak: {consecutive_real_losses}")
                                    
                                    # TRIGGER GHOST MODE?
                                    if consecutive_real_losses >= 2:
                                        ghost_mode_active = True
                                        virtual_win_streak = 0
                                        print(">>> [ALERT] 2 LOSSES DETECTED. ACTIVATING GHOST PROTOCOL. <<<")

                            # Log to History
                            UI_HISTORY.appendleft({
                                "period": curr_issue, 
                                "pred": pred_label, 
                                "result": last_win_status,
                                "bankroll": round(current_bankroll, 2)
                            })
                            
                        else:
                            # Handling COOLDOWN/SKIP
                            print(f"\n[RESULT] {curr_issue} was {real_outcome} (Skipped)")
                            if cooldown_counter > 0:
                                cooldown_counter -= 1
                                print(f"[COOL] Profit Lock Remaining: {cooldown_counter}")

                    # =========================================================
                    # D. PROFIT LOCK CHECK (5 Wins -> Sleep)
                    # =========================================================
                    if consecutive_wins >= 5 and cooldown_counter == 0 and not ghost_mode_active:
                        cooldown_counter = 6
                        consecutive_wins = 0
                        print(f"\n[PROFIT] 5 WINS HIT! Locking profit. Cooling down 6 rounds.")

                    # =========================================================
                    # E. GENERATE NEXT PREDICTION
                    # =========================================================
                    if len(RAM_HISTORY) >= MIN_DATA_REQUIRED:
                        next_issue = str(int(curr_issue) + 1)
                        
                        # 1. IS PROFIT LOCK ACTIVE?
                        if cooldown_counter > 0:
                            last_prediction = {
                                "issue": next_issue, "label": "COOLDOWN", "stake": 0, "conf": 0,
                                "level": "PAUSED", "reason": "Profit Lock", "strategy": "REST"
                            }
                            print(f"[PRED] Target: {next_issue} | PROFIT LOCK ACTIVE")
                            
                        # 2. NORMAL / GHOST PREDICTION
                        else:
                            try:
                                res = ultraAIPredict(list(RAM_HISTORY), current_bankroll, last_prediction['label'])
                                
                                # IF GHOST MODE IS ON, FORCE STAKE TO 0
                                final_stake = res['positionsize']
                                final_level = res['level']
                                final_reason = res.get('reason', '')
                                
                                if ghost_mode_active:
                                    final_stake = 0
                                    final_level = "👻 GHOST"
                                    final_reason = f"Virtual Test {virtual_win_streak}/2"
                                
                                last_prediction = {
                                    "issue": next_issue, 
                                    "label": res['finalDecision'], 
                                    "stake": final_stake,
                                    "conf": res['confidence'], 
                                    "level": final_level, 
                                    "reason": final_reason
                                }
                                
                                log_tag = "[GHOST]" if ghost_mode_active else "[REAL]"
                                print(f"{log_tag} Target: {next_issue} | Decision: {last_prediction['label']} | Conf: {last_prediction['conf']:.0%}")
                                
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
