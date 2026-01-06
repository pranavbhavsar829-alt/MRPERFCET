"""
TITAN V700 - SOVEREIGN FETCHER ENGINE (V2026.9 - FULL PRODUCTION FIX)
================================================================
Integrated with Mimetype Fix, Warmup Logic, and Server-Loop Sync.
================================================================
"""

import aiohttp
import asyncio
import json
import logging
import os
import sys
import time
import sqlite3
from collections import deque
from datetime import datetime
from typing import List, Dict, Optional, Any

# --- CUSTOM LOGGING CONFIGURATION ---
LOG_FORMAT = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, stream=sys.stdout)
logger = logging.getLogger("TITAN_FETCHER")

# --- CORE SYSTEM ENGINE IMPORT ---
try:
    from prediction_engine import ultraAIPredict
    
    def get_outcome_from_number(n):
        """Standard WinGo mapping: 0-4 Small, 5-9 Big."""
        try:
            val = int(float(n))
            return "SMALL" if 0 <= val <= 4 else "BIG"
        except: return None
    logger.info("Logic Engine (TITAN V700) linked successfully.")
except ImportError as e:
    logger.critical(f"FATAL: prediction_engine.py missing! Details: {e}")
    sys.exit(1)

# --- API CONFIGURATION ---
TARGET_URL = "https://api-wo4u.onrender.com/api/get_history"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
    "Origin": "https://draw.ar-lottery01.com",
    "Referer": "https://draw.ar-lottery01.com/"
}

# --- PERSISTENT STORAGE PATHS (RENDER COMPATIBLE) ---
if os.path.exists('/var/lib/data'):
    BASE_STORAGE_PATH = '/var/lib/data'
else:
    BASE_STORAGE_PATH = os.path.abspath(os.path.dirname(__file__))

DB_FILE = os.path.join(BASE_STORAGE_PATH, 'ar_lottery_history.db')
DASHBOARD_PATH = os.path.join(BASE_STORAGE_PATH, 'dashboard_data.json')

# --- OPERATIONAL PARAMETERS ---
HISTORY_RETENTION_LIMIT = 2000
WARMUP_TARGET = 50        
MIN_BRAIN_CAPACITY = 40   # Engines like 'reversion' need this much data
LIVE_POLL_INTERVAL = 2.0  

# --- SHARED GLOBAL STATE ---
class FetcherState:
    def __init__(self):
        self.bankroll = 10000.0
        self.ram_history = deque(maxlen=HISTORY_RETENTION_LIMIT)
        self.ui_history = deque(maxlen=50)
        self.wins = 0
        self.losses = 0
        self.last_processed_issue = None
        self.active_prediction = {
            "issue": None, "label": "WAITING", "stake": 0, "conf": 0,
            "level": "---", "reason": "System Warming Up...", "strategy": "WARMUP"
        }
        self.last_win_status = "NONE"

state = FetcherState()

# =============================================================================
# DATABASE LAYER
# =============================================================================

def ensure_db_setup():
    """Initializes SQLite to ensure data persists across restarts."""
    conn = sqlite3.connect(DB_FILE)
    conn.execute('''CREATE TABLE IF NOT EXISTS results 
                    (issue TEXT PRIMARY KEY, code INTEGER, fetch_time TEXT)''')
    conn.commit()
    conn.close()

async def save_to_db(issue: str, code: int):
    """Saves records to SQLite."""
    try:
        conn = sqlite3.connect(DB_FILE)
        conn.execute("INSERT OR IGNORE INTO results (issue, code, fetch_time) VALUES (?, ?, ?)", 
                       (str(issue), int(code), str(datetime.now())))
        conn.commit()
        conn.close()
    except Exception as e:
        logger.error(f"DB Save Error: {e}")

async def load_db_to_ram():
    """Loads history from DB into RAM and returns the record count."""
    state.ram_history.clear()
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute(f"SELECT issue, code FROM results ORDER BY issue DESC LIMIT {HISTORY_RETENTION_LIMIT}")
        rows = cursor.fetchall()
        conn.close()
        for r in reversed(rows):
            state.ram_history.append({'issue': str(r[0]), 'actual_number': int(r[1])})
        return len(state.ram_history)
    except Exception as e:
        logger.error(f"DB Load Error: {e}")
        return 0

# =============================================================================
# UI & DASHBOARD SYNC
# =============================================================================

async def sync_dashboard(status: str, timer: int):
    """Updates the JSON file consumed by server.py."""
    total = state.wins + state.losses
    acc = f"{(state.wins / total) * 100:.1f}%" if total > 0 else "0.0%"
    
    payload = {
        "period": state.active_prediction['issue'] or "---",
        "prediction": state.active_prediction['label'],
        "confidence": f"{state.active_prediction.get('conf', 0)*100:.1f}%",
        "stake": state.active_prediction['stake'],
        "level": state.active_prediction.get('level', '---'),
        "bankroll": state.bankroll,
        "lastresult_status": state.last_win_status,
        "status_text": status,
        "timer": timer,
        "data_size": len(state.ram_history),
        "stats": {"wins": state.wins, "losses": state.losses, "accuracy": acc},
        "history": list(state.ui_history),
        "timestamp": time.time()
    }

    try:
        temp_path = DASHBOARD_PATH + ".tmp"
        with open(temp_path, "w") as f:
            json.dump(payload, f, indent=2)
        os.replace(temp_path, DASHBOARD_PATH)
    except Exception as e:
        logger.error(f"Dashboard Sync Error: {e}")

# =============================================================================
# NETWORK LAYER
# =============================================================================

async def execute_api_fetch(session: aiohttp.ClientSession, limit: int, page=1) -> List[Dict]:
    """Fetches data using GET with octet-stream decoding fix."""
    params = {
        "pageSize": limit,
        "pageNo": page,
        "typeId": 1,
        "language": 0,
        "timestamp": int(time.time() * 1000)
    }

    try:
        async with session.get(TARGET_URL, headers=HEADERS, params=params, timeout=10) as response:
            if response.status == 200:
                # content_type=None bypasses strict mimetype checking
                json_data = await response.json(content_type=None)
                return json_data.get('data', {}).get('list', []) or json_data.get('list', [])
            else:
                logger.warning(f"API Rejection: Status {response.status}")
    except Exception as e:
        logger.error(f"Network Error: {e}")
    
    return []

# =============================================================================
# MAIN OPERATIONAL LOOP (Renamed to main_loop for server.py compatibility)
# =============================================================================

async def main_loop():
    """Main loop for server-side integration."""
    ensure_db_setup()
    current_count = await load_db_to_ram()
    
    async with aiohttp.ClientSession() as session:
        
        # --- PHASE 1: FORCED WARMUP SYNC ---
        if current_count < WARMUP_TARGET:
            logger.info(f"[WARMUP] Current data: {current_count}. Syncing {WARMUP_TARGET} records...")
            for p in range(1, 6): # Fetch up to 5 pages
                await sync_dashboard(f"WARMUP: PAGE {p}", 0)
                batch = await execute_api_fetch(session, 20, page=p)
                if batch:
                    for item in batch:
                        iss = str(item.get('issueNumber') or item.get('issue'))
                        num = int(item.get('number') or item.get('result'))
                        await save_to_db(iss, num)
                
                current_count = await load_db_to_ram()
                if current_count >= WARMUP_TARGET:
                    break
            logger.info(f"[WARMUP] Completed. Memory size: {current_count}")

        logger.info(f"--- TITAN V700 LIVE MONITORING ACTIVE ---")
        
        while True:
            raw_list = await execute_api_fetch(session, 10)
            
            if raw_list:
                # Sync any missed rounds into memory
                for item in reversed(raw_list):
                    iss = str(item.get('issueNumber') or item.get('issue'))
                    num = int(item.get('number') or item.get('result'))
                    if not any(d['issue'] == iss for d in state.ram_history):
                        await save_to_db(iss, num)
                        state.ram_history.append({'issue': iss, 'actual_number': num})

                latest = raw_list[0]
                curr_issue = str(latest.get('issueNumber') or latest.get('issue'))
                curr_num = int(latest.get('number') or latest.get('result'))
                
                seconds_remaining = 60 - datetime.now().second
                await sync_dashboard("LIVE SYNCED", seconds_remaining)
                
                if curr_issue != state.last_processed_issue:
                    logger.info(f"[NEW DATA] {curr_issue} Result: {curr_num}")
                    
                    # Verify Prediction Accuracy
                    if state.active_prediction['issue'] == curr_issue:
                        real_outcome = get_outcome_from_number(curr_num)
                        pred_label = state.active_prediction['label']
                        if pred_label not in ["WAITING", "SKIP"]:
                            win = (pred_label == real_outcome)
                            state.last_win_status = "WIN" if win else "LOSS"
                            if win: state.wins += 1
                            else: state.losses += 1
                            state.ui_history.appendleft({"period": curr_issue, "pred": pred_label, "result": state.last_win_status})

                    # Run Engine Prediction (Requires 40+ Records)
                    if len(state.ram_history) >= MIN_BRAIN_CAPACITY:
                        next_issue = str(int(curr_issue) + 1)
                        try:
                            # ultraAIPredict call
                            ai_output = ultraAIPredict(list(state.ram_history), state.bankroll, get_outcome_from_number(curr_num))
                            state.active_prediction = {
                                "issue": next_issue,
                                "label": ai_output['finalDecision'],
                                "stake": ai_output['positionsize'],
                                "conf": ai_output['confidence'],
                                "level": ai_output['level'],
                                "reason": ai_output.get('reason', 'Processing signals...')
                            }
                            logger.info(f"[TITAN] Target: {next_issue} | Signal: {ai_output['finalDecision']}")
                        except Exception as e:
                            logger.error(f"Logic Engine Error: {e}")
                    else:
                        logger.warning(f"[BUFFER] Data {len(state.ram_history)}/40. Engines waiting.")
                    
                    state.last_processed_issue = curr_issue
            
            await asyncio.sleep(LIVE_POLL_INTERVAL)

if __name__ == '__main__':
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    try:
        asyncio.run(main_loop())
    except KeyboardInterrupt:
        logger.info("Shutdown requested.")
