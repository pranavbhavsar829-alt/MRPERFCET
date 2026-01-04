import aiohttp
import asyncio
import json
import sqlite3
import time
import sys
import os
import logging
from collections import deque
from datetime import datetime

# --- LOGGING SETUP ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("TITAN_FETCHER")

# --- IMPORT ENGINE ---
try:
    from prediction_engine import ultraAIPredict
    logger.info("TITAN V700 ENGINE LINKED SUCCESSFULLY.")
except ImportError:
    logger.error("prediction_engine.py is missing! Predictions will fail.")

# --- CONFIGURATION ---
# Replace with your actual working API URL
API_URL = "https://harshpredictor.site/api/api.php"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "application/json",
    "Connection": "keep-alive",
    "Cache-Control": "no-cache"
}

# --- RENDER PERSISTENCE PATHS ---
# Ensures server and fetcher share the exact same JSON file on persistent storage
if os.path.exists('/var/lib/data'):
    BASE_DIR = '/var/lib/data'
    logger.info("Using Render Persistent Disk: /var/lib/data")
elif os.path.exists('/data'):
    BASE_DIR = '/data'
    logger.info("Using Render Persistent Disk: /data")
else:
    BASE_DIR = os.path.abspath(os.path.dirname(__file__))
    logger.info(f"Using Local Storage: {BASE_DIR}")

DASHBOARD_FILE = os.path.join(BASE_DIR, 'dashboard_data.json')
LOCAL_DB_FILE = os.path.join(BASE_DIR, 'history_backup.json')

# --- SETTINGS ---
HISTORY_LIMIT = 2000       
INITIAL_FETCH_SIZE = 1000  
LIVE_FETCH_SIZE = 10        
RECONNECT_DELAY = 5        

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

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_outcome_from_number(n):
    """Determines BIG/SMALL based on standard rules (0-4 Small, 5-9 Big)"""
    try:
        val = int(float(n))
        if 0 <= val <= 4: return "SMALL"
        if 5 <= val <= 9: return "BIG"
    except (ValueError, TypeError): 
        pass
    return None

def save_history_to_disk():
    """Backup RAM_HISTORY to disk to prevent data loss on restarts."""
    try:
        with open(LOCAL_DB_FILE, "w") as f:
            json.dump(list(RAM_HISTORY), f)
    except Exception as e:
        logger.error(f"Failed to backup history: {e}")

def load_history_from_disk():
    """Load previous data on startup if API fails to provide full history."""
    if os.path.exists(LOCAL_DB_FILE):
        try:
            with open(LOCAL_DB_FILE, "r") as f:
                data = json.load(f)
                for item in data:
                    RAM_HISTORY.append(item)
            logger.info(f"Restored {len(RAM_HISTORY)} records from disk.")
        except Exception as e:
            logger.error(f"Disk restore failed: {e}")

def update_dashboard(status_text="IDLE", timer_val=0):
    """Writes system state to JSON for the Web UI to read."""
    total = session_wins + session_losses
    acc = f"{(session_wins/total)*100:.1f}%" if total > 0 else "0.0%"
    
    data = {
        "period": last_prediction['issue'] if last_prediction['issue'] else "---",
        "prediction": last_prediction['label'],
        "confidence": f"{last_prediction.get('conf', 0)*100:.1f}%",
        "stake": last_prediction['stake'],
        "level": last_prediction.get('level', '---'),
        "strategy": last_prediction.get('strategy', 'STANDARD'),
        "reason": last_prediction.get('reason', 'Syncing...'),
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
        # Atomic write to prevent file corruption
        temp_file = DASHBOARD_FILE + ".tmp"
        with open(temp_file, "w") as f: 
            json.dump(data, f)
        os.replace(temp_file, DASHBOARD_FILE)
    except Exception as e: 
        logger.error(f"Dashboard Update Failed: {e}")

# =============================================================================
# DATA ACQUISITION
# =============================================================================

async def fetch_api_data(session, limit=10):
    """Robust fetcher that handles common lottery JSON structures."""
    try:
        params = {'pageSize': limit, 'page': 1}
        async with session.get(API_URL, headers=HEADERS, params=params, timeout=15) as response:
            if response.status == 200:
                # Use content_type=None to handle misconfigured servers sending text/html
                d = await response.json(content_type=None)
                
                # Dynamic extraction: handle different JSON formats
                data_list = []
                if 'data' in d and 'list' in d['data']:
                    data_list = d['data']['list']
                elif 'list' in d:
                    data_list = d['list']
                elif isinstance(d, list):
                    data_list = d
                    
                return data_list
            else:
                logger.warning(f"API returned status {response.status}")
    except asyncio.TimeoutError:
        logger.error("API Connection Timeout")
    except Exception as e:
        logger.error(f"Fetch Connection failed: {e}")
    return None

# =============================================================================
# CORE PROCESSING LOOP
# =============================================================================

async def main_loop():
    global currentbankroll, last_prediction, last_win_status, session_wins, session_losses
    logger.info("TITAN V700 FETCHER INITIALIZED.")
    
    last_processed_issue = None
    load_history_from_disk()
    
    async with aiohttp.ClientSession() as session:
        # Initial Bootstrapping
        update_dashboard("BOOTING...", 0)
        startup_data = await fetch_api_data(session, INITIAL_FETCH_SIZE)
        
        if startup_data:
            for item in reversed(startup_data):
                try:
                    issue = str(item.get('issueNumber') or item.get('issue'))
                    num = int(item.get('number') or item.get('result'))
                    
                    if not RAM_HISTORY or RAM_HISTORY[-1]['issue'] != issue:
                        RAM_HISTORY.append({'issue': issue, 'actual_number': num})
                except: continue
            
            logger.info(f"History Sync Complete. Memory: {len(RAM_HISTORY)} items.")
            if len(RAM_HISTORY) > 0:
                last_processed_issue = RAM_HISTORY[-1]['issue']
        
        # Real-time Monitoring Loop
        while True:
            try:
                raw_list = await fetch_api_data(session, LIVE_FETCH_SIZE)
                
                if raw_list:
                    latest = raw_list[0]
                    curr_issue = str(latest.get('issueNumber') or latest.get('issue'))
                    curr_num = int(latest.get('number') or latest.get('result'))
                    
                    now = datetime.now()
                    seconds_left = 60 - now.second
                    update_dashboard("LIVE SYNC", seconds_left)
                    
                    if curr_issue != last_processed_issue:
                        logger.info(f"New Period Detected: {curr_issue}")
                        
                        # Store to memory
                        if not RAM_HISTORY or RAM_HISTORY[-1]['issue'] != curr_issue:
                            RAM_HISTORY.append({'issue': curr_issue, 'actual_number': curr_num})
                            save_history_to_disk()

                        # --- PHASE A: WIN/LOSS VERIFICATION ---
                        if last_prediction['issue'] == curr_issue:
                            real_outcome = get_outcome_from_number(curr_num)
                            predicted = last_prediction['label']
                            
                            if predicted not in ["SKIP", "WAITING"]:
                                is_win = (predicted == real_outcome)
                                
                                # Accuracy update
                                if is_win:
                                    session_wins += 1
                                    last_win_status = "WIN"
                                    logger.info(f"RESULT: WIN | Period: {curr_issue}")
                                else:
                                    session_losses += 1
                                    last_win_status = "LOSS"
                                    logger.info(f"RESULT: LOSS | Period: {curr_issue}")
                                
                                # Add to UI History
                                UI_HISTORY.appendleft({
                                    "period": curr_issue, 
                                    "pred": predicted, 
                                    "result": last_win_status
                                })

                        # --- PHASE B: ENGINE PREDICTION ---
                        next_issue = str(int(curr_issue) + 1)
                        
                        # Wait for a few seconds for data stability
                        for i in range(3, 0, -1):
                            update_dashboard(f"THINKING... {i}", seconds_left)
                            await asyncio.sleep(1)
                            seconds_left = 60 - datetime.now().second

                        if len(RAM_HISTORY) >= 15:
                            try:
                                # Trigger the Sovereign AI
                                ai_res = ultraAIPredict(
                                    list(RAM_HISTORY), 
                                    currentbankroll, 
                                    get_outcome_from_number(curr_num)
                                )
                                
                                last_prediction = {
                                    "issue": next_issue, 
                                    "label": ai_res['finalDecision'], 
                                    "stake": ai_res['positionsize'],
                                    "conf": ai_res['confidence'],
                                    "level": ai_res['level'],
                                    "strategy": "SOVEREIGN",
                                    "reason": ai_res.get('reason', 'Processing...')
                                }
                                logger.info(f"NEXT: {next_issue} | PRED: {last_prediction['label']}")
                            except Exception as e:
                                logger.error(f"Engine Crash: {e}")
                                last_prediction['label'] = "SKIP"
                                last_prediction['reason'] = "Internal Engine Error"
                        
                        last_processed_issue = curr_issue

                else:
                    logger.warning("No data received from API. Retrying...")
                    update_dashboard("CONNECTION LOST", 0)
                    await asyncio.sleep(RECONNECT_DELAY)

                await asyncio.sleep(2.0) # Check every 2 seconds

            except Exception as e:
                logger.error(f"Global Loop Error: {e}")
                await asyncio.sleep(RECONNECT_DELAY)

if __name__ == '__main__':
    # Optimization for Windows local testing
    if sys.platform == 'win32': 
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    try:
        asyncio.run(main_loop())
    except KeyboardInterrupt:
        logger.info("System shutting down gracefully.")
