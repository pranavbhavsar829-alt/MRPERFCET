from flask import Flask, render_template_string, jsonify, request, session, redirect, url_for
import sqlite3
import secrets
import os
import json
import functools
import uuid
import hmac
import hashlib
import base64
import time
from datetime import datetime, timedelta
from database_handler import create_connection, DB_PATH, ensure_setup, cleanup_inactive_sessions

app = Flask(__name__)
app.secret_key = "TITAN_SECURE_KEY_CHANGE_THIS" 
ADMIN_PASSWORD = "admin" 

# --- CRITICAL: SECURITY CONFIGURATION ---
# This SECRET must match the one in your local_generator.py EXACTLY.
OFFLINE_SECRET = "TITAN_OFFLINE_SECRET_CODE_123"

# Master Key (Always works, even if DB is wiped)
MASTER_KEY = "TITAN-PERM-ADMIN" 

DASHBOARD_FILE = os.path.join(DB_PATH, 'dashboard_data.json')
ensure_setup()

# --- AUTH LOGIC ---
def login_required(f):
    @functools.wraps(f)
    def wrapped(*args, **kwargs):
        if not session.get('authenticated'): return redirect(url_for('login'))
        return f(*args, **kwargs)
    return wrapped

def admin_required(f):
    @functools.wraps(f)
    def wrapped(*args, **kwargs):
        if not session.get('is_admin'): return redirect(url_for('admin_login'))
        return f(*args, **kwargs)
    return wrapped

# --- 1. STATELESS KEY VERIFICATION (OFFLINE KEYS) ---
def verify_stateless_key(key_input):
    try:
        # Format: TITAN-PAYLOAD-SIGNATURE
        parts = key_input.split('-')
        if len(parts) != 3 or parts[0] != "TITAN":
            return False, "INVALID FORMAT"
        
        payload_encoded = parts[1]
        signature_input = parts[2]
        
        # Decode Payload
        try:
            padding = len(payload_encoded) % 4
            if padding: payload_encoded += '=' * (4 - padding)
            payload = base64.urlsafe_b64decode(payload_encoded).decode()
            timestamp_str, max_dev_str = payload.split('|')
            expiry_ts = int(timestamp_str)
            max_dev = int(max_dev_str)
        except:
            return False, "CORRUPT KEY"

        # Verify Signature
        expected_sig = hmac.new(
            OFFLINE_SECRET.encode(), 
            payload.encode(), 
            hashlib.sha256
        ).hexdigest()[:8].upper()
        
        if not hmac.compare_digest(expected_sig, signature_input):
            return False, "FAKE KEY DETECTED"
            
        # Check Expiration
        if time.time() > expiry_ts:
            return False, "KEY EXPIRED"
            
        return True, "OK"
        
    except Exception as e:
        return False, "ERROR"

# --- 2. MAIN LOGIN LOGIC ---
def validate_key_and_login(key_input):
    cleanup_inactive_sessions()
    
    # A. MASTER KEY CHECK
    if key_input == MASTER_KEY:
        session['session_uuid'] = str(uuid.uuid4())
        return True, "OK"

    # B. STATELESS KEY CHECK (Offline)
    if key_input.startswith("TITAN-"):
        is_valid, msg = verify_stateless_key(key_input)
        if is_valid:
            # Key is valid mathematically.
            # We treat it as a valid session without needing a DB lookup for the key itself.
            my_session_id = session.get('session_uuid') or str(uuid.uuid4())
            session['session_uuid'] = my_session_id
            
            # Record session for counting (optional, best effort)
            try:
                conn = create_connection()
                conn.execute("INSERT OR REPLACE INTO active_sessions (session_id, key_code, last_seen) VALUES (?, ?, ?)", (my_session_id, key_input, datetime.now()))
                conn.commit()
                conn.close()
            except: pass
            
            return True, "OK"
        else:
            return False, msg

    # C. DATABASE KEY CHECK (Legacy/Online Keys)
    conn = create_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM access_keys WHERE key_code = ? AND is_active = 1", (key_input,))
    row = cur.fetchone()
    if not row:
        conn.close()
        return False, "INVALID KEY"
    expires_at = row[4]
    if expires_at and datetime.now() > datetime.strptime(expires_at, '%Y-%m-%d %H:%M:%S.%f'):
        conn.close()
        return False, "KEY EXPIRED"
    
    max_devices = row[5] if len(row) > 5 else 1
    cur.execute("SELECT COUNT(*) FROM active_sessions WHERE key_code = ?", (key_input,))
    current_users = cur.fetchone()[0]
    my_session_id = session.get('session_uuid')
    
    if current_users < max_devices or (my_session_id and check_is_active(my_session_id)):
        new_uuid = my_session_id if my_session_id else str(uuid.uuid4())
        session['session_uuid'] = new_uuid
        conn.execute("INSERT OR REPLACE INTO active_sessions (session_id, key_code, last_seen) VALUES (?, ?, ?)", (new_uuid, key_input, datetime.now()))
        conn.commit()
        conn.close()
        return True, "OK"
    else:
        conn.close()
        return False, f"MAX DEVICES REACHED ({current_users}/{max_devices})"

def check_is_active(sess_id):
    conn = create_connection()
    try:
        row = conn.execute("SELECT * FROM active_sessions WHERE session_id = ?", (sess_id,)).fetchone()
    except: return False
    conn.close()
    return row is not None

# --- ROUTES ---
@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        key = request.form.get('key', '').strip()
        is_valid, msg = validate_key_and_login(key)
        if is_valid:
            session['authenticated'] = True
            session['user_key'] = key
            return redirect(url_for('index'))
        else: error = msg
    return f"""<body style="background:#000; color:#0f0; display:flex; justify-content:center; align-items:center; height:100vh; font-family:monospace; flex-direction:column;">
            <h1>🔒 LOCKED SYSTEM</h1><p style="color:red">{error if error else ''}</p>
            <form method="post"><input type="text" name="key" placeholder="LICENSE KEY" style="padding:10px; width:250px; text-align:center;"><br><br><button style="padding:10px 20px; font-weight:bold; cursor:pointer;">UNLOCK</button></form></body>"""

@app.route('/heartbeat', methods=['POST'])
def heartbeat():
    if session.get('authenticated') and session.get('session_uuid'):
        try:
            conn = create_connection()
            conn.execute("UPDATE active_sessions SET last_seen = ? WHERE session_id = ?", (datetime.now(), session['session_uuid']))
            conn.commit()
            conn.close()
        except: pass
        return jsonify({"status": "ok"})
    return jsonify({"status": "ignored"})

@app.route('/admin_login', methods=['GET', 'POST'])
def admin_login():
    if request.method == 'POST':
        if request.form.get('password') == ADMIN_PASSWORD:
            session['is_admin'] = True
            return redirect(url_for('admin_panel'))
    return '<form method="post" style="text-align:center; margin-top:50px;"><input type="password" name="password" placeholder="Pass"><button>Login</button></form>'

@app.route('/admin')
@admin_required
def admin_panel():
    cleanup_inactive_sessions()
    conn = create_connection()
    keys = conn.execute("SELECT * FROM access_keys ORDER BY created_at DESC").fetchall()
    conn.close()
    rows = ""
    for k in keys:
        expiry = k[4][:16] if k[4] else "Lifetime" 
        max_dev = k[5] if len(k) > 5 else 1
        rows += f"<tr><td>{k[0]}</td><td>{k[1]}</td><td>{expiry}</td><td>Max: {max_dev}</td><td>Online DB</td></tr>"
    return f"""<body style="font-family:monospace; padding:20px;"><h1>ADMIN</h1>
            <div style="background:#eee; padding:10px;">
            <p><strong>SYSTEM MODE:</strong> Hybrid (Offline Keys Enabled)</p>
            <p>Use <code>local_generator.py</code> on your PC to create permanent keys.</p>
            </div>
            <h3>Database Keys (Will be deleted if server sleeps)</h3>
            <table border="1" cellpadding="5" style="width:100%; margin-top:20px;">{rows}</table></body>"""

@app.route('/logout')
def logout():
    if session.get('session_uuid'):
        try:
            conn = create_connection()
            conn.execute("DELETE FROM active_sessions WHERE session_id = ?", (session['session_uuid'],))
            conn.commit()
            conn.close()
        except: pass
    session.clear()
    return redirect(url_for('login'))

@app.route('/data')
@login_required
def data():
    try:
        if os.path.exists(DASHBOARD_FILE):
            with open(DASHBOARD_FILE, 'r') as f:
                return jsonify(json.load(f))
    except: pass
    return jsonify({"period": "---", "prediction": "LOADING", "timer": 0, "stats": {"wins":0, "losses":0, "accuracy":"0%"}})

@app.route('/')
@login_required
def index():
    return render_template_string(HTML_TEMPLATE)

# --- PRO UI TEMPLATE ---
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>TITAN AI V4</title>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;700;900&family=JetBrains+Mono:wght@400;700&display=swap');
        
        :root {
            --bg: #050505;
            --card: #111;
            --text: #fff;
            --accent: #00ff41;
            --loss: #ff0055;
            --dim: #444;
        }

        body {
            background-color: var(--bg);
            color: var(--text);
            font-family: 'Inter', sans-serif;
            margin: 0;
            padding: 20px;
            display: flex;
            flex-direction: column;
            align-items: center;
            min-height: 100vh;
        }

        .header {
            width: 100%;
            max-width: 480px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
        }
        
        .logo { font-family: 'JetBrains Mono'; font-weight: 900; letter-spacing: -1px; font-size: 20px; }
        .logout-btn { color: var(--loss); text-decoration: none; font-size: 12px; border: 1px solid var(--loss); padding: 4px 10px; border-radius: 4px; }

        .dashboard-grid {
            width: 100%;
            max-width: 480px;
            display: grid;
            grid-template-columns: 1fr 1fr 1fr;
            gap: 10px;
            margin-bottom: 15px;
        }

        .stat-card {
            background: var(--card);
            padding: 15px;
            border-radius: 12px;
            text-align: center;
            border: 1px solid #222;
        }
        .stat-label { font-size: 10px; color: #888; text-transform: uppercase; letter-spacing: 1px; }
        .stat-val { font-size: 18px; font-weight: 900; margin-top: 5px; font-family: 'JetBrains Mono'; }
        .c-green { color: var(--accent); }
        .c-red { color: var(--loss); }

        .main-card {
            width: 100%;
            max-width: 480px;
            background: var(--card);
            border-radius: 16px;
            padding: 25px;
            text-align: center;
            border: 1px solid #222;
            box-shadow: 0 10px 30px rgba(0,0,0,0.5);
            margin-bottom: 20px;
            position: relative;
            overflow: hidden;
        }
        
        .timer-bar {
            position: absolute;
            top: 0; left: 0; height: 4px; background: var(--accent);
            width: 100%; transition: width 0.5s linear;
        }

        .period-display { font-family: 'JetBrains Mono'; color: #666; font-size: 14px; margin-bottom: 10px; }
        
        .prediction-display {
            font-size: 56px;
            font-weight: 900;
            text-transform: uppercase;
            margin: 10px 0;
            line-height: 1;
        }
        
        .res-big { color: var(--accent); text-shadow: 0 0 30px rgba(0, 255, 65, 0.15); }
        .res-small { color: var(--loss); text-shadow: 0 0 30px rgba(255, 0, 85, 0.15); }
        .res-wait { color: #333; }
        
        .countdown {
            font-family: 'JetBrains Mono';
            font-size: 24px;
            font-weight: bold;
            color: #fff;
            margin-top: 10px;
        }
        .countdown.danger { color: #ffaa00; }

        .history-list {
            width: 100%;
            max-width: 480px;
            background: var(--card);
            border-radius: 12px;
            border: 1px solid #222;
            overflow-y: auto;
            max-height: 350px;
        }
        
        .history-list::-webkit-scrollbar { width: 6px; }
        .history-list::-webkit-scrollbar-track { background: #111; }
        .history-list::-webkit-scrollbar-thumb { background: #333; border-radius: 3px; }

        .history-item {
            display: flex;
            justify-content: space-between;
            padding: 12px 20px;
            border-bottom: 1px solid #1a1a1a;
            font-size: 13px;
            font-family: 'JetBrains Mono';
        }
        .history-item:last-child { border-bottom: none; }
        .h-period { color: #666; }
        .h-win { color: var(--accent); }
        .h-loss { color: var(--loss); }
        .h-pending { color: #888; }

    </style>
</head>
<body>

    <div class="header">
        <div class="logo">TITAN AI <span style="color:var(--accent)">V4</span></div>
        <a href="/logout" class="logout-btn">LOGOUT</a>
    </div>

    <div class="dashboard-grid">
        <div class="stat-card">
            <div class="stat-label">WINS</div>
            <div class="stat-val c-green" id="stat-wins">0</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">LOSSES</div>
            <div class="stat-val c-red" id="stat-loss">0</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">ACCURACY</div>
            <div class="stat-val" id="stat-acc">0%</div>
        </div>
    </div>

    <div class="main-card">
        <div class="timer-bar" id="timer-bar"></div>
        <div class="period-display" id="period">PERIOD: LOADING...</div>
        
        <div id="prediction" class="prediction-display res-wait">---</div>
        
        <div class="countdown" id="countdown">00:00</div>
        <div style="font-size: 10px; color: #555; margin-top: 5px;" id="status">SYNCING SERVER...</div>
    </div>

    <div style="width:100%; max-width:480px; margin-bottom:5px; font-size:11px; color:#555; text-transform:uppercase; letter-spacing:1px; font-weight:bold;">Recent 50 Outcomes</div>
    <div class="history-list" id="history-box">
        <div class="history-item" style="justify-content:center; color:#444;">No history yet...</div>
    </div>

    <script>
        // --- HEARTBEAT ---
        setInterval(() => { fetch('/heartbeat', { method: 'POST' }); }, 30000);

        function updateData() {
            fetch('/data')
                .then(r => r.json())
                .then(d => {
                    // 1. Update Header Info
                    document.getElementById('period').innerText = "PERIOD: " + d.period;
                    document.getElementById('status').innerText = d.status_text;

                    // 2. Update Prediction
                    const p = document.getElementById('prediction');
                    p.innerText = d.prediction;
                    p.className = 'prediction-display';
                    if (d.prediction === 'BIG') p.classList.add('res-big');
                    else if (d.prediction === 'SMALL') p.classList.add('res-small');
                    else p.classList.add('res-wait');

                    // 3. Update Timer
                    const timeLeft = d.timer || 0;
                    const min = Math.floor(timeLeft / 60);
                    const sec = timeLeft % 60;
                    const timeStr = `00:${sec < 10 ? '0' : ''}${sec}`;
                    const cdEl = document.getElementById('countdown');
                    cdEl.innerText = timeStr;
                    
                    if(timeLeft < 10) cdEl.classList.add('danger');
                    else cdEl.classList.remove('danger');

                    const percent = (timeLeft / 60) * 100;
                    document.getElementById('timer-bar').style.width = percent + "%";

                    // 4. Update Stats
                    if (d.stats) {
                        document.getElementById('stat-wins').innerText = d.stats.wins;
                        document.getElementById('stat-loss').innerText = d.stats.losses;
                        document.getElementById('stat-acc').innerText = d.stats.accuracy;
                    }

                    // 5. Update History List
                    const histBox = document.getElementById('history-box');
                    if (d.history && d.history.length > 0) {
                        let html = "";
                        d.history.forEach(h => {
                            let resClass = 'h-pending';
                            let icon = '⏳';
                            if (h.result === 'WIN') { resClass = 'h-win'; icon = '✅'; }
                            if (h.result === 'LOSS') { resClass = 'h-loss'; icon = '❌'; }
                            if (h.result === 'SKIP') { resClass = 'h-pending'; icon = '⛔'; }

                            html += `
                                <div class="history-item">
                                    <span class="h-period">${h.period}</span>
                                    <span style="font-weight:bold;">${h.pred}</span>
                                    <span class="${resClass}">${icon} ${h.result}</span>
                                </div>
                            `;
                        });
                        histBox.innerHTML = html;
                    }
                })
                .catch(err => console.log(err));
        }

        setInterval(updateData, 500);
        updateData();
    </script>
</body>
</html>
"""

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
