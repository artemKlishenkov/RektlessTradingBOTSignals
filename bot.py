import vk_api
from vk_api.bot_longpoll import VkBotLongPoll, VkBotEventType
from vk_api.keyboard import VkKeyboard, VkKeyboardColor
from vk_api.upload import VkUpload
import sqlite3
import random
import time
import os
import tempfile
import threading
import json
from datetime import datetime
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import requests
import sys
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# ================= CONFIG =================
# Загружаем из переменных окружения Railway
GROUP_TOKEN = os.getenv("GROUP_TOKEN")
USER_TOKEN = os.getenv("USER_TOKEN")
GROUP_ID = os.getenv("GROUP_ID")
CHANNEL_ID = os.getenv("CHANNEL_ID")

# Создаем папку для данных
DATA_DIR = os.getenv("DATA_DIR", "/app/data")
os.makedirs(DATA_DIR, exist_ok=True)

# Путь к базе данных
DATABASE_PATH = os.getenv("DATABASE_PATH", os.path.join(DATA_DIR, "trading_bot.db"))

# Если переменных нет, пробуем config.json
if not GROUP_TOKEN:
    try:
        with open('config.json', 'r', encoding='utf-8') as f:
            CONFIG = json.load(f)
            GROUP_TOKEN = CONFIG.get('GROUP_TOKEN')
            USER_TOKEN = CONFIG.get('USER_TOKEN')
            GROUP_ID = CONFIG.get('GROUP_ID')
            CHANNEL_ID = CONFIG.get('CHANNEL_ID')
    except:
        pass

# Проверяем наличие токенов
if not GROUP_TOKEN or not USER_TOKEN:
    print("ERROR: Missing tokens! Set environment variables GROUP_TOKEN and USER_TOKEN")
    sys.exit(1)

# Преобразуем ID в int
GROUP_ID = int(GROUP_ID) if GROUP_ID else None
CHANNEL_ID = int(CHANNEL_ID) if CHANNEL_ID else None

print(f"Starting bot with GROUP_ID: {GROUP_ID}, CHANNEL_ID: {CHANNEL_ID}")
print(f"Data directory: {DATA_DIR}")
print(f"Database path: {DATABASE_PATH}")

# ================= VK INIT =================
try:
    vk_session = vk_api.VkApi(token=GROUP_TOKEN)
    vk = vk_session.get_api()
    longpoll = VkBotLongPoll(vk_session, GROUP_ID)
    print("VK Bot session initialized")
    
    user_session = vk_api.VkApi(token=USER_TOKEN)
    vk_user = user_session.get_api()
    user_upload = VkUpload(user_session)
    print("VK User session initialized")
except Exception as e:
    print(f"VK init error: {e}")
    sys.exit(1)

# ================= DB =================
try:
    conn = sqlite3.connect(DATABASE_PATH, check_same_thread=False)
    cursor = conn.cursor()
    print("Database connected successfully")
except Exception as e:
    print(f"Database connection error: {e}")
    sys.exit(1)

cursor.execute('''
CREATE TABLE IF NOT EXISTS subscribers (
    user_id INTEGER PRIMARY KEY,
    is_subscribed INTEGER DEFAULT 1
)
''')

cursor.execute('''
CREATE TABLE IF NOT EXISTS signals_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT,
    direction TEXT,
    entry REAL,
    tp1 REAL,
    tp2 REAL,
    tp3 REAL,
    sl REAL,
    confidence REAL,
    risk_reward REAL,
    status TEXT DEFAULT 'open',
    close_price REAL,
    result TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
''')
conn.commit()
print("Database initialized")

# ================= BYBIT =================
SYMBOLS = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT']
BASE_URL = 'https://api.bybit.com'
INTERVAL = '1'
CHECK_INTERVAL = 300

# ================= ML MODEL =================
model = RandomForestClassifier(n_estimators=100, random_state=42)
scaler = StandardScaler()

# ================= HELPER FUNCTIONS =================
def fetch_ohlcv(symbol, limit=250, retries=3):
    for attempt in range(retries):
        try:
            url = f'{BASE_URL}/v5/market/kline?category=linear&symbol={symbol}&interval={INTERVAL}&limit={limit}'
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            data_json = resp.json()
            
            if data_json.get('retCode') != 0 or not data_json.get('result', {}).get('list'):
                print(f"[{symbol}] No data from Bybit: {data_json.get('retMsg')}")
                time.sleep(2)
                continue
                
            kline_list = data_json['result']['list'][::-1]
            df = pd.DataFrame(kline_list, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'])
            
            df['close'] = df['close'].astype(float)
            df['open'] = df['open'].astype(float)
            df['high'] = df['high'].astype(float)
            df['low'] = df['low'].astype(float)
            df['volume'] = df['volume'].astype(float)
            df['timestamp'] = pd.to_datetime(pd.to_numeric(df['timestamp']), unit='ms')
            
            return df
        except Exception as e:
            print(f"Fetch OHLCV error for {symbol}, attempt {attempt+1}: {e}")
            time.sleep(2)
    return None

def calculate_indicators(df):
    df['EMA9'] = df['close'].ewm(span=9, adjust=False).mean()
    df['EMA21'] = df['close'].ewm(span=21, adjust=False).mean()
    df['EMA200'] = df['close'].ewm(span=200, adjust=False).mean()
    
    df['MACD'] = df['close'].ewm(span=12, adjust=False).mean() - df['close'].ewm(span=26, adjust=False).mean()
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    df['Vol_SMA20'] = df['volume'].rolling(20).mean()
    df['ATR'] = df['high'].rolling(14).max() - df['low'].rolling(14).min()
    
    delta = df['close'].diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    roll_up = up.rolling(14).mean()
    roll_down = down.rolling(14).mean()
    RS = roll_up / roll_down
    df['RSI'] = 100.0 - (100.0 / (1.0 + RS))
    return df

def format_signal_post(signal):
    direction_emoji = "🟢" if signal['direction'] == 'LONG' else "🔴"
    return f"""{direction_emoji} НОВЫЙ СИГНАЛ {direction_emoji}

📊 {signal['symbol']} - {signal['direction']}

💰 Вход: {signal['entry']:.2f}

🎯 Цели:
  TP1: {signal['tp1']:.2f}
  TP2: {signal['tp2']:.2f}
  TP3: {signal['tp3']:.2f}

🛑 Стоп-лосс: {signal['sl']:.2f}

📈 Уверенность: {signal['confidence']:.0f}%
⚖️ Risk/Reward: 1:{signal['risk_reward']}

🕐 {datetime.now().strftime('%H:%M %d.%m.%Y')}"""

def create_chart(signal, df):
    try:
        fig, ax = plt.subplots(figsize=(12, 6))
        
        plot_df = df.tail(100)
        
        ax.plot(plot_df['timestamp'], plot_df['close'], label='Price', color='black', linewidth=1.5)
        ax.plot(plot_df['timestamp'], plot_df['EMA200'], label='EMA200', color='blue', alpha=0.7, linewidth=1)
        
        ax.axhline(signal['entry'], color='green', linestyle='-', linewidth=2, label=f'Entry: {signal["entry"]:.2f}')
        ax.axhline(signal['tp1'], color='lime', linestyle='--', linewidth=1.5, label=f'TP1: {signal["tp1"]:.2f}')
        ax.axhline(signal['tp2'], color='lime', linestyle=':', linewidth=1.5, alpha=0.7)
        ax.axhline(signal['tp3'], color='lime', linestyle=':', linewidth=1.5, alpha=0.5)
        ax.axhline(signal['sl'], color='red', linestyle='--', linewidth=2, label=f'SL: {signal["sl"]:.2f}')
        
        ax.set_title(f"{signal['symbol']} - {signal['direction']} SIGNAL", fontsize=14, fontweight='bold')
        ax.set_xlabel('Time')
        ax.set_ylabel('Price (USDT)')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Создаем временную папку для графиков
        temp_dir = os.path.join(DATA_DIR, "temp")
        os.makedirs(temp_dir, exist_ok=True)
        
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png', dir=temp_dir)
        plt.savefig(temp_file.name, dpi=100, bbox_inches='tight')
        plt.close(fig)
        
        return temp_file.name
    except Exception as e:
        print(f"Chart error: {e}")
        return None

def publish_to_community(text, photo_path=None):
    try:
        attachment = None
        if photo_path and os.path.exists(photo_path):
            photo = user_upload.photo_wall(photos=photo_path)[0]
            attachment = f"photo{photo['owner_id']}_{photo['id']}"
            print("Photo uploaded")
        
        if attachment:
            vk_user.wall.post(owner_id=-CHANNEL_ID, message=text, attachments=attachment, from_group=1)
        else:
            vk_user.wall.post(owner_id=-CHANNEL_ID, message=text, from_group=1)
        
        print("Post published to community")
        return True
    except Exception as e:
        print(f'Publish error: {e}')
        return False

def get_keyboard(user_id):
    kb = VkKeyboard(one_time=False)
    cursor.execute("SELECT is_subscribed FROM subscribers WHERE user_id=?", (user_id,))
    r = cursor.fetchone()
    if r and r[0]:
        kb.add_button("Отписаться", color=VkKeyboardColor.NEGATIVE)
    else:
        kb.add_button("Подписаться", color=VkKeyboardColor.POSITIVE)
    kb.add_line()
    kb.add_button("Статистика", color=VkKeyboardColor.SECONDARY)
    return kb.get_keyboard()

# ================= SIGNAL LOGIC =================
def generate_signal(symbol):
    df = fetch_ohlcv(symbol, limit=250)
    if df is None or len(df) < 200:
        return None, None
        
    df = calculate_indicators(df)
    df.dropna(inplace=True)
    
    last_close = df['close'].iloc[-1]
    ema9 = df['EMA9'].iloc[-1]
    ema21 = df['EMA21'].iloc[-1]
    ema200 = df['EMA200'].iloc[-1]
    macd = df['MACD'].iloc[-1]
    macd_signal = df['MACD_Signal'].iloc[-1]
    rsi = df['RSI'].iloc[-1]
    atr = df['ATR'].iloc[-1]
    vol = df['volume'].iloc[-1]
    vol_sma = df['Vol_SMA20'].iloc[-1]
    
    direction = None
    
    is_long = (ema9 > ema21) and (last_close > ema200) and (macd > macd_signal) and (50 < rsi < 70) and (vol > vol_sma)
    is_short = (ema9 < ema21) and (last_close < ema200) and (macd < macd_signal) and (30 < rsi < 50) and (vol > vol_sma)
    
    if is_long:
        direction = 'LONG'
    elif is_short:
        direction = 'SHORT'
        
    if not direction:
        return None, df
        
    entry = last_close
    
    if direction == 'LONG':
        sl = entry - (atr * 1.5)
        tp1 = entry + (atr * 1.5)
        tp2 = entry + (atr * 2.5)
        tp3 = entry + (atr * 4.0)
    else:
        sl = entry + (atr * 1.5)
        tp1 = entry - (atr * 1.5)
        tp2 = entry - (atr * 2.5)
        tp3 = entry - (atr * 4.0)
        
    confidence = 65 + (rsi - 30) / 40 * 30 if direction == 'LONG' else 65 + (70 - rsi) / 40 * 30
    confidence = min(max(confidence, 60), 95)
    
    risk_diff = abs(entry - sl)
    risk_reward = round(abs(tp1 - entry) / risk_diff, 1) if risk_diff > 0 else 0
    
    signal = {
        'symbol': symbol,
        'direction': direction,
        'entry': round(entry, 2),
        'tp1': round(tp1, 2),
        'tp2': round(tp2, 2),
        'tp3': round(tp3, 2),
        'sl': round(sl, 2),
        'confidence': confidence,
        'risk_reward': risk_reward
    }
    return signal, df

# ================= THREADS =================
def signal_loop():
    print("Signal loop started")
    while True:
        try:
            for symbol in SYMBOLS:
                signal, df = generate_signal(symbol)
                if signal and signal['confidence'] >= 65:
                    print(f"Signal found: {symbol} {signal['direction']} confidence: {signal['confidence']:.0f}%")
                    
                    photo_path = create_chart(signal, df)
                    post_text = format_signal_post(signal)
                    
                    if publish_to_community(post_text, photo_path):
                        cursor.execute("""INSERT INTO signals_history 
                            (symbol, direction, entry, tp1, tp2, tp3, sl, confidence, risk_reward) 
                            VALUES (?,?,?,?,?,?,?,?,?)""",
                            (signal['symbol'], signal['direction'], signal['entry'],
                             signal['tp1'], signal['tp2'], signal['tp3'], 
                             signal['sl'], signal['confidence'], signal['risk_reward']))
                        conn.commit()
                        print(f"Signal saved to DB")
                    
                    if photo_path and os.path.exists(photo_path):
                        os.remove(photo_path)
                        
        except Exception as e:
            print(f"Signal loop error: {e}")
            
        time.sleep(CHECK_INTERVAL)

def check_results_loop():
    print("Results check loop started")
    while True:
        try:
            cursor.execute("SELECT * FROM signals_history WHERE status='open'")
            rows = cursor.fetchall()
            for r in rows:
                id, symbol, direction, entry, tp1, tp2, tp3, sl, confidence, risk_reward, status, _, _, _ = r
                df = fetch_ohlcv(symbol, limit=2)
                if df is None:
                    continue
                last_price = df['close'].iloc[-1]
                result = None
                close_price = last_price
                
                if direction == 'LONG':
                    if last_price >= tp1:
                        result = 'win'
                    elif last_price <= sl:
                        result = 'loss'
                else:
                    if last_price <= tp1:
                        result = 'win'
                    elif last_price >= sl:
                        result = 'loss'
                        
                if result:
                    cursor.execute("UPDATE signals_history SET status='closed', close_price=?, result=? WHERE id=?",
                                   (close_price, result, id))
                    conn.commit()
                    text = f"🏁 Signal result: {symbol} {direction}\nClosed at: {close_price:.2f}\nResult: {result.upper()}"
                    publish_to_community(text)
            time.sleep(60)
        except Exception as e:
            print(f"Results check error: {e}")
            time.sleep(60)

# ================= VK HANDLER =================
def handle_message(user_id, text):
    text = text.lower().strip()
    try:
        if text == 'подписаться':
            cursor.execute("INSERT OR REPLACE INTO subscribers(user_id, is_subscribed) VALUES(?,1)", (user_id,))
            conn.commit()
            vk.messages.send(user_id=user_id, message='✅ Вы подписаны на сигналы!', 
                           random_id=random.randint(1, 2**31), keyboard=get_keyboard(user_id))
        elif text == 'отписаться':
            cursor.execute("UPDATE subscribers SET is_subscribed=0 WHERE user_id=?", (user_id,))
            conn.commit()
            vk.messages.send(user_id=user_id, message='❌ Вы отписаны от сигналов', 
                           random_id=random.randint(1, 2**31), keyboard=get_keyboard(user_id))
        elif text == 'статистика':
            cursor.execute("SELECT COUNT(*) FROM signals_history")
            total = cursor.fetchone()[0] or 0
            cursor.execute("SELECT COUNT(*) FROM signals_history WHERE result='win'")
            wins = cursor.fetchone()[0] or 0
            winrate = (wins/total*100) if total > 0 else 0
            vk.messages.send(user_id=user_id, 
                message=f"📊 Статистика:\n\nВсего сигналов: {total}\nУспешных: {wins}\nВинрейт: {winrate:.1f}%", 
                random_id=random.randint(1, 2**31), keyboard=get_keyboard(user_id))
        else:
            vk.messages.send(user_id=user_id, 
                message='Используйте кнопки меню 👇', 
                random_id=random.randint(1, 2**31), keyboard=get_keyboard(user_id))
    except Exception as e:
        print(f"Handle message error: {e}")

# ================= MAIN =================
if __name__ == '__main__':
    print("="*50)
    print("TRADING SIGNAL BOT STARTED")
    print("="*50)
    print(f"Group ID: {GROUP_ID}")
    print(f"Channel ID: {CHANNEL_ID}")
    print(f"Data directory: {DATA_DIR}")
    print(f"Database: {DATABASE_PATH}")
    print("="*50)
    
    # Запускаем потоки
    threading.Thread(target=signal_loop, daemon=True).start()
    threading.Thread(target=check_results_loop, daemon=True).start()
    
    print("Bot started successfully!")
    
    # Основной цикл для сообщений
    while True:
        try:
            for event in longpoll.listen():
                if event.type == VkBotEventType.MESSAGE_NEW:
                    msg = event.message
                    handle_message(msg['from_id'], msg.get('text', ''))
        except Exception as e:
            print(f"Connection error: {e}")
            time.sleep(5)