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
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
from xgboost import XGBClassifier
import logging

# ================= CONFIG =================
GROUP_TOKEN = os.getenv("GROUP_TOKEN")
USER_TOKEN = os.getenv("USER_TOKEN")
GROUP_ID = os.getenv("GROUP_ID")
CHANNEL_ID = os.getenv("CHANNEL_ID")

DATA_DIR = os.getenv("DATA_DIR", "/app/data")
os.makedirs(DATA_DIR, exist_ok=True)
DATABASE_PATH = os.getenv("DATABASE_PATH", os.path.join(DATA_DIR, "trading_bot.db"))

LOG_DIR = os.path.join(DATA_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    filename=os.path.join(LOG_DIR, "bot.log"),
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

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

if not GROUP_TOKEN or not USER_TOKEN:
    print("ERROR: Missing tokens!")
    sys.exit(1)

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
    logging.error(f"VK init error: {e}", exc_info=True)
    sys.exit(1)

# ================= DB =================
try:
    conn = sqlite3.connect(DATABASE_PATH, check_same_thread=False)
    cursor = conn.cursor()
    print("Database connected successfully")
except Exception as e:
    logging.error(f"Database connection error: {e}", exc_info=True)
    sys.exit(1)

cursor.execute('''CREATE TABLE IF NOT EXISTS subscribers (
    user_id INTEGER PRIMARY KEY,
    is_subscribed INTEGER DEFAULT 1
)''')
cursor.execute('''CREATE TABLE IF NOT EXISTS signals_history (
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
)''')
conn.commit()
print("Database initialized")

# ================= SYMBOLS =================
SYMBOLS = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT']
CHECK_INTERVAL = 300

#xgb
xgb_model = XGBClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

# ================= ML MODELS =================
rf_model = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42)
scaler_rf = StandardScaler()
scaler_xgb = StandardScaler()
MODEL_RF_TRAINED = False
MODEL_LSTM_TRAINED = False
lstm_model = None
SEQ_LEN = 20
MODEL_XGB_TRAINED = False

# ================= DATA FETCH =================
def fetch_bybit(symbol, limit=250):
    url = f"https://api.bybit.com/v5/market/kline?category=linear&symbol={symbol}&interval=1&limit={limit}"
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        data = r.json()
        k = data['result']['list'][::-1]
        df = pd.DataFrame(k, columns=['timestamp','open','high','low','close','volume','turnover'])
        df = df[['timestamp','open','high','low','close','volume']]
        df['timestamp'] = pd.to_datetime(df['timestamp'].astype(float), unit='ms')
        for col in ['open','high','low','close','volume']:
            df[col] = df[col].astype(float)
        return df
    except:
        return None

def fetch_binance(symbol, limit=250, interval='1m'):
    url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        data = r.json()
        df = pd.DataFrame(data, columns=[
            'timestamp','open','high','low','close','volume','_','_','_','_','_','_'
        ])
        df = df[['timestamp','open','high','low','close','volume']]
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        for col in ['open','high','low','close','volume']:
            df[col] = df[col].astype(float)
        return df
    except:
        return None

def fetch_ohlcv(symbol, limit=250):
    df = fetch_bybit(symbol, limit)
    if df is not None:
        return df
    df = fetch_binance(symbol, limit)
    return df
    

def fetch_multi_tf(symbol):
    df_1m = fetch_binance(symbol, limit=250, interval='1m')
    df_5m = fetch_binance(symbol, limit=250, interval='5m')
    df_15m = fetch_binance(symbol, limit=250, interval='15m')   

    if df_1m is None or df_5m is None or df_15m is None:
        return None, None, None

    return df_1m, df_5m, df_15m

# ================= INDICATORS =================
def calculate_indicators(df):
    df['EMA20'] = df['close'].ewm(span=20).mean()
    df['EMA50'] = df['close'].ewm(span=50).mean()
    df['EMA200'] = df['close'].ewm(span=200).mean()
    df['MACD'] = df['close'].ewm(span=12).mean() - df['close'].ewm(span=26).mean()
    df['MACD_signal'] = df['MACD'].ewm(span=9).mean()
    delta = df['close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    df['ATR'] = (df['high'] - df['low']).rolling(14).mean()
    df['VOL_SMA'] = df['volume'].rolling(20).mean()
    df['HH'] = df['high'].rolling(20).max()
    df['LL'] = df['low'].rolling(20).min()

    df['BB_middle'] = df['close'].rolling(20).mean()
    df['BB_std'] = df['close'].rolling(20).std()
    df['BB_upper'] = df['BB_middle'] + 2*df['BB_std']
    df['BB_lower'] = df['BB_middle'] - 2*df['BB_std']
    
    df['STO_k'] = (df['close'] - df['LL']) / (df['HH'] - df['LL']) * 100
    df['STO_d'] = df['STO_k'].rolling(3).mean()

    high = df['high']
    low = df['low']
    close = df['close']
    df['tr'] = pd.concat([high - low, abs(high - close.shift()), abs(low - close.shift())], axis=1).max(axis=1)
    df['atr14'] = df['tr'].rolling(14).mean()
    df['up_move'] = high - high.shift()
    df['down_move'] = low.shift() - low
    df['plus_dm'] = np.where((df['up_move'] > df['down_move']) & (df['up_move'] > 0), df['up_move'], 0)
    df['minus_dm'] = np.where((df['down_move'] > df['up_move']) & (df['down_move'] > 0), df['down_move'], 0)
    df['plus_di'] = 100 * (df['plus_dm'].rolling(14).sum() / df['atr14'])
    df['minus_di'] = 100 * (df['minus_dm'].rolling(14).sum() / df['atr14'])
    df['ADX'] = 100 * abs(df['plus_di'] - df['minus_di']) / (df['plus_di'] + df['minus_di'])
    return df

def create_features(df):
    df = df.copy()
    df['return'] = df['close'].pct_change()
    df['volatility'] = df['return'].rolling(10).std()
    df['ema20'] = df['close'].ewm(span=20).mean()
    df['ema50'] = df['close'].ewm(span=50).mean()
    df['ema_diff'] = df['ema20'] - df['ema50']
    df['rsi'] = df['RSI']
    df['macd'] = df['MACD']
    df['volume_ratio'] = df['volume'] / df['VOL_SMA']
    df.dropna(inplace=True)
    return df

def create_target(df, horizon=5):
    df = df.copy()
    horizon = int(horizon)  # 👈 фикс
    df['future_close'] = df['close'].shift(-horizon)
    df['target'] = (df['future_close'] > df['close']).astype(int)
    df.dropna(inplace=True)
    return df

# ================= TRAINING =================
def train_xgb(symbol="BTCUSDT"):
    global MODEL_XGB_TRAINED

    df = fetch_ohlcv(symbol, limit=1000)
    if df is None or len(df) < 300:
        return
        

    df = calculate_indicators(df)
    df = create_features(df)
    df = create_target(df)
    

    features = ['return','volatility','ema_diff','rsi','macd','volume_ratio']

    X = df[features]
    y = df['target']
    
    X_scaled = scaler_rf.fit_transform(X)
    depth, lr = auto_tune_xgb(X_scaled, y)
    xgb_model.set_params(max_depth=depth, learning_rate=lr)
    xgb_model.fit(X_scaled, y)

    MODEL_XGB_TRAINED = True
    print("XGBoost trained")


def train_rf(symbol="BTCUSDT"):
    global MODEL_RF_TRAINED
    df = fetch_ohlcv(symbol, limit=1000)
    if df is None or len(df) < 300:
        return
    df = calculate_indicators(df)
    df = create_features(df)
    df = create_target(df)
    features = ['return','volatility','ema_diff','rsi','macd','volume_ratio']
    X = df[features]
    y = df['target']
    X_scaled = scaler_xgb.fit_transform(X)
    rf_model.fit(X_scaled, y)
    MODEL_RF_TRAINED = True
    print(f"RandomForest trained on {len(X)} samples")

def train_lstm(symbol="BTCUSDT"):
    global lstm_model, MODEL_LSTM_TRAINED, SEQ_LEN  # 👈 добавлено

    df = fetch_ohlcv(symbol, limit=1000)
    if df is None or len(df) < 50:
        print("Not enough data for LSTM training")
        return

    df = calculate_indicators(df)
    df = create_features(df)
    df = create_target(df)

    features = ['return', 'volatility', 'ema_diff', 'rsi', 'macd', 'volume_ratio']
    data = df[features].values

    SEQ_LEN = int(SEQ_LEN)  # гарантируем целое число

    X, y = [], []
    for i in range(int(len(data) - SEQ_LEN)):
        X.append(data[i:i+SEQ_LEN])
        y.append(df['target'].iloc[i+SEQ_LEN])
    X = np.array(X)
    y = np.array(y)

    lstm_model = Sequential([
        LSTM(50, input_shape=(X.shape[1], X.shape[2])),
        Dense(1, activation='sigmoid')
    ])
    lstm_model.compile(optimizer=Adam(0.001), loss='binary_crossentropy', metrics=['accuracy'])
    lstm_model.fit(X, y, epochs=5, batch_size=32, verbose=0)

    MODEL_LSTM_TRAINED = True
    print(f"LSTM trained on {len(X)} sequences")

def training_loop():
    while True:
        try:
            print("Training RandomForest...")
            train_rf()
            print("Training LSTM...")
            train_lstm()
            print("Training XGBoost...")
            train_xgb()     
        except Exception as e:
            logging.error(f"training_loop error: {e}", exc_info=True)
        time.sleep(3600)

# ================= PREDICTION =================
def predict_signal(df):
    if not (MODEL_RF_TRAINED and MODEL_LSTM_TRAINED and MODEL_XGB_TRAINED):
        return None, None

    df = calculate_indicators(df)
    df = create_features(df)

    last = df.iloc[-1:]
    features = ['return','volatility','ema_diff','rsi','macd','volume_ratio']

    X_rf = scaler_rf.transform(last[features])
    X_xgb = scaler_xgb.transform(last[features])
    rf_prob = rf_model.predict_proba(X_rf)[0][1]
    xgb_prob = xgb_model.predict_proba(X_xgb)[0][1]

    lstm_data = df[features].values
    if len(lstm_data) < SEQ_LEN:
        return None, None

    lstm_input = np.expand_dims(lstm_data[-SEQ_LEN:], axis=0)
    lstm_prob = lstm_model.predict(lstm_input, verbose=0)[0][0]
    lstm_w = np.clip(lstm_w, 0.1, 0.9)
    rf_w = 1 - lstm_w   

    regime = detect_market_regime(df)

    # динамические веса по волатильности
    vol = df['return'].rolling(20).std().iloc[-1]

    if regime == "trend":
        lstm_w = min(0.9, 0.5 + vol*50)
        rf_w = 1 - lstm_w
    elif regime == "range":
        rf_w = min(0.9, 0.5 + (0.01 - vol)*50)
        lstm_w = 1 - rf_w
    else:
        return None, None  # флэт

    final_prob = rf_prob * rf_w + lstm_prob * lstm_w + xgb_prob * 0.1

    last_row = df.iloc[-1]

    # ❌ флэт
    if last_row['volatility'] < 0.001:
        return None, None

    # ❌ слабый RSI
    if 45 < last_row['rsi'] < 55:
        return None, None

    # 🎯 более строгие пороги
    if final_prob > 0.67:
        return "LONG", final_prob
    elif final_prob < 0.33:
        return "SHORT", 1 - final_prob

    return None, None

def generate_equity_curve():
    cursor.execute("""
        SELECT entry, tp1, sl, result 
        FROM signals_history 
        WHERE status='closed'
    """)
    rows = cursor.fetchall()

    equity = [0]
    balance = 0

    for entry, tp1, sl, result in rows:
        risk = abs(entry - sl)
        reward = abs(tp1 - entry)

        if result == "TP1":
            balance += reward / risk
        elif result == "SL":
            balance -= 1

        equity.append(balance)

    plt.figure()
    plt.plot(equity)
    plt.title("Equity Curve")
    plt.grid()

    path = os.path.join(DATA_DIR, "equity.png")
    plt.savefig(path)
    plt.close()

    return path

# ================= SIGNAL GENERATION =================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def generate_signal(symbol):
    # Загружаем данные по нескольким таймфреймам
    df, df5, df15 = fetch_multi_tf(symbol)
    if not check_data_valid(df) or not check_data_valid(df5) or not check_data_valid(df15):
        return None, None
    
    # Рассчитываем индикаторы
    df = calculate_indicators(df)
    df5 = calculate_indicators(df5)
    df15 = calculate_indicators(df15)
    df.dropna(inplace=True)

    last = df.iloc[-1]
    last_5m = df5.iloc[-1]
    last_15m = df15.iloc[-1]

    # ================= Тренд =================
    trend_up = last['EMA50'] > last['EMA200']
    trend_down = last['EMA50'] < last['EMA200']
    trend_5m_up = last_5m['EMA50'] > last_5m['EMA200']
    trend_5m_down = last_5m['EMA50'] < last_5m['EMA200']
    trend_15m_up = last_15m['EMA50'] > last_15m['EMA200']
    trend_15m_down = last_15m['EMA50'] < last_15m['EMA200']

    # ================= Моментум =================
    momentum_up = last['MACD'] > last['MACD_signal'] and last['RSI'] > 55
    momentum_down = last['MACD'] < last['MACD_signal'] and last['RSI'] < 45

    # ================= Фильтры =================
    volume_ok = last['volume'] > last['VOL_SMA']
    breakout_up = last['close'] > last['HH']
    breakout_down = last['close'] < last['LL']
    strong_trend = last['ADX'] > 20
    bb_long = last['close'] < last['BB_lower']
    bb_short = last['close'] > last['BB_upper']
    stoch_long = last['STO_k'] < 20
    stoch_short = last['STO_k'] > 80

    # ================= Стратегия =================
    strategy_direction = None
    if trend_up and momentum_up and volume_ok and breakout_up and trend_5m_up and trend_15m_up and strong_trend and (bb_long or stoch_long):
        strategy_direction = "LONG"
    elif trend_down and momentum_down and volume_ok and breakout_down and trend_5m_down and trend_15m_down and strong_trend and (bb_short or stoch_short):
        strategy_direction = "SHORT"

    # ================= ML прогноз =================
    ml_direction, ml_conf = predict_signal(df)
    if not strategy_direction or not ml_direction:
        return None, df
    if strategy_direction != ml_direction:
        return None, df

    # ================= Точки входа и цели =================
    entry = last['close']
    atr = last['ATR']
    if ml_direction == "LONG":
        sl = entry - atr * 1.5
        tp1 = entry + atr * 2
        tp2 = entry + atr * 3
        tp3 = entry + atr * 5
    else:
        sl = entry + atr * 1.5
        tp1 = entry - atr * 2
        tp2 = entry - atr * 3
        tp3 = entry - atr * 5

    # ================= Уверенность =================
    momentum_strength = (last['MACD'] / atr) + (last['STO_k'] / 100)
    confidence = round(min(ml_conf * momentum_strength * 100, 100), 1)
    if confidence < 70:
        return None, df

    # ================= Риск/Возврат =================
    risk_reward = round(abs(tp1 - entry) / abs(entry - sl), 2)
    if risk_reward < 1.5:
        return None, df

    # ================= Формируем сигнал =================
    signal = {
        'symbol': symbol,
        'direction': ml_direction,
        'entry': round(entry, 2),
        'tp1': round(tp1, 2),
        'tp2': round(tp2, 2),
        'tp3': round(tp3, 2),
        'sl': round(sl, 2),
        'confidence': confidence,
        'risk_reward': risk_reward
    }

    # ================= Создаем график =================
    chart_path = f"./charts/{symbol}_signal.png"
    plt.figure(figsize=(12,6))
    plt.plot(df['close'], label='Close', color='blue')
    plt.plot(df['EMA50'], label='EMA50', color='orange')
    plt.plot(df['EMA200'], label='EMA200', color='red')
    plt.fill_between(df.index, df['BB_lower'], df['BB_upper'], color='gray', alpha=0.2, label='BB')
    plt.scatter(df.index[-1], entry, marker='^' if ml_direction=="LONG" else 'v', color='green', s=150, label='Entry')
    plt.title(f"{symbol} Signal: {ml_direction} | Confidence: {confidence}% | R/R: {risk_reward}")
    plt.legend()
    plt.savefig(chart_path)
    plt.close()
    signal['chart'] = chart_path

    return signal, df

def auto_tune_xgb(X, y):
    best_score = 0
    best_params = None

    for depth in [3, 5, 7]:
        for lr in [0.01, 0.05, 0.1]:
            model = XGBClassifier(
                max_depth=depth,
                learning_rate=lr,
                n_estimators=150
            )
            model.fit(X, y)
            score = model.score(X, y)

            if score > best_score:
                best_score = score
                best_params = (depth, lr)

    print("Best XGB:", best_params, best_score)
    return best_params

def detect_market_regime(df):
    last = df.iloc[-1]

    trend_strength = abs(last['EMA50'] - last['EMA200']) / last['close']
    volatility = df['return'].rolling(20).std().iloc[-1]

    if trend_strength > 0.01:
        return "trend"
    elif volatility < 0.001:
        return "flat"
    else:
        return "range"

def get_stats():
    cursor.execute("""
        SELECT entry, tp1, sl, result, direction 
        FROM signals_history 
        WHERE status='closed'
    """)
    rows = cursor.fetchall()

    total = len(rows)
    if total == 0:
        return "Нет данных"

    balance = 0
    wins = 0
    losses = 0

    for entry, tp1, sl, result, direction in rows:
        risk = abs(entry - sl)
        reward = abs(tp1 - entry)

        if result == "TP1":
            balance += reward / risk
            wins += 1
        elif result == "SL":
            balance -= 1
            losses += 1

    winrate = wins / total * 100

    return f"""📊 СТАТИСТИКА

Сделок: {total}
Winrate: {winrate:.1f}%
PnL (R): {balance:.2f}
Wins: {wins} | Losses: {losses}"""

# ================= VK + GRAPHICS =================
def format_signal_post(signal):
    direction_emoji = "🟢" if signal['direction']=='LONG' else "🔴"
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

def create_advanced_chart(signal, df):
    try:
        import matplotlib.dates as mdates
        
        # Берем последние 100 свечей
        plot_df = df.tail(100).copy()
        
        # Вычисляем дополнительные индикаторы
        plot_df['BB_middle'] = plot_df['close'].rolling(20).mean()
        plot_df['BB_std'] = plot_df['close'].rolling(20).std()
        plot_df['BB_upper'] = plot_df['BB_middle'] + 2*plot_df['BB_std']
        plot_df['BB_lower'] = plot_df['BB_middle'] - 2*plot_df['BB_std']
        
        # ADX
        high_low = plot_df['high'] - plot_df['low']
        high_close = np.abs(plot_df['high'] - plot_df['close'].shift())
        low_close = np.abs(plot_df['low'] - plot_df['close'].shift())
        tr = high_low.combine(high_close, max).combine(low_close, max)
        plot_df['+DM'] = plot_df['high'].diff()
        plot_df['-DM'] = -plot_df['low'].diff()
        plot_df['+DM'] = np.where((plot_df['+DM']>plot_df['-DM']) & (plot_df['+DM']>0), plot_df['+DM'], 0)
        plot_df['-DM'] = np.where((plot_df['-DM']>plot_df['+DM']) & (plot_df['-DM']>0), plot_df['-DM'], 0)
        tr14 = tr.rolling(14).sum()
        plus_DI = 100 * (plot_df['+DM'].rolling(14).sum() / tr14)
        minus_DI = 100 * (plot_df['-DM'].rolling(14).sum() / tr14)
        plot_df['ADX'] = abs(plus_DI - minus_DI) / (plus_DI + minus_DI) * 100

        # Stochastic
        plot_df['lowest_low'] = plot_df['low'].rolling(14).min()
        plot_df['highest_high'] = plot_df['high'].rolling(14).max()
        plot_df['Stoch'] = 100 * (plot_df['close'] - plot_df['lowest_low']) / (plot_df['highest_high'] - plot_df['lowest_low'])

        # Создаем фигуру с 2 subplot
        fig, (ax1, ax2) = plt.subplots(2,1, figsize=(14,8), sharex=True, gridspec_kw={'height_ratios':[2,1]})
        
        # Верхний график: цена + EMA + Bollinger + Entry/SL/TP
        ax1.plot(plot_df['timestamp'], plot_df['close'], color='black', label='Price', linewidth=1.5)
        ax1.plot(plot_df['timestamp'], plot_df['EMA50'], color='blue', label='EMA50', alpha=0.7)
        ax1.plot(plot_df['timestamp'], plot_df['EMA200'], color='purple', label='EMA200', alpha=0.7)
        ax1.plot(plot_df['timestamp'], plot_df['BB_upper'], color='orange', linestyle='--', alpha=0.7)
        ax1.plot(plot_df['timestamp'], plot_df['BB_lower'], color='orange', linestyle='--', alpha=0.7)
        
        # Entry / TP / SL
        ax1.axhline(signal['entry'], color='green', linestyle='-', linewidth=2, label=f'Entry {signal["entry"]:.2f}')
        ax1.axhline(signal['tp1'], color='lime', linestyle='--', linewidth=1.5, label=f'TP1 {signal["tp1"]:.2f}')
        ax1.axhline(signal['tp2'], color='lime', linestyle=':', linewidth=1.5, alpha=0.7)
        ax1.axhline(signal['tp3'], color='lime', linestyle=':', linewidth=1.5, alpha=0.5)
        ax1.axhline(signal['sl'], color='red', linestyle='--', linewidth=2, label=f'SL {signal["sl"]:.2f}')
        
        ax1.set_ylabel("Price")
        ax1.set_title(f"{signal['symbol']} - {signal['direction']} SIGNAL", fontsize=14, fontweight='bold')
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # Нижний график: MACD, RSI, ADX, Stochastic
        ax2.plot(plot_df['timestamp'], plot_df['MACD'], color='blue', label='MACD')
        ax2.plot(plot_df['timestamp'], plot_df['MACD_signal'], color='red', label='MACD Signal')
        ax2.plot(plot_df['timestamp'], plot_df['rsi'], color='purple', label='RSI')
        ax2.plot(plot_df['timestamp'], plot_df['ADX'], color='orange', label='ADX')
        ax2.plot(plot_df['timestamp'], plot_df['Stoch'], color='green', label='Stoch')
        
        ax2.axhline(70, color='grey', linestyle='--', linewidth=0.5)
        ax2.axhline(30, color='grey', linestyle='--', linewidth=0.5)
        
        ax2.set_ylabel("Indicators")
        ax2.legend(loc='upper left')
        ax2.grid(True, alpha=0.3)
        
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        plt.xticks(rotation=30)
        plt.tight_layout()
        
        # Сохраняем во временный файл
        temp_dir = os.path.join(DATA_DIR, "temp")
        os.makedirs(temp_dir, exist_ok=True)
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png', dir=temp_dir)
        plt.savefig(temp_file.name, dpi=120, bbox_inches='tight')
        plt.close(fig)
        return temp_file.name
    except Exception as e:
        logging.error(f"Advanced chart creation error: {e}", exc_info=True)
        return None

def publish_to_community(vk_session, signal):
    try:
        attachments = []
        if signal.get('chart'):
            upload = vk_session.photos.getMessagesUploadServer(peer_id=signal['symbol'])
            with open(signal['chart'], 'rb') as file:
                response = requests.post(upload['upload_url'], files={'photo': file}).json()
            photo = vk_session.photos.saveMessagesPhoto(**response)[0]
            attachments.append(f"photo{photo['owner_id']}_{photo['id']}")

        message = (
            f"{signal['symbol']} | {signal['direction']}\n"
            f"Entry: {signal['entry']:.2f}\n"
            f"SL: {signal['sl']:.2f}\n"
            f"TP1: {signal['tp1']:.2f}, TP2: {signal['tp2']:.2f}, TP3: {signal['tp3']:.2f}\n"
            f"Confidence: {signal['confidence']*100:.1f}%"
        )

        vk_session.messages.send(peer_id=signal['symbol'],
                                 message=message,
                                 attachment=",".join(attachments),
                                 random_id=0)
        logging.info(f"Signal published with chart: {signal['symbol']}")
    except Exception as e:
        logging.error(f"Error publishing signal: {e}", exc_info=True)

# ================= SIGNAL LOOP =================
def signal_loop():
    logging.info("Signal loop started")
    print("Signal loop started")

    while True:
        try:
            for symbol in SYMBOLS:
                logging.info(f"Checking symbol: {symbol}")

                # Пропускаем, если недавно был сигнал
                if recently_had_signal(symbol):
                    logging.info(f"Skipping {symbol}, recent signal exists")
                    continue

                # Получаем OHLCV данные
                df = fetch_ohlcv(symbol)
                if not check_data_valid(df):
                    logging.warning(f"Skipping {symbol}, invalid or insufficient data")
                    continue

                # Генерация сигнала
                signal, df = generate_signal(symbol)

                if signal is None:
                    logging.info(f"No valid signal for {symbol}")
                    continue

                # ДЕТАЛЬНОЕ ЛОГИРОВАНИЕ ИНДИКАТОРОВ
                last = df.iloc[-1]
                logging.info(
                    f"{symbol} indicators -> EMA50: {last['EMA50']:.2f}, EMA200: {last['EMA200']:.2f}, "
                    f"MACD: {last['MACD']:.2f}, MACD_signal: {last['MACD_signal']:.2f}, "
                    f"RSI: {last['RSI']:.2f}, ADX: {last['ADX']:.2f}, "
                    f"BB_upper: {last['BB_upper']:.2f}, BB_lower: {last['BB_lower']:.2f}, "
                    f"STO_k: {last['STO_k']:.2f}, volume: {last['volume']:.2f}, VOL_SMA: {last['VOL_SMA']:.2f}"
                )

                logging.info(f"Signal found: {symbol} {signal['direction']} {signal['confidence']:.0f}%")
                print(f"Signal found: {symbol} {signal['direction']} {signal['confidence']:.0f}%")

                # Создание графика
                try:
                    photo_path = create_chart(signal, df)
                except Exception as e:
                    logging.error(f"Failed to create chart for {symbol}: {e}")
                    photo_path = None

                # Форматирование текста для публикации
                try:
                    post_text = format_signal_post(signal)
                except Exception as e:
                    logging.error(f"Failed to format post for {symbol}: {e}")
                    post_text = None

                # Публикация в сообщество VK
                if post_text and publish_to_community(post_text, photo_path):
                    try:
                        cursor.execute("""
                            INSERT INTO signals_history
                            (symbol,direction,entry,tp1,tp2,tp3,sl,confidence,risk_reward)
                            VALUES (?,?,?,?,?,?,?,?,?)
                        """, (
                            signal['symbol'],
                            signal['direction'],
                            signal['entry'],
                            signal['tp1'],
                            signal['tp2'],
                            signal['tp3'],
                            signal['sl'],
                            signal['confidence'],
                            signal['risk_reward']
                        ))
                        conn.commit()
                        logging.info(f"Signal saved to database: {symbol} {signal['direction']}")
                    except Exception as e:
                        logging.error(f"Failed to save signal to DB: {e}", exc_info=True)

                # Удаляем временный файл графика
                if photo_path and os.path.exists(photo_path):
                    os.remove(photo_path)
                    logging.info(f"Deleted chart file for {symbol}")

        except Exception as e:
            logging.error(f"Signal loop error: {e}", exc_info=True)

        # Пауза перед следующей проверкой
        time.sleep(CHECK_INTERVAL)

# Запуск в отдельном потоке
threading.Thread(target=signal_loop, daemon=True).start()

# ================= RESULTS CHECK LOOP =================
def check_results_loop():
    print("Results check loop started")
    while True:
        try:
            cursor.execute("SELECT * FROM signals_history WHERE status='open'")
            rows = cursor.fetchall()
            for r in rows:
                id, symbol, direction, entry, tp1, tp2, tp3, sl, confidence, rr, status, _, _, _ = r
                df = fetch_ohlcv(symbol, limit=2)
                if not check_data_valid(df):
                    continue
                last_price = df['close'].iloc[-1]
                result = None
                if direction=="LONG":
                    if last_price >= tp1:
                        result = "TP1"
                    elif last_price <= sl:
                        result = "SL"
                elif direction=="SHORT":
                    if last_price <= tp1:
                        result = "TP1"
                    elif last_price >= sl:
                        result = "SL"
                if result:
                    cursor.execute("UPDATE signals_history SET status='closed', close_price=?, result=? WHERE id=?",
                                   (last_price, result, id))
                    conn.commit()
                if result:
                    cursor.execute("UPDATE signals_history SET status='closed', close_price=?, result=? WHERE id=?",
                                    (last_price, result, id))
                    conn.commit()
                    logging.info(f"Signal closed: {symbol} {direction} {result} at {last_price}")
        except Exception as e:
            logging.error(f"check_results_loop error: {e}", exc_info=True)
        time.sleep(60)

def get_keyboard():
    keyboard = VkKeyboard(one_time=False)
    keyboard.add_button("📊 Статистика", color=VkKeyboardColor.PRIMARY)
    keyboard.add_line()
    keyboard.add_button("Старт", color=VkKeyboardColor.POSITIVE)
    keyboard.add_button("Стоп", color=VkKeyboardColor.NEGATIVE)
    return keyboard.get_keyboard()

# ================= VK MESSAGE HANDLER =================
def handle_message(user_id, text):
    text = text.lower()

    if "стат" in text:
        stats = get_stats()
        vk.messages.send(
            user_id=user_id,
            message=stats,
            keyboard=get_keyboard(),
            random_id=random.randint(0, 1_000_000)
        )

    elif "стоп" in text:
        cursor.execute("UPDATE subscribers SET is_subscribed=0 WHERE user_id=?", (user_id,))
        conn.commit()

        vk.messages.send(
            user_id=user_id,
            message="Вы отписаны",
            keyboard=get_keyboard(),
            random_id=random.randint(0, 1_000_000)
        )

    elif "старт" in text:
        cursor.execute("INSERT OR REPLACE INTO subscribers VALUES (?,1)", (user_id,))
        conn.commit()

        vk.messages.send(
            user_id=user_id,
            message="Вы подписаны",
            keyboard=get_keyboard(),
            random_id=random.randint(0, 1_000_000)
        )
    elif "график" in text:
        path = generate_equity_curve()
        vk.messages.send(
            user_id=user_id,
            message="📈 Кривая доходности",
            attachment=upload_photo(path),
            random_id=random.randint(0, 1_000_000)
        )

    else:
        vk.messages.send(
            user_id=user_id,
            message="Используй кнопки 👇",
            keyboard=get_keyboard(),
            random_id=random.randint(0, 1_000_000)
        )

def upload_photo(path):
    photo = user_upload.photo_messages(path)[0]
    return f"photo{photo['owner_id']}_{photo['id']}"

def recently_had_signal(symbol, minutes=30):
    cursor.execute("""
        SELECT created_at FROM signals_history 
        WHERE symbol=? 
        ORDER BY created_at DESC LIMIT 1
    """, (symbol,))
    
    row = cursor.fetchone()
    if not row:
        return False

    last_time = datetime.fromisoformat(row[0])
    return (datetime.now() - last_time).total_seconds() < minutes * 60

def check_data_valid(df):
    if df is None or len(df) < 50:
        logging.warning("Data fetch failed or too short")
        return False
    if df['close'].isnull().any():
        logging.warning("Data contains NaN")
        return False
    return True


# ================= MAIN =================
if __name__ == '__main__':
    print("="*50)
    print("TRADING SIGNAL BOT STARTED")
    print("="*50)
    
    stop_event = threading.Event()
    
    # Потоки
    threading.Thread(target=signal_loop, daemon=True).start()
    threading.Thread(target=check_results_loop, daemon=True).start()
    threading.Thread(target=training_loop, daemon=True).start()
    
    print("Bot started successfully!")
    
    # Основной цикл VK сообщений
    try:
        while not stop_event.is_set():
            for event in longpoll.listen():
                if event.type == VkBotEventType.MESSAGE_NEW:
                    msg = event.message
                    handle_message(msg['from_id'], msg.get('text', ''))
    except KeyboardInterrupt:
        print("Bot stopped by user")
    except Exception as e:
        logging.error(f"Fatal error in main loop error: {e}", exc_info=True)
        import traceback
        traceback.print_exc()

