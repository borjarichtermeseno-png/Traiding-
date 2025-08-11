# app.py - pega todo esto en GitHub exactamente
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="AI Trading Bot Avanzado", layout="wide")
st.title("🤖 AI Trading Bot Avanzado")
st.markdown("Bot demo: indicadores técnicos, IA simple y backtest visual.")

# --- Inputs (en la barra lateral) ---
st.sidebar.header("Configuración")
asset = st.sidebar.selectbox("Activo (ej: AAPL, BTC-USD, EURUSD=X, GC=F)", ["AAPL","MSFT","EURUSD=X","GC=F","BTC-USD"])
period = st.sidebar.selectbox("Periodo", ["1mo","3mo","6mo","1y","2y"], index=3)
interval = st.sidebar.selectbox("Intervalo (yfinance)", ["1d"], index=0)
initial_capital = st.sidebar.number_input("Capital inicial (demo)", value=10000, step=1000)

# --- Descargar datos ---
st.info("Descargando datos... espera un momento")
data = yf.download(asset, period=period, interval=interval, progress=False)
if data.empty:
    st.error("No se pudieron descargar datos para ese activo / periodo. Prueba otro.")
    st.stop()

# --- Indicadores (SMA y RSI sencillo) ---
data = data.dropna()
data['SMA_20'] = data['Close'].rolling(window=20).mean()
data['SMA_50'] = data['Close'].rolling(window=50).mean()

# RSI simple (sin librería externa)
delta = data['Close'].diff()
up = delta.clip(lower=0)
down = -1 * delta.clip(upper=0)
roll_up = up.rolling(14).mean()
roll_down = down.rolling(14).mean()
rs = roll_up / (roll_down + 1e-9)
data['RSI'] = 100.0 - (100.0 / (1.0 + rs))

data = data.dropna()

# --- Gráfico principal: velas + SMA ---
fig = go.Figure()
fig.add_trace(go.Candlestick(x=data.index, open=data['Open'], high=data['High'],
                             low=data['Low'], close=data['Close'], name='Precio'))
fig.add_trace(go.Scatter(x=data.index, y=data['SMA_20'], line=dict(color='blue', width=1), name='SMA 20'))
fig.add_trace(go.Scatter(x=data.index, y=data['SMA_50'], line=dict(color='orange', width=1), name='SMA 50'))
st.plotly_chart(fig, use_container_width=True)

# --- Preparar datos para IA simple ---
data['Return'] = data['Close'].pct_change()
data['Target'] = np.where(data['Return'].shift(-1) > 0, 1, 0)  # sube el siguiente periodo?
data = data.dropna()

features = ['Open','High','Low','Close','Volume','SMA_20','SMA_50','RSI']
X = data[features]
y = data['Target']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Entrenar modelo rápido y ligero
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, shuffle=False)
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)
score = model.score(X_test, y_test)

st.subheader("🔎 Resultados del modelo (prueba)")
st.write(f"Precisión en test: {score*100:.2f}%")

# --- Señal actual ---
last = X_scaled[-1].reshape(1, -1)
pred = model.predict(last)[0]
prob = model.predict_proba(last)[0][pred]
if pred == 1:
    st.success(f"📈 Señal actual: COMPRAR  ({prob*100:.2f}% confianza)")
else:
    st.error(f"📉 Señal actual: VENDER / NO ENTRAR  ({prob*100:.2f}% confianza)")

# --- Backtest simple (ejecución simulada) ---
capital = float(initial_capital)
position = 0  # unidades
entry_price = 0
history = []

for i in range(len(data)):
    sig = model.predict(X_scaled[i].reshape(1, -1))[0]
    price = float(data['Close'].iloc[i])
    date = data.index[i]
    if sig == 1 and position == 0:
        # comprar 1 unidad (demo)
        position = 1
        entry_price = price
        capital -= price
        history.append({'date': date, 'action': 'BUY', 'price': price, 'capital': capital + position*price})
    elif sig == 0 and position == 1:
        # vender
        position = 0
        pnl = price - entry_price
        capital += price
        history.append({'date': date, 'action': 'SELL', 'price': price, 'pnl': pnl, 'capital': capital})

# Mostrar capital final
final_net = capital + (position*price if position>0 else 0)
st.subheader("📊 Backtest (simulación sencilla)")
st.metric("Capital inicial", f"{initial_capital:.2f} USD")
st.metric("Capital final (simulado)", f"{final_net:.2f} USD")

# Gráfico de evolución de capital (si hay operaciones)
if len(history) > 0:
    df_ops = pd.DataFrame(history)
    # grafico simple de capital a lo largo del tiempo
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=df_ops['date'], y=df_ops['capital'], mode='lines+markers', name='Capital'))
    st.plotly_chart(fig2, use_container_width=True)
    st.subheader("Operaciones registradas")
    st.dataframe(df_ops.tail(50))
    # permitir descarga CSV
    csv = df_ops.to_csv(index=False).encode('utf-8')
    st.download_button("Descargar historial de operaciones (CSV)", data=csv, file_name='operaciones_backtest.csv', mime='text/csv')
else:
    st.info("No se registraron operaciones en el backtest con esta configuración.")
