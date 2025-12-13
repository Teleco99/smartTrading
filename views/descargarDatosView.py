from datetime import datetime

import streamlit as st
import yfinance as yf
import os

# Crear carpeta "data" si no existe
if not os.path.exists('data'):
    os.makedirs('data')

# ----------------------------
# Función para descargar datos
# ----------------------------
def descargar_y_guardar(symbol, timeframe, year_month):
    year, month = map(int, year_month.split('-'))
    start_date = f"{year}-{month:02d}-01"
    if month == 12:
        end_date = f"{year+1}-01-01"
    else:
        end_date = f"{year}-{month+1:02d}-01"

    # Descargar datos
    print(f"Descargando datos de {start_date} a {end_date}")
    df = yf.download(symbol, interval=timeframe, start=start_date, end=end_date, progress=False)
 
    # Procesar datos
    if df is not None and not df.empty:
        df = df.reset_index()
        df['DATE'] = df['Datetime'].dt.strftime('%Y.%m.%d')
        df['TIME'] = df['Datetime'].dt.strftime('%H:%M:%S')
        df_clean = df[['DATE', 'TIME', 'Close']]
        df_clean.columns = ['<DATE>', '<TIME>', '<CLOSE>']

        if df.empty:
            st.error("No se encontraron datos para esa combinación.")
            return
    else:
        st.error("No se encontraron datos para esa combinación.")
        return

    # Guardar CSV
    filename = f"data/{symbol.upper()}_{timeframe}_{year_month}.csv"
    df_clean.to_csv(filename, index=False, sep='\t')

    st.success(f"Datos guardados correctamente en: {filename}")

# ----------------------------
# Streamlit UI
# ----------------------------
st.title("📈 Descargador de datos históricos")

SYMBOL_MAP = {
    "IBEX 35": "^IBEX",
    "S&P 500": "^GSPC",
    "Dow Jones": "^DJI",
    "EUR/USD": "EURUSD=X",
    "EUR/GBP": "EURGBP=X",
    "GBP/USD": "GBPUSD=X",
    "Ethereum (ETH)": "ETH-USD",
    "Bitcoin (BTC)": "BTC-USD",
    "Polkadot (DOT)": "DOT-USD",
    "NVIDIA": "NVDA",
    "Apple": "AAPL",
    "Google": "GOOGL",
    "AMD": "AMD",
    "Amazon": "AMZN"
}

# Mostrar nombres legibles
display_names = list(SYMBOL_MAP.keys())
selection = st.selectbox("Selecciona un activo:", options=display_names)
symbol = SYMBOL_MAP[selection]

timeframe = st.selectbox(
    "Temporalidad",
    options=["1m", "5m", "15m", "30m", "1h", "1d"], 
    index=1
)

# Seleccionar mes (opciones predefinidas)
año_actual = datetime.now().year
mes_actual = datetime.now().month
meses_disponibles = [f"{año_actual}-{m:02d}" for m in range(1, mes_actual + 1)]

year_month = st.selectbox(
    "Mes completo:",
    options=meses_disponibles
)

if st.button("Descargar y Guardar CSV"):
    descargar_y_guardar(symbol, timeframe, year_month)
