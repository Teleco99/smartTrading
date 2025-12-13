from simulation.DataLoader import DataLoader
from simulation.OptimizerController import OptimizerController
from simulation.Visualizer import Visualizer
import streamlit as st
import datetime as dt
import json
import os
import pandas as pd
import numpy as np


def safe_to_dataframe(x, debug=False):
    if debug:
        # Mostrar debug antes de cambiar nada
        st.write("--------------------------------")
        st.write("Tipo original:", type(x).__name__)

    try:
        arr = np.array(x)
    except Exception as e:
        st.error(f"No se pudo convertir a array: {e}")
        return pd.DataFrame()

    if debug:
        st.write("Tipo tras np.array:", type(arr).__name__)
        st.write("ndim:", arr.ndim)
        st.write("shape:", arr.shape)

    # Escalar → 1x1
    if arr.ndim == 0:
        arr = arr.reshape(1, 1)

    # 1D → convertir en columna
    elif arr.ndim == 1:
        arr = arr.reshape(-1, 1)

    # 2D → OK
    elif arr.ndim == 2:
        pass

    # Más dimensiones → forzar a 2D
    else:
        arr = arr.reshape(arr.shape[0], -1)

    if debug:
        st.write("shape final:", arr.shape)
        st.write("--------------------------------")

    return pd.DataFrame(arr)

def format_result_dict(d):
    formatted = {}
    for k, v in d.items():
        if isinstance(v, (float, int)):   # números → formatear
            formatted[k] = f"{v:.2f}"
        else:
            formatted[k] = v              # otros tipos → dejar igual
    return formatted

def streamlit_progress(value, message=""):
    progress_bar.progress(value, text=message)

def data_callback(tag, pos, X, y):
    st.write(f"🔍 Datos de {tag} en t = {pos}")

    # Convertimos siempre a DF (pero mostrando el debug dentro)
    df_X = safe_to_dataframe(X).round(2)
    df_y = safe_to_dataframe(y).round(2)

    st.write("📌 Entradas (X)")
    st.dataframe(df_X)

    st.write("🎯 Salidas (y)")
    st.dataframe(df_y)

def extraer_temporalidad(nombre_archivo):
    partes = nombre_archivo.replace(".csv", "").split("_")
    if len(partes) >= 2:
        return partes[1].lower()
    return ""


# Mapa entre opción del usuario y fragmento esperado en el nombre del archivo
freq_map = {
    "5min": "5m",
    "15min": "15m",
    "1h": "1h",
    "1D": "1D"
}


st.title("Optimización de Indicadores Predictivos")

tecnica = st.selectbox("Selecciona indicador a optimizar:", ["RSI", "MACD", "Regresión Lineal", "Red Neuronal", "Random Forest"])

col1, col2 = st.columns([2, 1])
with col2:
    selected_freq = st.selectbox("Temporalidad:", ["5min", "15min", "1h", "1D"], index=1)

# Convertimos la temporalidad seleccionada al código que aparece en los nombres de archivo
freq_code = freq_map[selected_freq]

# Carga de archivo que contengan la temporalidad
data_folder = "data"
csv_files = [
    f for f in os.listdir(data_folder)
    if f.endswith(".csv") and extraer_temporalidad(f) == freq_code.lower()
]

# Selector de archivo (ya filtrado)
with col1:
    selected_file = st.selectbox("Selecciona archivo:", csv_files)

col1, col2 = st.columns(2)
with col1:
    start_time = st.time_input("Hora de apertura", value=dt.time(13, 30))
with col2:
    end_time = st.time_input("Hora de cierre", value=dt.time(20, 0))

# Cargar datos
if selected_file:
    file_path = os.path.join(data_folder, selected_file)
    loader = DataLoader(file_path, freq=selected_freq)
    loader.load_data()
    st.success(f"{selected_file} cargado correctamente")

    if loader.data is not None:
        # Rango de fechas disponibles
        start_date = loader.data.index.min()
        end_date = loader.data.index.max()

        col1, col2 = st.columns(2)
        with col1:
            fecha_inicio = st.date_input("Desde", value=start_date.date(), min_value=start_date.date(), max_value=end_date.date())
        with col2:
            fecha_fin = st.date_input("Hasta", value=end_date.date(), min_value=start_date.date(), max_value=end_date.date())

        # Filtro y validación
        filtered = loader.get_filtered_data(fecha_inicio, fecha_fin)
        filtered = filtered.between_time(start_time.strftime("%H:%M"), end_time.strftime("%H:%M"))

        if len(filtered) < 100:
            st.error("⚠️ Se requieren al menos 100 datos para optimizar")
        else:
            st.plotly_chart(Visualizer.plot_interactive_price(filtered), use_container_width=True)
            st.success(f"{len(filtered)} puntos disponibles para optimización")

            progress_placeholder = st.empty()
            progress_bar = progress_placeholder.empty()

            if st.button("Optimizar"):
                with st.spinner("Optimizando..."):

                    if tecnica == "RSI":
                        result = OptimizerController.optimize_rsi(filtered, progress_callback=streamlit_progress)
                    elif tecnica == "MACD":
                        result = OptimizerController.optimize_macd(filtered, progress_callback=streamlit_progress)
                    elif tecnica == "Regresión Lineal":
                        result = OptimizerController.optimize_regression(filtered, progress_callback=streamlit_progress, data_callback=data_callback)
                    elif tecnica == "Red Neuronal":
                        result = OptimizerController.optimize_neural_network(filtered, progress_callback=streamlit_progress, data_callback=data_callback)
                    elif tecnica == "Random Forest":
                        result = OptimizerController.optimize_random_forest(filtered, progress_callback=streamlit_progress, data_callback=data_callback)

                    progress_placeholder.empty()
                    st.write("📊 Resultado óptimo:", format_result_dict(result))

                    # Guardar resultados
                    os.makedirs("optimizaciones", exist_ok=True)
                    file_name = f"{os.path.splitext(selected_file)[0]}_{tecnica.replace(' ', '_')}.json"
                    path = os.path.join("optimizaciones", file_name)

                    with open(path, 'w') as f:
                        json.dump(result, f, indent=2)

                    st.success(f"✅ Resultado guardado como '{file_name}' en la carpeta 'optimizaciones'")

