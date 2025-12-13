from simulation.Visualizer import Visualizer
from simulation.SimulationController import SimulationController
from metrics.Metrics import Metrics

import pandas as pd
import numpy as np
import streamlit as st  # type: ignore


placeholder = st.empty()

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

def data_callback(tag, pos, X, y):
    st.write(f"🔍 Datos de {tag} en t = {pos}")

    # Convertimos siempre a DF (pero mostrando el debug dentro)
    df_X = safe_to_dataframe(X).round(2)
    df_y = safe_to_dataframe(y).round(2)

    st.write("📌 Entradas (X)")
    st.dataframe(df_X)

    st.write("🎯 Salidas (y)")
    st.dataframe(df_y)

with placeholder.container():
    required_keys = ["test_data", "training_data", "selected_strategy"]
    for key in required_keys:
        if key not in st.session_state:
            st.error(f"Falta '{key}'. Vuelve a la pantalla principal.")
            st.stop()

    capital_por_operacion = st.session_state.capital_por_operacion
    horario_permitido = st.session_state.horario_permitido
    training_data = st.session_state.training_data
    test_data = st.session_state.test_data
    estrategia = st.session_state.estrategia

    simulation = SimulationController(capital_por_operacion=capital_por_operacion, training_data=training_data, test_data=test_data, horario_permitido=horario_permitido)

    if st.session_state.get("simulando", False):
        status = st.empty()
        status.info("⌛ Simulando estrategia...")

        progress_bar = st.progress(0)

        num_tecnicas = len([parte.strip() for parte in estrategia.split("+")])
        step_fraction = 1 / num_tecnicas

        def make_progress_callback():
            def callback(frac, message=""): 
                    progress_bar.progress(min(100, int((frac * step_fraction) * 100)))
                    if message:
                        status.info(message)
            return callback

        strategy_map = {
            "RSI": simulation.run_rsi_strategy,
            "MACD": simulation.run_macd_strategy,
            "RSI + MACD": simulation.run_rsi_macd_strategy,
            "RSI + Regresión Lineal": simulation.run_rsi_regresion_strategy,
            "MACD + Regresión Lineal": simulation.run_macd_regresion_strategy,
            "RSI + MACD + Regresión Lineal": simulation.run_rsi_macd_regresion_strategy,
            "RSI + Neural Network": simulation.run_rsi_neuralNetwork_strategy,
            "MACD + Neural Network": simulation.run_macd_neuralNetwork_strategy,
            "RSI + MACD + Neural Network": simulation.run_rsi_macd_neuralNetwork_strategy,
            "RSI + Random Forest": simulation.run_rsi_randomForest_strategy,
        }

        if estrategia in strategy_map:
            operaciones = strategy_map[estrategia](progress_callback=make_progress_callback(), data_callback=data_callback)
        else:
            st.warning(f"Estrategia con {key} no implementada.")
            st.stop()

        if not operaciones:
            st.info("No se generaron operaciones con las señales actuales.")

        metricas = Metrics(operaciones)
        resumen = metricas.resumen()

        status.empty()
        progress_bar.empty()
        st.toast("✅ Simulación completada", icon="🎯")

        st.session_state.simulando = False

    st.subheader("Gráficos interactivo")

    fig_training = Visualizer.plot_interactive_combined(training_data, title="Precios de entrenamiento")
    fig_test = Visualizer.plot_interactive_combined(test_data, signals=simulation.signals, title="Precios de test")
    st.plotly_chart(fig_training, use_container_width=True)
    st.plotly_chart(fig_test, use_container_width=True)

    st.subheader("Operaciones simuladas")
    st.dataframe(operaciones)

    st.subheader("Resumen de métricas")
    st.dataframe(resumen, use_container_width=True)
    
    st.page_link("views/homeView.py", label="Volver")    

           

