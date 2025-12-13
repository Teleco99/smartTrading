from darwinex.DwxController import DwxController  
from simulation.Visualizer import Visualizer
from streamlit_autorefresh import st_autorefresh
from views.holders import controller_holder, status_holder
from collections import defaultdict
from threading import Thread

import streamlit as st
import pandas as pd
import json
import os

def start_controller_thread(symbol, timeframe, strategy, capital, pasos_por_delante, optimization_params, callback):
    def run():
        try:
            controller = DwxController(
                symbol=symbol,
                timeframe=timeframe,
                strategy=strategy,
                capital_por_operacion=capital,
                sampling_rate=pasos_por_delante,
                optimization_params=optimization_params,
                make_callback=callback
            )

            controller_holder["controller"] = controller
        except Exception as e:
            print(f"[ERROR en hilo controller]: {e}")
    Thread(target=run).start()

def make_status_callback():
    def callback(frac, base_message):
        percentage = int(frac * 100)
        status_holder["status"] = f"{base_message} {percentage}%"
    return callback

def cerrarConexion(): 
    dwx = controller_holder.get("controller", None)

    if dwx is None:
        dwx = st.session_state["dwx"]

    if dwx is not None:
        # Paramos bucle de dwx_client
        dwx.dwx.stop()

        # Eliminar referencias del controlador
        del st.session_state["dwx"]

        # Eliminar referencia global si existe
        if "controller" in controller_holder:
            controller_holder["controller"] = None  

        # Reiniciar todos los estados
        for key in list(st.session_state.keys()):
            if key != "text":
                del st.session_state[key]

        # Forzar recolección (opcional, paranoia extra)
        import gc
        gc.collect()

        print("❌ Controlador borrado de memoria.")

    # Evita que se dispare antes de limpiar todo
    st.session_state.init_Metatrader = False
    st.session_state.text = "⚠️ Conexión cerrada. Pulsa 'Iniciar conexión' para volver a empezar."

    st.rerun()  

def extraer_tecnicas(estrategia: str):
    return [t.strip() for t in estrategia.split("+")]

TIMEFRAME_MILLISECONDS = {
    "M1": 60 * 1000,
    "M5": 5 * 60 * 1000,
    "M15": 15 * 60 * 1000,
    "H1": 60 * 60 * 1000,
    "D1": 24 * 60 * 60 * 1000,
}


st.title("Sistema de trading automático")

subheader_placeholder = st.empty()
form_placeholder = st.empty()
warning_placeholder = st.empty()
graph_placeholder = st.empty()
button_placeholder = st.empty()

if "init_Metatrader" not in st.session_state:
    st.session_state.init_Metatrader = False
    st.session_state.text = f"⚠️ Metatrader no iniciado"

if not st.session_state.init_Metatrader:
    with form_placeholder.container():
        # Selección de símbolo y timeframe ---
        symbols = ["ESP35", "SP500m", "EURUSD", "GBPUSD"]
        timeframes = ["M1", "M5", "M15", "H1", "D1"]
        pasos_por_delantes = [3, 5]

        col1, col2, col3 = st.columns([1, 1, 1])

        symbol = col1.selectbox("Símbolo:", symbols, index=2)
        timeframe = col2.selectbox("Temporalidad:", timeframes, index=1)
        pasos_por_delante = col3.selectbox("Pasos por delante:", pasos_por_delantes, index=0)

        col1, col2 = st.columns([1, 1])

        strategy = col1.selectbox("Selecciona estrategia:", [
            "RSI", "MACD", "RSI + MACD", 
            "RSI + Regresión Lineal", "MACD + Regresión Lineal", "RSI + MACD + Regresión Lineal", 
            "RSI + Neural Network", "MACD + Neural Network", "RSI + MACD + Neural Network"
        ])
        capital_por_operacion = col2.number_input("Capital por operación (€)", min_value=1000, value=10000, step=1000)
        
        hiperparam_source = st.radio("Origen de los hiperparámetros:", ["Últimos datos (en vivo)", "Archivo de optimización (.json)"])

        opt_folder = "optimizaciones"
        opt_json_files = [f for f in os.listdir(opt_folder) if f.endswith(".json")]

        optimization_params = {}

        if hiperparam_source == "Archivo de optimización (.json)":
            tecnicas = extraer_tecnicas(strategy)
            opt_folder = "optimizaciones"

            # Asegurar que todos los archivos sean .json
            opt_json_files = [f for f in os.listdir(opt_folder) if f.lower().endswith(".json")]

            # Buscar todos los archivos que coincidan por técnica
            files_por_tecnica = defaultdict(list)

            for tecnica in tecnicas:
                tecnica_slug = tecnica.replace(" ", "_").lower()
                for f in opt_json_files:
                    if tecnica_slug in f.lower():
                        files_por_tecnica[tecnica].append(f)

            # Mostrar un selectbox por técnica y cargar el archivo elegido
            for tecnica in tecnicas:
                archivos = files_por_tecnica.get(tecnica, [])
                if archivos:
                    selected_file = st.selectbox(f"Selecciona JSON para {tecnica}:", archivos)

                    with open(os.path.join(opt_folder, selected_file), "r") as f:
                        optimization_params[tecnica] = json.load(f)
                else:
                    st.warning(f"⚠️ No se encontraron archivos de optimización para {tecnica}")

        st.session_state.symbol = symbol
        st.session_state.timeframe = timeframe
        st.session_state.strategy = strategy
        st.session_state.capital_por_operacion = capital_por_operacion
        st.session_state.pasos_por_delante = pasos_por_delante
        st.session_state.optimization_params = optimization_params

    if button_placeholder.button("Iniciar conexión con Metatrader"):
        form_placeholder.empty()
        
        st.session_state.init_Metatrader = True
        st.session_state.text = f"⏳ Iniciando conexión..."
        st.session_state.timeframe_milliseconds = TIMEFRAME_MILLISECONDS[st.session_state.timeframe]
        st.session_state.last_index = None

        start_controller_thread(
            st.session_state.symbol, 
            st.session_state.timeframe, 
            st.session_state.strategy, 
            st.session_state.capital_por_operacion, 
            st.session_state.pasos_por_delante,
            st.session_state.optimization_params,
            make_status_callback
            )

dwx = None

# Actualización cada N segundos
UPDATE_INTERVAL = 1000  # milisegundos

# Limite de actualizaciones para evitar bucle infinito
UPDATE_LIMIT = 10

warning_placeholder.info(st.session_state.text)

if st.session_state.init_Metatrader:
    subheader_placeholder.subheader(f"Estrategia: {st.session_state.strategy}")

    # Esperamos que el hilo haya puesto el controller
    dwx_temporal = controller_holder.get("controller", None)

    # Si el hilo lo ha creado y aún no lo hemos guardado en session_state
    if dwx_temporal is not None and "dwx" not in st.session_state:
        st.session_state.dwx = dwx_temporal
        dwx = dwx_temporal
        print("✅ Controlador asignado desde hilo.")
    
    # Si ya tenemos uno en session_state, lo usamos
    elif "dwx" in st.session_state:
        dwx = st.session_state.dwx
        print("📦 Usando controlador persistente de session_state.")

    # Si aún no está listo, esperamos con auto-refresh
    else:
        print("⌛ Esperando a que el hilo cree el controlador...")
        st_autorefresh(interval=UPDATE_INTERVAL, limit=UPDATE_LIMIT)

if "dwx" in st.session_state and st.session_state.dwx is not None:
    dwx = st.session_state.dwx

    if dwx.error == "OPEN_ORDER_LOTSIZE_OUT_OF_RANGE":
        lote_estandar = 100_000
        min_lote = 0.01
        st.session_state.text = f"✖️ Error: Se ha intentado efectuar una orden pero el capital por operación es demasiado bajo. El capital mínimo es {dwx.ask * lote_estandar * min_lote}"

    if not dwx.data.empty:
        data = dwx.data
        signals = dwx.signals

        st.session_state.last_index = data[~data['<CLOSE>'].isna()].index[-1].tz_localize(None)

        # Calcular tiempo desde el ultimo punto hasta ahora
        last_index = st.session_state.last_index

        # Calcular tiempo para la siguiente vela
        frecuencia = pd.Timedelta(milliseconds=st.session_state.timeframe_milliseconds)
        fecha_esperada = last_index + (frecuencia * 2)
        
        now = (pd.Timestamp.utcnow() + pd.Timedelta(hours=3)).tz_localize(None)

        # Tiempo restante hasta que la nueva vela debería estar formada
        diff_ms = (fecha_esperada - now) / pd.Timedelta(milliseconds=1)

        # Se añaden 5 segundos de margen
        UPDATE_GRAPH = diff_ms + 5000 

        print(f"Ultimo index disponible: {last_index}")
        print(f"Fecha esperada: {fecha_esperada}")
        print(f"Fecha actual: {now}")
        print(f"Segundos hasta proxima actualización: {UPDATE_GRAPH / 1000}")

        warning_placeholder.empty()

        # Generamos los gráficos
        fig = Visualizer.plot_interactive_combined(data, signals, operaciones=dwx.operaciones, title="Precios en tiempo real")

        # Mostramos los gráficos
        graph_placeholder.plotly_chart(fig, use_container_width=True)

        st.subheader("Operaciones realizadas")
        st.dataframe(dwx.operaciones)

        button_placeholder.empty()

        if UPDATE_GRAPH > 0:
            st_autorefresh(interval=UPDATE_GRAPH)
        elif button_placeholder.button("Actualizar Metatrader"):
            st.rerun()

        if st.button("Cerrar conexión"):
            st.session_state.text = "⚠️ Conexión cerrada. Pulsa 'Iniciar conexión' para volver a empezar."
            cerrarConexion()
    else:
        st.session_state.text = "✖️ Error: MetaTrader no conectado. Inicia Metatrader y carga DWX_Server"

        if dwx.error == "HISTORIC_DATA":
            st.session_state.text = "✖️ Error: Actualmente el mercado esta cerrado. Prueba con otro símbolo"

        cerrarConexion()


    
 