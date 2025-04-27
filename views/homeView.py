from simulation.DataLoader import DataLoader
from simulation.Visualizer import Visualizer

import streamlit as st # type: ignore
import datetime as dt
import os

def get_flags_from_strategy(strategy_name):
    return {
        "rsi": "RSI" in strategy_name,
        "macd": "MACD" in strategy_name,
        "regresion": "Regresión" in strategy_name,
        "neuralNetwork": "Neural Network" in strategy_name
    }

print("Navegando a Home")

home_placeholder = st.empty()

with home_placeholder.container():
    # Título de la app
    st.title("Simulador de Estrategias de Trading")

    capital_por_operacion = st.number_input("Capital por operación (€)", min_value=100, value=100, step=100)

    estrategia = st.selectbox("Selecciona estrategia:", 
                            ["RSI", "MACD", "RSI + MACD", 
                            "RSI + Regresión Lineal", "MACD + Regresión Lineal", "RSI + MACD + Regresión Lineal", 
                            "RSI + Neural Network", "MACD + Neural Network", "RSI + MACD + Neural Network"])

    # Carpeta de datos locales
    data_folder = "data"

    # Obtener lista de archivos CSV
    csv_files = [f for f in os.listdir(data_folder) if f.endswith(".csv")]

    col1, col2 = st.columns([2, 1])

    with col1:
        # Selección de archivo
        selected_file = st.selectbox(
            "Selecciona un archivo de datos:", 
            csv_files, 
            index=0  # por defecto ESP35 5min
        )

    with col2:
        selected_freq = st.selectbox(
            "Temporalidad:",
            options=["1min", "5min", "15min", "30min", "1h", "1D"],
            index=1  # por defecto 5min
        )

    # Cargar archivo seleccionado
    if selected_file:
        file_path = os.path.join(data_folder, selected_file)
        data_loader = DataLoader(file_path, freq=selected_freq)
        data_loader.load_data()

        st.success(f"Archivo cargado: {selected_file}")

        start_date = data_loader.data.index.min()
        end_date = data_loader.data.index.max()
        data = data_loader.get_filtered_data(start_date, end_date)

        fig = Visualizer.plot_interactive_price(data)
        st.plotly_chart(fig, use_container_width=True)

        # Mostrar información general
        st.write("Rango de fechas disponible:")
        st.write(f"Desde: {data_loader.data.index.min()}  Hasta: {data_loader.data.index.max()}")

        # Selección de fechas desde el frontend 
        col1, col2 = st.columns(2)

        with col1:
            subcol1, subcol2 = col1.columns([2, 1])

            with subcol1:      
                start_date_training = st.date_input("Inicio entrenamiento", value=dt.date(2025, 3, 3))
                start_date_test = st.date_input("Inicio test", value=dt.date(2025, 3, 4))

            with subcol2:  
                start_time_training = st.time_input("Hora training i", value=dt.time(8, 0, 0), label_visibility="hidden")
                start_time_test = st.time_input("Hora test i", value=dt.time(8, 0, 0), label_visibility="hidden")

        with col2:
            subcol1, subcol2 = col2.columns([2, 1])

            with subcol1: 
                end_date_training = st.date_input("Fin entrenamiento", value=dt.date(2025, 3, 3))
                end_date_test = st.date_input("Fin test", value=dt.date(2025, 3, 4))

            with subcol2: 
                end_time_training = st.time_input("Hora training f", value=dt.time(17, 0, 0), label_visibility="hidden")
                end_time_test = st.time_input("Hora test f", value=dt.time(17, 0, 0), label_visibility="hidden")

        start_training = dt.datetime.combine(start_date_training, start_time_training)
        end_training = dt.datetime.combine(end_date_training, end_time_training)

        start_test = dt.datetime.combine(start_date_test, start_time_test)
        end_test = dt.datetime.combine(end_date_test, end_time_test)

        # Cargar datos filtrados
        try:
            training_data = data_loader.get_filtered_data(start_training, end_training)
            test_data = data_loader.get_filtered_data(start_test, end_test)

            st.subheader(f"Datos de entrenamiento: {len(training_data)} puntos")
            st.dataframe(training_data.head()["<CLOSE>"])

            st.subheader(f"Datos de test: {len(test_data)} puntos")
            st.dataframe(test_data.head()["<CLOSE>"])

        except Exception as e:
            st.error(f"Error al filtrar las fechas: {e}")

        flags = get_flags_from_strategy(estrategia)

        st.session_state.capital_por_operacion = capital_por_operacion
        st.session_state.selected_strategy = estrategia
        st.session_state.training_data = training_data
        st.session_state.test_data = test_data
        st.session_state.flags = flags

        st.session_state.current_view = "simulation"
        st.session_state.simulando = True

        st.page_link("views/simulationView.py", label="Simular estrategia")

        
            


