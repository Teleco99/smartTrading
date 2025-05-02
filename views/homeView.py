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

    col1, col2 = st.columns([1, 1])

    with col1:
        start_time_permitido = st.time_input("Hora de apertura diario: ", value=dt.time(13, 30, 0))
    with col2:
        end_time_permitido = st.time_input("Hora de cierre diario: ", value=dt.time(20, 0, 0))

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

        # Intervalo total
        with col1:
            start_datetime = st.date_input("Fecha de inicio: ", value=dt.datetime(2025, 3, 10, 0, 0, 0))

        with col2:
            end_datetime = st.date_input("Fecha de fin: ", value=dt.datetime(2025, 3, 11, 0, 0, 0))

        # Porcentaje de entrenamiento
        train_percentage = st.slider("Porcentaje de datos para entrenamiento", min_value=50, max_value=90, value=70, step=10)

        # Filtrar fechas
        data = data_loader.get_filtered_data(start_datetime, end_datetime)

        # Calcular el punto de corte
        data_daily = data.between_time(start_time_permitido, end_time_permitido)
        cutoff_idx = int(len(data_daily) * (train_percentage / 100))

        # Cargar datos filtrados
        training_data = data_daily.iloc[:cutoff_idx]
        test_data = data_daily.iloc[cutoff_idx:]

        st.subheader(f"Datos de entrenamiento: {len(training_data)} puntos")
        st.dataframe(training_data.head()["<CLOSE>"])

        st.subheader(f"Datos de test: {len(test_data)} puntos")
        st.dataframe(test_data.head()["<CLOSE>"])

        data_daily = data.between_time(start_time_permitido, end_time_permitido)

        if(len(data_daily) < 55):
            st.error("Se necesitan al menos 55 puntos de entrenamiento")
        else:
            flags = get_flags_from_strategy(estrategia)

            st.session_state.capital_por_operacion = capital_por_operacion
            st.session_state.selected_strategy = estrategia
            st.session_state.training_data = training_data
            st.session_state.test_data = test_data
            st.session_state.horario_permitido = (start_time_permitido.strftime("%H:%M"), end_time_permitido.strftime("%H:%M"))
            st.session_state.flags = flags

            st.session_state.current_view = "simulation"
            st.session_state.simulando = True

            st.page_link("views/simulationView.py", label="Simular estrategia")

        
            


