from simulation.DataLoader import DataLoader
from simulation.Visualizer import Visualizer
from datetime import timedelta

import streamlit as st # type: ignore
import datetime as dt
import os, glob
import pandas as pd

os.makedirs("optimizaciones", exist_ok=True)

def tecnica_key(tecnica: str) -> str:
    return tecnica.replace(" ", "_")

def elegir_fila_opt_desde_tabla(df: pd.DataFrame, titulo: str, default_row: int = 0) -> dict:
    st.subheader(titulo)

    # Intento: selección nativa desde st.dataframe (click en la fila)
    try:
        ev = st.dataframe(
            df,
            use_container_width=True,
            on_select="rerun",
            selection_mode="single-row",
        )
        rows = ev.selection.rows if ev is not None else [] # type: ignore
        row_idx = rows[0] if rows else default_row

    except TypeError:
        # Fallback si tu Streamlit no tiene on_select/selection_mode
        st.warning("Tu Streamlit no soporta selección de filas en st.dataframe. Usando selector de fila (fallback).")
        st.dataframe(df, use_container_width=True)
        row_idx = st.selectbox("Fila a usar:", list(range(len(df))), index=min(default_row, max(0, len(df)-1)))

    row_idx = max(0, min(int(row_idx), len(df) - 1))
    params = df.iloc[row_idx].to_dict()

    st.caption(f"✅ Fila seleccionada: {row_idx}")
    st.table(pd.DataFrame(params.items(), columns=["Parámetro", "Valor"]))

    return params

def list_opt_csv(selected_file: str, tecnica: str) -> list[str]:
    base = os.path.splitext(os.path.basename(selected_file))[0]
    key = tecnica_key(tecnica)

    pattern = os.path.join("optimizaciones", f"{base}_*{key}*.csv")
    matches = sorted(glob.glob(pattern), key=lambda p: os.path.getmtime(p))
    return matches  # ordenado por fecha (más antiguo -> más reciente)

def params_from_df_row(df: pd.DataFrame, row_idx: int = 0) -> dict:
    row_idx = max(0, min(row_idx, len(df) - 1))
    return df.iloc[row_idx].to_dict()

def best_params_from_csv(csv_path: str) -> dict:
    df = pd.read_csv(csv_path)

    first = df.iloc[0]

    return first.to_dict()    

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


home_placeholder = st.empty()

with home_placeholder.container():
    # Título de la app
    st.title("Simulador de Estrategias de Trading")

    capital_operacion = st.number_input("Capital por operación (€)", min_value=100, value=1000, step=100)
    
    col1, col2 = st.columns([2, 1])
    with col2:
        selected_freq = st.selectbox("Temporalidad:", ["5min", "15min", "1h", "1D"], index=1)

    # Convertimos la temporalidad seleccionada al código que aparece en los nombres de archivo
    freq_code = freq_map[selected_freq]

    indicador = st.selectbox(
        "Selecciona indicador:",
        ["RSI", "MACD"]
    )

    modelo = st.selectbox(
        "Selecciona modelo:",
        ["Ninguno", "Regresión Lineal", "Red Neuronal", "Random Forest"]
    )

    estrategia = indicador if modelo == "Ninguno" else f"{indicador} + {modelo}"
    
    if modelo != "Ninguno":
        match selected_freq:
            case "5min":
                unidad = "horas"
                min_freq = 4
            case "15min":
                unidad = "horas"
                min_freq = 8
            case "1h":
                unidad = "horas"
                min_freq = 24
            case "1D":
                unidad = "días"
                min_freq = 24

        habilitar_reentreno = st.checkbox("Habilitar reentrenamiento periódico del modelo", value=True)
        if habilitar_reentreno:
            frecuencia_reentrenamiento = st.number_input(f"Frecuencia de reentrenamiento ({unidad})", min_value=min_freq, value=min_freq, step=1)

            # Transformamos frecuencia de unidad a número de puntos
            match selected_freq:
                case "5min":
                    frecuencia_reentrenamiento = frecuencia_reentrenamiento * 12
                case "15min":
                    frecuencia_reentrenamiento = frecuencia_reentrenamiento * 4
        else:
            frecuencia_reentrenamiento = 0
    else:
        frecuencia_reentrenamiento = 0

    # Carga de archivo que contengan la temporalidad
    data_folder = "data"
    csv_files = [
        f for f in os.listdir(data_folder)
        if f.endswith(".csv") and extraer_temporalidad(f) == freq_code.lower()
    ]

    # Selector de archivo (ya filtrado)
    with col1:
        selected_file = st.selectbox("Selecciona archivo:", csv_files)

    col1, col2 = st.columns([1, 1])

    # Cargar archivo seleccionado
    if selected_file:
        file_path = os.path.join(data_folder, selected_file)
        data_loader = DataLoader(file_path, freq=selected_freq)
        data_loader.load_data()

        opt_ind_list = list_opt_csv(selected_file, indicador)  # RSI / MACD
        opt_mod_list = [] if modelo == "Ninguno" else list_opt_csv(selected_file, modelo)  # Red Neuronal / etc.

        usar_opt_ind = False
        usar_opt_mod = False

        chosen_ind_path = None
        chosen_mod_path = None

        chosen_ind_row = 0
        chosen_mod_row = 0

       # ---------- INDICADOR ----------
        best_ind_params = None
        if opt_ind_list:
            ind_df = pd.read_csv(opt_ind_list[0])
            usar_opt_ind = st.checkbox("Usar optimización de indicador", value=True)

            if usar_opt_ind and len(ind_df) > 0:
                best_ind_params = elegir_fila_opt_desde_tabla(ind_df, "Selecciona hiperparametros de indicador")

        else:
            st.caption(f"⛔ No hay optimización guardada para {indicador} en este dataset.")


        # ---------- MODELO ----------
        best_mod_params = None
        if modelo != "Ninguno":
            if opt_mod_list:
                mod_df = pd.read_csv(opt_mod_list[0])
                usar_opt_mod = st.checkbox("Usar optimización de modelo", value=True)

                if usar_opt_mod and len(mod_df) > 0:
                    best_mod_params = elegir_fila_opt_desde_tabla(mod_df, "Selecciona hiperparametros de modelo")
            else:
                st.caption(f"⛔ No hay optimización guardada para {modelo} en este dataset.")

        if data_loader.data is not None:
            start_date = data_loader.data.index.min()
            end_date = data_loader.data.index.max()
            data = data_loader.get_filtered_data(start_date, end_date)

            fig = Visualizer.plot_interactive_price(data)
            st.plotly_chart(fig, use_container_width=True)

            # Asegura que el índice sea datetime
            idx = pd.to_datetime(data_loader.data.index)

            min_dt = idx.min().to_pydatetime()
            max_dt = idx.max().to_pydatetime()

            # Mostrar información general
            st.subheader("Rango de fechas disponible:")
            st.info(f"Desde: {data_loader.data.index.min()}  Hasta: {data_loader.data.index.max()}")

            # Calcula el punto 70% del rango
            min_date = min_dt.date()
            max_date = max_dt.date()

            total_days = (max_date - min_date).days
            split_days = int(total_days * 0.7)          # entero
            training_end_default = min_date + timedelta(days=split_days)

            # Test empieza al día siguiente para no solapar:
            test_start_default = min(training_end_default + timedelta(days=1), max_date)

            # Selección de puntos de entrenamiento
            st.subheader(f"Datos de entrenamiento: ")
            col1, col2 = st.columns(2)
            with col1:
                training_start = st.date_input(
                    "Fecha de inicio:",
                    value=min_dt.date(),          # default = mínimo
                    min_value=min_dt.date(),      # límite inferior
                    max_value=max_dt.date()       # límite superior
                )
            with col2:
                training_end = st.date_input(
                    "Fecha de fin:",
                    value=training_end_default,   # default = 70% del rango
                    min_value=min_dt.date(),      # límite inferior
                    max_value=max_dt.date()       # límite superior
                )

            training_data = data_loader.get_filtered_data(training_start, training_end)
            st.write(f"Tamaño de dataset: {len(training_data)} puntos")
            st.dataframe(training_data.head()["<CLOSE>"])

            # Selección de puntos de test
            st.subheader(f"Datos de test: ")
            c3, c4 = st.columns(2)
            with c3:
                test_start = st.date_input(
                    "Fecha de inicio: ",
                    value=test_start_default,
                    min_value=min_dt.date(),
                    max_value=max_dt.date()
                )
            with c4:
                test_end = st.date_input(
                    "Fecha de fin: ",
                    value=max_dt.date(),
                    min_value=min_dt.date(),
                    max_value=max_dt.date()
                )
            
            test_data = data_loader.get_filtered_data(test_start, test_end)
            st.write(f"Tamaño de dataset: {len(test_data)} puntos")
            st.dataframe(test_data.head()["<CLOSE>"])

            min_points = max(frecuencia_reentrenamiento * 10, 240)
            if len(data) < min_points:
                st.error(f"⚠️ Se requieren al menos {min_points} puntos para simular")
            else:
                st.session_state.capital_operacion = capital_operacion
                st.session_state.estrategia = estrategia
                st.session_state.training_data = training_data
                st.session_state.test_data = test_data
                st.session_state.frecuencia_entrenamiento = frecuencia_reentrenamiento
                st.session_state.best_ind_params = best_ind_params if usar_opt_ind and opt_ind_list else None
                st.session_state.best_mod_params = best_mod_params if usar_opt_mod and opt_mod_list else None

                st.session_state.current_view = "simulation"

                if st.button("Simular estrategia"):
                    st.session_state.simulando = True
                    st.switch_page("views/simulationView.py")

        
            


