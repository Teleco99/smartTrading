from simulation.Visualizer import Visualizer
from simulation.Simulation import Simulation
from metrics.Metrics import Metrics

import streamlit as st  # type: ignore

def get_strategy_key(flags):
    return tuple([
        flags.get("rsi", False),
        flags.get("macd", False),
        flags.get("regresion", False),
        flags.get("neuralNetwork", False)
    ])

print("Navegando a Simulation")

placeholder = st.empty()

with placeholder.container():
    required_keys = ["test_data", "training_data", "selected_strategy", "flags"]
    for key in required_keys:
        if key not in st.session_state:
            st.error(f"Falta '{key}'. Vuelve a la pantalla principal.")
            st.stop()

    capital_por_operacion = st.session_state.capital_por_operacion
    horario_permitido = st.session_state.horario_permitido
    training_data = st.session_state.training_data
    test_data = st.session_state.test_data
    flags = st.session_state.flags

    simulation = Simulation(capital_por_operacion=capital_por_operacion, training_data=training_data, test_data=test_data, horario_permitido=horario_permitido)

    if st.session_state.get("simulando", False):
        status = st.empty()
        status.info("⌛ Simulando estrategia...")

        progress_bar = st.progress(0)

        # Cálculo de proporción de carga por flag activo
        active_flags = [k for k, v in flags.items() if v]
        num_flags = len(active_flags)
        step_fraction = 1 / num_flags
        offset = 0.0

        def make_progress_callback(offset_local):
            return lambda frac, message="": (
                    progress_bar.progress(min(100, int((offset_local + frac * step_fraction) * 100))),
                    status.info(message) if message else None
                )

        strategy_map = {
            (True, False, False, False): simulation.run_rsi_strategy,
            (False, True, False, False): simulation.run_macd_strategy,
            (True, True, False, False): simulation.run_rsi_macd_strategy,
            (True, False, True, False): simulation.run_rsi_regresion_strategy,
            (False, True, True, False): simulation.run_macd_regresion_strategy,
            (True, True, True, False): simulation.run_rsi_macd_regresion_strategy,
            (True, False, False, True): simulation.run_rsi_neuralNetwork_strategy,
            (False, True, False, True): simulation.run_macd_neuralNetwork_strategy,
            (True, True, False, True): simulation.run_rsi_macd_neuralNetwork_strategy,
        }

        key = get_strategy_key(flags)
        if key in strategy_map:
            operaciones = strategy_map[key](progress_callback=make_progress_callback(offset))
            offset += step_fraction
        else:
            st.warning(f"Estrategia con flags {key} no implementada.")
            st.stop()

        if not operaciones:
            st.info("No se generaron operaciones con las señales actuales.")
            exit

        metricas = Metrics(operaciones)
        resumen = metricas.resumen()

        status.empty()
        progress_bar.empty()
        st.toast("✅ Simulación completada", icon="🎯")

        st.session_state.simulando = False

    st.subheader("Gráfico interactivo")

    fig_training = Visualizer.plot_interactive_combined(training_data, title="Precios de entrenamiento")
    fig_test = Visualizer.plot_interactive_combined(test_data, flags=flags, signals=simulation.signals, title="Precios de test")
    st.plotly_chart(fig_training, use_container_width=True)
    st.plotly_chart(fig_test, use_container_width=True)

    st.subheader("Operaciones simuladas")
    st.dataframe(operaciones)

    st.subheader("Resumen de métricas")
    st.dataframe(resumen, use_container_width=True)
    
    st.page_link("views/homeView.py", label="Volver")    

        

           

