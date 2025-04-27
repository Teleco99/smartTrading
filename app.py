import streamlit as st # type: ignore

pg = st.navigation([
    st.Page("views/homeView.py", title="Inicio", icon="🏠"),
    st.Page("views/descargarDatosView.py", title="Descargar Precios", icon="📥"),
    st.Page("views/simulationView.py", title="Simulación", icon="📊"),
], position="sidebar")  

# Ejecutar la página seleccionada
pg.run()