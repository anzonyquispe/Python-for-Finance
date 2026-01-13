import streamlit as st
from datetime import datetime, timedelta

st.title("Widgets de Fecha")

# DATE INPUT
st.subheader("Date Input")
fecha_inicio = st.date_input(
    "Fecha de inicio:",
    value=datetime.now() - timedelta(days=365)
)
fecha_fin = st.date_input(
    "Fecha de fin:",
    value=datetime.now()
)
st.write(f"Periodo: {fecha_inicio} a {fecha_fin}")

# TIME INPUT
st.subheader("Time Input")
hora = st.time_input(
    "Hora de ejecucion:",
    value=datetime.now().time()
)
st.write(f"Hora seleccionada: {hora}")