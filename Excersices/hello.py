import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.title("Graficos con Plotly")

# Datos de ejemplo
np.random.seed(42)
fechas = pd.date_range("2023-01-01", periods=252, freq="D")
precio_inicial = 100
retornos = np.random.normal(0.0005, 0.02, len(fechas))
precios = precio_inicial * np.cumprod(1 + retornos)

df = pd.DataFrame({
    "Fecha": fechas,
    "Precio": precios,
    "Retorno": retornos
})

# 1. Grafico de Lineas Interactivo
st.subheader("1. Grafico de Lineas")

fig_linea = go.Figure()
fig_linea.add_trace(go.Scatter(
    x=df["Fecha"],
    y=df["Precio"],
    mode="lines",
    name="Precio",
    line=dict(color="#1E3A8A", width=2),
    hovertemplate="Fecha: %{x}<br>Precio: $%{y:.2f}<extra></extra>"
))

fig_linea.update_layout(
    title="Evolucion del Precio",
    xaxis_title="Fecha",
    yaxis_title="Precio ($)",
    hovermode="x unified",
    height=400
)

st.plotly_chart(fig_linea, use_container_width=True)

# 2. Grafico de Velas (Candlestick)
st.subheader("2. Grafico de Velas")

# Crear datos OHLC
df_ohlc = pd.DataFrame({
    "Date": fechas,
    "Open": precios * np.random.uniform(0.99, 1.01, len(fechas)),
    "High": precios * np.random.uniform(1.01, 1.03, len(fechas)),
    "Low": precios * np.random.uniform(0.97, 0.99, len(fechas)),
    "Close": precios
})

fig_velas = go.Figure(data=[go.Candlestick(
    x=df_ohlc["Date"],
    open=df_ohlc["Open"],
    high=df_ohlc["High"],
    low=df_ohlc["Low"],
    close=df_ohlc["Close"],
    increasing_line_color="green",
    decreasing_line_color="red"
)])

fig_velas.update_layout(
    title="Grafico de Velas",
    xaxis_title="Fecha",
    yaxis_title="Precio",
    xaxis_rangeslider_visible=False,
    height=400
)

st.plotly_chart(fig_velas, use_container_width=True)

# 3. Histograma de Retornos
st.subheader("3. Distribucion de Retornos")

fig_hist = px.histogram(
    df, x="Retorno",
    nbins=50,
    title="Distribucion de Retornos Diarios",
    labels={"Retorno": "Retorno Diario"},
    color_discrete_sequence=["#10B981"]
)

fig_hist.add_vline(x=0, line_dash="dash", line_color="red")
fig_hist.update_layout(height=400)

st.plotly_chart(fig_hist, use_container_width=True)

# 4. Subplots
st.subheader("4. Dashboard con Subplots")

fig_sub = make_subplots(
    rows=2, cols=2,
    subplot_titles=("Precio", "Retornos", "Distribucion", "Volumen")
)

# Precio
fig_sub.add_trace(
    go.Scatter(x=df["Fecha"], y=df["Precio"], mode="lines"),
    row=1, col=1
)

# Retornos
fig_sub.add_trace(
    go.Bar(x=df["Fecha"], y=df["Retorno"]),
    row=1, col=2
)

# Distribucion
fig_sub.add_trace(
    go.Histogram(x=df["Retorno"]),
    row=2, col=1
)

# Volumen simulado
volumen = np.random.randint(1000000, 5000000, len(fechas))
fig_sub.add_trace(
    go.Bar(x=df["Fecha"], y=volumen),
    row=2, col=2
)

fig_sub.update_layout(height=600, showlegend=False)

st.plotly_chart(fig_sub, use_container_width=True)