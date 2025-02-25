import streamlit as st
import os
import joblib
import zipfile
import pandas as pd
import seaborn as sns
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import gzip
import pickle
import io
from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder



st.markdown("<h1 style='color: blue;'>Hola, Streamlit con HTML!</h1>", unsafe_allow_html=True)

st.markdown(
    """
    <style>
        .custom-text {
            color: red;
            font-size: 20px;
            font-weight: bold;
        }
    </style>
    <p class='custom-text'>Texto con estilo personalizado</p>
    """,
    unsafe_allow_html=True
)


# Simulación de datos
data = np.random.randn(100, 2)

# Creando el layout del dashboard con columnas
col1, col2 = st.columns(2)

# Gráfico 1 en la primera columna
with col1:
    st.subheader("Gráfico 1")
    fig, ax = plt.subplots()
    ax.hist(data[:, 0], bins=20, color="blue", alpha=0.7)
    st.pyplot(fig)

# Gráfico 2 en la segunda columna
with col2:
    st.subheader("Gráfico 2")
    fig, ax = plt.subplots()
    ax.scatter(data[:, 0], data[:, 1], color="red")
    st.pyplot(fig)




with st.container():
    st.subheader("Fila 1 - Gráfico de dispersión")
    fig, ax = plt.subplots()
    ax.scatter(data[:, 0], data[:, 1], color="green")
    st.pyplot(fig)

with st.container():
    st.subheader("Fila 2 - Histograma")
    fig, ax = plt.subplots()
    ax.hist(data[:, 1], bins=15, color="purple", alpha=0.6)
    st.pyplot(fig)


with st.grid(expand=True) as grid:
    cell1 = grid.cell()
    cell2 = grid.cell()
    cell3 = grid.cell()
    cell4 = grid.cell()

with cell1:
    st.subheader("Gráfico 1")
    st.line_chart(np.random.randn(20, 2))

with cell2:
    st.subheader("Gráfico 2")
    st.bar_chart(np.random.rand(10))

with cell3:
    st.subheader("Gráfico 3")
    st.area_chart(np.random.randn(30, 2))

with cell4:
    st.subheader("Gráfico 4")
    st.line_chart(np.random.randn(15, 3))










