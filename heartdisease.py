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
col1, col2, col3 = st.columns(3)

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

# Gráfico 2 en la segunda columna
with col3:
    st.subheader("Gráfico 3")
    fig, ax = plt.subplots()
    ax.scatter(data[:, 0], data[:, 1], color="green")
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



# Simulación de datos
data = np.random.randn(100, 2)

# Título del Dashboard
st.title("Dashboard con 8 Gráficos (2 Filas x 4 Columnas)")

# Primera fila (4 columnas)
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.subheader("Gráfico 1 - Histograma")
    fig, ax = plt.subplots()
    ax.hist(data[:, 0], bins=20, color="blue", alpha=0.7)
    st.pyplot(fig)

with col2:
    st.subheader("Gráfico 2 - Dispersión")
    fig, ax = plt.subplots()
    ax.scatter(data[:, 0], data[:, 1], color="red")
    st.pyplot(fig)

with col3:
    st.subheader("Gráfico 3 - tiempo")
    fig, ax = plt.subplots()
    ax.scatter(data[:, 0], data[:, 1], color="red")
    st.pyplot(fig)

with col4:
    st.subheader("Gráfico 4 - Barras")
    st.bar_chart(np.random.rand(10))

# Segunda fila (4 columnas)
col5, col6, col7, col8 = st.columns(4)

with col5:
    st.subheader("Gráfico 5 - Área")
    fig, ax = plt.subplots()
    ax.scatter(data[:, 0], data[:, 1], color="red")
    st.pyplot(fig)

with col6:
    st.subheader("Gráfico 6 - Histograma")
    fig, ax = plt.subplots()
    ax.hist(data[:, 1], bins=15, color="green", alpha=0.6)
    st.pyplot(fig)

with col7:
    st.subheader("Gráfico 7 - Dispersión")
    fig, ax = plt.subplots()
    ax.scatter(data[:, 1], data[:, 0], color="purple")
    st.pyplot(fig)

with col8:
    st.subheader("Gráfico 8 - Barras")
    fig, ax = plt.subplots()
    ax.scatter(data[:, 0], data[:, 1], color="red")
    st.pyplot(fig)









