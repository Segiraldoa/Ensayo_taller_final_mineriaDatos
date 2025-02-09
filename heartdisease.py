import streamlit as st
import os
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


def load_classic_model():
    filename = "model_trained_classifier.pkl.gz"
    with gzip.open(filename, "rb") as f:
        model = pickle.load(f)
    return model

heartdisease = pd.read_csv('heartdisease.csv')
#Modelo Clasico
if st.sidebar.checkbox("Utilizar arboles de decisión"): 
    st.write("### Arboles de decisión")
    st.write("""
    El modelo utilizado consiste en un arbol con una profundidad de 3.
    La base de datos fue codificada con One Hot Encoder y los datos no fueron escalados.
    """)
    model_classic=load_classic_model()

    st.write("### Indique si desea hacer una predicción de manera manual o usar datos por defecto")
    selected_column = st.selectbox("Selecciona un método para la predicción", ['Por defecto','Manual'])
    zip_path = "modelo_entrenado_comprimido.zip"
    extract_path = "modelo_descomprimido"
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
        # st.success("Descompresión completada.")
    except zipfile.BadZipFile:
        st.error("Error: El archivo ZIP está corrupto o no es un archivo ZIP válido.")
    except zipfile.LargeZipFile:
        st.error("Error: El archivo ZIP es demasiado grande y requiere compatibilidad con ZIP64.")
    except Exception as e:
        st.error(f"Error durante la descompresión: {str(e)}")

    model_path = None
    for root, _, files in os.walk(extract_path):
        for file in files:
            if file.endswith(".h5"):
                model_path = os.path.join(root, file)
                break
                
    if model_path:
        # Cargar el modelo
        model = tf.keras.models.load_model(model_path)
        #st.success("Modelo cargado correctamente.")
        X = heartdisease.iloc[:, :-1]
        y = heartdisease['Cath']
        X_encoded = pd.get_dummies(X, drop_first=True,dtype= int)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_encoded)
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)
   
    if selected_column=='Por defecto':
        # Buscar el archivo del modelo dentro de la carpeta extraída
        st.write("### Indique los datos por defecto que desea uasr para la predicción")
        data_model = st.selectbox("Selecciona un método para la predicción", ['Datos 1','Datos 2','Datos 3','Datos 4','Datos 5'])

        if data_model=='Datos 1':
            input_data = X_train[0].reshape(1, -1)  # Excluir la última columna si es la etiqueta
            st.write("Datos de entrada:", input_data)

        if data_model=='Datos 2':
            input_data = X_train[1].reshape(1, -1)  # Excluir la última columna si es la etiqueta
            st.write("Datos de entrada:", input_data)

        if data_model=='Datos 3':
            input_data = X_train[2].reshape(1, -1)  # Excluir la última columna si es la etiqueta
            st.write("Datos de entrada:", input_data)

        if data_model=='Datos 4':
            input_data = X_train[3].reshape(1, -1)  # Excluir la última columna si es la etiqueta
            st.write("Datos de entrada:", input_data)

        if data_model=='Datos 5':
            input_data = X_train[4].reshape(1, -1)  # Excluir la última columna si es la etiqueta
            st.write("Datos de entrada:", input_data)

        # Realizar predicción
        prediction = model_classic.predict(input_data) # np.argmax(model_classic.predict(input_data))
        # prediction = model.predict(argmax(input_data))
        st.write("datos entrada:", input_data)
        
        st.write("Predicción del modelo:", prediction)
            
    if selected_column=='Manual':
        # Definir nombres de columnas
        column_names = [
            "Age", "Weight", "Length", "Sex", "BMI", "DM", "HTN", "Current Smoker", "EX-Smoker", "FH", "Obesity", "CRF", "CVA",
            "Airway disease", "Thyroid Disease", "CHF", "DLP", "BP", "PR", "Edema", "Weak Peripheral Pulse", "Lung rales",
            "Systolic Murmur", "Diastolic Murmur", "Typical Chest Pain", "Dyspnea", "Function Class", "Atypical", "Nonanginal",
            "Exertional CP", "LowTH Ang", "Q Wave", "St Elevation", "St Depression", "Tinversion", "LVH", "Poor R Progression",
            "BBB", "FBS", "CR", "TG", "LDL", "HDL", "BUN", "ESR", "HB", "K", "Na", "WBC", "Lymph", "Neut", "PLT", "EF-TTE",
            "Region RWMA", "Cath"
        ]
        
        # Variables categóricas y sus opciones
        categorical_columns = {
            "Sex": ["Male", "Female"],
            "DM": [0,1],
            "HTN":[0,1],
            "Current Smoker": [0, 1],
            "EX-Smoker": [0, 1],
            "FH": [0, 1],
            "Obesity": ["Y", "N"],
            "CRF": ["Y", "N"],
            "CVA": ["Y", "N"],
            "Airway disease": ["Y", "N"],
            "Thyroid Disease": ["Y", "N"],
            "CHF": ["Y", "N"],
            "Edema": [0,1],
            
            "Region RWMA": ["Normal", "Abnormal"],  
            "Cath": ["Normal", "Disease"]  
        }
        
        # Crear DataFrame inicial con valores numéricos en 0 y categóricos con el primer valor de la lista
        data = {col: [0.0] for col in column_names}  # Inicializar numéricos en 0
        for col in categorical_columns:
            data[col] = [categorical_columns[col][0]]  # Inicializar con el primer valor de la lista
        
        df = pd.DataFrame(data)
        
        # Convertir columnas categóricas a tipo "category" para que se muestren como dropdown en st.data_editor
        for col in categorical_columns:
            df[col] = df[col].astype("category")
        
        # Mostrar la tabla editable en Streamlit
        st.write("### Introduce los datos para la predicción:")
        edited_df = st.data_editor(df, key="editable_table")
        
        # Mostrar la tabla actualizada
        st.write("#### Datos ingresados:")
        st.write(edited_df)
        
        # Botón para generar la predicción
        if st.button("Realizar predicción"):
            st.write("Procesando los datos para la predicción...")
        
            # Convertir variables categóricas a valores numéricos
            for col in categorical_columns:
                edited_df[col] = edited_df[col].apply(lambda x: 1 if x in ["Yes", "Male", "Abnormal", "Disease"] else 0)
        
            # Convertir DataFrame a numpy para pasarlo al modelo
            input_data = edited_df.to_numpy()
        
            # Aquí iría la llamada al modelo de predicción (simulación)
            prediction = np.random.rand()  # Simulación de predicción
        
            st.write("### Predicción realizada:")
            st.write(prediction)

    
    
# Modelo de redes neuronales
if st.sidebar.checkbox("Utilizar redes Neuronales"): 
    st.write("### Redes Neuronales")
    st.write("ADADASD")
    
    st.write("""
    El modelo utilizado consiste en una red neuronal de una capa con 32 neuronas de entrada.
    La base de datos fue codificada con One Hot Encoder y estandarizada con StandardScaler.
    """)

    # Extracción del 
    zip_path = "modelo_entrenado_comprimido.zip"
    extract_path = "modelo_descomprimido"
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
        # st.success("Descompresión completada.")
    except zipfile.BadZipFile:
        st.error("Error: El archivo ZIP está corrupto o no es un archivo ZIP válido.")
    except zipfile.LargeZipFile:
        st.error("Error: El archivo ZIP es demasiado grande y requiere compatibilidad con ZIP64.")
    except Exception as e:
        st.error(f"Error durante la descompresión: {str(e)}")

    st.write("### Indique si desea hacer una predicción de manera manual o usar datos por defecto")
    selected_column = st.selectbox("Selecciona un método para la predicción", ['Por defecto','Manual'])
    if selected_column=='Por defecto':
        # Buscar el archivo del modelo dentro de la carpeta extraída
        model_path = None
        for root, _, files in os.walk(extract_path):
            for file in files:
                if file.endswith(".h5"):
                    model_path = os.path.join(root, file)
                    break
                    
        if model_path:
            # Cargar el modelo
            model = tf.keras.models.load_model(model_path)
            #st.success("Modelo cargado correctamente.")
            X = heartdisease.iloc[:, :-1]
            y = heartdisease['Cath']
            X_encoded = pd.get_dummies(X, drop_first=True,dtype= int)
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_encoded)
            label_encoder = LabelEncoder()
            y_encoded = label_encoder.fit_transform(y)
            X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)

            st.write("### Indique los datos por defecto que desea uasr para la predicción")
            data_model = st.selectbox("Selecciona un método para la predicción", ['Datos 1','Datos 2','Datos 3','Datos 4','Datos 5'])

            if data_model=='Datos 1':
                input_data = X_train[0].reshape(1, -1)  # Excluir la última columna si es la etiqueta
                st.write("Datos de entrada:", input_data)

            if data_model=='Datos 2':
                input_data = X_train[1].reshape(1, -1)  # Excluir la última columna si es la etiqueta
                st.write("Datos de entrada:", input_data)

            if data_model=='Datos 3':
                input_data = X_train[2].reshape(1, -1)  # Excluir la última columna si es la etiqueta
                st.write("Datos de entrada:", input_data)

            if data_model=='Datos 4':
                input_data = X_train[3].reshape(1, -1)  # Excluir la última columna si es la etiqueta
                st.write("Datos de entrada:", input_data)

            if data_model=='Datos 5':
                input_data = X_train[4].reshape(1, -1)  # Excluir la última columna si es la etiqueta
                st.write("Datos de entrada:", input_data)

            
            # Realizar predicción
            prediction = np.argmax(model.predict(input_data))
            # prediction = model.predict(argmax(input_data))
            st.write("Predicción del modelo:", prediction)
        else:
            st.error("No se encontró un archivo .h5 en el ZIP. Verifica el contenido.")
            
    if selected_column=='Manual':
        # Buscar el archivo del modelo dentro de la carpeta extraída
        model_path = None
        for root, _, files in os.walk(extract_path):
            for file in files:
                if file.endswith(".h5"):
                    model_path = os.path.join(root, file)
                    break
                    
        if model_path:
            # Cargar el modelo
            model = tf.keras.models.load_model(model_path)
            #st.success("Modelo cargado correctamente.")
            X = heartdisease.iloc[:, :-1]
            y = heartdisease['Cath']
            X_encoded = pd.get_dummies(X, drop_first=True,dtype= int)
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_encoded)
            label_encoder = LabelEncoder()
            y_encoded = label_encoder.fit_transform(y)
            X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)

        # Definir nombres de columnas
        column_names = [
            "Age", "Weight", "Length", "Sex", "BMI", "DM", "HTN", "Current Smoker", "EX-Smoker", "FH", "Obesity", "CRF", "CVA",
            "Airway disease", "Thyroid Disease", "CHF", "DLP", "BP", "PR", "Edema", "Weak Peripheral Pulse", "Lung rales",
            "Systolic Murmur", "Diastolic Murmur", "Typical Chest Pain", "Dyspnea", "Function Class", "Atypical", "Nonanginal",
            "Exertional CP", "LowTH Ang", "Q Wave", "St Elevation", "St Depression", "Tinversion", "LVH", "Poor R Progression",
            "BBB", "FBS", "CR", "TG", "LDL", "HDL", "BUN", "ESR", "HB", "K", "Na", "WBC", "Lymph", "Neut", "PLT", "EF-TTE",
            "Region RWMA", "Cath"
        ]
        
        # Variables categóricas y sus opciones
        categorical_columns = {
            "Sex": ["Male", "Female"],
            "DM": [0,1],
            "HTN":[0,1],
            "Current Smoker": [0, 1],
            "EX-Smoker": [0, 1],
            "FH": [0, 1],
            "Obesity": ["Y", "N"],
            "CRF": ["Y", "N"],
            "CVA": ["Y", "N"],
            "Airway disease": ["Y", "N"],
            "Thyroid Disease": ["Y", "N"],
            "CHF": ["Y", "N"],
            "Edema": [0,1],
            
            "Region RWMA": ["Normal", "Abnormal"],  
            "Cath": ["Normal", "Disease"]  
        }
        
        # Crear DataFrame inicial con valores numéricos en 0 y categóricos con el primer valor de la lista
        data = {col: [0.0] for col in column_names}  # Inicializar numéricos en 0
        for col in categorical_columns:
            data[col] = [categorical_columns[col][0]]  # Inicializar con el primer valor de la lista
        
        df = pd.DataFrame(data)
        
        # Convertir columnas categóricas a tipo "category" para que se muestren como dropdown en st.data_editor
        for col in categorical_columns:
            df[col] = df[col].astype("category")
        
        # Mostrar la tabla editable en Streamlit
        st.write("### Introduce los datos para la predicción:")
        edited_df = st.data_editor(df, key="editable_table")
        
        # Mostrar la tabla actualizada
        st.write("#### Datos ingresados:")
        st.write(edited_df)
        
        # Botón para generar la predicción
        if st.button("Realizar predicción"):
            st.write("Procesando los datos para la predicción...")
        
            # Convertir variables categóricas a valores numéricos
            for col in categorical_columns:
                edited_df[col] = edited_df[col].apply(lambda x: 1 if x in ["Yes", "Male", "Abnormal", "Disease"] else 0)
        
            # Convertir DataFrame a numpy para pasarlo al modelo
            input_data = edited_df.to_numpy()
        
            # Aquí iría la llamada al modelo de predicción (simulación)
            prediction = np.random.rand()  # Simulación de predicción
        
            st.write("### Predicción realizada:")
            st.write(prediction)









