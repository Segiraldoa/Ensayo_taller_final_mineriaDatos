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

# Título de la aplicación
st.title("Exploración de datos: Heart Disease")
st.image('heartdisease.jpg', caption="Problemas del corazón")
# Descripción inicial

st.write("""
### ¡Bienvenidos!
Esta aplicación interactiva permite explorar el dataset de Heart Disease.
Puedes:
1. Ver los primeros registros.
2. Consultar información general del dataset.
3. Mirar las estadisticas descriptivas.
4. Identificar los datos faltantes.
5. Analizar la frecuencia de las columnas.
6. Observar la información de cada paciente.
7. Explorar la matriz de correlación.
8. Generar graficos dinámicos.

Y además, transformar los datos mediante la imputación de datos faltantes, la codificación de variables categóricas y la estandarización de los datos.
""")

# Sección para explorar el dataset
st.sidebar.header("Exploración de datos")

# Mostrar las primeras filas dinámicamente
if st.sidebar.checkbox("Mostrar primeras filas"):
    n_rows = st.sidebar.slider("Número de filas a mostrar:", 1, 50, 5)
    st.write(f"### Primeras {n_rows} filas del dataset")
    st.write(heartdisease.head(n_rows))


# Mostrar información del dataset
import io
if st.sidebar.checkbox("Mostrar información del dataset"):
    st.write("### Información del dataset")

    # Capturar la salida de info() en un buffer
    buffer = io.StringIO()
    heartdisease.info(buf=buffer)
    
    # Procesar la salida para estructurarla mejor
    info_text = buffer.getvalue().split("\n")  # Dividir en líneas
    info_text = [line.strip() for line in info_text if line.strip()]  # Quitar espacios vacíos
    
    # Extraer información clave
    filas_columnas = info_text[0]  # Primera línea con shape
    columnas_info = info_text[3:]  # A partir de la cuarta línea están las columnas

    # Mostrar filas y columnas
    st.write(f"**{filas_columnas}**")

    # Convertir la información de columnas en un DataFrame
    column_data = []
    for line in columnas_info:
        parts = line.split()  # Separar por espacios
        if len(parts) >= 3:
            column_name = parts[1]  # Nombre de la columna
            non_null_count = parts[2]  # Cantidad de valores no nulos
            dtype = parts[-1]  # Tipo de dato
            column_data.append([column_name, non_null_count, dtype])

    df_info = pd.DataFrame(column_data, columns=["Columna", "No Nulos", "Tipo de Dato"]).iloc[2:]
    memory_values = df_info.iloc[-1].values
    memorie_use = " ".join(str(value) for value in memory_values)
    # Mostrar la tabla en Streamlit
    st.dataframe(df_info.iloc[:-2])
    st.write(f"Uso en memoria {memorie_use}")

# Estadísticas descriptivas
if st.sidebar.checkbox("Mostrar estadísticas descriptivas"):
    st.write("### Estadísticas descriptivas")
    st.write(heartdisease.describe())
    
# Datos faltantes
if st.sidebar.checkbox("Mostrar datos faltantes"):
    st.write("### Datos faltantes por columna")
    selected_column = st.selectbox("Selecciona una columna para ver los datos faltantes:", heartdisease.columns)

    # Calcular datos faltantes
    missing_values = heartdisease[selected_column].isnull().sum()
    total_values = len(heartdisease[selected_column])
    missing_percentage = (missing_values / total_values) * 100

    # Mostrar resultado
    st.write(f"### Información de la columna: `{selected_column}`")
    st.write(f"- **Valores totales:** {total_values}")
    st.write(f"- **Valores faltantes:** {missing_values} ({missing_percentage:.2f}%)")
    
    if st.button("Mostrar todos los valores faltantes"):
        missing_total = heartdisease.isnull().sum()
        missing_total_df = pd.DataFrame({"Columna": missing_total.index, "Valores Faltantes": missing_total.values})
        
        # Filtrar solo las columnas con valores faltantes
        missing_total_df = missing_total_df[missing_total_df["Valores Faltantes"] > 0]
        st.write(missing_total_df)

#Frecuencia Columnas
if st.sidebar.checkbox("Frecuencia columnas"):
    st.write("### Frecuencia por columna")
    columna_seleccionada = st.selectbox("Selecciona una columna para ver su frecuencia:", heartdisease.columns)
    st.write(heartdisease[columna_seleccionada].value_counts())
    if st.button("Mostrar valor más frecuente"):
        st.write(heartdisease[columna_seleccionada].mode()[0])

#Informacion por paciente
if st.sidebar.checkbox("Información paciente"):
    st.write("### Informacion por paciente")
    row_index = st.number_input("Ingresa el índice de la fila a visualizar:", min_value=0, max_value=len(heartdisease)-1, step=1)

    if st.button("Mostrar fila seleccionada"):
        st.write(f"### Datos de la fila `{row_index}`")
        st.dataframe(heartdisease.iloc[[row_index]].iloc[:, 1:])

#Matriz de correlacion
if st.sidebar.checkbox("Matriz de correlacion"):
    st.write("### Matriz de correlacion")
    # Filtrar solo las columnas numéricas
    heartdisease_num = heartdisease.select_dtypes(include=['float64', 'int64'])
    variables_objetivo = ['Age','Weight','Length', 'BMI', 'BP','PR','FBS','CR','TG','LDL','HDL','BUN','ESR','HB','K','Na','WBC','Lymph','Neut','PLT','EF-TTE']
    # Calcular la matriz de correlación
    correlacion = heartdisease_num[variables_objetivo].corr()
    # Create a mask using numpy's triu function
    mask = np.triu(np.ones_like(correlacion, dtype=bool))
    # Configuración de la gráfica
    # Create a masked heatmap
    plt.figure(figsize = (10,8))
    plt.rcParams.update({'font.size': 12})
    sns.heatmap(correlacion, cmap = 'coolwarm', annot_kws={"size": 7},vmin = -1, vmax = 1, center = 0, annot=True, fmt=".2f", square=True, linewidths=.5, mask = mask)
    plt.show()

    #plt.figure(figsize=(10, 8))  # Tamaño de la figura
    #sns.heatmap(correlacion, annot=True, cmap='coolwarm', fmt='.2f', cbar=True, square=True)
    
    # Título de la gráfica
    plt.title('Matriz de Correlación de Heart Disease')
    
    # Mostrar la gráfica en Streamlit
    st.pyplot(plt)

# Sección para gráficos dinámicos
if st.sidebar.checkbox("Gráficos dinámicos"):

    # Selección de variables para el gráfico
    x_var = st.sidebar.selectbox("Selecciona la variable X:", heartdisease.columns)
    y_var = st.sidebar.selectbox("Selecciona la variable Y:", heartdisease.columns)
    
    # Tipo de gráfico
    chart_type = st.sidebar.radio(
        "Selecciona el tipo de gráfico:",
        ("Dispersión", "Histograma", "Boxplot")
    )
    
    # Mostrar el gráfico
    st.write("### Gráficos")
    if chart_type == "Dispersión":
        st.write(f"#### Gráfico de dispersión: {x_var} vs {y_var}")
        fig, ax = plt.subplots()
        sns.scatterplot(data=heartdisease, x=x_var, y=y_var, ax=ax)
        st.pyplot(fig)
    elif chart_type == "Histograma":
        st.write(f"#### Histograma de {x_var}")
        fig, ax = plt.subplots()
        sns.histplot(heartdisease[x_var], bins=30, kde=True, ax=ax)
        st.pyplot(fig)
    elif chart_type == "Boxplot":
        st.write(f"#### Boxplot de {y_var} por {x_var}")
        fig, ax = plt.subplots()
        sns.boxplot(data=heartdisease, x=x_var, y=y_var, ax=ax)
        st.pyplot(fig)

st.sidebar.header("Transformacion datos")
# Copiar el DataFrame para evitar modificar el original
if 'heartdisease_copy' not in st.session_state:
    st.session_state.heartdisease_copy = heartdisease.copy()

if st.sidebar.checkbox("Datos categoricos"):
    # Estrategias de codificación disponibles
    estrategias2 = ['Ordinal Encoder', 'OneHot Encoder']
    
    # Crear un selectbox para seleccionar la estrategia de codificación
    strategy2 = st.selectbox('Selecciona una estrategia de codificación:', estrategias2, index=0)
    
    # Función para aplicar la codificación
    def apply_encoding(data, strategy):
        categorical_cols = data.select_dtypes(exclude=['int64', 'float64']).columns
        st.write(f'{categorical_cols}')
        st.write(f'{len(categorical_cols)}')
        if len(categorical_cols) == 0:
            st.warning("No hay columnas categóricas en los datos.")
            return data
    
        data_copy = data.copy()
    
        if strategy2 == 'Ordinal Encoder':
            encoder = OrdinalEncoder()
            data_copy[categorical_cols] = encoder.fit_transform(data_copy[categorical_cols])
        elif strategy2 == 'OneHot Encoder':
            encoder = OneHotEncoder(sparse_output=False)
            encoded_data = pd.DataFrame(encoder.fit_transform(data_copy[categorical_cols]),
                                        columns=encoder.get_feature_names_out(categorical_cols),
                                        index=data_copy.index)
            data_copy = data_copy.drop(categorical_cols, axis=1)
            data_copy = pd.concat([data_copy, encoded_data], axis=1)
    
        return data_copy
    
    # Botón para aplicar la estrategia de codificación
    if st.button('Aplicar Estrategia de Codificación'):
        encoded_data = apply_encoding(heartdisease, strategy2)
        
        # Mostrar los datos codificados
        st.write(f"Vista previa de los datos codificados usando '{strategy2}':")
        st.dataframe(encoded_data.head())
        st.write(f"Información de los datos codificados:")
        st.write(encoded_data.info())
        st.session_state.heartdisease_copy = encoded_data.copy()

if st.sidebar.checkbox("Escalado de datos"):
    st.write("### Escalado de datos")
    # Estrategias disponibles
    estrategias1 = ['Standard Scaler', 'MinMax Scaler', 'Robust Scaler']

    # Crear selectbox para seleccionar estrategia
    strategy = st.selectbox('Selecciona una estrategia de escalado:', estrategias1, index=0)
    
    # Función para aplicar el escalado
    def apply_scaling(data, strategy):
        numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns
    
        if len(numeric_cols) == 0:
            st.warning("No hay columnas numéricas en los datos.")
            return data
    
        data_copy = data.copy()
    
        if strategy == 'Standard Scaler':
            scaler = StandardScaler()
            data_copy[numeric_cols] = scaler.fit_transform(data_copy[numeric_cols])
        elif strategy == 'MinMax Scaler':
            scaler = MinMaxScaler()
            data_copy[numeric_cols] = scaler.fit_transform(data_copy[numeric_cols])
        elif strategy == 'Robust Scaler':
            scaler = RobustScaler()
            data_copy[numeric_cols] = scaler.fit_transform(data_copy[numeric_cols])
    
        return data_copy
    
    # Botón para aplicar la estrategia
    if st.button('Aplicar Estrategia de Escalado'):
        
        scaled_data = apply_scaling(st.session_state.heartdisease_copy, strategy)
        
        # Mostrar los datos escalados
        st.write(f"Vista previa de los datos escalados usando '{strategy}':")
        st.dataframe(scaled_data.head())

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
        st.write("Predicción del modelo:", prediction)
            
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









