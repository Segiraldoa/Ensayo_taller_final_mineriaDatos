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


def load_encoder():
    with open("onehot_encoder_5.pkl", "rb") as f:
            encoder = pickle.load(f)
    with open("numerical_columns_2.pkl", "rb") as f:
            numerical_columns = pickle.load(f)
    return encoder, numerical_columns

def load_model_1():
    filename = 'model_trained_classifier.pkl.gz'
    with gzip.open(filename, 'rb') as f:
        model1 = pickle.load(f)
    return model1
    
def load_model_2():
    filename = 'best_model.pkl.gz'
    with gzip.open(filename, 'rb') as f:
        model2 = pickle.load(f)
    return model2

model1=load_model_1()

model2=load_model_2()

def load_model_1():
    filename = 'model_trained_classifier.pkl.gz'
    with gzip.open(filename, 'rb') as f:
        model1 = pickle.load(f)
    return model1
    
def load_model_2():
    filename = 'best_model.pkl.gz'
    with gzip.open(filename, 'rb') as f:
        model2 = pickle.load(f)
    return model2

model1=load_model_1()

model2=load_model_2()

def datos_pordefecto1(data_model):
    n=int(data_model[-1])               
    prediction1 = int(model1.predict(df.iloc[n,:].to_frame().T))
    if prediction1==1 and int(y_test[n])==1:
        st.write("Predicción del modelo:","Cath", prediction1)
        st.write("Clasificación real:","Cath", y_test[n])
        st.write("¡El modelo acertó!")                    
    elif prediction1==0 and int(y_test[n])==0:
        st.write("Predicción del modelo:","Normal", prediction1)
        st.write("Clasificación real:","Normal", y_test[n])
        st.write("¡El modelo acertó!")
    else:
        st.write("Predicción del modelo:", prediction1)
        st.write("Clasificación real", y_test[n])
        st.write("¡El modelo falló!")
        
def datos_pordefecto2(data_model):
    n=int(data_model[-1])               
    prediction2 = int(np.argmax(model2.predict(df.iloc[n,:].to_frame().T)))
    if prediction2==1 and int(y_test[n])==1:
        st.write("Predicción del modelo:","Cath", prediction2)
        st.write("Clasificación real","Cath", y_test[n])
        st.write("¡El modelo acertó!")                    
    elif prediction2==0 and int(y_test[n])==0:
        st.write("Predicción del modelo:","Normal", prediction2)
        st.write("Clasificación real","Normal", y_test[n])
        st.write("¡El modelo acertó!")
    else:
        st.write("Predicción del modelo:", prediction2)
        st.write("Clasificación real", y_test[n])
        st.write("¡El modelo falló!")


column_names = [
            "Age", "Weight", "Length", "Sex", "BMI", "DM", "HTN", "Current Smoker", 
    "EX-Smoker", "FH", "Obesity", "CRF", "CVA",
            "Airway disease", "Thyroid Disease", "CHF", "DLP", "BP", "PR", "Edema", 
    "Weak Peripheral Pulse", "Lung rales",
            "Systolic Murmur", "Diastolic Murmur", "Typical Chest Pain", "Dyspnea", 
    "Function Class", "Atypical", "Nonanginal",
            "Exertional CP", "LowTH Ang", "Q Wave", "St Elevation", "St Depression", 
    "Tinversion", "LVH", "Poor R Progression",
            "BBB", "FBS", "CR", "TG", "LDL", "HDL", "BUN", "ESR", "HB", "K", "Na", 
    "WBC", "Lymph", "Neut", "PLT", "EF-TTE", "VHD",
            "Region RWMA"
        ]
categorical_columns = {
            "Sex": ["Male", "Female"], "DM": [0,1], "HTN": [0,1], "Current Smoker": [0, 1],"EX-Smoker": [0, 1],"FH": [0, 1],"Obesity": ["Y", "N"],
            "CRF": ["Y", "N"],"CVA": ["Y", "N"],"Airway disease": ["Y", "N"],"Thyroid Disease": ["Y", "N"],"CHF": ["Y", "N"],"DLP":["Y","N"],"Edema": [0,1],
            "Weak Peripheral Pulse": ["Y","N"],"Lung rales": ["Y","N"],"Systolic Murmur": ["Y","N"],"Diastolic Murmur": ["Y","N"],"Typical Chest Pain": [0,1],
            "Dyspnea": ["Y","N"],"Function Class": [0,1,2,3],"Atypical": ["Y","N"],"Nonanginal": ["Y","N"], "Exertional CP":["N","Y"],"LowTH Ang": ["Y","N"],"Q Wave": [0,1],
            "St Elevation": [0,1],"St Depression": [0, 1],"Tinversion": [0, 1],"LVH": ["Y", "N"],"Poor R Progression": ["Y", "N"],"BBB": ["LBBB", "N","RBBB"], 
            "Region RWMA": [0,1,2,3,4],"VHD": ["mild","Moderate","N","Severe"]
        }
column_types = {
    "Age": "Edad en años.", "Length": "Estatura en cm.", "Weight": "Peso en kg.", "Sex": "Sexo de la persona.",
    "BMI": "Índice de masa corporal.", "DM": "Diabetes Mellitus.", "HTN": "Hipertensión.", "Current Smoker": "Fumador actual.",
    "EX-Smoker": "Ex-fumador.", "FH": "Historial familiar.", "Obesity": "Obesidad.", "CRF": "Insuficiencia renal crónica.",
    "DLP": "Dislipidemia.", "CHF": "Insuficiencia cardíaca congestiva.", "Thyroid Disease": "Enfermedad tiroidea.",
    "Airway disease": "Enfermedad de las vías respiratorias.", "CVA": "Accidente cerebrovascular.", "Typical Chest Pain": "Dolor torácico típico.",
    "Edema": "Edema.", "Diastolic Murmur": "Soplo diastólico.", "Systolic Murmur": "Soplo sistólico.", "Dyspnea": "Disnea.",
    "Function Class": "Clase funcional.", "PR": "Pulso en ppm.", "BP": "Presión arterial en mmHg.", "Weak Peripheral Pulse": "Pulso periférico débil.",
    "Lung rales": "Estertores pulmonares.", "Atypical": "Dolor torácico atípico.", "Nonanginal": "Dolor torácico no anginoso.",
    "Exertional CP": "Dolor torácico por esfuerzo.", "LowTH Ang": "Angina de umbral bajo.", "Q Wave": "Onda Q.", "St Elevation": "Elevación del segmento ST.",
    "St Depression": "Depresión del segmento ST.", "Tinversion": "Inversión de la onda T.", "Poor R Progression": "Mala progresión de la onda R.",
    "BBB": "Bloqueo de rama.", "BUN": "Nitrógeno ureico en sangre.", "ESR": "Velocidad de sedimentación globular.", "HB": "Hemoglobina.",
    "WBC": "Recuento de glóbulos blancos.", "Lymph": "Linfocitos.", "Neut": "Neutrófilos.", "PLT": "Plaquetas.",
    "LVH": "Hipertrofia ventricular izquierda.", "Na": "Sodio.", "K": "Potasio.", "HDL": "Lipoproteínas de alta densidad.",
    "LDL": "Lipoproteínas de baja densidad.", "TG": "Triglicéridos.", "CR": "Creatinina en mg/dl.", "FBS": "Glucosa en ayunas en mg/dl.",
    "EF-TTE": "Fracción de eyección en porcentaje.", "Region RWMA": "Anormalidades del movimiento regional de la pared.", "VHD":"Enfermedad valvular del corazón."
}

heartdisease = pd.read_csv('heartdisease.csv')

X = heartdisease.iloc[:, :-1]
y = heartdisease['Cath']
X_encoded = pd.get_dummies(X, drop_first=True,dtype= int)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y_encoded, test_size=0.2, random_state=42)
df_defecto=X_test.copy()

df=X_test.copy()

#Modelo Clasico
if st.sidebar.checkbox("Utilizar arboles de decisión"):
    st.write("### Arboles de decisión")
    st.write("""El modelo utilizado consiste en un arbol con una profundidad de 3.
    La base de datos fue codificada con One Hot Encoder y los datos no fueron escalados.""")
    st.write("### Indique si desea hacer una predicción de manera manual, usar datos por defecto o cargar una fila desde un archivo Excel")
    selected_column = st.selectbox("Selecciona un método para la predicción", ['Por defecto','Manual','Cargar desde Excel'],key="madelo1_metodo_prediccion")
    
    if selected_column=='Por defecto':
        # Buscar el archivo del modelo dentro de la carpeta extraída
        st.write("### Indique los datos por defecto que desea usar para la predicción")
        data_model1 = st.selectbox("Selecciona un método para la predicción", ['Datos 1','Datos 2','Datos 3','Datos 4','Datos 5','Datos 6','Datos 7','Datos 8','Datos 9','Datos 10'],key="modelo1_eleccion_datos")
        datos_pordefecto1(data_model1)
        
    elif selected_column=='Manual':             
         # Título de la aplicación
        st.write("### Formulario de ingreso de datos para predicción")
        
        # Crear el formulario
        input_data = {}
        num_columns = 3  # Definir el número de columnas para organizar los campos
        
        # Inicializar session_state si no existe
        if "inputs" not in st.session_state:
            st.session_state["inputs"] = {col: "0.0" for col in column_names}
        
        for i in range(0, len(column_names), num_columns):
            cols = st.columns(num_columns)
            
            for j, col in enumerate(column_names[i:i+num_columns]):
                widget_key = f"input_{col}"  # Generar clave única para cada input
                
                if col in categorical_columns:
                    # Asegurar que la clave existe en session_state
                    if widget_key not in st.session_state:
                        st.session_state[widget_key] = categorical_columns[col][0]
        
                    # Seleccionar el índice correcto
                    input_value = cols[j].selectbox(
                        f"{col}", 
                        options=categorical_columns[col], 
                        index=categorical_columns[col].index(st.session_state[widget_key]) 
                        if st.session_state[widget_key] in categorical_columns[col] else 0, 
                        help=column_types.get(col, ""),
                        key=widget_key  # Evita duplicados en selectbox
                    )
                
                else:
                    # Tomar el valor desde session_state["inputs"]
                    input_value = cols[j].text_input(
                        f"{col}", 
                        value=str(st.session_state["inputs"][col]),  
                        help=column_types.get(col, ""),
                        key=f"input_{col}_modelo1"  # Se asegura de que cada campo tenga un key único
                    )
        
                    # Convertir a float
                    try:
                        input_value = float(input_value)
                    except ValueError:
                        input_value = 0.0
        
                    # Guardar en session_state["inputs"]
                    st.session_state["inputs"][col] = str(input_value)
        
                # Guardar en input_data
                input_data[col] = input_value
        
        st.write("### Datos ingresados")
        # Convertir datos para evitar errores
        processed_data = [
            str(value) if col in categorical_columns else float(value) 
            for col, value in input_data.items()
        ]
        
        # Convertir a numpy array
        input_array = np.array(processed_data, dtype=object)

        if st.button("Realizar predicción",key="modelo1_predic"):
            st.write("Procesando los datos para la predicción...")
            # Mostrar los datos originales
            st.write(" **Datos originales:**")
            st.write(input_array)
            encoder, numerical_columns = load_encoder()
            # Simulación de datos nuevos
            new_data = input_array   
            if not isinstance(new_data, pd.DataFrame):
                new_data = pd.DataFrame([new_data], columns=column_names)
            
            # Seleccionar solo las variables categóricas
            new_data_categorical = new_data.loc[:, encoder.feature_names_in_]
            # Separar variables numéricas y categóricas
            new_data_numerical = new_data[numerical_columns]  # Mantiene solo las numéricas            
            # Codificar las variables categóricas
            encoded_array = encoder.transform(new_data_categorical)            
            # Convertir la salida a DataFrame con nombres de columnas codificadas
            encoded_df = pd.DataFrame(encoded_array, columns=encoder.get_feature_names_out())            
            # Concatenar las variables numéricas con las categóricas codificadas
            final_data = pd.concat([new_data_numerical, encoded_df], axis=1)    
            prediction=model1.predict(final_data)
            if prediction==1:
                st.write("Predicción del modelo:","Cath", prediction)
            else:
                st.write("Predicción del modelo:","Normal", prediction)
                
    elif selected_column == 'Cargar desde Excel':
        st.write("### Cargar archivo Excel para la predicción")
        uploaded_file = st.file_uploader("Cargar archivo Excel", type=["xlsx"])
        
        if uploaded_file is not None:
            # Leer el archivo Excel
            df_excel = pd.read_excel(uploaded_file)
            
            # Mostrar las primeras filas para ver los datos
            st.write("### Datos cargados del archivo Excel:")
            st.write(df_excel.head())
            
            # Pedir al usuario que seleccione la fila (podría ser por índice o número de fila)
            row_number = st.number_input("Selecciona el número de fila para la predicción", min_value=0, max_value=len(df_excel)-1, value=0)
            
            # Seleccionar la fila correspondiente
            selected_row = df_excel.iloc[row_number, :]
            
            # Mostrar la fila seleccionada
            st.write("### Fila seleccionada para la predicción:")
            st.write(selected_row)

            # Preparar los datos para la predicción: aplicar One Hot Encoder y separación de variables numéricas
            encoder, numerical_columns = load_encoder()

            # Separar variables categóricas y numéricas
            new_data_categorical = selected_row[encoder.feature_names_in_].to_frame().T  # Convertir a DataFrame
            new_data_numerical = selected_row[numerical_columns].to_frame().T  # Convertir a DataFrame

            # Codificar las variables categóricas
            encoded_array = encoder.transform(new_data_categorical)
            st.write("array: ",encoded_array)

            # Convertir la salida a DataFrame con nombres de columnas codificadas
            encoded_df = pd.DataFrame(encoded_array, columns=encoder.get_feature_names_out())
            st.write("df: ",encoded_df)

            # Concatenar las variables numéricas con las categóricas codificadas
            final_data = pd.concat([new_data_numerical, encoded_df], axis=1)
            st.write("Final: ",final_data)

            # Realizar la predicción
            prediction = model1.predict(final_data)

            if prediction == 1:
                st.write("Predicción del modelo:","Cath", prediction)
            else:
                st.write("Predicción del modelo:","Normal", prediction)

# Modelo de redes neuronales
if st.sidebar.checkbox("Utilizar redes Neuronales"): 
    st.write("### Redes Neuronales")    
    st.write("""
    El modelo utilizado consiste en una red neuronal de una capa con 32 neuronas de entrada.
    La base de datos fue codificada con One Hot Encoder y estandarizada con StandardScaler.
    """)
    st.write("### Indique si desea hacer una predicción de manera manual, usar datos por defecto o cargar una fila desde un archivo Excel")
    selected_column = st.selectbox("Selecciona un método para la predicción", ['Por defecto','Manual','Cargar desde Excel'],key="madelo2_metodo_prediccion")
    
    if selected_column=='Por defecto':             
        st.write("### Indique los datos por defecto que desea uasr para la predicción")
        data_model2 = st.selectbox("Selecciona un método para la predicción", ['Datos 1','Datos 2','Datos 3','Datos 4','Datos 5','Datos 6','Datos 7','Datos 8','Datos 9','Datos 10'],key="modelo2_eleccion_datos")
        datos_pordefecto2(data_model2) 
    
    elif selected_column=='Manual':
        # Título de la aplicación
        st.write("### Formulario de ingreso de datos para predicción")
        
        # Crear el formulario
        input_data = {}
        num_columns = 3  # Definir el número de columnas para organizar los campos
        
        # Inicializar session_state si no existe
        if "inputs" not in st.session_state:
            st.session_state["inputs"] = {col: "0.0" for col in column_names}
        
        for i in range(0, len(column_names), num_columns):
            cols = st.columns(num_columns)
            
            for j, col in enumerate(column_names[i:i+num_columns]):
                widget_key = f"input_{col}"  # Generar clave única para cada input
                
                if col in categorical_columns:
                    # Asegurar que la clave existe en session_state
                    if widget_key not in st.session_state:
                        st.session_state[widget_key] = categorical_columns[col][0]
        
                    # Seleccionar el índice correcto
                    input_value = cols[j].selectbox(
                        f"{col}", 
                        options=categorical_columns[col], 
                        index=categorical_columns[col].index(st.session_state[widget_key]) 
                        if st.session_state[widget_key] in categorical_columns[col] else 0, 
                        help=column_types.get(col, ""),
                        key=f"input_{col}_modelo02"  # Evita duplicados en selectbox
                    )
                
                else:
                    # Tomar el valor desde session_state["inputs"]
                    input_value = cols[j].text_input(
                        f"{col}", 
                        value=str(st.session_state["inputs"][col]),  
                        help=column_types.get(col, ""),
                        key=f"input_{col}_modelo2"  # Se asegura de que cada campo tenga un key único
                    )
        
                    # Convertir a float
                    try:
                        input_value = float(input_value)
                    except ValueError:
                        input_value = 0.0
        
                    # Guardar en session_state["inputs"]
                    st.session_state["inputs"][col] = str(input_value)
        
                # Guardar en input_data
                input_data[col] = input_value
        
        st.write("### Datos ingresados")
        # Convertir datos para evitar errores
        processed_data = [
            str(value) if col in categorical_columns else float(value) 
            for col, value in input_data.items()
        ]
        
        # Convertir a numpy array
        input_array = np.array(processed_data, dtype=object)

        if st.button("Realizar predicción",key="modelo2_predic"):
            st.write("Procesando los datos para la predicción...")
            # Mostrar los datos originales
            st.write(" **Datos originales:**")
            st.write(input_array)
            encoder, numerical_columns = load_encoder()
            # Simulación de datos nuevos
            new_data = input_array   
            if not isinstance(new_data, pd.DataFrame):
                new_data = pd.DataFrame([new_data], columns=column_names)
            # Seleccionar solo las variables categóricas
            new_data_categorical = new_data.loc[:, encoder.feature_names_in_]
            # Separar variables numéricas y categóricas
            new_data_numerical = new_data[numerical_columns]  # Mantiene solo las numéricas            
            # Codificar las variables categóricas
            encoded_array = encoder.transform(new_data_categorical)            
            # Convertir la salida a DataFrame con nombres de columnas codificadas
            encoded_df = pd.DataFrame(encoded_array, columns=encoder.get_feature_names_out())            
            # Concatenar las variables numéricas con las categóricas codificadas
            final_data = pd.concat([new_data_numerical, encoded_df], axis=1)  
            prediction=np.argmax(model2.predict(final_data))
            if prediction==1:
                st.write("Predicción del modelo:","Cath", prediction)
            else:
                st.write("Predicción del modelo:","Normal", prediction)

    elif selected_column == 'Cargar desde Excel':
        st.write("### Cargar archivo Excel para la predicción")
        uploaded_file = st.file_uploader("Cargar archivo Excel", type=["xlsx"])
        
        if uploaded_file is not None:
            # Leer el archivo Excel
            df_excel = pd.read_excel(uploaded_file)
            
            # Mostrar las primeras filas para ver los datos
            st.write("### Datos cargados del archivo Excel:")
            st.write(df_excel.head())
            
            # Pedir al usuario que seleccione la fila (podría ser por índice o número de fila)
            row_number = st.number_input("Selecciona el número de fila para la predicción", min_value=0, max_value=len(df_excel)-1, value=0)
            
            # Seleccionar la fila correspondiente
            selected_row = df_excel.iloc[row_number, :]
            
            # # Mostrar la fila seleccionada
            # st.write("### Fila seleccionada para la predicción:")
            # st.write(selected_row)

            # # Preparar los datos para la predicción: aplicar One Hot Encoder y separación de variables numéricas
            # encoder, numerical_columns = load_encoder()

            # # Separar variables categóricas y numéricas
            # new_data_categorical = selected_row[encoder.feature_names_in_].to_frame().T  # Convertir a DataFrame
            # new_data_numerical = selected_row[numerical_columns].to_frame().T  # Convertir a DataFrame

            # # Codificar las variables categóricas
            # encoded_array = encoder.transform(new_data_categorical)
            # st.write("array: ",encoded_array)

            # # Convertir la salida a DataFrame con nombres de columnas codificadas
            # encoded_df = pd.DataFrame(encoded_array, columns=encoder.get_feature_names_out())
            # st.write("df: ",encoded_df)

            # # Concatenar las variables numéricas con las categóricas codificadas
            # final_data = pd.concat([new_data_numerical, encoded_df], axis=1)
            # st.write("Final: ",type(final_data))
            

            # # Realizar la predicción
            # prediction = np.argmax(model2.predict(final_data))

            # if prediction == 1:
            #     st.write("Predicción del modelo:","Cath", prediction)
            # else:
            #     st.write("Predicción del modelo:","Normal", prediction)

            if st.button("Realizar predicción",key="modelo2_predic_excel"):
                st.write("Procesando los datos para la predicción...")
                # Mostrar los datos originales
                st.write(" **Datos originales:**")
                # st.write(input_array)
                encoder, numerical_columns = load_encoder()
                # Simulación de datos nuevos
                new_data = selected_row
                if not isinstance(new_data, pd.DataFrame):
                    new_data = pd.DataFrame([new_data], columns=column_names)
                # Seleccionar solo las variables categóricas
                new_data_categorical = new_data.loc[:, encoder.feature_names_in_]
                # Separar variables numéricas y categóricas
                new_data_numerical = new_data[numerical_columns]  # Mantiene solo las numéricas            
                # Codificar las variables categóricas
                encoded_array = encoder.transform(new_data_categorical)            
                # Convertir la salida a DataFrame con nombres de columnas codificadas
                encoded_df = pd.DataFrame(encoded_array, columns=encoder.get_feature_names_out())            
                # Concatenar las variables numéricas con las categóricas codificadas
                final_data = pd.concat([new_data_numerical, encoded_df], axis=1)  
                prediction=np.argmax(model2.predict(final_data))
                st.write(prediction)
                if prediction==1:
                    st.write("Predicción del modelo:","Cath", prediction)
                else:
                    st.write("Predicción del modelo:","Normal", prediction)
    


# additional_params = {
#     'Depth': 1,
#     'Epochs': 11,
#     'Batch Size': 58,
#     'Accuracy': 0.704918,
#     'Loss': 0.6126
# }

# # Colocar el checkbox en la barra lateral
# if st.sidebar.checkbox("Mostrar hiperparámetros del modelo"):
#     st.write("#### Hiperparámetros del modelo")
    
#     # Mostrar los hiperparámetros del modelo 1 (modelo de clasificación - árbol de decisión)
#     if hasattr(model1, 'get_params'):
#         st.write("##### Hiperparámetros del modelo de clasificación (sklearn)")
        
#         model1_params = model1.get_params()  # Extraer los hiperparámetros del modelo
        
#         # Convertir los hiperparámetros a un formato adecuado para una tabla
#         model1_params_table = [(key, value) for key, value in model1_params.items()] 
        
#         # Limpiar los valores None o <NA> y reemplazarlos con un guion o valor vacío
#         cleaned_model1_params = [
#             (key, value if value is not None and value != "<NA>" else "-") 
#             for key, value in model1_params_table
#         ]
        
#         # Mostrar los parámetros del modelo 1 como una tabla
#         model1_params_df = pd.DataFrame(cleaned_model1_params, columns=["Hiperparámetro", "Valor"])
        
#         # Establecer el ancho de las columnas para que se ajusten adecuadamente
#         model1_params_df.style.set_properties(subset=["Hiperparámetro", "Valor"], width="300px")
        
#         # Mostrar la tabla con estilo
#         st.dataframe(model1_params_df, use_container_width=True)
        
#         # Agregar tabla con el Accuracy del modelo de árbol de decisión (con 6 decimales)
#         st.write("##### Accuracy del modelo de clasificación (Árbol de Decisión)")
#         accuracy_params = {
#             "Accuracy": f"{0.836065:.6f}"  # Formatear el Accuracy con 6 decimales
#         }
#         accuracy_df = pd.DataFrame(list(accuracy_params.items()), columns=["Métrica", "Valor"])
#         st.dataframe(accuracy_df, use_container_width=True)
    
#     # Mostrar los hiperparámetros del modelo 2 (modelo de red neuronal)
#     if hasattr(model2, 'get_config'):
#         st.write("##### Hiperparámetros del modelo de red neuronal (TensorFlow/Keras)")
        
#         # Obtener los hiperparámetros de la red neuronal
#         model2_params = []
#         for layer in model2.layers:
#             layer_info = {
#                 "Capa": layer._class.name_,  # Nombre de la capa (ej. Dense, Conv2D)
#                 "Hiperparámetros": layer.get_config()  # Obtiene la configuración de la capa
#             }
#             model2_params.append(layer_info)
        
#         # Crear un diccionario para almacenar los hiperparámetros de cada capa
#         layers_info = {}
        
#         for i, layer in enumerate(model2_params):
#             layer_name = f"Capa {i+1} ({layer['Capa']})"
#             layer_config = layer["Hiperparámetros"]
            
#             for param, value in layer_config.items():
#                 if param not in layers_info:
#                     layers_info[param] = []
#                 layers_info[param].append(value)

#         # Convertir el diccionario en un DataFrame con los hiperparámetros como filas
#         model2_params_df = pd.DataFrame(layers_info)
        
#         # Transponer la tabla para que las capas estén como columnas y los hiperparámetros como filas
#         model2_params_df = model2_params_df.transpose()
        
#         # Renombrar las columnas para reflejar el número de capa
#         model2_params_df.columns = [f"Capa {i+1}" for i in range(len(model2_params))]
        
#         # Establecer el ancho de las columnas para que se ajusten adecuadamente
#         model2_params_df.style.set_properties(subset=model2_params_df.columns, width="300px")
        
#         # Mostrar la tabla con estilo
#         st.dataframe(model2_params_df, use_container_width=True)
        
#         # Obtener el learning rate
#         if hasattr(model2, 'optimizer'):
#             optimizer = model2.optimizer
#             if hasattr(optimizer, 'lr'):  # Para versiones más antiguas de Keras
#                 learning_rate = optimizer.lr.numpy()
#             elif hasattr(optimizer, 'learning_rate'):  # Para versiones más recientes de TensorFlow
#                 learning_rate = optimizer.learning_rate.numpy()
                        
#         # Agregar el learning rate a los parámetros generales
#         additional_params['Learning Rate'] = learning_rate
        
#         # Crear un DataFrame para los parámetros generales
#         additional_params_df = pd.DataFrame(list(additional_params.items()), columns=["Hiperparámetro", "Valor"])
        
#         # Ajustar los decimales de los valores para que se muestren con hasta 6 decimales
#         def format_value(value):
#             if isinstance(value, (float, int)):
#                 # Si el valor tiene decimales, mostrarlo con 6 decimales, de lo contrario, mostrarlo como entero
#                 if value.is_integer():
#                     return f"{int(value)}"  # Mostrar como entero si no tiene decimales
#                 return f"{value:.6f}"  # Mostrar con 6 decimales
#             return value  # Para valores no numéricos, devolver tal cual
        
#         additional_params_df["Valor"] = additional_params_df["Valor"].apply(format_value)
        
#         # Mostrar la tabla de los parámetros generales
#         st.write("##### Parámetros Generales del Modelo")
#         st.dataframe(additional_params_df, use_container_width=True)
    
#     else:
#         st.write("El modelo no tiene el método get_config() disponible.")
