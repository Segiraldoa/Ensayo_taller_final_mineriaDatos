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

def datos_pordefecto1(data_model):
    n=int(data_model[-1])               
    prediction1 = int(model1.predict(df.iloc[n,:].to_frame().T))
    st.write("Valor del modelo: ",model1.predict(df.iloc[n,:].to_frame().T))
    st.write("Valor de la predicción",prediction1)
    st.write(int(y_test[n]))
    if prediction1==1 and int(y_test[n])==1:
        st.write("Predicción del modelo:","Cath", prediction1)
        st.write("Clasificación real:","Cath", y_test[n])
        st.write("¡El modelo acertó!1")                    
    elif prediction1==0 and int(y_test[n])==0:
        st.write("Predicción del modelo:","Normal", prediction1)
        st.write("Clasificación real:","Normal", y_test[n])
        st.write("¡El modelo acertó!2")
    else:
        st.write("Predicción del modelo:", prediction1)
        st.write("Clasificación real", y_test[n])
        st.write("¡El modelo falló!3")
        
def datos_pordefecto2(data_model):
    n=int(data_model[-1])               
    prediction2 = int(np.argmax(model2.predict(df.iloc[n,:].to_frame().T)))
    st.write(n)
    st.write(model2.predict(df.iloc[n,:].to_frame().T))
    if prediction2==1 and y_test[n]==1:
        st.write("Predicción del modelo:","Cath", prediction2)
        st.write("Clasificación real","Cath", y_test[n])
        st.write("¡El modelo acertó!")                    
    elif prediction2==0 and y_test[n]==0:
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
    "WBC", "Lymph", "Neut", "PLT", "EF-TTE",
            "Region RWMA"
        ]
categorical_columns = {
            "Sex": ["Male", "Female"],"DM": [0,1],"HTN":[0,1],"Current Smoker": [0, 1],"EX-Smoker": [0, 1],"FH": [0, 1],"Obesity": ["Y", "N"],
            "CRF": ["Y", "N"],"CVA": ["Y", "N"],"Airway disease": ["Y", "N"],"Thyroid Disease": ["Y", "N"],"CHF": ["Y", "N"],"Edema": [0,1],
            "Weak Peripheral Pulse": ["Y","N"],"Lung rales": ["Y","N"],"Systolic Murmur": ["Y","N"],"Diastolic Murmur": ["Y","N"],"Typical Chest Pain": [0,1],
            "Dyspnea": ["Y","N"],"Function Class": [0,1,2,3],"Atypical": ["Y","N"],"Nonanginal": ["Y","N"],"LowTH Ang": ["Y","N"],"Q Wave": [0,1],
            "St Elevation": [0,1],"St Depression": [0, 1],"Tinversion": [0, 1],"LVH": ["Y", "N"],"Poor R Progression": ["Y", "N"],"BBB": ["LBBB", "N","RBBB"], 
            "Region RWMA": [0,1,2,3,4],"VHD": ["mild","Moderate","N","Severe"]
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
# df_first_row = df.iloc[0,:].to_frame().T # Estos son los valores por defecto y no deben pasar por encoder

#Modelo Clasico
if st.sidebar.checkbox("Utilizar arboles de decisión"):
    st.write("### Arboles de decisión")
    st.write("""El modelo utilizado consiste en un arbol con una profundidad de 3.
    La base de datos fue codificada con One Hot Encoder y los datos no fueron escalados.""")
    st.write("### Indique si desea hacer una predicción de manera manual o usar datos por defecto")
    selected_column = st.selectbox("Selecciona un método para la predicción", ['Por defecto','Manual'],key="madelo1_metodo_prediccion")
    
    if selected_column=='Por defecto':
        # Buscar el archivo del modelo dentro de la carpeta extraída
        st.write("### Indique los datos por defecto que desea uasr para la predicción")
        data_model1 = st.selectbox("Selecciona un método para la predicción", ['Datos 1','Datos 2','Datos 3','Datos 4','Datos 5','Datos 6','Datos 7','Datos 8','Datos 9','Datos 10'],key="modelo1_eleccion_datos")
        datos_pordefecto1(data_model1)
        
    if selected_column=='Manual':

        ###############################################################################################   ################################################################################################
        # Nombres de las columnas (según el dataset boston_housing)
        # columns = column_names
        
        # column_types = {
        #     "Age": "Edad en años.",
        #     "Length": "Estatura en cm.",
        #     "Weight": "Peso en kg.",
        #     "Sex": "Sexo de la persona.",
        #     "BMI": "Índice de masa corporal.",
        #     "DM": "Diabetes Mellitus.",
        #     "HTN": "Hipertensión.",
        #     "Current Smoker": "Fumador actual.",
        #     "EX-Smoker": "Ex-fumador.",
        #     "FH": "Historial familiar.",
        #     "Obesity": "Obesidad.",
        #     "CRF": "insuficiencia renal crónica.",
        #     "DLP": "Dislipidemia.",
        #     "CHF": "Insuficiencia cardíaca congestiva.",
        #     "Thyroid Disease": "Enfermedad tiroidea.",
        #     "Airway disease": "Enfermedad de las vías respiratorias.",
        #     "CVA": "Accidente cerebrovascular.",
        #     "Typical Chest Pain": "Dolor torácico típico.",
        #     "Edema": "Edema.",
        #     "Diastolic Murmur": "Soplo diastólico.",
        #     "Systolic Murmur": "Soplo sistólico.",
        #     "Dyspnea": "Disnea.",
        #     "Function Class": "Clase funcional.",
        #     "PR": "Pulso en ppm.",
        #     "BP": "Presión arterial en mmHg.",
        #     "Weak Peripheral Pulse": "Pulso periférico débil.",
        #     "Lung rales": "Estertores pulmonares.",
        #     "Atypical": "Dolor torácico atípico.",
        #     "Nonanginal": "Dolor torácico no anginoso.",
        #     "Exertional CP": "Dolor torácico por esfuerzo.",
        #     "LowTH Ang": "Angina de umbral bajo.",
        #     "Q Wave": "Onda Q.",
        #     "St Elevation": "Elevación del segmento ST.",
        #     "St Depression": "Depresión del segmento ST.",
        #     "Tinversion": "Inversión de la onda T.",
        #     "Poor R Progression": "Mala progresión de la onda R",
        #     "BBB": "Bloqueo de rama.",
        #     "BUN": "Nitrógeno ureico en sangre.",
        #     "ESR": "Velocidad de sedimentación globular.",
        #     "HB": "Hemoglobina.",
        #     "WBC": "Recuento de glóbulos blancos.",
        #     "Lymph": "Linfocitos.",
        #     "Neut": "Neutrófilos.",
        #     "PLT": "Plaquetas.",
        #     "LVH": "hipertrofia ventricular izquierda.",
        #     "Na": "Sodio.",
        #     "K": "Potasio.",
        #     "HDL": "lipoproteínas de alta densidad.",
        #     "LDL": "lipoproteínas de baja densidad.",
        #     "TG": "triglicéridos.",
        #     "Cr": "Creatinina en mg/dl.",
        #     "FBS": "Glucosa en ayuanas en mg/dl.",
        #     "EF-TTE": "Fracción de eyección en porcentaje.",
        #     "RWMA": "Anormalidades del movimiento regional de la pared."
        # }
        
        # # Categorías disponibles para las variables categóricas
        # chas_options = [0, 1]  # CHAS solo puede ser 0 o 1
        # rad_options = list(range(1, 25))  # Suponiendo que RAD es un índice con valores entre 1 y 24
    
        
        # # Explicación breve
        # st.write("#### Datos de la vivienda")
        # st.write("Introduce los datos de la vivienda para estimar su precio promedio.")
        
        # # Inicializar las variables en st.session_state si no están presentes
        # for col in columns:
        #     if f"input_{col}" not in st.session_state:
        #         st.session_state[f"input_{col}"] = 0.0  # Inicializar cada variable individualmente con valor 0.0
    
        # # Organizar la entrada en forma de tabla con 6 columnas
        # input_data = {}
    
        # # Número de columnas que queremos
        # num_columns = 3
    
        # # Crear número de filas necesario según el número de variables
        # for i in range(0, len(columns), num_columns):
        #     # Crear 3 columnas para cada fila
        #     cols = st.columns(num_columns)
            
        #     for j, col in enumerate(columns[i:i+num_columns]):
        #         # Usamos un selectbox para CHAS y RAD, y text_input para el resto de las variables
        #         if col == "CHAS":
        #             input_value = cols[j].selectbox(
        #                 f"Ingrese el valor para {col} (0 o 1)", 
        #                 options=chas_options,
        #                 help=column_types[col]
        #             )
        #         elif col == "RAD":
        #             input_value = cols[j].selectbox(
        #                 f"Ingrese el valor para {col} (1-24)", 
        #                 options=rad_options,
        #                 help=column_types[col]
        #             )
        #         else:
        #             # Para otras variables numéricas, seguimos usando text_input
        #             input_value = cols[j].text_input(
        #                 f"Ingrese el valor para {col}",
        #                 value=str(st.session_state[f'input_{col}']),  # Como texto para evitar botones
        #                 help=column_types[col]  # Mostrar el tipo de la variable
        #             )
    
        #         # Convertir el valor ingresado a número (si es válido)
        #         try:
        #             input_value = float(input_value) if input_value else 0.0  # Usar 0.0 si no se ingresa valor
        #         except ValueError:
        #             input_value = 0.0  # En caso de que no se ingrese un número válido
    
        #         # Guardamos el valor en session_state
        #         st.session_state[f'input_{col}'] = input_value
        #         input_data[col] = input_value  # Guardar el valor en el diccionario de entrada
    
    
    ####################################################################################################################################################################################################################################################################################

        # Definir las columnas del dataset
        column_names = [
            "Age", "Weight", "Length", "Sex", "BMI", "DM", "HTN", "Current Smoker", 
            "EX-Smoker", "FH", "Obesity", "CRF", "CVA", "Airway disease", "Thyroid Disease", "CHF", "DLP", "BP", "PR", "Edema", 
            "Weak Peripheral Pulse", "Lung rales", "Systolic Murmur", "Diastolic Murmur", "Typical Chest Pain", "Dyspnea", 
            "Function Class", "Atypical", "Nonanginal", "Exertional CP", "LowTH Ang", "Q Wave", "St Elevation", "St Depression", 
            "Tinversion", "LVH", "Poor R Progression", "BBB", "FBS", "CR", "TG", "LDL", "HDL", "BUN", "ESR", "HB", "K", "Na", 
            "WBC", "Lymph", "Neut", "PLT", "EF-TTE", "Region RWMA"
        ]
        
        # Definir las variables categóricas
        categorical_columns = {
            "Sex": ["Male", "Female"], "DM": [0,1], "HTN": [0,1], "Current Smoker": [0, 1], "EX-Smoker": [0, 1], "FH": [0, 1], 
            "Obesity": ["Y", "N"], "CRF": ["Y", "N"], "CVA": ["Y", "N"], "Airway disease": ["Y", "N"], "Thyroid Disease": ["Y", "N"],
            "CHF": ["Y", "N"], "Edema": [0,1], "Weak Peripheral Pulse": ["Y","N"], "Lung rales": ["Y","N"], 
            "Systolic Murmur": ["Y","N"], "Diastolic Murmur": ["Y","N"], "Typical Chest Pain": [0,1], "Dyspnea": ["Y","N"],
            "Function Class": [0,1,2,3], "Atypical": ["Y","N"], "Nonanginal": ["Y","N"], "LowTH Ang": ["Y","N"], 
            "Q Wave": [0,1], "St Elevation": [0,1], "St Depression": [0,1], "Tinversion": [0,1], "LVH": ["Y", "N"], 
            "Poor R Progression": ["Y", "N"], "BBB": ["LBBB", "N","RBBB"], "Region RWMA": [0,1,2,3,4]
        }
        
        # Diccionario con descripciones de cada variable
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
            "EF-TTE": "Fracción de eyección en porcentaje.", "Region RWMA": "Anormalidades del movimiento regional de la pared."
        }
        
        # Título de la aplicación
        # Título de la aplicación
        st.write("### Formulario de ingreso de datos para predicción")
        
        # Crear el formulario
        input_data = {}
        num_columns = 3  # Definir el número de columnas para organizar los campos
        
        for i in range(0, len(column_names), num_columns):
            cols = st.columns(num_columns)
            for j, col in enumerate(column_names[i:i+num_columns]):
                if col in categorical_columns:
                    # Inicializar con el primer valor de la lista si no está en session_state
                    if f"input_{col}" not in st.session_state:
                        st.session_state[f"input_{col}"] = categorical_columns[col][0]
        
                    input_value = cols[j].selectbox(
                        f"{col}", options=categorical_columns[col], 
                        index=categorical_columns[col].index(st.session_state[f"input_{col}"]),
                        help=column_types.get(col, "")
                    )
        
                else:
                    # Inicializar con 0.0 si no está en session_state
                    if f"input_{col}" not in st.session_state:
                        st.session_state[f"input_{col}"] = 0.0
        
                    input_value = cols[j].text_input(
                        f"{col}", value=str(st.session_state[f"input_{col}"]),
                        help=column_types.get(col, "")
                    )
        
                    try:
                        input_value = float(input_value)
                    except ValueError:
                        input_value = 0.0
        
                # Guardar el valor en session_state y en input_data
                st.session_state[f"input_{col}"] = input_value
                input_data[col] = input_value
        
        st.write("### Datos ingresados")
        
        # Procesar los datos en un formato adecuado
        processed_data = [
            str(value) if col in categorical_columns else float(value) 
            for col, value in input_data.items()
        ]
        
        # Convertir la lista en un numpy array
        input_array = np.array(processed_data, dtype=object)  # dtype=object mantiene tipos mixtos

        # st.json(input_data)



        

        


        ################################################################################################   ################################################################################################
     
        # # Crear DataFrame inicial con valores numéricos en 0 y categóricos con el primer valor de la lista
        # data = {col: [0.0] for col in column_names}  # Inicializar numéricos en 0
        # for col in categorical_columns:
        #     data[col] = [categorical_columns[col][0]]  # Inicializar con el primer valor de la lista
        # df = pd.DataFrame(data)
        # # Convertir columnas categóricas a tipo "category" para que se muestren como dropdown en st.data_editor
        # for col in categorical_columns:
        #     df[col] = df[col].astype("category")
        # # Mostrar la tabla editable en Streamlit
        # st.write("### Introduce los datos para la predicción:")
        # edited_df = st.data_editor(df, key="editable_table")
        # # Mostrar la tabla actualizada
        # st.write("#### Datos ingresados:")
        # st.write(edited_df)
        # Botón para generar la predicción

        if st.button("Realizar predicción"):
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
            # new_data_categorical = new_data[encoder.feature_names_in_]  # Mantiene solo las categóricas
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

# Modelo de redes neuronales
if st.sidebar.checkbox("Utilizar redes Neuronales"): 
    st.write("### Redes Neuronales")
    st.write("### dwd Neuronales")    
    st.write("""El modelo utilizado consiste en una red neuronal de una capa con 32 neuronas de entrada.
    La base de datos fue codificada con One Hot Encoder y estandarizada con StandardScaler.""")
    st.write("### Indique si desea hacer una predicción de manera manual o usar datos por defecto")
    selected_column = st.selectbox("Selecciona un método para la predicción", ['Por defecto','Manual'],key="modelo2_metodo_prediccion")
    if selected_column=='Por defecto':             
        st.write("### Indique los datos por defecto que desea uasr para la predicción")
        data_model2 = st.selectbox("Selecciona un método para la predicción", ['Datos 1','Datos 2','Datos 3','Datos 4','Datos 5','Datos 6','Datos 7','Datos 8','Datos 9','Datos 10'],key="modelo2_eleccion_datos")
        datos_pordefecto2(data_model2)        
        
    if selected_column=='Manual':
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
            # Mostrar los datos originales
            st.write(" **Datos originales:**")
            st.write(edited_df)            
            encoder, numerical_columns = load_encoder()            
            # Simulación de datos nuevos
            new_data = edited_df
            # Separar variables numéricas y categóricas
            new_data_categorical = new_data[encoder.feature_names_in_]  # Mantiene solo las categóricas
            new_data_numerical = new_data[numerical_columns]  # Mantiene solo las numéricas            
            # Codificar las variables categóricas
            encoded_array = encoder.transform(new_data_categorical)            
            # Convertir la salida a DataFrame con nombres de columnas codificadas
            encoded_df = pd.DataFrame(encoded_array, columns=encoder.get_feature_names_out())            
            # Concatenar las variables numéricas con las categóricas codificadas
            final_data = pd.concat([new_data_numerical, encoded_df], axis=1)    
            prediction=model2.predict(final_data)
            if prediction==1:
                st.write("Predicción del modelo:","Cath", prediction)
            else:
                st.write("Predicción del modelo:","Normal", prediction)
