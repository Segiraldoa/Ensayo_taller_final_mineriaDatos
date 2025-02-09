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


def load_encoder():
	with open("onehot_encoder_5.pkl", "rb") as f:
		encoder = pickle.load(f) 
	with open("numerical_columns_2.pkl", "rb") as f:
		numerical_columns = pickle.load(f) 
	return encoder, numerical_columns

def load_model_1():
    """Cargar el modelo y sus pesos desde el archivo model_weights.pkl."""
    # nombre de la red neuronalv4
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

column_names = [
            "Age", "Weight", "Length", "Sex", "BMI", "DM", "HTN", "Current Smoker", "EX-Smoker", "FH", "Obesity", "CRF", "CVA",
            "Airway disease", "Thyroid Disease", "CHF", "DLP", "BP", "PR", "Edema", "Weak Peripheral Pulse", "Lung rales",
            "Systolic Murmur", "Diastolic Murmur", "Typical Chest Pain", "Dyspnea", "Function Class", "Atypical", "Nonanginal",
            "Exertional CP", "LowTH Ang", "Q Wave", "St Elevation", "St Depression", "Tinversion", "LVH", "Poor R Progression",
            "BBB", "FBS", "CR", "TG", "LDL", "HDL", "BUN", "ESR", "HB", "K", "Na", "WBC", "Lymph", "Neut", "PLT", "EF-TTE",
            "Region RWMA"
        ]


heartdisease = pd.read_csv('heartdisease.csv')
            
#st.success("Modelo cargado correctamente.")
X = heartdisease.iloc[:, :-1]
y = heartdisease['Cath']
X_encoded = pd.get_dummies(X, drop_first=True,dtype= int)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y_encoded, test_size=0.2, random_state=42)

df=X_test.copy()
df_first_row = df.iloc[0,:].to_frame().T # Estos son los valores por defecto y no deben pasar por encoder
# st.write(df_first_row)
# st.write(y_test[0])


#Modelo Clasico
if st.sidebar.checkbox("Utilizar arboles de decisión"):
    st.write("### Arboles de decisión")
    st.write("""
    El modelo utilizado consiste en un arbol con una profundidad de 3.
    La base de datos fue codificada con One Hot Encoder y los datos no fueron escalados.
    """)
    
    st.write("### Indique si desea hacer una predicción de manera manual o usar datos por defecto")
    selected_column = st.selectbox("Selecciona un método para la predicción", ['Por defecto','Manual'])
    
    if selected_column=='Por defecto':
        # Buscar el archivo del modelo dentro de la carpeta extraída
        st.write("### Indique los datos por defecto que desea uasr para la predicción")
        data_model = st.selectbox("Selecciona un método para la predicción", ['Datos 1','Datos 2','Datos 3','Datos 4','Datos 5'])

        if data_model=='Datos 1':
            n=0
            prediction = model1.predict(df.iloc[n,:].to_frame().T)
            if prediction==1 and y_test[n]==1:
                st.write("Predicción del modelo:","Cath", prediction)
                st.write("Clasificación real","Cath", y_test[n])
                st.write("El modelo acertó")                    
            if prediction==0 and y_test[n]==0:
                st.write("Predicción del modelo:","Normal", prediction)
                st.write("Clasificación real","Normal", y_test[n])
                st.write("El modelo acertó")
            else:
                st.write("Predicción del modelo:", prediction)
                st.write("Clasificación real", y_test[n])
                st.write("El modelo falló")

        if data_model=='Datos 2':
            n=1
            prediction = model1.predict(df.iloc[n,:].to_frame().T)
            if prediction==1 and y_test[n]==1:
                st.write("Predicción del modelo:","Cath", prediction)
                st.write("Clasificación real","Cath", y_test[n])
                st.write("El modelo acertó")                    
            if prediction==0 and y_test[n]==0:
                st.write("Predicción del modelo:","Normal", prediction)
                st.write("Clasificación real","Normal", y_test[n])
                st.write("El modelo acertó")
            else:
                st.write("Predicción del modelo:", prediction)
                st.write("Clasificación real", y_test[n])
                st.write("El modelo falló")
        if data_model=='Datos 3':
            n=2
            prediction = model1.predict(df.iloc[n,:].to_frame().T)
            if prediction==1 and y_test[n]==1:
                st.write("Predicción del modelo:","Cath", prediction)
                st.write("Clasificación real","Cath", y_test[n])
                st.write("El modelo acertó")                    
            if prediction==0 and y_test[n]==0:
                st.write("Predicción del modelo:","Normal", prediction)
                st.write("Clasificación real","Normal", y_test[n])
                st.write("El modelo acertó")
            else:
                st.write("Predicción del modelo:", prediction)
                st.write("Clasificación real", y_test[n])
                st.write("El modelo falló")
        if data_model=='Datos 4':
            n=3
            prediction = model1.predict(df.iloc[n,:].to_frame().T)
            if prediction==1 and y_test[n]==1:
                st.write("Predicción del modelo:","Cath", prediction)
                st.write("Clasificación real","Cath", y_test[n])
                st.write("El modelo acertó")                    
            if prediction==0 and y_test[n]==0:
                st.write("Predicción del modelo:","Normal", prediction)
                st.write("Clasificación real","Normal", y_test[n])
                st.write("El modelo acertó")
            else:
                st.write("Predicción del modelo:", prediction)
                st.write("Clasificación real", y_test[n])
                st.write("El modelo falló")
        if data_model=='Datos 5':
            n=4
            prediction = model1.predict(df.iloc[n,:].to_frame().T)
            if prediction==1 and y_test[n]==1:
                st.write("Predicción del modelo:","Cath", prediction)
                st.write("Clasificación real","Cath", y_test[n])
                st.write("El modelo acertó")                    
            if prediction==0 and y_test[n]==0:
                st.write("Predicción del modelo:","Normal", prediction)
                st.write("Clasificación real","Normal", y_test[n])
                st.write("El modelo acertó")
            else:
                st.write("Predicción del modelo:", prediction)
                st.write("Clasificación real", y_test[n])
                st.write("El modelo falló")
            
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

			prediction=model1.predict(final_data)
			if prediction==1:
	            st.write("Predicción del modelo:","Cath", prediction)
			else:
				st.write("Predicción del modelo:","Normal", prediction)
    
# Modelo de redes neuronales
if st.sidebar.checkbox("Utilizar redes Neuronales"): 
    st.write("### Redes Neuronales")
    st.write("ADADASD")

        ###############################################################
        ###################################################################
    
    st.write("""
    El modelo utilizado consiste en una red neuronal de una capa con 32 neuronas de entrada.
    La base de datos fue codificada con One Hot Encoder y estandarizada con StandardScaler.
    """)

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
