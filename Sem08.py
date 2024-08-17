import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import plotly.express as px

#Título al app
st.title("K-Means Clustering con Streamlit")

#Subir archivo de Excel
uploaded_file = st.file_uploader("Sube un archivo Excel", type=["xlsx"])

if uploaded_file isnotNone:
    #Leer archivo Excel
    df = pd.read_excel(uploaded_file)

    st.write("### Vista previa de los datos")
    st.write(df.head())

    #seleccionar columnas categóricas
    categorical_columns = df.select_dtypes(include=['object']).columns.tolist()

    if categorical_columns:
        st.write("### Columnas categóricas identificadas")
        st.write(categorical_columns)

        #convertir columnas categóricas a dummies
        df = pd.get_dummies(df, columns=categorical_columns)
        st.write("### Datos después de la vonversión a dummies")
        st.write(df.head())
    else:
        st.write("No se encontraron columnas categóricas en los datos")
    
    #Normalizar los datos
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df)

    #Selección del número de clusters
    st.write("### Selecciona el número de clusters")
    num_clusters = st.slider("Número de cluster", min_value=2, max_value=10, value=3)

    #Aplicando el K-Means
    kmeans = kmeans(num_clusters=num_clusters, random_state=42)
    clusters = kmeans.fit_predict(df_scaled)