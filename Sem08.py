import streamlit as st
import pandas as pd


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