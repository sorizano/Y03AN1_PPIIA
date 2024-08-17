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