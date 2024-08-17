import streamlit as st
import pandas as pd


#Título al app
st.title("K-Means Clustering con Streamlit")

#Subir archivo de Excel
uploaded_file = st.file_uploader("Sube un archivo Excel", type=["xlsx"])