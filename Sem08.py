import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import plotly.express as px

# Título de la app
st.title("K-Means Clustering con Streamlit")

# Subir archivo de Excel
uploaded_file = st.file_uploader("Sube un archivo Excel", type=["xlsx"])

if uploaded_file is not None: 
    df = pd.read_excel(uploaded_file)

    st.write("### Vista previa de los datos")
    st.write(df.head())

    # Seleccionar columnas categóricas
    categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
    
    if categorical_columns:
        st.write("### Columnas categóricas identificadas")
        st.write(categorical_columns)

        # Convertir columnas categóricas a dummies
        df = pd.get_dummies(df, columns=categorical_columns)
        st.write("### Datos después de la conversión a dummies")
        st.write(df.head())
    else:
        st.write("No se encontraron columnas categóricas en los datos.")

    # Verificar valores faltantes
    if df.isnull().values.any():
        st.write("### Valores faltantes encontrados")
        st.write(df.isnull().sum())
        
        # Manejo de valores faltantes
        # Opción 1: Eliminar filas con valores faltantes
        # df = df.dropna()

        # Opción 2: Imputar valores faltantes (por ejemplo, con la media)
        df = df.fillna(df.mean())

        st.write("### Datos después de manejar valores faltantes")
        st.write(df.head())

    # Asegurar que todas las columnas sean numéricas
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Normalización de datos
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df.dropna())  # En caso de eliminar filas con NaN

    # Selección del número de clusters
    st.write("### Selecciona el número de clusters")
    num_clusters = st.slider("Número de clusters", min_value=2, max_value=10, value=3)

    # Aplicar K-Means
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    clusters = kmeans.fit_predict(df_scaled)

    # Añadir cluster al DataFrame original
    df['Cluster'] = clusters

    st.write("### Datos con el cluster asignado")
    st.write(df.head())

    # Visualización de los clusters (solo si hay dos dimensiones)
    if df_scaled.shape[1] >= 2:
        df_plot = pd.DataFrame(df_scaled, columns=[f'PC{i+1}'for i in range(df_scaled.shape[1])])
        df_plot['Cluster'] = clusters
        fig = px.scatter(df_plot, x='PC1', y='PC2', color='Cluster', title='Visualización de Clusters')
        st.plotly_chart(fig)
    else:
        st.write("Los datos deben tener al menos 2 columnas numéricas para visualizar los clusters.")
