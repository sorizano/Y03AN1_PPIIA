import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import plotly.express as px


# Título de la app
st.title("K-Means Clustering con Análisis PCA usando Streamlit")

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
        df = df.fillna(df.mean())

        st.write("### Datos después de manejar valores faltantes")
        st.write(df.head())

    # Asegurar que todas las columnas sean numéricas
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Normalización de datos
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df.dropna())

    # Aplicar PCA para entender las componentes principales
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(df_scaled)
    pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])

    st.write("### Varianza Explicada por cada Componente Principal:")
    st.write(pca.explained_variance_ratio_)

    st.write("### Cargas (Loadings) de las Variables en las Componentes Principales:")
    loadings = pd.DataFrame(pca.components_.T, columns=['PC1', 'PC2'], index=df.columns)
    st.write(loadings)

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

    # Visualización de los clusters utilizando PCA
    pca_df['Cluster'] = clusters
    fig = px.scatter(pca_df, x='PC1', y='PC2', color='Cluster', title='Visualización de Clusters usando PCA')
    st.plotly_chart(fig)

    # Preparar el archivo CSV en memoria
    csv = df.to_csv(index=False)

    # Crear botón de descarga
    st.download_button(
        label="Descargar CSV con Resultados",
        data=csv,
        file_name='resultados_cluster.csv',
        mime='text/csv'
    )