import streamlit as st
import pandas as pd
import altair as alt

# Set the title of the app
st.title("App Diego Flores") 

# Título de la app
st.title('Cargar y visualizar archivo CSV')

# Instrucción para que el usuario suba un archivo CSV
uploaded_file = st.file_uploader("Sube tu archivo CSV", type="csv")

# Si el usuario sube un archivo
if uploaded_file is not None:
    # Carga el archivo en un DataFrame
    df = pd.read_csv(uploaded_file)
    
    # Muestra los datos
    st.write("Vista previa de los primeros 5 registros del archivo CSV:")
    st.dataframe(df.head())  # Puedes cambiar el número de filas si lo deseas
    
    # Información básica del DataFrame
    st.write("Información del archivo CSV:")
    st.write(df.describe())
else:
    st.write("Por favor, sube un archivo CSV para visualizar los datos.")

