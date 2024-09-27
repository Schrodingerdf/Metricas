import streamlit as st
import pandas as pd

# Título de la app
st.title('AUTO-DOC')

# Inicializar una lista para almacenar los datos
if 'data' not in st.session_state:
    st.session_state['data'] = []

# Formulario para ingresar datos
with st.form(key='data_form'):
    nombre = st.text_input('Nombre:')
    edad = st.number_input('Edad:', min_value=0, max_value=120)
    ciudad = st.text_input('Ciudad:')
    
    # Botón para enviar el formulario
    submit_button = st.form_submit_button(label='Agregar datos')

# Si se presiona el botón, agregar los datos a la lista
if submit_button:
    if nombre and ciudad:  # Asegúrate de que los campos no estén vacíos
        nuevo_dato = {'Nombre': nombre, 'Edad': edad, 'Ciudad': ciudad}
        st.session_state['data'].append(nuevo_dato)
        st.success("¡Datos agregados exitosamente!")
    else:
        st.error("Por favor, completa todos los campos antes de agregar los datos.")

# Convertir los datos almacenados a un DataFrame de pandas
df = pd.DataFrame(st.session_state['data'])

# Si hay datos, mostrarlos
if not df.empty:
    st.write("Datos ingresados:")
    st.dataframe(df)
else:
    st.write("No se han ingresado datos aún.")

