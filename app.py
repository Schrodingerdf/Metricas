import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI

# Configura el modelo
llm_txt = ChatGoogleGenerativeAI(
    model='gemini-pro',
    google_api_key='AIzaSyCLY-K449EXP04NAMu2XEugi29HWGYdMlY',
    temperature=0.2
)

# Título de la aplicación
st.title("Chat con Gemini Pro")

# Caja de texto para la entrada del usuario
user_input = st.text_input("Escribe tu consulta aquí:")

# Botón para enviar la consulta
if st.button("Enviar"):
    if user_input:
        try:
            # Obtener la respuesta del modelo
            response = llm_txt(user_input)
            st.text_area("Respuesta de Gemini Pro:", value=response, height=300)
        except Exception as e:
            st.error(f"Se produjo un error: {e}")
    else:
        st.warning("Por favor, ingresa una consulta.")

# Agregar un pie de página
st.sidebar.info("Esta aplicación utiliza el modelo Gemini Pro de Google.")
