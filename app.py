import streamlit as st
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

# Configurar la clave de API de Google Gemini
GOOGLE_API_KEY = 'AIzaSyCLY-K449EXP04NAMu2XEugi29HWGYdMlY'  # Reemplazar con tu clave válida

# Inicializar el modelo de lenguaje Gemini para texto
llm_txt = ChatGoogleGenerativeAI(
    model='gemini-pro',
    google_api_key=GOOGLE_API_KEY,
    temperature=0.2
)

# Título de la aplicación en Streamlit
st.title("Chat con Gemini - API de Google")

# Entrada de texto del usuario
pregunta = st.text_input("Haz tu pregunta:", value="¿Qué es la inflación y cómo le fue al Perú en ese aspecto en el 2021?")

# Botón para enviar la consulta
if st.button("Consultar a Gemini"):
    # Hacer la consulta a Gemini
    with st.spinner('Consultando a Gemini...'):
        response_txt = llm_txt.invoke(pregunta)

    # Mostrar la respuesta de Gemini
    st.markdown(f"**Respuesta de Gemini:**\n\n{response_txt.content}")

