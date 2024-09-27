import streamlit as st
import openai  # Asegúrate de tener instalada la biblioteca openai

# Configura tu clave de API de OpenAI (puedes cambiarlo a Gemini si tienes una API de Gemini)
openai.api_key = 'tu_clave_de_api'

# Título de la app
st.title('Chatbot Interactivo')

# Inicializar el historial de chat si no existe en la sesión
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

# Función para obtener la respuesta del modelo de lenguaje
def obtener_respuesta(pregunta):
    # Llama al modelo GPT-4 de OpenAI (puedes usar cualquier modelo basado en API)
    response = openai.Completion.create(
        engine="text-davinci-003",  # Cambia esto al modelo que prefieras
        prompt=pregunta,
        max_tokens=150,
        temperature=0.7
    )
    return response['choices'][0]['text'].strip()

# Caja de texto para que el usuario ingrese su mensaje
pregunta_usuario = st.text_input("Haz tu pregunta:")

# Si el usuario escribe algo y presiona Enter
if pregunta_usuario:
    # Agrega la pregunta al historial
    st.session_state['chat_history'].append(f"Tú: {pregunta_usuario}")
    
    # Obtiene la respuesta del modelo
    respuesta = obtener_respuesta(pregunta_usuario)
    
    # Agrega la respuesta del chatbot al historial
    st.session_state['chat_history'].append(f"Chatbot: {respuesta}")
    
    # Limpiar la caja de texto después de enviar la pregunta
    st.experimental_rerun()

# Mostrar el historial de chat
if st.session_state['chat_history']:
    for mensaje in st.session_state['chat_history']:
        st.write(mensaje)
