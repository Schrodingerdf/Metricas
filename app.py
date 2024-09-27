import google.generativeai as genai
import streamlit as st
import json

# Define your API key and password
API_KEY = "AIzaSyCLY-K449EXP04NAMu2XEugi29HWGYdMlY"
PASSWORD = "112357"

# Function to initialize session state
def initialize_session_state():
    if 'password_correct' not in st.session_state:
        st.session_state.password_correct = False

# Main Streamlit app
def text_page():
    st.title("Métricas IA")

    # Initialize session state
    initialize_session_state()

    # Password input
    password = st.sidebar.text_input("Enter your password:", type="password")

    # Check if the password is correct
    if password == PASSWORD:
        st.session_state.password_correct = True
    else:
        if st.session_state.password_correct:
            st.sidebar.success("Password is correct!")
        else:
            st.sidebar.error("Incorrect password. Please try again.")
            st.stop()

    # Configure the Generative AI API with the provided key
    genai.configure(api_key=API_KEY)

    # Manual model configuration options
    temperature = 0.9
    top_p = 1.0
    top_k = 1
    max_output_tokens = 2048

    # Set up the model configuration dictionary manually
    generation_config = {
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "max_output_tokens": max_output_tokens,
    }

    safety_settings = "{}"  # Placeholder for safety settings, can be modified as needed
    safety_settings = json.loads(safety_settings)
        
    # Text input for the query
    prompt = 'Que es gemini?'
 
    # Initialize the generative model
    gemini = genai.GenerativeModel(
        model_name="gemini-pro",
        generation_config=generation_config,
        safety_settings=safety_settings
    )

    prompt_parts = [prompt]
    st.write(gemini.generate_content(prompt_parts).text)
    
# Run the Streamlit app
if __name__ == "__main__":
    text_page()
