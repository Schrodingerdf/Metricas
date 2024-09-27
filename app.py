import google.generativeai as genai
import streamlit as st
import pandas as pd
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
    st.title("Gemini-Diego Flores I.")

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

    # Set up the model configuration options
    temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.9, 0.1)
    top_p = st.sidebar.number_input("Top P", 0.0, 1.0, 1.0, 0.1)
    top_k = st.sidebar.number_input("Top K", 1, 100, 1)
    max_output_tokens = st.sidebar.number_input("Max Output Tokens", 1, 10000, 2048)

    # Set up the model configuration dictionary
    generation_config = {
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "max_output_tokens": max_output_tokens,
    }

    safety_settings = "{}"  # Placeholder for safety settings, can be modified as needed
    safety_settings = json.loads(safety_settings)
        
    # Text input for the query
    prompt = st.text_input("Ingresa tu pregunta:")

    # CSV File Uploader
    uploaded_file = st.file_uploader("Sube un archivo CSV", type=["csv"])

    if uploaded_file is not None:
        # Read the uploaded CSV file into a DataFrame
        df = pd.read_csv(uploaded_file)
        st.write("Contenido del archivo CSV:")
        st.dataframe(df)

    # Check if the query is provided
    if not prompt:
        st.error("Ingresa tu pregunta.")
        st.stop()

    # Initialize the generative model
    gemini = genai.GenerativeModel(
        model_name="gemini-pro",
        generation_config=generation_config,
        safety_settings=safety_settings
    )
                
    prompt_parts = [prompt]
    
    try:
        # Generate content using the model
        response = gemini.generate_content(prompt_parts)
        st.subheader("Gemini Response:")
        if response.text:
            st.write(response.text)
        else:
            st.write("No output from Gemini.")
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

# Run the Streamlit app
if __name__ == "__main__":
    text_page()
