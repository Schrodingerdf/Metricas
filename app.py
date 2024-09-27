# Función para inicializar los estados de sesión
def initialize_session_state():
    if 'password_correct' not in st.session_state:
        st.session_state.password_correct = False
    if 'uploaded_file' not in st.session_state:
        st.session_state.uploaded_file = None
    if 'messages' not in st.session_state:
        st.session_state.messages = []

# Función para mostrar el chat
def mostrar_chat(gemini):
    st.subheader("Chat con IA")

    # Campo de entrada para preguntas del usuario
    user_input = st.text_input("Escribe tu mensaje:")

    if user_input:
        # Agregar el mensaje del usuario a la lista
        st.session_state.messages.append({"role": "user", "content": user_input})

        # Generar respuesta usando Gemini
        prompt = f"Usuario: {user_input}\nAsistente:"
        response = gemini.generate_content([prompt]).text
        st.session_state.messages.append({"role": "assistant", "content": response})

    # Mostrar los mensajes en el chat
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.write(f"Tú: {msg['content']}")
        else:
            st.write(f"Asistente: {msg['content']}")

# Main Streamlit app
def text_page():
    st.title("Métricas IA")
    
    # Initialize session state
    initialize_session_state()

    # Password input
    password = st.sidebar.text_input("Enter your password:", type="password")

    if password == PASSWORD:
        st.session_state.password_correct = True
    else:
        if st.session_state.password_correct:
            st.sidebar.success("Password is correct!")
        else:
            st.sidebar.error("Incorrect password. Please try again.")
            st.stop()

    # Configuración de Google Generative AI con la API key
    genai.configure(api_key=API_KEY)

    # Configuración manual del modelo
    temperature = 0.9
    top_p = 1.0
    top_k = 1
    max_output_tokens = 2048

    generation_config = {
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "max_output_tokens": max_output_tokens,
    }

    safety_settings = "{}"
    safety_settings = json.loads(safety_settings)

    gemini = genai.GenerativeModel(
        model_name="gemini-pro",
        generation_config=generation_config,
        safety_settings=safety_settings
    )

    # Subir archivo CSV
    uploaded_file = st.file_uploader("Sube un archivo CSV", type=["csv"])

    # Si se sube un archivo, se guarda en el estado de la sesión
    if uploaded_file is not None:
        st.session_state.uploaded_file = uploaded_file

    # Utilizar el archivo guardado en el estado de la sesión
    if st.session_state.uploaded_file is not None:
        df = pd.read_csv(st.session_state.uploaded_file)
        st.write("Contenido del archivo CSV:")
        st.dataframe(df)

        # Selección de columnas para el análisis
        y_real_col = st.selectbox("Selecciona la columna Target:", df.columns)
        prob_col = st.selectbox("Selecciona la columna de probabilidades:", df.columns)
        filtro_col = st.selectbox("Selecciona la columna de filtro:", df.columns)

        # Mostrar análisis
        if st.button("Ejecutar análisis"):
            y_real_train = df[df[filtro_col] == 'train'][y_real_col]
            proba_train = df[df[filtro_col] == 'train'][prob_col]

            y_real_oot = df[df[filtro_col] == 'oot'][y_real_col]
            proba_oot = df[df[filtro_col] == 'oot'][prob_col]

            # KS Test
            st.subheader("Resultado del Test KS")
            ks_stat_train = evaluate_ks(y_real_train, proba_train)
            ks_stat_oot = evaluate_ks(y_real_oot, proba_oot)

            st.write(f"**Conjunto de Entrenamiento:** KS: {ks_stat_train.statistic:.4f} (p-value: {ks_stat_train.pvalue:.3e})")
            st.write(f"**Conjunto Fuera de Tiempo:** KS: {ks_stat_oot.statistic:.4f} (p-value: {ks_stat_oot.pvalue:.3e})")

            # Generación de conclusión con Gemini
            prompt = f"Haz una conclusión sobre la prueba de Kolmogorov para el entrenamiento con valores: {ks_stat_train} y fuera de tiempo con valores: {ks_stat_oot}. Si el p-valor es menor de 0.05 se puede concluir que el modelo está discriminando de forma adecuada, hazlo en máximo 2 párrafos."
            conclusion = gemini.generate_content([prompt]).text
            st.write("### Conclusión:")
            st.write(conclusion)

            # Métricas y matriz de confusión
            st.subheader("Matriz de Confusión y Métricas")
            calcular_metricas_y_graficar(y_real_train, proba_train)
            calcular_metricas_y_graficar(y_real_oot, proba_oot)

            # Veintiles
            st.subheader("Tabla de Eficiencia")
            calcular_veintiles(df[df[filtro_col] == 'train'], y_real_col, prob_col)
            calcular_veintiles(df[df[filtro_col] == 'oot'], y_real_col, prob_col)

    # Mostrar el chat al final de la página
    mostrar_chat(gemini)

# Run the Streamlit app
if __name__ == "__main__":
    text_page()
