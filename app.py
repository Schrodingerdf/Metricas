# Metricas
import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import ks_2samp
from sklearn.metrics import roc_curve
from sklearn import metrics
import matplotlib.pyplot as plt
import google.generativeai as genai
import json

# Define your API key and password
API_KEY = "AIzaSyCLY-K449EXP04NAMu2XEugi29HWGYdMlY"
PASSWORD = "112357"

# Function to initialize session state
def initialize_session_state():
    if 'password_correct' not in st.session_state:
        st.session_state.password_correct = False

# Función KS
def evaluate_ks(y_real, y_proba):
    df = pd.DataFrame({
        'real': y_real,
        'proba': y_proba
    })
    class0_proba = df.loc[df['real'] == 0, 'proba']
    class1_proba = df.loc[df['real'] == 1, 'proba']
    ks_result = ks_2samp(class0_proba, class1_proba)
    return ks_result

# Función para calcular el umbral óptimo
def calcular_umbral_optimo(y_real, proba):
    fpr, tpr, thresholds = roc_curve(y_real, proba)
    diferencia = tpr - fpr
    optimal_idx = np.argmax(diferencia)
    optimal_threshold = thresholds[optimal_idx]
    return optimal_threshold

# Función para calcular métricas y graficar la matriz de confusión
def calcular_metricas_y_graficar(y_real, proba):
    umbral_optimo = calcular_umbral_optimo(y_real, proba)
    y_pred = np.where(proba >= umbral_optimo, 1, 0)
    confusion_matrix = metrics.confusion_matrix(y_real, y_pred)
    
    # Graficar matriz de confusión
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=[False, True])
    fig, ax = plt.subplots(figsize=(8, 6))
    cm_display.plot(ax=ax, cmap=plt.cm.YlGnBu)
    st.pyplot(fig)
    
    # Calcular métricas
    Accuracy = metrics.accuracy_score(y_real, y_pred)
    Precision = metrics.precision_score(y_real, y_pred)
    Sensitivity_recall = metrics.recall_score(y_real, y_pred)
    Specificity = metrics.recall_score(y_real, y_pred, pos_label=0)
    F1_score = metrics.f1_score(y_real, y_pred)

    # Mostrar métricas
    st.write(f"Umbral óptimo: {umbral_optimo:.4f}")
    st.write(f"Accuracy: {Accuracy:.4f}")
    st.write(f"Precision: {Precision:.4f}")
    st.write(f"Sensitivity/Recall: {Sensitivity_recall:.4f}")
    st.write(f"Specificity: {Specificity:.4f}")
    st.write(f"F1 Score: {F1_score:.4f}")
    
    # Asegurarse de devolver las métricas
    return Accuracy, Precision, Sensitivity_recall, Specificity, F1_score

# Función para calcular veintiles
def calcular_veintiles(df, y_real_col, prob_col):
    percentiles = [i / 20 for i in range(1, 20)]
    veintiles = df[prob_col].quantile(percentiles)

    def clasificar_veintil(valor, veintiles):
        for i, veintil in enumerate(veintiles):
            if valor <= veintil:
                return i + 1
        return len(veintiles) + 1

    df['Veintil_prob'] = df[prob_col].apply(lambda x: clasificar_veintil(x, veintiles))

    df_0 = df[df[y_real_col] == 0]
    df_1 = df[df[y_real_col] == 1]
    
    counts_0 = df_0['Veintil_prob'].value_counts().sort_index()
    counts_1 = df_1['Veintil_prob'].value_counts().sort_index()

    veintil_df = pd.DataFrame({
        'Veintil_prob': range(1, 21),
        'N°Buenos': [counts_0.get(i, 0) for i in range(1, 21)],
        'N°Malos': [counts_1.get(i, 0) for i in range(1, 21)]
    }).set_index('Veintil_prob')

    veintil_df['Total_casos'] = veintil_df['N°Buenos'] + veintil_df['N°Malos']
    veintil_df['Buenos_acum'] = veintil_df['N°Buenos'].cumsum()
    veintil_df['Malos_acum'] = veintil_df['N°Malos'].cumsum()
    veintil_df['Total_acum'] = veintil_df['Total_casos'].cumsum()
    veintil_df['%Malo_Grupo'] = (veintil_df['N°Malos'] / veintil_df['Total_casos'] * 100).round(1).apply(lambda x: f"{x:.1f}%")
    veintil_df['%Buenos_acum'] = (veintil_df['Buenos_acum'] / veintil_df['N°Buenos'].sum() * 100).round(1).apply(lambda x: f"{x:.1f}%")
    veintil_df['%Malos_acum'] = (veintil_df['Malos_acum'] / veintil_df['N°Malos'].sum() * 100).round(1).apply(lambda x: f"{x:.1f}%")

    st.write(veintil_df)

# Main Streamlit app
def text_page():

    st.title("Métricas IA")
    st.write("¡Bienvenidos a Métricas IA! 🚀 Con nuestra herramienta, puedes calcular de manera rápida y sencilla las principales métricas de evaluación de modelos de scoring, todo potenciado con inteligencia artificial. 📊 Obtén análisis precisos y relevantes que te ayudarán a tomar decisiones informadas, evaluando el rendimiento de tus modelos desde la precisión hasta la matriz de confusión. Con Métricas IA, optimiza tus modelos y explora el potencial de tus datos de manera eficiente y efectiva. 💡✨")
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
        
    # Initialize the generative model
    gemini = genai.GenerativeModel(
        model_name="gemini-pro",
        generation_config=generation_config,
        safety_settings=safety_settings
    )

    # Subir archivo CSV
    uploaded_file = st.file_uploader("Sube un archivo CSV", type=["csv"])

    if uploaded_file is not None:
        # Leer CSV
        df = pd.read_csv(uploaded_file)
        st.write("Contenido del archivo CSV:")
        st.dataframe(df)

        # Seleccionar columnas para 'y_real' y 'proba'
        y_real_col = st.selectbox("Selecciona la columna Target:", df.columns)
        prob_col = st.selectbox("Selecciona la columna de probabilidades:", df.columns)
        filtro_col = st.selectbox("Selecciona la columna de filtro:", df.columns)

        # Mostrar análisis
        if st.button("Ejecutar análisis"):
            y_real_train = df[df[filtro_col] == 'train'][y_real_col]
            proba_train = df[df[filtro_col] == 'train'][prob_col]

            y_real_oot = df[df[filtro_col] == 'oot'][y_real_col]
            proba_oot = df[df[filtro_col] == 'oot'][prob_col]

            # KS
            st.subheader("Resultado del Test KS")

            # Explicación de la prueba KS
            st.write(
                "La prueba de Kolmogorov-Smirnov es una prueba de bondad de ajuste utilizada para determinar si una muestra proviene de una distribución específica. "
                "Compara la función de distribución empírica de la muestra con la función de distribución teórica esperada y calcula el valor d, que representa "
                "la máxima diferencia absoluta entre las dos funciones."
            )

            # Cálculo de la estadística KS para el conjunto de entrenamiento y fuera de tiempo
            ks_stat_train = evaluate_ks(y_real_train, proba_train)
            ks_stat_oot = evaluate_ks(y_real_oot, proba_oot)

            # Mostrar resultados para el conjunto de entrenamiento
            st.write(f"**Conjunto de Entrenamiento:**")
            st.write(f"KS: {ks_stat_train.statistic:.4f} (p-value: {ks_stat_train.pvalue:.3e})")

            # Mostrar resultados para el conjunto fuera de tiempo
            st.write(f"**Conjunto Fuera de Tiempo:**")
            st.write(f"KS: {ks_stat_oot.statistic:.4f} (p-value: {ks_stat_oot.pvalue:.3e})")

            # Generación de conclusión usando Gemini
            prompt = f"Haz una conclusión sobre la prueba de Kolmogorov para el entrenamiento con valores: {ks_stat_train} y fuera de tiempo con valores: {ks_stat_oot}. si el p-valor es menor de 0.05 se puede concluir que el modelo esta discriminando de forma adecuada,hazlo en maximo 2 parrafos"
            prompt_parts = [prompt]
            conclusion = gemini.generate_content(prompt_parts).text

            # Mostrar la conclusión generada
            st.write("### Conclusión:")
            st.write(conclusion)


            # Métricas y matriz de confusión
            st.subheader("Matriz de Confusión y Métricas")
            
            # Explicación breve
            st.write(
                "A continuación se presentan las matrices de confusión y las métricas asociadas para los conjuntos de entrenamiento y fuera de tiempo (OOT). "
                "Estas métricas permiten evaluar el rendimiento del modelo en cada uno de estos conjuntos de datos."
            )
            
            # Mostrar métricas y matriz de confusión para el conjunto de entrenamiento
            st.write("TRAIN:")
            accuracy_train, precision_train, sensitivity_train, specificity_train, f1_score_train = calcular_metricas_y_graficar(y_real_train, proba_train)
            
            # Generación de conclusión usando Gemini
            prompt = f"Haz una conclusión sobre los resultados de las siguientes métricas: Accuracy: {accuracy_train}, Precision: {precision_train}, Sensitivity: {sensitivity_train}, Specificity: {specificity_train}, F1 Score: {f1_score_train},hazlo en maximo 2 parrafos"
            prompt_parts = [prompt]
            conclusion = gemini.generate_content(prompt_parts).text
            
            # Mostrar la conclusión generada
            st.write("### Conclusión:")
            st.write(conclusion)

            
            st.write("OOT:")
            
            # Mostrar métricas y matriz de confusión para el conjunto de entrenamiento
            accuracy_train, precision_train, sensitivity_train, specificity_train, f1_score_train = calcular_metricas_y_graficar(y_real_oot, proba_oot)
            
            # Generación de conclusión usando Gemini
            prompt = f"Haz una conclusión sobre los resultados de las siguientes métricas: Accuracy: {accuracy_train}, Precision: {precision_train}, Sensitivity: {sensitivity_train}, Specificity: {specificity_train}, F1 Score: {f1_score_train},hazlo en maximo 2 parrafos"
            prompt_parts = [prompt]
            conclusion = gemini.generate_content(prompt_parts).text
            
            # Mostrar la conclusión generada
            st.write("### Conclusión:")
            st.write(conclusion)    

            # Veintiles
            st.subheader("Tabla de Eficiencia")

           # Explicación breve
            st.write(
                "Una tabla de eficiencia evalúa la precisión de un modelo predictivo al medir su capacidad para capturar correctamente un evento en segmentos de población ordenados. Es usada para validar modelos en áreas como crédito, riesgo o clasificación de clientes. "
                )
            
            st.write("Train:")
            calcular_veintiles(df[df[filtro_col] == 'train'], y_real_col, prob_col)
            st.write("OOT:")
            calcular_veintiles(df[df[filtro_col] == 'oot'], y_real_col, prob_col)

            # Inicializa la lista de mensajes en el estado de la sesión
            if 'messages' not in st.session_state:
                st.session_state.messages = []
            
            # Función para mostrar el chat
            def mostrar_chat():
                # Título del chat
                st.subheader("Chat con IA")
                
                # Campo de entrada para preguntas del usuario
                user_input = st.text_input("Escribe tu mensaje:")
                
                if user_input:
                    # Agregar el mensaje del usuario a la lista
                    st.session_state.messages.append({"role": "user", "content": user_input})
                    
                    # Generar respuesta usando Gemini
                    response = gemini.generate_content([user_input]).text
                    st.session_state.messages.append({"role": "assistant", "content": response})
            
                # Mostrar mensajes en el chat
                for msg in st.session_state.messages:
                    if msg["role"] == "user":
                        st.write(f"You: {msg['content']}")
                    else:
                        st.write(f"Assistant: {msg['content']}")
            
            # Llama a la función para mostrar el chat al final de tu página
            mostrar_chat()


# Run the Streamlit app
if __name__ == "__main__":
    text_page()
