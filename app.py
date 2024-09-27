# Métricas IA con Streamlit
import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import ks_2samp
from sklearn.metrics import roc_curve
from sklearn import metrics
import matplotlib.pyplot as plt
import google.generativeai as genai
import json

# Definir clave API y contraseña
API_KEY = "AIzaSyCLY-K449EXP04NAMu2XEugi29HWGYdMlY"
PASSWORD = "112357"

# Inicializar el estado de sesión
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

# Página principal
def text_page():
    st.title("Métricas IA")
    st.write("Bienvenido a la página de métricas alimentada por IA.")

    # Inicializar sesión
    initialize_session_state()

    # Verificación de contraseña
    password = st.sidebar.text_input("Enter your password:", type="password")
    if password == PASSWORD:
        st.session_state.password_correct = True
    else:
        st.sidebar.error("Incorrect password. Please try again.")
        st.stop()

    # Configurar API de Generative AI
    genai.configure(api_key=API_KEY)

    # Subir archivo CSV
    uploaded_file = st.file_uploader("Sube un archivo CSV", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("Contenido del archivo CSV:")
        st.dataframe(df)

        # Seleccionar columnas
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
            ks_stat_train = evaluate_ks(y_real_train, proba_train)
            ks_stat_oot = evaluate_ks(y_real_oot, proba_oot)
            st.write(f"**Conjunto de Entrenamiento KS:** {ks_stat_train.statistic:.4f} (p-value: {ks_stat_train.pvalue:.3e})")
            st.write(f"**Conjunto Fuera de Tiempo KS:** {ks_stat_oot.statistic:.4f} (p-value: {ks_stat_oot.pvalue:.3e})")

            # Conclusión generada con Gemini
            prompt = f"Concluye sobre el test KS: Train KS = {ks_stat_train}, OOT KS = {ks_stat_oot}."
            conclusion = genai.generate_content([prompt]).text
            st.write("### Conclusión KS:")
            st.write(conclusion)

            # Métricas y matriz de confusión
            st.subheader("Matriz de Confusión y Métricas")
            st.write("TRAIN:")
            calcular_metricas_y_graficar(y_real_train, proba_train)
            st.write("OOT:")
            calcular_metricas_y_graficar(y_real_oot, proba_oot)

            # Veintiles
            st.subheader("Tabla de Eficiencia")
            st.write("Train:")
            calcular_veintiles(df[df[filtro_col] == 'train'], y_real_col, prob_col)
            st.write("OOT:")
            calcular_veintiles(df[df[filtro_col] == 'oot'], y_real_col, prob_col)

    # Chat interactivo con el modelo IA
    st.subheader("Chat con el modelo IA")
    if "history" not in st.session_state:
        st.session_state["history"] = []

    # Crear cuadro de texto para la entrada del chat
