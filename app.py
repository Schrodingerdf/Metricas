import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn import metrics
from scipy.stats import ks_2samp

# Definir la contraseña correcta
PASSWORD = "112357"

# Función para evaluar la contraseña
def check_password():
    """Función que verifica la contraseña ingresada por el usuario."""
    st.sidebar.title("Autenticación")
    password = st.sidebar.text_input("Ingresa la contraseña:", type="password")
    
    if password == PASSWORD:
        return True
    elif password:
        st.sidebar.error("Contraseña incorrecta.")
        return False
    else:
        return False

# Función KS
def evaluate_ks(y_real, y_proba):
    df = pd.DataFrame({'real': y_real, 'proba': y_proba})
    class0_proba = df.loc[df['real'] == 0, 'proba']
    class1_proba = df.loc[df['real'] == 1, 'proba']
    ks_result = ks_2samp(class0_proba, class1_proba)
    st.write(f"KS: {ks_result.statistic:.4f} (p-value: {ks_result.pvalue:.3e})")
    return ks_result.statistic

# Función umbral óptimo
def calcular_umbral_optimo(y_real, proba):
    fpr, tpr, thresholds = roc_curve(y_real, proba)
    diferencia = tpr - fpr
    optimal_idx = np.argmax(diferencia)
    return thresholds[optimal_idx]

# Función de métricas y matriz de confusión
def calcular_metricas_y_graficar(y_real, proba):
    umbral_optimo = calcular_umbral_optimo(y_real, proba)
    y_pred = np.where(proba >= umbral_optimo, 1, 0)
    confusion_matrix = metrics.confusion_matrix(y_real, y_pred)
    
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=[False, True])
    plt.figure(figsize=(8, 6))
    cm_display.plot(cmap=plt.cm.YlGnBu)
    plt.title('Matriz de Confusión')
    st.pyplot(plt.gcf())
    
    accuracy = metrics.accuracy_score(y_real, y_pred)
    precision = metrics.precision_score(y_real, y_pred)
    recall = metrics.recall_score(y_real, y_pred)
    specificity = metrics.recall_score(y_real, y_pred, pos_label=0)
    f1_score = metrics.f1_score(y_real, y_pred)
    
    st.write(f"Umbral óptimo: {umbral_optimo:.4f}")
    st.write(f"Accuracy: {accuracy:.4f}")
    st.write(f"Precision: {precision:.4f}")
    st.write(f"Sensitivity (Recall): {recall:.4f}")
    st.write(f"Specificity: {specificity:.4f}")
    st.write(f"F1 Score: {f1_score:.4f}")

# Aplicación principal en Streamlit con autenticación
def main():
    st.title("Evaluación de Modelo con Filtros")

    # Verificar contraseña
    if check_password():
        # Subir el archivo CSV
        uploaded_file = st.file_uploader("Sube tu archivo CSV", type="csv")

        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.write("Datos cargados con éxito!")

            if 'filtro' in df.columns and 'target' in df.columns and 'predict' in df.columns:
                # Filtros para train y oot
                filtros = df['filtro'].unique()
                
                for filtro in filtros:
                    st.subheader(f"Resultados para filtro: {filtro}")
                    df_filtrado = df[df['filtro'] == filtro]
                    calcular_metricas_y_graficar(df_filtrado['target'], df_filtrado['predict'])
                    st.write("---")
            else:
                st.error("El archivo CSV debe contener las columnas 'filtro', 'target' y 'predict'.")
    else:
        st.warning("Por favor, ingresa la contraseña para acceder.")

if __name__ == '__main__':
    main()
