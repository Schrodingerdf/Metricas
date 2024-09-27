from fpdf import FPDF
import io

# Función para generar el PDF
def generar_pdf(ks_stat_train, ks_stat_oot, train_metrics, oot_metrics, veintil_train, veintil_oot, conclusion_train, conclusion_oot):
    pdf = FPDF()
    pdf.add_page()
    
    # Título
    pdf.set_font("Arial", size=12, style="B")
    pdf.cell(200, 10, txt="Reporte de Análisis de Modelo", ln=True, align="C")

    # Sección de KS Test
    pdf.set_font("Arial", size=10)
    pdf.ln(10)
    pdf.cell(200, 10, txt="Resultado del Test KS", ln=True)
    pdf.multi_cell(200, 10, f"Conjunto de Entrenamiento: KS: {ks_stat_train.statistic:.4f} (p-value: {ks_stat_train.pvalue:.3e})")
    pdf.multi_cell(200, 10, f"Conjunto Fuera de Tiempo: KS: {ks_stat_oot.statistic:.4f} (p-value: {ks_stat_oot.pvalue:.3e})")

    # Conclusión KS
    pdf.set_font("Arial", size=10, style="I")
    pdf.ln(5)
    pdf.multi_cell(200, 10, f"Conclusión KS: {conclusion_train}")
    
    # Sección de métricas para el conjunto de entrenamiento y OOT
    pdf.set_font("Arial", size=10)
    pdf.ln(10)
    pdf.cell(200, 10, txt="Métricas - Entrenamiento", ln=True)
    pdf.multi_cell(200, 10, f"Accuracy: {train_metrics[0]:.4f}, Precision: {train_metrics[1]:.4f}, Sensitivity: {train_metrics[2]:.4f}, Specificity: {train_metrics[3]:.4f}, F1 Score: {train_metrics[4]:.4f}")
    
    pdf.ln(5)
    pdf.cell(200, 10, txt="Métricas - OOT", ln=True)
    pdf.multi_cell(200, 10, f"Accuracy: {oot_metrics[0]:.4f}, Precision: {oot_metrics[1]:.4f}, Sensitivity: {oot_metrics[2]:.4f}, Specificity: {oot_metrics[3]:.4f}, F1 Score: {oot_metrics[4]:.4f}")
    
    # Conclusión
    pdf.set_font("Arial", size=10, style="I")
    pdf.ln(5)
    pdf.multi_cell(200, 10, f"Conclusión: {conclusion_oot}")

    # Tabla de veintiles
    pdf.set_font("Arial", size=10)
    pdf.ln(10)
    pdf.cell(200, 10, txt="Tabla de Eficiencia - Train", ln=True)
    pdf.multi_cell(200, 10, veintil_train.to_string(index=True))

    pdf.ln(5)
    pdf.cell(200, 10, txt="Tabla de Eficiencia - OOT", ln=True)
    pdf.multi_cell(200, 10, veintil_oot.to_string(index=True))

    return pdf

# Botón para exportar todo en PDF
if st.button("Exportar resultados a PDF"):
    veintil_train = calcular_veintiles(df[df[filtro_col] == 'train'], y_real_col, prob_col)
    veintil_oot = calcular_veintiles(df[df[filtro_col] == 'oot'], y_real_col, prob_col)
    
    # Llamar a la función de generar PDF
    pdf = generar_pdf(ks_stat_train, ks_stat_oot, 
                      (accuracy_train, precision_train, sensitivity_train, specificity_train, f1_score_train),
                      (accuracy_train, precision_train, sensitivity_train, specificity_train, f1_score_train), 
                      veintil_train, veintil_oot, conclusion_train, conclusion_oot)

    # Guardar PDF en un buffer
    pdf_output = io.BytesIO()
    pdf.output(pdf_output)
    pdf_output.seek(0)

    # Descargar PDF
    st.download_button(
        label="Descargar PDF",
        data=pdf_output,
        file_name="reporte_analisis.pdf",
        mime="application/pdf"
    )
