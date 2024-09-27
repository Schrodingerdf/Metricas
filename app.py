from langchain_google_genai import ChatGoogleGenerativeAI

# Configurar la clave de API de Google Gemini
GOOGLE_API_KEY = 'AIzaSyCLY-K449EXP04NAMu2XEugi29HWGYdMlY'  # Reemplazar con tu clave válida

# Inicializar el modelo de lenguaje Gemini para texto
llm_txt = ChatGoogleGenerativeAI(
    model='gemini-pro',
    google_api_key=GOOGLE_API_KEY,
    temperature=0.2
)

def main():
    print("Bienvenido al chat con Gemini. Escribe 'salir' para terminar.")
    
    while True:
        # Obtener la pregunta del usuario
        pregunta = input("Tú: ")
        
        # Verificar si el usuario quiere salir
        if pregunta.lower() == 'salir':
            print("Saliendo del chat. ¡Hasta luego!")
            break
        
        # Hacer la consulta a Gemini
        response_txt = llm_txt.invoke(pregunta)
        
        # Mostrar la respuesta de Gemini
        print(f"Gemini: {response_txt.content}")

if __name__ == "__main__":
    main()
