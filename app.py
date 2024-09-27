import subprocess
import os

def instalar_ollama():
    # Ejecutar el comando curl para instalar Ollama
    try:
        print("Instalando Ollama...")
        subprocess.run(["curl", "https://ollama.ai/install.sh", "|", "sh"], check=True)
        print("Ollama instalado correctamente.")
    except subprocess.CalledProcessError as e:
        print(f"Error durante la instalación de Ollama: {e}")

def iniciar_servidor_ollama():
    # Ejecutar el comando para iniciar el servidor Ollama
    try:
        print("Iniciando el servidor de Ollama...")
        subprocess.run(["ollama", "serve"], check=True)
        print("Servidor de Ollama iniciado correctamente.")
    except subprocess.CalledProcessError as e:
        print(f"Error al iniciar el servidor de Ollama: {e}")

def descargar_modelo(name_model):
    # Ejecutar el comando para descargar el modelo de Ollama
    try:
        print(f"Descargando el modelo {name_model}...")
        subprocess.run(["ollama", "pull", name_model], check=True)
        print(f"Modelo {name_model} descargado correctamente.")
    except subprocess.CalledProcessError as e:
        print(f"Error al descargar el modelo {name_model}: {e}")

if __name__ == "__main__":
    # Llamar a las funciones en orden
    instalar_ollama()
    iniciar_servidor_ollama()

    # Especificar el nombre del modelo que deseas descargar
    name_model = "phi3:medium"
    descargar_modelo(name_model)
