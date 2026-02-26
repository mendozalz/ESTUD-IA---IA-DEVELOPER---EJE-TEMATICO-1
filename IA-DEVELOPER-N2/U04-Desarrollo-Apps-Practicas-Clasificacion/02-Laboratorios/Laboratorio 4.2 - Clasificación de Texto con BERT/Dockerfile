# Laboratorio 4.2 - Dockerfile para API de Clasificación de Texto
FROM python:3.9-slim

# Establecer directorio de trabajo
WORKDIR /app

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copiar requirements
COPY requirements.txt .

# Instalar dependencias Python
RUN pip install --no-cache-dir -r requirements.txt

# Descargar recursos de NLTK
RUN python -m nltk.downloader punkt stopwords

# Copiar código fuente
COPY src/ ./src/
COPY models/ ./models/

# Crear directorios necesarios
RUN mkdir -p /app/data /app/logs /app/explanations

# Exponer puerto
EXPOSE 8000

# Comando para ejecutar la API
CMD ["python", "src/api.py"]
