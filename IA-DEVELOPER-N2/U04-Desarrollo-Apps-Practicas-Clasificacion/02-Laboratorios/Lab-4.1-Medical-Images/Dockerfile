# Laboratorio 4.1 - Dockerfile para API de Clasificación Médica
FROM python:3.9-slim

# Establecer directorio de trabajo
WORKDIR /app

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copiar requirements
COPY requirements.txt .

# Instalar dependencias Python
RUN pip install --no-cache-dir -r requirements.txt

# Copiar código fuente
COPY src/ ./src/
COPY models/ ./models/

# Crear directorios necesarios
RUN mkdir -p /app/data /app/logs

# Exponer puerto
EXPOSE 8000

# Comando para ejecutar la API
CMD ["python", "src/api.py"]
