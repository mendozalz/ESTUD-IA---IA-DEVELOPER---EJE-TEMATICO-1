# Guía Conceptual: Integración de IA en Aplicaciones Reales

## 📖 Fundamentos Teóricos

### 1. Despliegue de Modelos con APIs

#### Comparación de Frameworks en 2026:

| Framework | Ventajas | Casos de Uso | Ejemplo de Código |
|------------|----------|--------------|-------------------|
| **FastAPI** | - Alto rendimiento (async)<br>- Validación automática con Pydantic<br>- Documentación automática (Swagger/OpenAPI) | APIs para modelos en producción | ```python\nfrom fastapi import FastAPI\napp = FastAPI()\n@app.post(\"/predict\")\ndef predict(data: dict):\n    return {\"prediction\": model.predict(data)}\n``` |
| **Flask** | - Simple y flexible<br>- Gran ecosistema de extensiones<br>- Ideal para prototipos | Aplicaciones pequeñas o medianas | ```python\nfrom flask import Flask, request\napp = Flask(__name__)\n@app.route(\"/predict\", methods=[\"POST\"])\ndef predict():\n    data = request.json\n    return {\"prediction\": model.predict(data)}\n``` |
| **TensorFlow Serving** | - Optimizado para modelos TensorFlow<br>- Soporte para versionado de modelos<br>- Alto rendimiento en producción | Despliegue en entornos empresariales | ```bash\ndocker run -p 8501:8501 --mount type=bind,source=$(pwd)/model,target=/models/my_model -e MODEL_NAME=my_model -t tensorflow/serving\n``` |

#### Mejores Prácticas para APIs de IA:

**1. Diseño de Endpoints RESTful**
```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional

app = FastAPI(title="ML Model API", version="1.0.0")

class PredictionRequest(BaseModel):
    input_data: List[float]
    model_version: Optional[str] = "latest"

class PredictionResponse(BaseModel):
    prediction: float
    confidence: float
    model_version: str
    processing_time_ms: float

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    start_time = time.time()
    try:
        # Validar entrada
        if len(request.input_data) != 10:
            raise HTTPException(status_code=400, detail="Input must have 10 features")
        
        # Realizar predicción
        prediction = model.predict([request.input_data])
        confidence = max(prediction[0])
        
        processing_time = (time.time() - start_time) * 1000
        
        return PredictionResponse(
            prediction=prediction[0],
            confidence=confidence,
            model_version=request.model_version,
            processing_time_ms=processing_time
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

**2. Manejo de Archivos (Imágenes, Audio, Texto)**
```python
from fastapi import UploadFile, File
import io
from PIL import Image
import numpy as np

@app.post("/predict-image")
async def predict_image(file: UploadFile = File(...)):
    # Validar tipo de archivo
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    # Leer y procesar imagen
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert('RGB')
    image = image.resize((224, 224))
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    
    # Realizar predicción
    prediction = model.predict(image_array)
    
    return {
        "prediction": prediction.tolist(),
        "filename": file.filename,
        "size": len(contents)
    }
```

**3. Async/Await para Alto Rendimiento**
```python
import asyncio
import aiofiles
from concurrent.futures import ThreadPoolExecutor

executor = ThreadPoolExecutor(max_workers=4)

@app.post("/predict-batch")
async def predict_batch(files: List[UploadFile] = File(...)):
    async def process_file(file):
        contents = await file.read()
        # Procesamiento síncrono en thread pool
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            executor, 
            model.predict, 
            contents
        )
        return {"filename": file.filename, "result": result}
    
    # Procesar archivos en paralelo
    tasks = [process_file(file) for file in files]
    results = await asyncio.gather(*tasks)
    
    return {"results": results}
```

### 2. Metodología Windsor para Proyectos de IA

#### Fases de Windsor aplicadas a IA:

**Incepción:**
- Definir objetivos del proyecto (ej: "Desplegar un modelo de clasificación de imágenes con 95% de precisión")
- Identificar stakeholders (equipo de IA, desarrolladores front-end, usuarios finales)

**Iteraciones:**
- Sprint 1: Diseño de API y front-end básico
- Sprint 2: Integración del modelo con la API
- Sprint 3: Despliegue y monitoreo

**Liberación:**
- Despliegue en producción con rollback automático
- Documentación para usuarios y mantenimiento

**Retrospectiva:**
- Análisis de métricas (latencia, precisión, satisfacción del usuario)
- Lecciones aprendidas para futuros proyectos

#### Ejemplo de Tablero Windsor:

| Sprint 1: Diseño API | Sprint 2: Integración Modelo | Sprint 3: Despliegue |
|-----------------------------|--------------------------------|----------------------|
| - Definir endpoints | - Conectar modelo a API | - Configurar Kubernetes |
| - Crear mockups de front-end | - Validar integración | - Monitoreo con Prometheus |
| - Documentar API | - Optimizar rendimiento | - Pruebas de carga |

#### Plantilla de Planificación Windsor:

```markdown
# Planificación del Proyecto [Nombre] (Metodología Windsor)

## Objetivo
[Descripción clara del objetivo del proyecto]

## Stakeholders
- [Lista de stakeholders con roles y responsabilidades]

## Sprints

### Sprint 1: [Nombre del Sprint] (2 semanas)
**Objetivo:** [Objetivo específico del sprint]

**Actividades:**
- [ ] [Actividad 1]
- [ ] [Actividad 2]
- [ ] [Actividad 3]

**Entregables:**
- [Entregable 1]
- [Entregable 2]

**Medios de Verificación:**
- [Cómo se verificará que los entregables cumplen los requisitos]

### Sprint 2: [Nombre del Sprint] (2 semanas)
[Continuar con otros sprints...]

## Riesgos y Supuestos
- **Riesgo 1:** [Descripción] - [Mitigación]
- **Supuesto 1:** [Descripción] - [Verificación]

## Métricas de Éxito
- [Métrica 1]: [Valor objetivo]
- [Métrica 2]: [Valor objetivo]
```

### 3. Integración con Front-End

#### Opciones en 2026:

| Tecnología | Ventajas | Casos de Uso | Ejemplo de Integración |
|------------|----------|--------------|------------------------|
| **React** | - Componentes reutilizables<br>- Gran ecosistema<br>- Soporte para hooks | Aplicaciones web complejas | ```javascript\nfetch('/predict', {\n  method: 'POST',\n  body: JSON.stringify({image: base64Image})\n})\n.then(response => response.json())\n.then(data => setPrediction(data.prediction));\n``` |
| **Vue.js** | - Sintaxis simple<br>- Reactividad integrada<br>- Fácil curva de aprendizaje | Aplicaciones interactivas | ```javascript\naxios.post('/predict', {text: userInput})\n.then(response => {\n  this.prediction = response.data.prediction;\n});\n``` |
| **Streamlit** | - Rápido prototipado<br>- Integración nativa con Python<br>- Ideal para dashboards | Prototipos y dashboards | ```python\nimport streamlit as st\nimage = st.file_uploader(\"Subir imagen\")\nif image:\n    prediction = model.predict(preprocess(image))\n    st.write(f\"Predicción: {prediction}\")\n``` |

#### Patrones de Integración:

**1. React con FastAPI**
```javascript
// services/api.js
import axios from 'axios';

const API_BASE_URL = 'http://localhost:8000';

export const mlApi = {
  predict: async (data) => {
    const response = await axios.post(`${API_BASE_URL}/predict`, data);
    return response.data;
  },
  
  predictImage: async (file) => {
    const formData = new FormData();
    formData.append('file', file);
    
    const response = await axios.post(`${API_BASE_URL}/predict-image`, formData, {
      headers: { 'Content-Type': 'multipart/form-data' }
    });
    return response.data;
  }
};

// components/Predictor.jsx
import React, { useState } from 'react';
import { mlApi } from '../services/api';

function Predictor() {
  const [input, setInput] = useState('');
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);

  const handlePredict = async () => {
    setLoading(true);
    try {
      const result = await mlApi.predict({ input_data: input.split(',').map(Number) });
      setPrediction(result);
    } catch (error) {
      console.error('Error:', error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      <input 
        value={input} 
        onChange={(e) => setInput(e.target.value)}
        placeholder="Enter comma-separated values"
      />
      <button onClick={handlePredict} disabled={loading}>
        {loading ? 'Predicting...' : 'Predict'}
      </button>
      
      {prediction && (
        <div>
          <h3>Prediction: {prediction.prediction}</h3>
          <p>Confidence: {(prediction.confidence * 100).toFixed(2)}%</p>
          <p>Processing Time: {prediction.processing_time_ms.toFixed(2)}ms</p>
        </div>
      )}
    </div>
  );
}

export default Predictor;
```

**2. Vue.js con WebSockets**
```vue
<!-- components/RealTimePredictor.vue -->
<template>
  <div>
    <h2>Real-time Prediction</h2>
    <input v-model="inputText" placeholder="Enter text" />
    <button @click="predict">Predict</button>
    
    <div v-if="prediction">
      <h3>Result: {{ prediction.result }}</h3>
      <p>Confidence: {{ (prediction.confidence * 100).toFixed(2) }}%</p>
    </div>
    
    <div v-if="connectionStatus">
      <p>Status: {{ connectionStatus }}</p>
    </div>
  </div>
</template>

<script>
export default {
  data() {
    return {
      inputText: '',
      prediction: null,
      connectionStatus: 'Disconnected',
      socket: null
    };
  },
  
  mounted() {
    this.connectWebSocket();
  },
  
  methods: {
    connectWebSocket() {
      this.socket = new WebSocket('ws://localhost:8000/ws');
      
      this.socket.onopen = () => {
        this.connectionStatus = 'Connected';
      };
      
      this.socket.onmessage = (event) => {
        this.prediction = JSON.parse(event.data);
      };
      
      this.socket.onclose = () => {
        this.connectionStatus = 'Disconnected';
      };
    },
    
    predict() {
      if (this.socket && this.socket.readyState === WebSocket.OPEN) {
        this.socket.send(JSON.stringify({ text: this.inputText }));
      }
    }
  },
  
  beforeUnmount() {
    if (this.socket) {
      this.socket.close();
    }
  }
};
</script>
```

**3. Streamlit Dashboard**
```python
import streamlit as st
import pandas as pd
import plotly.express as px
import requests
import time

# Configuración de la página
st.set_page_config(page_title="ML Model Dashboard", layout="wide")

# Sidebar para configuración
st.sidebar.title("Model Configuration")
model_version = st.sidebar.selectbox("Model Version", ["v1.0", "v1.1", "v2.0"])
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5)

# Área principal
st.title("Real-time ML Model Dashboard")

# Sección de predicción
col1, col2 = st.columns(2)

with col1:
    st.header("Make Prediction")
    
    # Input para texto
    user_input = st.text_area("Enter text for analysis:", height=100)
    
    # Input para imagen
    uploaded_file = st.file_uploader("Upload an image", type=['jpg', 'png'])
    
    if st.button("Predict"):
        if user_input:
            # Predicción de texto
            with st.spinner("Analyzing text..."):
                response = requests.post(
                    "http://localhost:8000/predict-text",
                    json={"text": user_input, "threshold": confidence_threshold}
                )
                result = response.json()
                st.success(f"Sentiment: {result['sentiment']}")
                st.info(f"Confidence: {result['confidence']:.2f}")
        
        elif uploaded_file:
            # Predicción de imagen
            with st.spinner("Analyzing image..."):
                files = {"file": uploaded_file.getvalue()}
                response = requests.post(
                    "http://localhost:8000/predict-image",
                    files=files
                )
                result = response.json()
                st.success(f"Class: {result['class']}")
                st.info(f"Confidence: {result['confidence']:.2f}")

with col2:
    st.header("Model Performance")
    
    # Métricas en tiempo real
    if st.button("Refresh Metrics"):
        metrics_response = requests.get("http://localhost:8000/metrics")
        metrics = metrics_response.json()
        
        col2a, col2b = st.columns(2)
        with col2a:
            st.metric("Total Predictions", metrics['total_predictions'])
            st.metric("Avg Latency (ms)", f"{metrics['avg_latency']:.2f}")
        
        with col2b:
            st.metric("Accuracy", f"{metrics['accuracy']:.2%}")
            st.metric("Error Rate", f"{metrics['error_rate']:.2%}")
    
    # Gráfico de rendimiento
    if st.checkbox("Show Performance Chart"):
        # Datos de ejemplo
        performance_data = pd.DataFrame({
            'timestamp': pd.date_range(start='2024-01-01', periods=100, freq='H'),
            'latency': np.random.normal(50, 10, 100),
            'accuracy': np.random.normal(0.85, 0.05, 100)
        })
        
        fig = px.line(performance_data, x='timestamp', y=['latency', 'accuracy'],
                     title="Model Performance Over Time")
        st.plotly_chart(fig, use_container_width=True)

# Footer con información del modelo
st.sidebar.markdown("---")
st.sidebar.markdown("### Model Information")
st.sidebar.info(f"Current Model: {model_version}")
st.sidebar.info(f"Threshold: {confidence_threshold}")
```

### 4. Despliegue Escalable

#### Opciones en 2026:

| Tecnología | Ventajas | Casos de Uso | Ejemplo de Configuración |
|------------|----------|--------------|------------------------|
| **Docker** | - Contenerización consistente<br>- Fácil despliegue en cualquier entorno | Aplicaciones en contenedores | ```dockerfile\nFROM python:3.9\nCOPY . /app\nWORKDIR /app\nRUN pip install -r requirements.txt\nCMD [\"uvicorn\", \"app:app\", \"--host\", \"0.0.0.0\", \"--port\", \"8000\"]\n``` |
| **Kubernetes** | - Escalado automático<br>- Alta disponibilidad<br>- Gestión de microservicios | Aplicaciones empresariales | ```yaml\napiVersion: apps/v1\nkind: Deployment\nmetadata:\n  name: ia-app\nspec:\n  replicas: 3\n  template:\n    spec:\n      containers:\n      - name: app\n        image: ia-app:latest\n        ports:\n        - containerPort: 8000\n``` |
| **Serverless** | - Pago por uso<br>- Escalado automático<br>- Sin gestión de servidores | APIs con tráfico variable | ```bash\ngcloud functions deploy predict --runtime python39 --trigger-http --allow-unauthenticated\n``` |

#### Docker Multi-stage Build:
```dockerfile
# Stage 1: Build
FROM python:3.9-slim as builder

WORKDIR /app
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# Stage 2: Runtime
FROM python:3.9-slim

WORKDIR /app

# Copy only installed packages
COPY --from=builder /root/.local /root/.local

# Copy application code
COPY . .

# Make sure scripts installed in pip are in PATH
ENV PATH=/root/.local/bin:$PATH

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

#### Kubernetes Configuration Completa:
```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ml-api-deployment
  labels:
    app: ml-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: ml-api
  template:
    metadata:
      labels:
        app: ml-api
    spec:
      containers:
      - name: ml-api
        image: ml-api:latest
        ports:
        - containerPort: 8000
        env:
        - name: MODEL_PATH
          value: "/app/models"
        - name: REDIS_URL
          value: "redis://redis-service:6379"
        resources:
          requests:
            memory: "256Mi"
            cpu: "250m"
          limits:
            memory: "512Mi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5

---
# service.yaml
apiVersion: v1
kind: Service
metadata:
  name: ml-api-service
spec:
  selector:
    app: ml-api
  ports:
    - protocol: TCP
      port: 80
      targetPort: 8000
  type: LoadBalancer

---
# hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: ml-api-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: ml-api-deployment
  minReplicas: 3
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80

---
# ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: ml-api-ingress
  annotations:
    nginx.ingress.kubernetes.io/rewrite-target: /
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
spec:
  tls:
  - hosts:
    - api.mldomain.com
    secretName: ml-api-tls
  rules:
  - host: api.mldomain.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: ml-api-service
            port:
              number: 80
```

#### Serverless con AWS Lambda:
```python
# lambda_function.py
import json
import pickle
import numpy as np
import boto3

# Cargar modelo desde S3
def load_model():
    s3 = boto3.client('s3')
    response = s3.get_object(Bucket='ml-models', Key='model.pkl')
    model = pickle.loads(response['Body'].read())
    return model

# Global variables para reutilizar el modelo
model = None

def lambda_handler(event, context):
    global model
    
    # Inicializar modelo si no está cargado
    if model is None:
        model = load_model()
    
    try:
        # Parsear entrada
        body = json.loads(event['body'])
        
        # Validar entrada
        if 'input_data' not in body:
            return {
                'statusCode': 400,
                'body': json.dumps({'error': 'Missing input_data field'})
            }
        
        input_data = np.array(body['input_data'])
        
        # Realizar predicción
        prediction = model.predict(input_data.reshape(1, -1))
        confidence = max(prediction[0]) if len(prediction.shape) > 1 else prediction
        
        # Formatear respuesta
        response = {
            'prediction': prediction.tolist() if hasattr(prediction, 'tolist') else prediction,
            'confidence': float(confidence),
            'model_version': 'v1.0',
            'timestamp': context.aws_request_id
        }
        
        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps(response)
        }
        
    except Exception as e:
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }
```

### 5. Monitoreo y Observabilidad

#### Herramientas en 2026:

| Herramienta | Uso Principal | Ejemplo de Configuración |
|-------------|--------------|------------------------|
| **Prometheus** | Recopilar métricas en tiempo real (latencia, errores, uso de CPU) | ```yaml\nscrape_configs:\n  - job_name: 'ia-app'\n    static_configs:\n      - targets: ['app:8000']\n``` |
| **Grafana** | Visualizar métricas y crear dashboards | ```json\n{\n  \"title\": \"Latencia de API\",\n  \"targets\": [{\"expr\": \"histogram_quantile(0.95, sum(rate(api_latency_seconds_bucket[5m])) by (le))\", \"refId\": \"A\"}],\n  \"type\": \"graph\"\n}\n``` |
| **OpenTelemetry** | Trazabilidad distribuida para microservicios | ```python\nfrom opentelemetry import trace\ntracer = trace.get_tracer(__name__)\nwith tracer.start_as_current_span(\"predict\"):\n    return model.predict(data)\n``` |

#### Implementación de Métricas con Prometheus:
```python
# metrics.py
from prometheus_client import Counter, Histogram, Gauge, make_asgi_app
import time
import psutil
import threading

# Métricas principales
REQUEST_COUNT = Counter(
    'http_requests_total',
    'Total HTTP requests',
    ['method', 'endpoint', 'status_code']
)

REQUEST_LATENCY = Histogram(
    'http_request_duration_seconds',
    'HTTP request latency in seconds',
    ['method', 'endpoint']
)

MODEL_PREDICTION_COUNT = Counter(
    'model_predictions_total',
    'Total model predictions',
    ['model_name', 'status']
)

MODEL_PREDICTION_LATENCY = Histogram(
    'model_prediction_duration_seconds',
    'Model prediction latency in seconds',
    ['model_name']
)

SYSTEM_CPU_USAGE = Gauge(
    'system_cpu_usage_percent',
    'Current CPU usage percentage'
)

SYSTEM_MEMORY_USAGE = Gauge(
    'system_memory_usage_percent',
    'Current memory usage percentage'
)

# Función para actualizar métricas del sistema
def update_system_metrics():
    cpu_percent = psutil.cpu_percent()
    memory_percent = psutil.virtual_memory().percent
    
    SYSTEM_CPU_USAGE.set(cpu_percent)
    SYSTEM_MEMORY_USAGE.set(memory_percent)

# Actualizar métricas cada 5 segundos
def start_metrics_updater():
    while True:
        update_system_metrics()
        time.sleep(5)

# Iniciar thread para actualización de métricas
metrics_thread = threading.Thread(target=start_metrics_updater, daemon=True)
metrics_thread.start()

# Middleware para FastAPI
from fastapi import Request, Response
from fastapi.responses import JSONResponse
import json

async def metrics_middleware(request: Request, call_next):
    start_time = time.time()
    
    response = await call_next(request)
    
    # Registrar métricas
    method = request.method
    endpoint = request.url.path
    status_code = response.status_code
    
    REQUEST_COUNT.labels(method=method, endpoint=endpoint, status_code=status_code).inc()
    REQUEST_LATENCY.labels(method=method, endpoint=endpoint).observe(time.time() - start_time)
    
    return response

# Decorador para medir predicciones de modelos
def measure_prediction_latency(model_name):
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                MODEL_PREDICTION_COUNT.labels(model_name=model_name, status='success').inc()
                return result
            except Exception as e:
                MODEL_PREDICTION_COUNT.labels(model_name=model_name, status='error').inc()
                raise
            finally:
                MODEL_PREDICTION_LATENCY.labels(model_name=model_name).observe(time.time() - start_time)
        return wrapper
    return decorator
```

#### Configuración de Grafana Dashboards:
```json
{
  "dashboard": {
    "id": null,
    "title": "ML Model Monitoring Dashboard",
    "tags": ["ml", "monitoring"],
    "timezone": "browser",
    "panels": [
      {
        "id": 1,
        "title": "Request Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "sum(rate(http_requests_total[5m])) by (endpoint)",
            "legendFormat": "{{endpoint}}"
          }
        ],
        "gridPos": {"h": 8, "w": 12, "x": 0, "y": 0}
      },
      {
        "id": 2,
        "title": "Response Time (P95)",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, sum(rate(http_request_duration_seconds_bucket[5m])) by (le, endpoint))",
            "legendFormat": "P95 - {{endpoint}}"
          }
        ],
        "gridPos": {"h": 8, "w": 12, "x": 12, "y": 0}
      },
      {
        "id": 3,
        "title": "Model Prediction Latency",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, sum(rate(model_prediction_duration_seconds_bucket[5m])) by (le, model_name))",
            "legendFormat": "P95 - {{model_name}}"
          }
        ],
        "gridPos": {"h": 8, "w": 24, "x": 0, "y": 8}
      },
      {
        "id": 4,
        "title": "System Resources",
        "type": "stat",
        "targets": [
          {
            "expr": "system_cpu_usage_percent",
            "legendFormat": "CPU %"
          }
        ],
        "gridPos": {"h": 4, "w": 6, "x": 0, "y": 16}
      },
      {
        "id": 5,
        "title": "Memory Usage",
        "type": "stat",
        "targets": [
          {
            "expr": "system_memory_usage_percent",
            "legendFormat": "Memory %"
          }
        ],
        "gridPos": {"h": 4, "w": 6, "x": 6, "y": 16}
      }
    ],
    "time": {"from": "now-1h", "to": "now"},
    "refresh": "5s"
  }
}
```

#### Alertas con Prometheus:
```yaml
# alert_rules.yml
groups:
- name: ml_api_alerts
  rules:
  - alert: HighErrorRate
    expr: sum(rate(http_requests_total{status_code=~"5.."}[5m])) / sum(rate(http_requests_total[5m])) > 0.05
    for: 5m
    labels:
      severity: critical
    annotations:
      summary: "High error rate detected"
      description: "Error rate is {{ $value | humanizePercentage }} (threshold: 5%)"

  - alert: HighLatency
    expr: histogram_quantile(0.95, sum(rate(http_request_duration_seconds_bucket[5m])) by (le, endpoint)) > 1
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "High latency detected"
      description: "P95 latency is {{ $value }}s (threshold: 1s)"

  - alert: ModelPredictionFailure
    expr: sum(rate(model_predictions_total{status="error"}[5m])) > 0.1
    for: 2m
    labels:
      severity: critical
    annotations:
      summary: "Model prediction failures"
      description: "Model prediction error rate is {{ $value | humanizePercentage }}"

  - alert: HighCPUUsage
    expr: system_cpu_usage_percent > 80
    for: 10m
    labels:
      severity: warning
    annotations:
      summary: "High CPU usage"
      description: "CPU usage is {{ $value }}% (threshold: 80%)"

  - alert: HighMemoryUsage
    expr: system_memory_usage_percent > 85
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "High memory usage"
      description: "Memory usage is {{ $value }}% (threshold: 85%)"
```

#### OpenTelemetry Integration:
```python
# tracing.py
from opentelemetry import trace
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.requests import RequestsInstrumentor

def setup_tracing(service_name: str):
    # Configurar tracer provider
    trace.set_tracer_provider(TracerProvider())
    tracer = trace.get_tracer(__name__)
    
    # Configurar exportador Jaeger
    jaeger_exporter = JaegerExporter(
        agent_host_name="localhost",
        agent_port=6831,
    )
    
    # Añadir procesador de span
    span_processor = BatchSpanProcessor(jaeger_exporter)
    trace.get_tracer_provider().add_span_processor(span_processor)
    
    # Instrumentar FastAPI y requests
    FastAPIInstrumentor.instrument()
    RequestsInstrumentor.instrument()
    
    return tracer

# Decorador para spans personalizados
def trace_operation(operation_name: str):
    def decorator(func):
        def wrapper(*args, **kwargs):
            tracer = trace.get_tracer(__name__)
            with tracer.start_as_current_span(operation_name) as span:
                # Añadir atributos al span
                span.set_attribute("operation.name", operation_name)
                span.set_attribute("service.name", "ml-api")
                
                try:
                    result = func(*args, **kwargs)
                    span.set_attribute("operation.success", True)
                    return result
                except Exception as e:
                    span.set_attribute("operation.success", False)
                    span.set_attribute("operation.error", str(e))
                    raise
        return wrapper
    return decorator

# Uso en la aplicación
from fastapi import FastAPI

app = FastAPI()

# Configurar tracing
tracer = setup_tracing("ml-api-service")

@app.post("/predict")
@trace_operation("model_prediction")
async def predict(data: dict):
    # Lógica de predicción
    prediction = model.predict(data['input'])
    return {"prediction": prediction}
```

---

**Última Actualización**: Febrero 2026  
**Versión**: 1.0  
**Duración Estimada**: 10 horas de estudio teórico
