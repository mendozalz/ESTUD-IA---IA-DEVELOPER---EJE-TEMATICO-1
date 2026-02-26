# Laboratorio 6.1: Despliegue de Clasificación de Imágenes con FastAPI y React

## 🎯 Objetivos del Laboratorio

### Objetivo General
Crear una aplicación web completa para clasificación de productos de retail integrando un modelo de IA con FastAPI y React, aplicando la metodología Windsor para la gestión del proyecto.

### Objetivos Específicos
- Desarrollar una API RESTful con FastAPI para servir un modelo de clasificación de imágenes
- Crear una interfaz interactiva con React para subir imágenes y visualizar resultados
- Implementar la metodología Windsor para gestionar el ciclo de vida del proyecto
- Desplegar la aplicación en contenedores Docker y Kubernetes
- Configurar monitoreo con Prometheus y Grafana

## 📋 Marco Lógico del Proyecto

| Componente | Objetivo | Indicadores | Medios de Verificación |
|------------|----------|-------------|------------------------|
| **Planificación Windsor** | Estructurar el proyecto con metodología ágil | Sprints definidos con entregables claros | Documento windsor_plan.md con fases y objetivos |
| **API FastAPI** | Servir el modelo de clasificación de imágenes | Endpoints funcionales con validación | Código app.py con pruebas unitarias |
| **Front-end React** | Interfaz para subir imágenes y mostrar resultados | Componentes reutilizables y responsivos | Código React con tests de integración |
| **Contenerización** | Empaquetar la aplicación para despliegue | Imágenes Docker optimizadas | Dockerfile y docker-compose.yml |
| **Despliegue K8s** | Escalar la aplicación en producción | Pods funcionando con autoescalado | Configuración deployment.yaml |
| **Monitoreo** | Observabilidad de la aplicación en producción | Métricas y alertas configuradas | Dashboards de Grafana y reglas de Prometheus |

## 🛠️ Tecnologías y Herramientas

### Principales Tecnologías
- **FastAPI 2026.3**: Framework moderno para APIs
- **React 18.2**: Biblioteca para interfaces de usuario
- **TensorFlow 2.15**: Modelo de clasificación de imágenes
- **Docker 24.0**: Contenerización de aplicaciones
- **Kubernetes 1.28**: Orquestación en producción

### Dependencias Adicionales
- **Prometheus 3.0**: Monitoreo de métricas
- **Grafana 10.0**: Visualización de dashboards
- **Nginx**: Servidor web y balanceador de carga
- **Redis**: Caching de predicciones

## 📁 Estructura del Proyecto

```
Laboratorio 6.1 - Clasificación de Imágenes/
├── README.md                           # Guía del laboratorio
├── windsor_plan.md                    # Planificación con metodología Windsor
├── requirements.txt                    # Dependencias Python
├── docker-compose.yml                 # Configuración local
├── Dockerfile                          # Imagen Docker optimizada
├── k8s/                               # Configuración Kubernetes
│   ├── deployment.yaml                # Despliegue de aplicación
│   ├── service.yaml                   # Servicio de red
│   ├── ingress.yaml                   # Entrada externa
│   └── hpa.yaml                       # Autoescalado horizontal
├── backend/                           # API FastAPI
│   ├── app.py                         # Aplicación principal
│   ├── models/                        # Modelo de ML
│   │   ├── __init__.py
│   │   └── product_classifier.py     # Clasificador de productos
│   ├── utils/                         # Utilidades
│   │   ├── __init__.py
│   │   ├── image_processing.py        # Procesamiento de imágenes
│   │   └── metrics.py                 # Métricas Prometheus
│   └── tests/                         # Tests unitarios
│       ├── test_api.py
│       └── test_models.py
├── frontend/                          # Aplicación React
│   ├── package.json                   # Dependencias Node.js
│   ├── public/                        # Archivos estáticos
│   ├── src/                           # Código fuente
│   │   ├── components/                # Componentes React
│   │   │   ├── ImageUploader.jsx      # Subida de imágenes
│   │   │   ├── PredictionResult.jsx   # Resultados de predicción
│   │   │   └── MetricsDashboard.jsx    # Dashboard de métricas
│   │   ├── services/                  # Servicios API
│   │   │   └── api.js                 # Cliente HTTP
│   │   ├── utils/                     # Utilidades
│   │   │   └── imageUtils.js          # Procesamiento de imágenes
│   │   ├── App.jsx                    # Componente principal
│   │   └── index.js                   # Punto de entrada
│   └── tests/                         # Tests React
│       └── components/
├── monitoring/                        # Configuración de monitoreo
│   ├── prometheus/                    # Configuración Prometheus
│   │   ├── prometheus.yml             # Configuración principal
│   │   └── alert_rules.yml            # Reglas de alerta
│   └── grafana/                       # Dashboards Grafana
│       ├── dashboards/                # Definición de dashboards
│       └── provisioning/               # Scripts de aprovisionamiento
└── docs/                             # Documentación
    ├── api_documentation.md           # Documentación de API
    ├── deployment_guide.md           # Guía de despliegue
    └── troubleshooting.md             # Solución de problemas
```

## 🔧 Implementación Detallada

### Fase 1: Planificación con Metodología Windsor

#### Documento windsor_plan.md:
```markdown
# Planificación del Proyecto de Clasificación de Imágenes (Metodología Windsor)

## Objetivo
Desarrollar una aplicación web para clasificación automática de productos de retail usando IA.

## Stakeholders
- Equipo de IA (desarrollo del modelo y API)
- Desarrolladores Front-end (interfaz React)
- Equipo de Producto (requisitos de negocio)
- Equipo de Operaciones (despliegue y monitoreo)

## Sprints

### Sprint 1: Diseño de API y Front-End Básico (2 semanas)
**Objetivo:** Establecer la arquitectura básica y prototipos iniciales

**Actividades:**
- [ ] Definir endpoints de la API (/predict, /health, /metrics)
- [ ] Diseñar esquema de datos con Pydantic
- [ ] Crear mockups del front-end en React
- [ ] Configurar entorno de desarrollo Docker
- [ ] Implementar pipeline CI/CD básico

**Entregables:**
- Especificación de API con OpenAPI
- Mockups de interfaz de usuario
- Entorno Docker funcional
- Pipeline CI/CD configurado

**Medios de Verificación:**
- Documentación OpenAPI generada automáticamente
- Mockups aprobados por stakeholders
- Tests de integración del entorno

### Sprint 2: Integración del Modelo con la API (2 semanas)
**Objetivo:** Conectar el modelo de clasificación con la API FastAPI

**Actividades:**
- [ ] Implementar carga y preprocesamiento de imágenes
- [ ] Integrar modelo TensorFlow con FastAPI
- [ ] Implementar validación de entradas y manejo de errores
- [ ] Agregar logging y métricas básicas
- [ ] Escribir tests unitarios y de integración

**Entregables:**
- API funcional con endpoint /predict
- Modelo de clasificación integrado
- Suite de tests automatizados
- Documentación de API actualizada

**Medios de Verificación:**
- Tests de API pasando (95%+ cobertura)
- Métricas de rendimiento cumpliendo SLA
- Validación de entradas funcionando

### Sprint 3: Desarrollo del Front-End React (2 semanas)
**Objetivo:** Implementar la interfaz de usuario completa

**Actividades:**
- [ ] Desarrollar componente de subida de imágenes
- [ ] Implementar visualización de resultados
- [ ] Agregar manejo de estado y loading states
- [ ] Implementar diseño responsivo
- [ ] Integrar con API FastAPI

**Entregables:**
- Aplicación React funcional
- Componentes reutilizables
- Tests de componentes
- Documentación de uso

**Medios de Verificación:**
- Tests E2E pasando
- Interfaz responsiva en múltiples dispositivos
- Integración con API funcionando

### Sprint 4: Despliegue y Monitoreo (2 semanas)
**Objetivo:** Desplegar la aplicación en producción con monitoreo completo

**Actividades:**
- [ ] Configurar despliegue en Kubernetes
- [ ] Implementar métricas con Prometheus
- [ ] Configurar dashboards en Grafana
- [ ] Establecer alertas y notificaciones
- [ ] Realizar pruebas de carga y estrés

**Entregables:**
- Aplicación desplegada en Kubernetes
- Dashboards de monitoreo funcionales
- Reglas de alerta configuradas
- Reporte de pruebas de carga

**Medios de Verificación:**
- Aplicación accesible desde internet
- Métricas recolectadas correctamente
- Alertas funcionando
- Pruebas de carga cumpliendo requisitos

## Riesgos y Supuestos
- **Riesgo 1:** El modelo no cumple con la precisión requerida - **Mitigación:** Tener modelos de respaldo y fine-tuning continuo
- **Riesgo 2:** Problemas de rendimiento en producción - **Mitigación:** Implementar caching y optimización temprana
- **Supuesto 1:** El equipo tiene experiencia con FastAPI y React - **Verificación:** Evaluación de habilidades técnicas
- **Supuesto 2:** Infraestructura Kubernetes disponible - **Verificación:** Confirmación con equipo de operaciones

## Métricas de Éxito
- **Precisión del modelo:** >90% en dataset de prueba
- **Latencia de API:** <200ms para predicciones
- **Disponibilidad:** >99.5% uptime
- **Satisfacción del usuario:** >4.5/5 en encuestas
```

### Fase 2: API FastAPI Completa

#### app.py - Aplicación Principal:
```python
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
import time
import asyncio
from typing import List, Optional
import logging

from models.product_classifier import ProductClassifier
from utils.image_processing import ImageProcessor
from utils.metrics import (
    REQUEST_COUNT, REQUEST_LATENCY, PREDICTION_COUNT,
    PREDICTION_LATENCY, measure_prediction_latency
)

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Inicialización de FastAPI
app = FastAPI(
    title="Product Classification API",
    description="API for classifying product images using deep learning",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configurar CORS para React
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://yourdomain.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Servir archivos estáticos (para React build)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Inicializar componentes globales
classifier = ProductClassifier()
image_processor = ImageProcessor()

@app.on_event("startup")
async def startup_event():
    """Inicializar componentes al iniciar la aplicación"""
    logger.info("Starting Product Classification API...")
    
    # Cargar modelo
    try:
        classifier.load_model("models/product_classifier.h5")
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise
    
    # Cargar clases
    try:
        classifier.load_classes("models/classes.txt")
        logger.info("Classes loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load classes: {e}")
        raise

@app.get("/health")
async def health_check():
    """Endpoint de health check"""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "model_loaded": classifier.is_loaded(),
        "version": "1.0.0"
    }

@app.get("/ready")
async def readiness_check():
    """Endpoint de readiness check"""
    if not classifier.is_loaded():
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "status": "ready",
        "model_loaded": classifier.is_loaded(),
        "classes": len(classifier.get_classes()) if classifier.is_loaded() else 0
    }

@app.post("/predict")
@measure_prediction_latency("product_classifier")
async def predict_product(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    confidence_threshold: Optional[float] = 0.5
):
    """
    Clasificar una imagen de producto
    
    Args:
        file: Archivo de imagen a clasificar
        confidence_threshold: Umbral de confianza mínimo
    
    Returns:
        Predicción con clase y confianza
    """
    start_time = time.time()
    
    try:
        # Validar archivo
        if not file.content_type.startswith('image/'):
            raise HTTPException(
                status_code=400, 
                detail="File must be an image"
            )
        
        # Leer y procesar imagen
        contents = await file.read()
        processed_image = await image_processor.process_image(contents)
        
        # Realizar predicción
        prediction = classifier.predict(processed_image)
        
        # Aplicar umbral de confianza
        if prediction['confidence'] < confidence_threshold:
            prediction['class'] = "unknown"
            prediction['confidence'] = prediction['confidence']
        
        # Agregar metadata
        prediction.update({
            "filename": file.filename,
            "size": len(contents),
            "processing_time_ms": (time.time() - start_time) * 1000,
            "model_version": classifier.get_model_version()
        })
        
        # Tarea en background para logging
        background_tasks.add_task(
            log_prediction,
            file.filename,
            prediction['class'],
            prediction['confidence']
        )
        
        return prediction
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict-batch")
async def predict_batch(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    confidence_threshold: Optional[float] = 0.5
):
    """
    Clasificar múltiples imágenes de producto
    
    Args:
        files: Lista de archivos de imagen
        confidence_threshold: Umbral de confianza mínimo
    
    Returns:
        Lista de predicciones
    """
    if len(files) > 10:
        raise HTTPException(
            status_code=400,
            detail="Maximum 10 files allowed per batch"
        )
    
    results = []
    
    for file in files:
        try:
            # Validar archivo
            if not file.content_type.startswith('image/'):
                results.append({
                    "filename": file.filename,
                    "error": "File must be an image",
                    "class": None,
                    "confidence": None
                })
                continue
            
            # Procesar imagen
            contents = await file.read()
            processed_image = await image_processor.process_image(contents)
            
            # Realizar predicción
            prediction = classifier.predict(processed_image)
            
            # Aplicar umbral de confianza
            if prediction['confidence'] < confidence_threshold:
                prediction['class'] = "unknown"
            
            # Agregar metadata
            prediction.update({
                "filename": file.filename,
                "size": len(contents)
            })
            
            results.append(prediction)
            
        except Exception as e:
            logger.error(f"Batch prediction error for {file.filename}: {e}")
            results.append({
                "filename": file.filename,
                "error": str(e),
                "class": None,
                "confidence": None
            })
    
    # Tarea en background para logging
    background_tasks.add_task(
        log_batch_prediction,
        len(files),
        len([r for r in results if r.get('class') and r['class'] != 'unknown'])
    )
    
    return {
        "total_files": len(files),
        "successful_predictions": len([r for r in results if r.get('class')]),
        "results": results
    }

@app.get("/classes")
async def get_classes():
    """Obtener las clases disponibles para clasificación"""
    if not classifier.is_loaded():
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "classes": classifier.get_classes(),
        "total_classes": len(classifier.get_classes()),
        "model_version": classifier.get_model_version()
    }

@app.get("/metrics")
async def metrics():
    """Endpoint para métricas de Prometheus"""
    from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

# Tareas en background
async def log_prediction(filename: str, predicted_class: str, confidence: float):
    """Registrar predicción en logs"""
    logger.info(f"Prediction - File: {filename}, Class: {predicted_class}, Confidence: {confidence:.2f}")

async def log_batch_prediction(total_files: int, successful_predictions: int):
    """Registrar predicción batch en logs"""
    logger.info(f"Batch prediction - Total: {total_files}, Successful: {successful_predictions}")

# Middleware para métricas
@app.middleware("http")
async def add_metrics_middleware(request, call_next):
    start_time = time.time()
    
    response = await call_next(request)
    
    # Registrar métricas
    method = request.method
    endpoint = request.url.path
    status_code = response.status_code
    
    REQUEST_COUNT.labels(method=method, endpoint=endpoint, status_code=status_code).inc()
    REQUEST_LATENCY.labels(method=method, endpoint=endpoint).observe(time.time() - start_time)
    
    return response

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
```

### Fase 3: Front-end React Completo

#### src/components/ImageUploader.jsx:
```jsx
import React, { useState, useCallback } from 'react';
import { Upload, message, Spin, Progress } from 'antd';
import { InboxOutlined } from '@ant-design/icons';
import { api } from '../services/api';

const { Dragger } = Upload;

const ImageUploader = ({ onPredictionComplete, disabled = false }) => {
  const [uploading, setUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [previewImage, setPreviewImage] = useState(null);

  const handleUpload = useCallback(async (file) => {
    // Validar tipo de archivo
    const isImage = file.type.startsWith('image/');
    if (!isImage) {
      message.error('You can only upload image files!');
      return false;
    }

    // Validar tamaño (máximo 10MB)
    const isLt10M = file.size / 1024 / 1024 < 10;
    if (!isLt10M) {
      message.error('Image must smaller than 10MB!');
      return false;
    }

    // Crear preview
    const reader = new FileReader();
    reader.onload = (e) => {
      setPreviewImage(e.target.result);
    };
    reader.readAsDataURL(file);

    return false; // Evitar upload automático de Ant Design
  }, []);

  const handlePredict = useCallback(async (file) => {
    setUploading(true);
    setUploadProgress(0);

    try {
      // Simular progreso de upload
      const progressInterval = setInterval(() => {
        setUploadProgress(prev => {
          if (prev >= 90) {
            clearInterval(progressInterval);
            return 90;
          }
          return prev + 10;
        });
      }, 100);

      const formData = new FormData();
      formData.append('file', file);

      const response = await api.predict(formData);
      
      clearInterval(progressInterval);
      setUploadProgress(100);
      
      // Notificar éxito
      message.success('Image classified successfully!');
      
      // Pasar resultado al componente padre
      if (onPredictionComplete) {
        onPredictionComplete({
          ...response,
          filename: file.name,
          preview: previewImage
        });
      }

    } catch (error) {
      message.error('Failed to classify image: ' + error.message);
      console.error('Prediction error:', error);
    } finally {
      setUploading(false);
      setTimeout(() => setUploadProgress(0), 1000);
    }
  }, [onPredictionComplete, previewImage]);

  const uploadProps = {
    name: 'file',
    multiple: false,
    accept: 'image/*',
    beforeUpload: handleUpload,
    showUploadList: false,
  };

  return (
    <div className="image-uploader">
      <Dragger {...uploadProps} disabled={disabled || uploading}>
        <p className="ant-upload-drag-icon">
          <InboxOutlined />
        </p>
        <p className="ant-upload-text">
          Click or drag image to this area to upload
        </p>
        <p className="ant-upload-hint">
          Support for a single image upload. Strictly prohibit from uploading company data or other
          band files
        </p>
      </Dragger>

      {previewImage && (
        <div className="image-preview" style={{ marginTop: 16 }}>
          <img 
            src={previewImage} 
            alt="Preview" 
            style={{ 
              width: '200px', 
              height: '200px', 
              objectFit: 'cover',
              borderRadius: '8px'
            }} 
          />
        </div>
      )}

      {uploading && (
        <div style={{ marginTop: 16 }}>
          <Spin tip="Classifying image...">
            <div style={{ textAlign: 'center' }}>
              <Progress percent={uploadProgress} status="active" />
              <p style={{ marginTop: 8 }}>Analyzing image with AI...</p>
            </div>
          </Spin>
        </div>
      )}

      {!uploading && previewImage && (
        <div style={{ marginTop: 16, textAlign: 'center' }}>
          <button
            type="primary"
            onClick={() => {
              const fileInput = document.querySelector('input[type="file"]');
              if (fileInput && fileInput.files[0]) {
                handlePredict(fileInput.files[0]);
              }
            }}
            disabled={disabled}
          >
            Classify Image
          </button>
        </div>
      )}
    </div>
  );
};

export default ImageUploader;
```

#### src/components/PredictionResult.jsx:
```jsx
import React from 'react';
import { Card, Statistic, Tag, Progress, Timeline } from 'antd';
import { 
  CheckCircleOutlined, 
  ExclamationCircleOutlined,
  InfoCircleOutlined 
} from '@ant-design/icons';

const PredictionResult = ({ prediction, showDetails = true }) => {
  if (!prediction) {
    return (
      <Card title="No Prediction Yet" style={{ textAlign: 'center' }}>
        <p>Upload an image to see the classification result</p>
      </Card>
    );
  }

  const getConfidenceColor = (confidence) => {
    if (confidence >= 0.8) return '#52c41a';
    if (confidence >= 0.6) return '#faad14';
    return '#ff4d4f';
  };

  const getStatusIcon = (confidence) => {
    if (confidence >= 0.8) return <CheckCircleOutlined style={{ color: '#52c41a' }} />;
    if (confidence >= 0.6) return <ExclamationCircleOutlined style={{ color: '#faad14' }} />;
    return <InfoCircleOutlined style={{ color: '#ff4d4f' }} />;
  };

  return (
    <div className="prediction-result">
      <Card 
        title={
          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            {getStatusIcon(prediction.confidence)}
            Classification Result
          </div>
        }
        extra={
          <Tag color={getConfidenceColor(prediction.confidence)}>
            {(prediction.confidence * 100).toFixed(1)}% Confidence
          </Tag>
        }
      >
        <div style={{ marginBottom: 16 }}>
          <Statistic
            title="Predicted Class"
            value={prediction.class}
            valueStyle={{ 
              fontSize: '24px', 
              fontWeight: 'bold',
              color: '#1890ff'
            }}
          />
        </div>

        <div style={{ marginBottom: 16 }}>
          <p style={{ marginBottom: 8, fontWeight: 'bold' }}>Confidence Level</p>
          <Progress
            percent={prediction.confidence * 100}
            strokeColor={getConfidenceColor(prediction.confidence)}
            format={(percent) => `${percent.toFixed(1)}%`}
          />
        </div>

        {showDetails && (
          <Timeline>
            <Timeline.Item color="green">
              <strong>File:</strong> {prediction.filename}
            </Timeline.Item>
            <Timeline.Item color="blue">
              <strong>Size:</strong> {(prediction.size / 1024).toFixed(2)} KB
            </Timeline.Item>
            <Timeline.Item color="purple">
              <strong>Processing Time:</strong> {prediction.processing_time_ms.toFixed(2)} ms
            </Timeline.Item>
            <Timeline.Item color="orange">
              <strong>Model Version:</strong> {prediction.model_version}
            </Timeline.Item>
          </Timeline>
        )}

        {prediction.class === 'unknown' && (
          <div style={{ marginTop: 16, padding: 12, backgroundColor: '#fff2e8', borderRadius: 6 }}>
            <p style={{ margin: 0, color: '#d46b08' }}>
              <strong>Note:</strong> The confidence level is below the threshold. 
              The image could not be confidently classified.
            </p>
          </div>
        )}
      </Card>
    </div>
  );
};

export default PredictionResult;
```

### Fase 4: Despliegue con Kubernetes

#### k8s/deployment.yaml:
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: product-classifier-api
  labels:
    app: product-classifier
    component: api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: product-classifier
      component: api
  template:
    metadata:
      labels:
        app: product-classifier
        component: api
    spec:
      containers:
      - name: api
        image: product-classifier:latest
        ports:
        - containerPort: 8000
        env:
        - name: MODEL_PATH
          value: "/app/models"
        - name: REDIS_URL
          value: "redis://redis-service:6379"
        - name: LOG_LEVEL
          value: "INFO"
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
          timeoutSeconds: 5
          failureThreshold: 3
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
          timeoutSeconds: 3
          failureThreshold: 3
        volumeMounts:
        - name: model-volume
          mountPath: /app/models
          readOnly: true
      volumes:
      - name: model-volume
        persistentVolumeClaim:
          claimName: model-pvc

---
apiVersion: v1
kind: Service
metadata:
  name: product-classifier-service
  labels:
    app: product-classifier
    component: api
spec:
  selector:
    app: product-classifier
    component: api
  ports:
    - name: http
      port: 80
      targetPort: 8000
      protocol: TCP
  type: ClusterIP

---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: model-pvc
spec:
  accessModes:
    - ReadOnlyMany
  resources:
    requests:
      storage: 1Gi
  storageClassName: standard
```

## 📊 Entregables del Laboratorio

### 1. Código Fuente Completo
- **windsor_plan.md**: Planificación detallada con metodología Windsor
- **backend/app.py**: API FastAPI completa con todas las funcionalidades
- **frontend/src/**: Aplicación React completa con componentes
- **k8s/**: Configuración completa de Kubernetes

### 2. Configuración de Despliegue
- **Dockerfile**: Imagen Docker optimizada multi-stage
- **docker-compose.yml**: Configuración para desarrollo local
- **k8s/**: Manifiestos completos para producción

### 3. Monitoreo y Observabilidad
- **monitoring/prometheus/**: Configuración de Prometheus
- **monitoring/grafana/**: Dashboards y alertas
- **backend/utils/metrics.py**: Implementación de métricas

### 4. Tests y Validación
- **backend/tests/**: Suite de tests unitarios y de integración
- **frontend/tests/**: Tests de componentes React
- **k8s/tests/**: Tests de despliegue

### 5. Documentación
- **docs/api_documentation.md**: Documentación completa de API
- **docs/deployment_guide.md**: Guía paso a paso de despliegue
- **docs/troubleshooting.md**: Solución de problemas comunes

## 🎯 Criterios de Evaluación

### Componente Técnico (60%)
- **Funcionalidad de la API** (20%): Todos los endpoints funcionando correctamente
- **Integración Front-End** (20%): Aplicación React completa y responsiva
- **Despliegue Escalable** (20%): Configuración Kubernetes con autoescalado

### Componente Metodológico (40%)
- **Aplicación de Windsor** (20%): Planificación y ejecución según metodología
- **Monitoreo y Observabilidad** (20%): Métricas, dashboards y alertas funcionales

### Métricas de Éxito
- **Precisión del Modelo**: >90% en dataset de prueba
- **Latencia de API**: <200ms para predicciones individuales
- **Disponibilidad**: >99.5% uptime en producción
- **Cobertura de Tests**: >90% para backend y frontend

## 🚀 Extensiones y Mejoras

### Opciones Avanzadas
1. **Streaming con WebSockets**: Predicciones en tiempo real
2. **Caching Inteligente**: Redis con políticas de expiración
3. **A/B Testing**: Comparación de diferentes versiones de modelos
4. **Edge Computing**: Despliegue en edge locations con Cloudflare Workers

### Aplicaciones en Producción
1. **CI/CD Avanzado**: Pipeline con GitOps y ArgoCD
2. **Security Hardening**: Autenticación OAuth2 y rate limiting
3. **Multi-Region Despliegue**: Alta disponibilidad geográfica
4. **Cost Optimization**: Autoscaling basado en métricas de negocio

---

**Duración Estimada**: 8-10 horas  
**Dificultad**: Intermedia-Avanzada  
**Prerrequisitos**: Conocimientos de FastAPI, React, Docker y Kubernetes
