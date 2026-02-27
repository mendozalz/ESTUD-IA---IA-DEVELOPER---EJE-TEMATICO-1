"""
Caso de Uso 3 - Monitoreo en Tiempo Real con OpenTelemetry
Fase 1: Instrumentación Completa de API Médica
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import time
import numpy as np
import tensorflow as tf
from PIL import Image
import io
import base64
import json
import logging
from datetime import datetime

# OpenTelemetry imports
from opentelemetry import trace, metrics
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import (
    ConsoleMetricExporter,
    PeriodicExportingMetricReader,
)
from opentelemetry.exporter.prometheus import PrometheusMetricReader
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.requests import RequestsInstrumentor
from opentelemetry.sdk.resources import Resource
from opentelemetry.semconv.resource import ResourceAttributes

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MedicalImageClassifier:
    """
    Clase para clasificación de imágenes médicas con monitoreo completo
    """
    
    def __init__(self):
        self.model = None
        self.class_names = ['normal', 'pneumonia', 'covid', 'tumor']
        self.confidence_threshold = 0.7
        
    def load_model(self):
        """Cargar modelo pre-entrenado"""
        try:
            # En producción, cargar modelo real
            # self.model = tf.keras.models.load_model('models/medical_classifier.h5')
            
            # Para demo, crear un modelo dummy
            self.model = self._create_demo_model()
            logger.info("Modelo cargado exitosamente")
            return True
        except Exception as e:
            logger.error(f"Error cargando modelo: {e}")
            return False
    
    def _create_demo_model(self):
        """Crear modelo demo para clasificación"""
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(224, 224, 3)),
            tf.keras.layers.Conv2D(32, 3, activation='relu'),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(64, 3, activation='relu'),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(len(self.class_names), activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def preprocess_image(self, image_data: bytes) -> np.ndarray:
        """Preprocesar imagen para el modelo"""
        try:
            # Decodificar imagen
            image = Image.open(io.BytesIO(image_data))
            
            # Convertir a RGB si es necesario
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Redimensionar
            image = image.resize((224, 224))
            
            # Normalizar
            image_array = np.array(image) / 255.0
            
            # Expandir dimensiones para batch
            image_array = np.expand_dims(image_array, axis=0)
            
            return image_array
            
        except Exception as e:
            logger.error(f"Error preprocesando imagen: {e}")
            raise HTTPException(status_code=400, detail="Error procesando imagen")
    
    def predict(self, image_array: np.ndarray) -> Dict[str, Any]:
        """Realizar predicción"""
        try:
            # Realizar predicción
            predictions = self.model.predict(image_array, verbose=0)
            
            # Obtener clase predicha y confianza
            predicted_class_idx = np.argmax(predictions[0])
            confidence = float(np.max(predictions[0]))
            predicted_class = self.class_names[predicted_class_idx]
            
            # Obtener probabilidades para todas las clases
            class_probabilities = {
                self.class_names[i]: float(predictions[0][i])
                for i in range(len(self.class_names))
            }
            
            return {
                "predicted_class": predicted_class,
                "confidence": confidence,
                "class_probabilities": class_probabilities,
                "prediction_id": f"pred_{int(time.time())}",
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error en predicción: {e}")
            raise HTTPException(status_code=500, detail="Error en predicción")

class MonitoringSystem:
    """
    Sistema de monitoreo con OpenTelemetry y Prometheus
    """
    
    def __init__(self, service_name: str = "medical-classifier"):
        self.service_name = service_name
        self.setup_opentelemetry()
        
    def setup_opentelemetry(self):
        """Configurar OpenTelemetry"""
        # Configurar recurso
        resource = Resource.create(
            attributes={
                ResourceAttributes.SERVICE_NAME: self.service_name,
                ResourceAttributes.SERVICE_VERSION: "1.0.0",
                "environment": "production"
            }
        )
        
        # Configurar tracing
        trace_provider = TracerProvider(resource=resource)
        trace_provider.add_span_processor(
            BatchSpanProcessor(ConsoleSpanExporter())
        )
        trace.set_tracer_provider(trace_provider)
        
        # Configurar métricas
        meter_provider = MeterProvider(resource=resource)
        
        # Exportador para Prometheus
        prometheus_reader = PrometheusMetricReader()
        meter_provider.add_metric_reader(prometheus_reader)
        
        # Exportador para consola (debug)
        console_reader = PeriodicExportingMetricReader(
            ConsoleMetricExporter(),
            export_interval_millis=10000
        )
        meter_provider.add_metric_reader(console_reader)
        
        # Obtener tracer y meter
        self.tracer = trace.get_tracer(__name__)
        self.meter = meter_provider.get_meter(__name__)
        
        # Crear métricas personalizadas
        self.create_metrics()
        
        # Instrumentar FastAPI y Requests
        self.instrument_libraries()
        
        logger.info("OpenTelemetry configurado exitosamente")
    
    def create_metrics(self):
        """Crear métricas personalizadas"""
        # Contadores
        self.request_counter = self.meter.create_counter(
            name="http_requests_total",
            description="Total number of HTTP requests"
        )
        
        self.prediction_counter = self.meter.create_counter(
            name="predictions_total",
            description="Total number of predictions"
        )
        
        self.error_counter = self.meter.create_counter(
            name="http_errors_total",
            description="Total number of HTTP errors"
        )
        
        # Histogramas
        self.request_duration = self.meter.create_histogram(
            name="http_request_duration_seconds",
            description="HTTP request duration in seconds",
            unit="s"
        )
        
        self.prediction_confidence = self.meter.create_histogram(
            name="prediction_confidence",
            description="Prediction confidence scores",
            unit="1"
        )
        
        # Gauges
        self.active_requests = self.meter.create_up_down_counter(
            name="http_active_requests",
            description="Number of active HTTP requests"
        )
        
        self.model_accuracy = self.meter.create_gauge(
            name="model_accuracy",
            description="Current model accuracy",
            unit="1"
        )
        
        logger.info("Métricas personalizadas creadas")
    
    def instrument_libraries(self):
        """Instrumentar bibliotecas automáticamente"""
        # Instrumentar FastAPI
        FastAPIInstrumentor().instrument()
        
        # Instrumentar Requests
        RequestsInstrumentor().instrument()
        
        logger.info("Bibliotecas instrumentadas")
    
    def record_request_start(self, method: str, endpoint: str):
        """Registrar inicio de solicitud"""
        self.active_requests.add(1)
        return time.time()
    
    def record_request_end(self, start_time: float, method: str, endpoint: str, status_code: int):
        """Registrar fin de solicitud"""
        duration = time.time() - start_time
        
        # Actualizar contadores
        self.request_counter.add(1, {
            "method": method,
            "endpoint": endpoint,
            "status_code": str(status_code)
        })
        
        if status_code >= 400:
            self.error_counter.add(1, {
                "method": method,
                "endpoint": endpoint,
                "status_code": str(status_code)
            })
        
        # Registrar duración
        self.request_duration.record(duration, {
            "method": method,
            "endpoint": endpoint
        })
        
        self.active_requests.add(-1)
        
        return duration
    
    def record_prediction(self, prediction_result: Dict[str, Any]):
        """Registrar predicción"""
        self.prediction_counter.add(1, {
            "predicted_class": prediction_result["predicted_class"],
            "confidence_range": self._get_confidence_range(prediction_result["confidence"])
        })
        
        # Registrar confianza
        self.prediction_confidence.record(prediction_result["confidence"], {
            "predicted_class": prediction_result["predicted_class"]
        })
    
    def _get_confidence_range(self, confidence: float) -> str:
        """Obtener rango de confianza"""
        if confidence >= 0.9:
            return "high"
        elif confidence >= 0.7:
            return "medium"
        else:
            return "low"
    
    def create_span(self, name: str, attributes: Optional[Dict[str, Any]] = None):
        """Crear span de tracing"""
        return self.tracer.start_as_current_span(name, attributes=attributes)

# Modelos Pydantic
class ImageRequest(BaseModel):
    image_data: str  # Base64 encoded image
    patient_id: Optional[str] = None
    study_id: Optional[str] = None

class PredictionResponse(BaseModel):
    predicted_class: str
    confidence: float
    class_probabilities: Dict[str, float]
    prediction_id: str
    timestamp: str
    processing_time_ms: float

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    uptime_seconds: float
    active_requests: int

# Inicializar componentes
classifier = MedicalImageClassifier()
monitoring = MonitoringSystem()

# Crear aplicación FastAPI
app = FastAPI(
    title="Medical Image Classification API",
    description="API para clasificación de imágenes médicas con monitoreo completo",
    version="1.0.0"
)

# Variables globales
start_time = time.time()

@app.on_event("startup")
async def startup_event():
    """Evento de inicio de la aplicación"""
    logger.info("Iniciando API de clasificación médica...")
    
    # Cargar modelo
    if not classifier.load_model():
        raise RuntimeError("No se pudo cargar el modelo")
    
    logger.info("API iniciada exitosamente")

@app.on_event("shutdown")
async def shutdown_event():
    """Evento de cierre de la aplicación"""
    logger.info("Cerrando API de clasificación médica...")

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    uptime = time.time() - start_time
    
    return HealthResponse(
        status="healthy",
        model_loaded=classifier.model is not None,
        uptime_seconds=uptime,
        active_requests=0  # En producción, obtener de métricas
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict_image(request: ImageRequest):
    """Endpoint principal de predicción"""
    request_start = monitoring.record_request_start("POST", "/predict")
    
    with monitoring.create_span("image_prediction", {
        "patient_id": request.patient_id,
        "study_id": request.study_id
    }):
        try:
            # Decodificar imagen base64
            image_data = base64.b64decode(request.image_data)
            
            # Preprocesar imagen
            with monitoring.create_span("image_preprocessing"):
                image_array = classifier.preprocess_image(image_data)
            
            # Realizar predicción
            with monitoring.create_span("model_inference"):
                prediction_result = classifier.predict(image_array)
            
            # Registrar métricas
            monitoring.record_prediction(prediction_result)
            
            # Calcular tiempo de procesamiento
            processing_time = (time.time() - request_start) * 1000
            prediction_result["processing_time_ms"] = processing_time
            
            # Registrar duración de la solicitud
            monitoring.record_request_end(request_start, "POST", "/predict", 200)
            
            logger.info(f"Predicción exitosa: {prediction_result['predicted_class']} "
                       f"(confianza: {prediction_result['confidence']:.3f})")
            
            return prediction_result
            
        except HTTPException:
            # Re-lanzar excepciones HTTP
            monitoring.record_request_end(request_start, "POST", "/predict", 400)
            raise
        except Exception as e:
            logger.error(f"Error en predicción: {e}")
            monitoring.record_request_end(request_start, "POST", "/predict", 500)
            raise HTTPException(status_code=500, detail="Error interno del servidor")

@app.get("/metrics")
async def metrics_endpoint():
    """Endpoint para métricas de Prometheus"""
    # OpenTelemetry Prometheus exporter expone automáticamente en /metrics
    # Este endpoint es para compatibilidad con herramientas que esperan /metrics
    from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
    return generate_latest(), 200, {'Content-Type': CONTENT_TYPE_LATEST}

@app.get("/predictions/stats")
async def get_prediction_stats():
    """Obtener estadísticas de predicciones"""
    # En producción, obtener de base de datos o sistema de almacenamiento
    stats = {
        "total_predictions": 1000,  # Ejemplo
        "average_confidence": 0.85,
        "class_distribution": {
            "normal": 400,
            "pneumonia": 300,
            "covid": 200,
            "tumor": 100
        },
        "last_24h_predictions": 150,
        "uptime_hours": (time.time() - start_time) / 3600
    }
    
    return stats

@app.get("/model/info")
async def get_model_info():
    """Obtener información del modelo"""
    return {
        "model_name": "Medical Image Classifier",
        "version": "1.0.0",
        "classes": classifier.class_names,
        "input_shape": (224, 224, 3),
        "confidence_threshold": classifier.confidence_threshold,
        "model_loaded": classifier.model is not None,
        "last_updated": datetime.now().isoformat()
    }

if __name__ == "__main__":
    import uvicorn
    
    logger.info("Iniciando servidor de API médica...")
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
