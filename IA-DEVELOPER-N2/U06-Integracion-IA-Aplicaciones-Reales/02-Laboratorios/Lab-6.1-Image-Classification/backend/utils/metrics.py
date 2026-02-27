from prometheus_client import Counter, Histogram, Gauge, start_http_server
import time
import logging
from typing import Dict, Any
from functools import wraps

logger = logging.getLogger(__name__)

# Métricas de API
REQUEST_COUNT = Counter(
    'api_requests_total',
    'Total number of API requests',
    ['method', 'endpoint', 'status_code']
)

REQUEST_LATENCY = Histogram(
    'api_request_duration_seconds',
    'API request duration in seconds',
    ['method', 'endpoint']
)

PREDICTION_COUNT = Counter(
    'model_predictions_total',
    'Total number of model predictions',
    ['model', 'status']
)

MODEL_ACCURACY = Gauge(
    'model_accuracy',
    'Current model accuracy',
    ['model']
)

ACTIVE_CONNECTIONS = Gauge(
    'active_connections',
    'Number of active connections'
)

# Métricas del sistema
CPU_USAGE = Gauge(
    'cpu_usage_percent',
    'CPU usage percentage'
)

MEMORY_USAGE = Gauge(
    'memory_usage_percent',
    'Memory usage percentage'
)

DISK_USAGE = Gauge(
    'disk_usage_percent',
    'Disk usage percentage'
)

# Métricas de imágenes
IMAGE_SIZE_BYTES = Histogram(
    'image_size_bytes',
    'Size of uploaded images in bytes',
    buckets=[1024, 4096, 16384, 65536, 262144, 1048576, 4194304, 16777216]
)

IMAGE_PROCESSING_TIME = Histogram(
    'image_processing_duration_seconds',
    'Time taken to process images',
    ['operation']
)

PREDICTION_CONFIDENCE = Histogram(
    'prediction_confidence',
    'Confidence scores of predictions',
    ['model', 'class_name']
)

class MetricsCollector:
    def __init__(self):
        self.start_time = time.time()
        
    def record_request(self, method: str, endpoint: str, status_code: int, duration: float):
        """Registrar una petición API"""
        REQUEST_COUNT.labels(
            method=method,
            endpoint=endpoint,
            status_code=status_code
        ).inc()
        
        REQUEST_LATENCY.labels(
            method=method,
            endpoint=endpoint
        ).observe(duration)
    
    def record_prediction(self, model: str, status: str, confidence: float = None):
        """Registrar una predicción"""
        PREDICTION_COUNT.labels(
            model=model,
            status=status
        ).inc()
        
        if confidence is not None:
            PREDICTION_CONFIDENCE.labels(
                model=model,
                class_name='predicted'
            ).observe(confidence)
    
    def record_image_processing(self, operation: str, duration: float, size_bytes: int):
        """Registrar procesamiento de imagen"""
        IMAGE_PROCESSING_TIME.labels(operation=operation).observe(duration)
        IMAGE_SIZE_BYTES.observe(size_bytes)
    
    def update_system_metrics(self):
        """Actualizar métricas del sistema"""
        try:
            import psutil
            
            # CPU
            cpu_percent = psutil.cpu_percent(interval=1)
            CPU_USAGE.set(cpu_percent)
            
            # Memory
            memory = psutil.virtual_memory()
            MEMORY_USAGE.set(memory.percent)
            
            # Disk
            disk = psutil.disk_usage('/')
            DISK_USAGE.set(disk.percent)
            
        except Exception as e:
            logger.error(f"Error updating system metrics: {e}")
    
    def update_model_accuracy(self, model: str, accuracy: float):
        """Actualizar accuracy del modelo"""
        MODEL_ACCURACY.labels(model=model).set(accuracy)
    
    def increment_active_connections(self):
        """Incrementar conexiones activas"""
        ACTIVE_CONNECTIONS.inc()
    
    def decrement_active_connections(self):
        """Decrementar conexiones activas"""
        ACTIVE_CONNECTIONS.dec()
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Obtener resumen de métricas"""
        try:
            import psutil
            
            return {
                "uptime_seconds": time.time() - self.start_time,
                "system": {
                    "cpu_percent": psutil.cpu_percent(),
                    "memory_percent": psutil.virtual_memory().percent,
                    "disk_percent": psutil.disk_usage('/').percent
                },
                "api": {
                    "total_requests": REQUEST_COUNT._value._value,
                    "total_predictions": PREDICTION_COUNT._value._value,
                    "active_connections": ACTIVE_CONNECTIONS._value._value
                }
            }
        except Exception as e:
            logger.error(f"Error getting metrics summary: {e}")
            return {}

def track_metrics(metric_type: str = 'api'):
    """Decorador para trackear métricas automáticamente"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            
            try:
                result = await func(*args, **kwargs)
                duration = time.time() - start_time
                
                if metric_type == 'api':
                    # Extraer información de la petición
                    request = kwargs.get('request') or args[0] if args else None
                    if request:
                        method = request.method
                        endpoint = request.url.path
                        status_code = 200  # Por defecto para éxito
                        
                        REQUEST_COUNT.labels(
                            method=method,
                            endpoint=endpoint,
                            status_code=status_code
                        ).inc()
                        
                        REQUEST_LATENCY.labels(
                            method=method,
                            endpoint=endpoint
                        ).observe(duration)
                
                return result
                
            except Exception as e:
                duration = time.time() - start_time
                
                if metric_type == 'api':
                    request = kwargs.get('request') or args[0] if args else None
                    if request:
                        method = request.method
                        endpoint = request.url.path
                        status_code = 500
                        
                        REQUEST_COUNT.labels(
                            method=method,
                            endpoint=endpoint,
                            status_code=status_code
                        ).inc()
                        
                        REQUEST_LATENCY.labels(
                            method=method,
                            endpoint=endpoint
                        ).observe(duration)
                
                raise
        
        return wrapper
    return decorator

def start_metrics_server(port: int = 8001):
    """Iniciar servidor de métricas"""
    try:
        start_http_server(port)
        logger.info(f"Metrics server started on port {port}")
    except Exception as e:
        logger.error(f"Failed to start metrics server: {e}")

# Inicializar colector de métricas
metrics_collector = MetricsCollector()
