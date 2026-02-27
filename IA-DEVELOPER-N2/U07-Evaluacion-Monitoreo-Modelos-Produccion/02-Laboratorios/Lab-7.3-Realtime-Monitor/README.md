# Laboratorio 7.3: Monitoreo en Tiempo Real con OpenTelemetry y Prometheus

## 🎯 Contexto
Un modelo de clasificación de imágenes médicas en producción requiere monitoreo en tiempo real de latencia de predicciones, precisión por clase, y uso de recursos (CPU, memoria).

## 🎯 Objetivos

- Instrumentar la API con OpenTelemetry
- Configurar Prometheus para recopilar métricas
- Crear dashboards en Grafana para visualización

## 📋 Marco Lógico del Laboratorio

| Componente | Objetivo | Indicadores | Medios de Verificación |
|------------|----------|-------------|------------------------|
| **Instrumentación** | Implementar tracing y métricas | API instrumentada con OpenTelemetry | Código con spans y métricas |
| **Recolección** | Configurar Prometheus para scraping | Métricas disponibles en /metrics | Configuración de Prometheus |
| **Visualización** | Crear dashboards en Grafana | Dashboards funcionales con datos en tiempo real | Configuración de Grafana |
| **Alertas** | Configurar notificaciones automáticas | Alertas activas para umbrales críticos | Reglas de alerta configuradas |

## 🛠️ Tecnologías Utilizadas

- **OpenTelemetry**: Instrumentación distribuida
- **Prometheus**: Recolección de métricas
- **Grafana**: Visualización y alertas
- **FastAPI**: API para servir el modelo
- **TensorFlow**: Modelo de clasificación médica
- **Docker**: Contenerización

## 📁 Estructura del Laboratorio

```
Laboratorio 7.3 - Monitoreo en Tiempo Real con OpenTelemetry y Prometheus/
├── README.md                           # Esta guía
├── requirements.txt                     # Dependencias Python
├── otel_instrumentation.py             # API instrumentada con OpenTelemetry
├── prometheus.yml                      # Configuración de Prometheus
├── grafana_dashboard.json             # Dashboard de Grafana
├── docker-compose.yml                  # Orquestación de servicios
├── models/                            # Modelos de clasificación
│   └── medical_classifier.h5          # Modelo de imágenes médicas
├── data/                              # Datos de ejemplo
│   ├── medical_images/                # Imágenes médicas de prueba
│   └── test_samples.json             # Metadatos de prueba
├── configs/                           # Configuraciones
│   ├── prometheus/                    # Configuración de Prometheus
│   └── grafana/                       # Dashboards y alertas
└── outputs/                           # Resultados generados
    ├── traces/                        # Logs de tracing
    └── metrics/                       # Exportación de métricas
```

## 🚀 Implementación Paso a Paso

### Paso 1: Configuración del Entorno

```bash
pip install opentelemetry-sdk==1.20.0 opentelemetry-exporter-prometheus==1.20.0 prometheus-client==0.17.1 fastapi==0.103.0 tensorflow==2.15.0 pillow==10.0.0 numpy==1.24.0
```

### Paso 2: Instrumentación con OpenTelemetry

El script `otel_instrumentation.py` implementa una API completamente instrumentada:

```python
from fastapi import FastAPI
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import (
    ConsoleMetricExporter,
    PeriodicExportingMetricReader,
)
from opentelemetry.exporter.prometheus import PrometheusMetricReader
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
import time

# Configurar OpenTelemetry
trace_provider = TracerProvider()
trace_provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
trace.set_tracer_provider(trace_provider)

meter_provider = MeterProvider()
meter_provider.add_metric_reader(
    PeriodicExportingMetricReader(ConsoleMetricExporter(), 5000)
)
meter_provider.add_metric_reader(PrometheusMetricReader())

app = FastAPI()
FastAPIInstrumentor.instrument_app(app)
tracer = trace.get_tracer(__name__)
meter = meter_provider.get_meter(__name__)

# Métricas personalizadas
request_counter = meter.create_counter(
    name="request_count",
    description="Number of requests"
)
latency_histogram = meter.create_histogram(
    name="request_latency_seconds",
    description="Request latency in seconds"
)
prediction_counter = meter.create_counter(
    name="prediction_count",
    description="Number of predictions by class",
)

@app.post("/predict")
async def predict(image: dict):
    start_time = time.time()
    with tracer.start_as_current_span("prediction"):
        request_counter.add(1, {"route": "/predict"})

        # Simular predicción (en producción: model.predict())
        time.sleep(0.1)  # Simular procesamiento
        predicted_class = "tumor" if image["class_id"] % 2 == 0 else "normal"

        latency_histogram.record(time.time() - start_time, {"route": "/predict"})
        prediction_counter.add(1, {"class": predicted_class})

        return {"class": predicted_class, "confidence": 0.95}
```

### Paso 3: Configuración de Prometheus

Archivo `prometheus.yml`:

```yaml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'fastapi'
    scrape_interval: 5s
    static_configs:
      - targets: ['localhost:8000']
        labels:
          service: 'medical-image-classifier'
```

### Paso 4: Dashboard en Grafana

Archivo `grafana_dashboard.json`:

```json
{
  "title": "Monitoreo de Modelo de Clasificación Médica",
  "panels": [
    {
      "title": "Solicitudes por Segundo",
      "type": "graph",
      "targets": [
        {
          "expr": "rate(request_count[1m])",
          "legendFormat": "{{route}}"
        }
      ]
    },
    {
      "title": "Latencia de Predicciones (P95)",
      "type": "stat",
      "targets": [
        {
          "expr": "histogram_quantile(0.95, sum(rate(request_latency_seconds_bucket[5m])) by (le, route))",
          "legendFormat": ""
        }
      ]
    },
    {
      "title": "Predicciones por Clase",
      "type": "piechart",
      "targets": [
        {
          "expr": "sum(prediction_count) by (class)",
          "legendFormat": "{{class}}"
        }
      ]
    }
  ]
}
```

### Paso 5: Orquestación con Docker Compose

Archivo `docker-compose.yml`:

```yaml
version: '3.8'

services:
  api:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./models:/app/models
      - ./data:/app/data
    environment:
      - OTEL_EXPORTER_PROMETHEUS_PORT=9464
      - OTEL_RESOURCE_ATTRIBUTES=service.name=medical-classifier
    depends_on:
      - prometheus

  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./configs/prometheus:/etc/prometheus
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    volumes:
      - grafana_data:/var/lib/grafana
      - ./configs/grafana:/etc/grafana/provisioning
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    depends_on:
      - prometheus

volumes:
  prometheus_data:
  grafana_data:
```

## 📊 Resultados Esperados

### Sistema de Monitoreo Completo
- **API instrumentada** con OpenTelemetry en http://localhost:8000
- **Métricas Prometheus** disponibles en http://localhost:9090/metrics
- **Prometheus UI** accesible en http://localhost:9090
- **Grafana Dashboard** accesible en http://localhost:3000
- **Tracing distribuido** funcionando con spans detallados

### Métricas Monitoreadas
- **Request Count**: Número total de solicitudes por endpoint
- **Request Latency**: Histograma de latencias con percentiles
- **Prediction Count**: Contador de predicciones por clase
- **Error Rate**: Tasa de errores del sistema
- **Resource Usage**: CPU y memoria del contenedor

### Visualizaciones Disponibles
- **Time Series**: Gráficos de métricas en tiempo real
- **Histograms**: Distribución de latencias
- **Pie Charts**: Distribución de predicciones por clase
- **Heatmaps**: Patrones de uso por hora del día

## 🔍 Análisis e Interpretación

### Métricas Clave de Rendimiento

1. **Latencia (P95)**: Percentil 95 del tiempo de respuesta
   - < 100ms: Excelente
   - 100-500ms: Aceptable
   - > 500ms: Necesita optimización

2. **Throughput**: Solicitudes por segundo
   - Depende del caso de uso
   - Monitorear picos y valles

3. **Error Rate**: Tasa de errores
   - < 1%: Excelente
   - 1-5%: Aceptable
   - > 5%: Crítico

### Patrones de Uso

- **Horas pico**: Mayor carga durante horarios de consulta
- **Distribución por clase**: Balance en predicciones
- **Tendencias temporales**: Evolución del rendimiento

### Alertas Configuradas

| Métrica | Umbral | Severidad | Acción |
|-----------|---------|------------|----------|
| Latencia P95 > 1s | 1000ms | Alta | Escalar horizontalmente |
| Error Rate > 5% | 5% | Crítica | Investigar inmediatamente |
| CPU > 80% | 80% | Media | Optimizar código |
| Memory > 85% | 85% | Alta | Reiniciar servicio |

## 📌 Entregables del Laboratorio

| Entregable | Descripción | Formato |
|-------------|-------------|-----------|
| Instrumentación con OTel | Código para rastrear métricas con OpenTelemetry | otel_instrumentation.py |
| Configuración de Prometheus | Archivo de configuración para scraping | prometheus.yml |
| Dashboard de Grafana | Visualización de métricas en tiempo real | grafana_dashboard.json |
| Orquestación | Docker Compose para despliegue completo | docker-compose.yml |
| Documentación | Guía para configurar el monitoreo | README.md |

## 🎯 Criterios de Evaluación

- **Funcionalidad** (40%): Sistema completo funcionando correctamente
- **Instrumentación** (25%): Métricas y tracing implementados
- **Visualización** (20%): Dashboards claros e informativos
- **Monitoreo** (15%): Sistema de alertas configurado

## 🚀 Extensión y Mejoras

### Mejoras Sugeridas
1. **Métricas Avanzadas**: Incluir métricas de negocio específicas
2. **Alertas Inteligentes**: Machine learning para detección de anomalías
3. **Distributed Tracing**: Seguimiento entre múltiples microservicios
4. **Auto-scaling**: Escalado automático basado en métricas

### Aplicaciones en Producción
1. **SLA Monitoring**: Monitoreo de acuerdos de nivel de servicio
2. **Capacity Planning**: Planificación de capacidad basada en tendencias
3. **Cost Optimization**: Optimización de recursos basada en uso
4. **Compliance Monitoring**: Monitoreo de requisitos regulatorios

---

**Duración Estimada**: 8-10 horas  
**Nivel de Dificultad**: Intermedia  
**Prerrequisitos**: Conocimientos de APIs, sistemas de monitoreo, contenerización
