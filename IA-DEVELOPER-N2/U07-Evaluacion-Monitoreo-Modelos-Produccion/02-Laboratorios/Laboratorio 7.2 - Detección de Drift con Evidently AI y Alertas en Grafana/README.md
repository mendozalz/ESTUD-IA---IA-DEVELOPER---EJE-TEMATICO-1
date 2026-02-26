# Laboratorio 7.2: Detección de Drift con Evidently AI y Alertas en Grafana

## 🎯 Contexto
Un modelo de predicción de demanda en un retailer muestra signos de degradación. Se implementará un sistema para detectar drift en los datos de entrada, generar alertas en Grafana cuando se supere un umbral, y automatizar el reentrenamiento con Airflow.

## 🎯 Objetivos

- Detectar drift en los datos de entrada
- Generar alertas en Grafana cuando se supere un umbral
- Automatizar el reentrenamiento con Airflow

## 📋 Marco Lógico del Laboratorio

| Componente | Objetivo | Indicadores | Medios de Verificación |
|------------|----------|-------------|------------------------|
| **Detección de Drift** | Implementar sistema de detección automática | Drift detectado en <24h | API con endpoints de monitoreo |
| **Alertas en Grafana** | Configurar notificaciones automáticas | Alertas funcionando con <5min delay | Configuración de Grafana y reglas |
| **Reentrenamiento Automático** | Automatizar actualización del modelo | Pipeline ejecutado cuando drift > umbral | DAG de Airflow funcional |
| **Monitoreo Continuo** | Mantener vigilancia constante | Sistema operativo 24/7 | Logs y métricas disponibles |

## 🛠️ Tecnologías Utilizadas

- **Evidently AI**: Detección de drift y monitoreo de modelos
- **FastAPI**: API para servir el modelo y métricas
- **Prometheus**: Recolección de métricas
- **Grafana**: Visualización y alertas
- **Apache Airflow**: Orquestación de pipelines
- **Docker**: Contenerización de servicios

## 📁 Estructura del Laboratorio

```
Laboratorio 7.2 - Detección de Drift con Evidently AI y Alertas en Grafana/
├── README.md                           # Esta guía
├── requirements.txt                     # Dependencias Python
├── drift_api.py                        # API con métricas de drift
├── grafana_alerts.yml                  # Configuración de alertas
├── retraining_pipeline.py               # Pipeline de Airflow
├── docker-compose.yml                   # Orquestación de servicios
├── data/                              # Datos de referencia y producción
│   ├── reference_data.csv             # Datos de referencia históricos
│   ├── current_data.csv               # Datos recientes
│   └── demand_data_sample.csv        # Datos de ejemplo
├── configs/                           # Configuraciones
│   ├── grafana/                       # Dashboards y alertas
│   ├── prometheus/                    # Configuración de Prometheus
│   └── airflow/                       # DAGs y configuración
├── notebooks/                         # Jupyter notebooks
│   └── drift_analysis.ipynb          # Análisis interactivo
└── outputs/                           # Resultados generados
    ├── drift_reports/                 # Reportes de drift
    ├── alerts/                        # Logs de alertas
    └── retraining_logs/               # Logs de reentrenamiento
```

## 🚀 Implementación Paso a Paso

### Paso 1: Configuración del Entorno

```bash
pip install evidently==0.3.1 prometheus-client==0.17.1 fastapi==0.103.0 grafana-api==0.7.0 apache-airflow==2.7.0 pandas==2.0.0 numpy==1.24.0 scikit-learn==1.3.0 uvicorn==0.24.0
```

### Paso 2: API con FastAPI y Métricas de Drift

El script `drift_api.py` implementa una API que sirve un modelo de predicción de demanda y monitorea el drift:

```python
from fastapi import FastAPI, Request
from prometheus_client import make_asgi_app, Gauge
from evidently.report import Report
from evidently.metrics import DatasetDriftMetric, ColumnDriftMetric
import pandas as pd
import numpy as np
import json

app = FastAPI()
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

# Métricas de Prometheus
DRIFT_SCORE = Gauge('dataset_drift_score', 'Dataset drift score')
COLUMN_DRIFT_SCORES = Gauge('column_drift_scores', 'Column drift scores', ['column'])

# Datos de referencia (guardados en producción)
ref_data = pd.read_csv('data/reference_data.csv')

@app.post("/predict")
async def predict(request: Request):
    # Obtener datos de la solicitud
    data = await request.json()
    current_data = pd.DataFrame([data])

    # Calcular drift
    report = Report(metrics=[
        DatasetDriftMetric(),
        ColumnDriftMetric(column_name='demand'),
        ColumnDriftMetric(column_name='price')
    ])
    report.run(reference_data=ref_data, current_data=current_data)

    # Actualizar métricas de Prometheus
    DRIFT_SCORE.set(report.metrics[0].result['drift_score'])
    for metric in report.metrics[1:]:
        COLUMN_DRIFT_SCORES.labels(column=metric.column_name).set(metric.result['drift_score'])

    # Hacer predicción (simplificado)
    prediction = np.random.normal() if np.random.random() > 0.1 else np.random.normal() + 2
    return {"prediction": float(prediction)}

@app.get("/drift_report")
async def get_drift_report():
    # Generar reporte de drift
    report = Report(metrics=[DatasetDriftMetric()])
    current_data = pd.read_csv('data/current_data.csv')
    report.run(reference_data=ref_data, current_data=current_data)
    return json.loads(report.json())
```

### Paso 3: Configuración de Alertas en Grafana

Archivo `grafana_alerts.yml`:

```yaml
apiVersion: 1
groups:
- name: model-drift-alerts
  rules:
  - alert: HighDatasetDrift
    expr: dataset_drift_score > 0.2
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "High dataset drift detected ({{ $value }})"
      description: "The dataset drift score is {{ $value }}, which is above threshold of 0.2."

  - alert: HighColumnDrift
    expr: column_drift_scores > 0.3
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "High drift detected in column {{ $labels.column }} ({{ $value }})"
      description: "The drift score for column {{ $labels.column }} is {{ $value }}, which is above threshold of 0.3."
```

### Paso 4: Pipeline de Reentrenamiento con Airflow

El script `retraining_pipeline.py` define un DAG de Airflow para automatizar el reentrenamiento:

```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta
import requests

def check_drift():
    response = requests.get("http://localhost:8000/drift_report").json()
    drift_score = response['metrics'][0]['result']['drift_score']
    return drift_score > 0.2  # Umbral para reentrenamiento

def retrain_model():
    # Lógica para reentrenar el modelo con nuevos datos
    print("Reentrenando modelo...")
    # En producción, esto incluiría:
    # 1. Cargar nuevos datos
    # 2. Entrenar modelo
    # 3. Evaluar
    # 4. Guardar en MLflow
    return True

def update_reference_data():
    # Actualizar datos de referencia con los últimos datos "buenos"
    print("Actualizando datos de referencia...")
    return True

with DAG(
    dag_id="model_retraining",
    start_date=datetime(2026, 1, 1),
    schedule_interval="@daily",
    catchup=False
) as dag:
    check_drift_task = PythonOperator(
        task_id="check_drift",
        python_callable=check_drift
    )

    retrain_task = PythonOperator(
        task_id="retrain_model",
        python_callable=retrain_model
    )

    update_ref_task = PythonOperator(
        task_id="update_reference_data",
        python_callable=update_reference_data
    )

    check_drift_task >> retrain_task >> update_ref_task
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
      - ./data:/app/data
      - ./outputs:/app/outputs
    environment:
      - PROMETHEUS_MULTIPROC_DIR=/tmp
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
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'

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

  airflow:
    image: apache/airflow:2.7.0
    ports:
      - "8080:8080"
    volumes:
      - ./configs/airflow:/opt/airflow/dags
      - airflow_data:/opt/airflow
    environment:
      - AIRFLOW__CORE__EXECUTOR=LocalExecutor
      - AIRFLOW__CORE__SQL_ALCHEMY_CONN=sqlite:///opt/airflow/airflow.db
      - AIRFLOW__CORE__FERNET_KEY=your-fernet-key
    depends_on:
      - api

volumes:
  prometheus_data:
  grafana_data:
  airflow_data:
```

## 📊 Resultados Esperados

### Sistema de Monitoreo Funcional
- **API de predicción** operativa en http://localhost:8000
- **Métricas de drift** disponibles en http://localhost:9090
- **Dashboard de Grafana** accesible en http://localhost:3000
- **Alertas automáticas** configuradas y funcionando

### Detección de Drift
- **Dataset Drift Score**: Monitoreo continuo
- **Column Drift**: Análisis por características individuales
- **Umbral de alerta**: 0.2 para dataset drift, 0.3 para column drift
- **Tiempo de respuesta**: <5 minutos para detección

### Automatización de Reentrenamiento
- **Trigger automático** cuando drift > umbral
- **Pipeline de Airflow** ejecutándose diariamente
- **Actualización de datos** de referencia después de reentrenamiento exitoso
- **Logs completos** de todo el proceso

## 🔍 Análisis e Interpretación

### Tipos de Drift Detectados

1. **Data Drift**: Cambios en la distribución de datos de entrada
   - Causas comunes: Cambios estacionales, nuevos productos, eventos externos
   - Detección: Evidently DatasetDriftMetric

2. **Feature Drift**: Cambios en características específicas
   - Ejemplo: Cambios en patrones de demanda por producto
   - Detección: Evidently ColumnDriftMetric

### Umbrales y Acciones

| Nivel de Drift | Score | Acción Recomendada |
|-----------------|--------|-------------------|
| Bajo | < 0.1 | Monitoreo continuo |
| Medio | 0.1 - 0.2 | Investigación manual |
| Alto | 0.2 - 0.3 | Preparar reentrenamiento |
| Crítico | > 0.3 | Reentrenamiento inmediato |

### Métricas de Monitoreo

- **Drift Score**: Métrica principal de detección
- **Prediction Latency**: Tiempo de respuesta del modelo
- **Request Rate**: Frecuencia de solicitudes
- **Error Rate**: Tasa de errores en predicciones
- **Model Performance**: Métricas de calidad del modelo

## 📌 Entregables del Laboratorio

| Entregable | Descripción | Formato |
|-------------|-------------|-----------|
| API con Métricas de Drift | FastAPI con endpoints para monitoreo | drift_api.py |
| Configuración de Alertas | Reglas para Grafana | grafana_alerts.yml |
| Pipeline de Reentrenamiento | Airflow DAG para automatizar el reentrenamiento | retraining_pipeline.py |
| Orquestación de Servicios | Docker Compose para despliegue completo | docker-compose.yml |
| Datos de Ejemplo | Datos de referencia y producción | data/ |
| Documentación | Guía para configurar el monitoreo y reentrenamiento | README.md |

## 🎯 Criterios de Evaluación

- **Funcionalidad** (40%): Sistema completo funcionando correctamente
- **Detección de Drift** (25%): Detección precisa y oportuna
- **Automatización** (20%): Pipeline de reentrenamiento automático
- **Monitoreo** (15%): Dashboards y alertas configurados

## 🚀 Extensión y Mejoras

### Mejoras Sugeridas
1. **Modelos Avanzados**: Usar métodos más sofisticados de detección de drift
2. **Alertas Multi-canal**: Email, Slack, SMS para notificaciones
3. **Análisis de Causa Raíz**: Investigar automáticamente causas de drift
4. **A/B Testing**: Comparar modelos antes y después del reentrenamiento

### Aplicaciones en Producción
1. **Monitoreo Multi-modelo**: Extender a múltiples modelos en producción
2. **Escalado Horizontal**: Manejar alta carga de predicciones
3. **Integración con MLflow**: Tracking completo del ciclo de vida
4. **Dashboard Unificado**: Vista consolidada de todos los modelos

---

**Duración Estimada**: 10-12 horas  
**Nivel de Dificultad**: Avanzada  
**Prerrequisitos**: Conocimientos de APIs, Docker, sistemas de monitoreo
