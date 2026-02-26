# Unidad 7: Evaluación y Monitoreo de Modelos en Producción (2026)

## 🎯 Título
"Métricas Avanzadas, Detección de Drift y Reentrenamiento Automático: Manteniendo Modelos de IA en Producción con Excelencia"

## 🎯 Objetivos de Aprendizaje

Al finalizar esta unidad, los estudiantes podrán:

- Evaluar modelos en producción usando métricas avanzadas (ROC-AUC, precision-recall, F1-score)
- Detectar y cuantificar drift (concept drift, data drift) con herramientas como Evidently AI, TensorFlow Data Validation y Alibi Detect
- Implementar sistemas de monitoreo en tiempo real con Prometheus, Grafana y OpenTelemetry
- Automatizar el reentrenamiento de modelos usando MLflow, TensorFlow Extended (TFX) y Airflow
- Aplicar la metodología Windsor para gestionar el ciclo de vida completo de modelos en producción
- Optimizar el rendimiento de modelos en producción con técnicas como A/B testing y shadow deployment

## 📌 Contexto Tecnológico (2026)

En 2026, el monitoreo y evaluación de modelos de IA en producción es crítico debido a:

- Cambios en los datos (ej: nuevos patrones de fraude, cambios en el comportamiento de usuarios)
- Degradación del rendimiento (ej: caída en precisión, aumento en latencia)
- Requisitos regulatorios (ej: GDPR, explicabilidad de decisiones)
- Escalabilidad (millones de predicciones por día)

### Herramientas clave en 2026:

| Herramienta | Versión | Uso Principal |
|-------------|----------|----------------|
| Evidently AI | 2026.2 | Detección de drift y monitoreo de modelos |
| TensorFlow Data Validation | 2.0 | Validación de esquemas y calidad de datos |
| Alibi Detect | 0.12 | Detección de outliers y concept drift |
| Prometheus | 3.0 | Recopilar métricas en tiempo real |
| Grafana | 10.0 | Visualización de métricas y alertas |
| MLflow | 3.0 | Gestión del ciclo de vida de modelos (experimentación, despliegue, monitoreo) |
| TensorFlow Extended (TFX) | 2.0 | Pipelines end-to-end para reentrenamiento automático |
| Apache Airflow | 3.0 | Orquestación de pipelines de reentrenamiento |
| OpenTelemetry | 1.0 | Trazabilidad distribuida para microservicios |
| Weights & Biases | 2.12 | Tracking de experimentos y monitoreo de modelos |
| Arize | 2026.1 | Plataforma para observabilidad de modelos en producción |
| Fiddler | 2026.2 | Monitoreo y explicabilidad de modelos |

## 📁 Estructura de la Unidad

```
U07-Evaluacion-Monitoreo-Modelos-Produccion/
├── README.md                           # Guía principal de la unidad
├── 01-Guía/                          # Guía conceptual
│   └── README.md
├── 02-Laboratorios/                    # Laboratorios prácticos
│   ├── Laboratorio 7.1 - Evaluación de Modelos con Métricas Avanzadas/
│   ├── Laboratorio 7.2 - Detección de Drift con Evidently AI/
│   ├── Laboratorio 7.3 - Monitoreo en Tiempo Real con OpenTelemetry/
│   ├── Laboratorio 7.4 - Reentrenamiento Automático con MLflow y TFX/
│   └── Laboratorio 7.5 - Explicabilidad y Cumplimiento con SHAP y Arize/
├── 03-Recursos/                        # Recursos de aprendizaje
│   └── README.md
└── 04-Buenas Prácticas/               # Mejores prácticas
    └── README.md
```

## 🚀 Laboratorios Prácticos

### 🔧 Laboratorio 7.1: Evaluación de Modelos con Métricas Avanzadas y TensorFlow Model Analysis
- **Contexto**: Modelo de clasificación de fraudes en producción
- **Objetivos**: Calcular ROC-AUC, PR-AUC, análisis por subgrupos
- **Herramientas**: TensorFlow Model Analysis, scikit-learn

### 🔧 Laboratorio 7.2: Detección de Drift con Evidently AI y Alertas en Grafana
- **Contexto**: Modelo de predicción de demanda en retail
- **Objetivos**: Detectar drift, generar alertas, automatizar reentrenamiento
- **Herramientas**: Evidently AI, Grafana, Airflow

### 🔧 Laboratorio 7.3: Monitoreo en Tiempo Real con OpenTelemetry y Prometheus
- **Contexto**: Modelo de clasificación de imágenes médicas
- **Objetivos**: Instrumentar API, configurar dashboards
- **Herramientas**: OpenTelemetry, Prometheus, Grafana

### 🔧 Laboratorio 7.4: Reentrenamiento Automático con MLflow y TFX
- **Contexto**: Modelo de recomendación de productos en e-commerce
- **Objetivos**: Pipeline automático, validación, despliegue seguro
- **Herramientas**: MLflow, TFX, Kubernetes

### 🔧 Laboratorio 7.5: Explicabilidad y Cumplimiento con SHAP y Arize
- **Contexto**: Modelo de aprobación de préstamos
- **Objetivos**: Explicaciones individuales, auditorías automáticas
- **Herramientas**: SHAP, Arize, documentación regulatoria

## 📊 Proyecto Integrador

### Sistema de Monitoreo y Reentrenamiento para un Modelo de Crédito

**Contexto**: Un banco quiere implementar un sistema completo de monitoreo y reentrenamiento para su modelo de aprobación de créditos.

**Arquitectura**:
```
[Solicitudes de Crédito] → [API FastAPI] → [Modelo de Crédito] → 
[Monitoreo (Evidently, Prometheus)] → [Reentrenamiento (TFX, Airflow)] → 
[Explicaciones (SHAP, Arize)]
```

**Requisitos**:
- Evaluación continua con TFMA
- Detección de drift con Evidently
- Reentrenamiento automático con TFX
- Explicabilidad con SHAP y Arize
- Monitoreo en tiempo real con Prometheus y Grafana

## 📈 Evaluación y Entregables

### 📋 Entregables por Laboratorio

| Laboratorio | Entregables Principales | Tecnologías |
|-------------|----------------------|-------------|
| 7.1 | Scripts de evaluación, reportes TFMA, análisis por subgrupos | TensorFlow, scikit-learn |
| 7.2 | API con métricas de drift, alertas Grafana, pipeline Airflow | Evidently, Grafana, Airflow |
| 7.3 | API instrumentada, configuración Prometheus, dashboards | OpenTelemetry, Prometheus |
| 7.4 | Pipeline TFX, módulo de entrenamiento, tracking MLflow | TFX, MLflow, Kubernetes |
| 7.5 | Explicaciones SHAP, integración Arize, auditorías | SHAP, Arize, documentación |

### 🎯 Criterios de Evaluación

- **Correctitud técnica**: Implementación funcional de todos los componentes
- **Calidad del código**: Código limpio, documentado, siguiendo buenas prácticas
- **Monitoreo efectivo**: Detección oportuna de problemas y degradación
- **Automatización robusta**: Pipelines de reentrenamiento confiables
- **Explicabilidad clara**: Decisiones del modelo explicadas y auditables

---

**Duración Estimada de la Unidad**: 40-50 horas  
**Nivel de Dificultad**: Avanzado  
**Prerrequisitos**: Conocimientos de ML, APIs, contenerización y orquestación
