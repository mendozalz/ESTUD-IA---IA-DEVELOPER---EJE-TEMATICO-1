# Buenas Prácticas - Evaluación y Monitoreo de Modelos en Producción

## 📋 Introducción

Este documento establece las mejores prácticas y metodologías para la evaluación y monitoreo de modelos de IA en producción, utilizando el marco lógico como metodología fundamental para garantizar sistemas robustos, escalables y confiables.

## 🎯 Metodología de Marco Lógico

### **Definición del Marco Lógico**
El marco lógico es una herramienta de planificación y gestión que estructura los proyectos mediante una matriz de objetivos, indicadores, medios de verificación y supuestos críticos.

### **Componentes del Marco Lógico**

#### **1. Jerarquía de Objetivos**
```
Fin → Propósito → Componentes → Actividades
```

- **Fin**: Objetivo de desarrollo al que contribuye el proyecto
- **Propósito**: Efecto directo esperado al completar el proyecto
- **Componentes**: Resultados específicos que el proyecto debe producir
- **Actividades**: Tareas necesarias para producir los componentes

#### **2. Matriz del Marco Lógico**
| Nivel | Objetivos | Indicadores Verificables | Medios de Verificación | Supuestos Críticos |
|-------|-----------|-------------------------|----------------------|-------------------|
| **Fin** | Impacto a largo plazo | KPIs de negocio | Reportes ejecutivos | Condiciones externas |
| **Propósito** | Efecto directo | Métricas de éxito | Dashboards | Factores internos |
| **Componentes** | Resultados entregables | Especificaciones técnicas | Documentación | Recursos disponibles |
| **Actividades** | Tareas ejecutadas | Cronograma cumplido | Logs y reports | Capacitación equipo |

## 🔹 Buenas Prácticas por Componente

### **1. Evaluación de Modelos**

#### ✅ Qué Hacer
- **Implementar métricas avanzadas** (ROC-AUC, PR-AUC, F1-score)
- **Analizar por subgrupos** (regiones, demografía, tiempo)
- **Validar continuamente** el rendimiento del modelo
- **Documentar criterios** de evaluación

#### 🔧 Cómo Hacerlo
```python
# Ejemplo: Evaluación completa con TFMA
import tensorflow_model_analysis as tfma
from tensorflow_model_analysis.view.render_slicing_metrics import render_slicing_metrics

eval_config = tfma.EvalConfig(
    model_specs=[tfma.ModelSpec(label_key='approved')],
    metrics_specs=[
        tfma.MetricConfig(class_name='AUC'),
        tfma.MetricConfig(class_name='Precision'),
        tfma.MetricConfig(class_name='Recall'),
        tfma.MetricConfig(class_name='ExampleCount')
    ],
    slicing_specs=[
        tfma.SlicingSpec(),
        tfma.SlicingSpec(feature_keys=['region']),
        tfma.SlicingSpec(feature_keys=['income_level'])
    ]
)

eval_result = tfma.run_model_analysis(
    tfma.load_model('credit_model'),
    data_location='eval.tfrecord',
    eval_config=eval_config,
    output_path='tfma_output'
)

render_slicing_metrics(eval_result, output_file='evaluation_report.html')
```

#### 📊 Aplicación al Proyecto Integral
- **Evaluar modelos** con métricas apropiadas para el dominio
- **Analizar rendimiento** por segmentos de usuarios
- **Generar reportes** automáticos de evaluación
- **Establecer umbrales** para acción correctiva

### **2. Detección de Drift**

#### ✅ Qué Hacer
- **Implementar detección** de data drift y concept drift
- **Configurar umbrales** basados en impacto del negocio
- **Automatizar alertas** cuando se detecte drift
- **Documentar procedimientos** de respuesta

#### 🔧 Cómo Hacerlo
```python
# Ejemplo: Sistema de detección de drift con Evidently
from evidently.report import Report
from evidently.metrics import DatasetDriftMetric, ColumnDriftMetric
from evidently.test_suite import TestSuite

class DriftDetector:
    def __init__(self, reference_data, drift_threshold=0.1):
        self.reference_data = reference_data
        self.drift_threshold = drift_threshold
        self.drift_history = []
    
    def detect_drift(self, current_data):
        # Crear reporte de drift
        drift_report = Report(metrics=[
            DatasetDriftMetric(),
            ColumnDriftMetric(column_name='feature1'),
            ColumnDriftMetric(column_name='feature2')
        ])
        
        drift_report.run(
            reference_data=self.reference_data,
            current_data=current_data
        )
        
        # Evaluar drift
        dataset_drift = drift_report.metrics[0].result['drift_score']
        
        # Registrar en historial
        self.drift_history.append({
            'timestamp': datetime.now(),
            'drift_score': dataset_drift,
            'column_drift': {
                metric.column_name: metric.result['drift_score']
                for metric in drift_report.metrics[1:]
            }
        })
        
        return {
            'drift_detected': dataset_drift > self.drift_threshold,
            'drift_score': dataset_drift,
            'report': drift_report
        }
    
    def get_drift_trend(self):
        """Analizar tendencia de drift en el tiempo"""
        if len(self.drift_history) < 2:
            return None
        
        recent_scores = [d['drift_score'] for d in self.drift_history[-10:]]
        trend = 'increasing' if recent_scores[-1] > recent_scores[0] else 'decreasing'
        
        return {
            'trend': trend,
            'recent_average': sum(recent_scores) / len(recent_scores),
            'current': recent_scores[-1]
        }
```

#### 📊 Aplicación al Proyecto Integral
- **Monitorear datos** de entrada continuamente
- **Detectar cambios** en distribuciones y relaciones
- **Responder proactivamente** a drift detectado
- **Mantener histórico** para análisis de tendencias

### **3. Monitoreo en Tiempo Real**

#### ✅ Qué Hacer
- **Instrumentar aplicaciones** con OpenTelemetry
- **Configurar Prometheus** para recolectar métricas
- **Crear dashboards** en Grafana para visualización
- **Establecer alertas** automáticas y escalonadas

#### 🔧 Cómo Hacerlo
```python
# Ejemplo: Instrumentación completa con OpenTelemetry
from opentelemetry import trace, metrics
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.exporter.prometheus import PrometheusMetricReader
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

class ProductionMonitor:
    def __init__(self, service_name: str):
        self.service_name = service_name
        self.setup_tracing()
        self.setup_metrics()
    
    def setup_tracing(self):
        """Configurar tracing distribuido"""
        trace.set_tracer_provider(TracerProvider())
        self.tracer = trace.get_tracer(__name__)
    
    def setup_metrics(self):
        """Configurar métricas personalizadas"""
        meter_provider = MeterProvider()
        meter_provider.add_metric_reader(PrometheusMetricReader())
        
        self.meter = meter_provider.get_meter(__name__)
        
        # Métricas clave
        self.request_counter = self.meter.create_counter(
            "requests_total",
            "Total number of requests"
        )
        
        self.latency_histogram = self.meter.create_histogram(
            "request_duration_seconds",
            "Request duration"
        )
        
        self.accuracy_gauge = self.meter.create_up_down_counter(
            "model_accuracy",
            "Model accuracy"
        )
    
    def record_prediction(self, prediction_time: float, accuracy: float):
        """Registrar métricas de predicción"""
        self.request_counter.add(1)
        self.latency_histogram.record(prediction_time)
        self.accuracy_gauge.add(accuracy)
    
    def trace_prediction(self, input_data: dict, prediction: dict):
        """Crear span de tracing para predicción"""
        with self.tracer.start_as_current_span("prediction") as span:
            span.set_attribute("input_size", len(str(input_data)))
            span.set_attribute("prediction_confidence", prediction.get("confidence"))
            span.set_attribute("service.name", self.service_name)

# Instrumentar FastAPI
FastAPIInstrumentor.instrument_app(app)
```

#### 📊 Aplicación al Proyecto Integral
- **Monitorear latencia**, throughput y errores
- **Visualizar métricas** en dashboards accionables
- **Configurar alertas** escalonadas por severidad
- **Mantener trazabilidad** distribuida completa

### **4. Reentrenamiento Automático**

#### ✅ Qué Hacer
- **Implementar pipelines** de reentrenamiento con TFX
- **Configurar triggers** automáticos (drift, rendimiento, tiempo)
- **Validar modelos** antes de despliegue
- **Implementar despliegue** seguro (canary, shadow)

#### 🔧 Cómo Hacerlo
```python
# Ejemplo: Pipeline de reentrenamiento automático
from tfx.orchestration import pipeline
from tfx.components import (
    CsvExampleGen, StatisticsGen, SchemaGen, Transform,
    Trainer, Evaluator, Pusher
)
from tfx.dsl.components.common import resolver
from tfx.dsl.input_resolution.strategies.latest_blessed_model import LatestBlessedModelStrategy

class AutoRetrainingPipeline:
    def __init__(self, pipeline_name: str):
        self.pipeline_name = pipeline_name
    
    def create_pipeline(self):
        """Crear pipeline completo de reentrenamiento"""
        
        # 1. Ingestión de datos
        example_gen = CsvExampleGen(input_base='data/new')
        
        # 2. Estadísticas y validación de esquema
        statistics_gen = StatisticsGen(examples=example_gen.outputs['examples'])
        schema_gen = SchemaGen(statistics=statistics_gen.outputs['statistics'])
        
        # 3. Transformación de datos
        transform = Transform(
            examples=example_gen.outputs['examples'],
            schema=schema_gen.outputs['schema']
        )
        
        # 4. Resolver modelo actual
        model_resolver = resolver.Resolver(
            strategy_class=LatestBlessedModelStrategy,
            model=standard_artifacts.Model,
            model_blessing=standard_artifacts.ModelBlessing
        ).with_id('latest_model_resolver')
        
        # 5. Entrenamiento
        trainer = Trainer(
            module_file='trainer_module.py',
            examples=transform.outputs['transformed_examples'],
            schema=schema_gen.outputs['schema'],
            transform_graph=transform.outputs['transform_graph']
        )
        
        # 6. Evaluación comparativa
        evaluator = Evaluator(
            examples=example_gen.outputs['examples'],
            model=trainer.outputs['model'],
            baseline_model=model_resolver.outputs['model']
        )
        
        # 7. Despliegue condicional
        pusher = Pusher(
            model=trainer.outputs['model'],
            model_blessing=evaluator.outputs['blessing']
        )
        
        return pipeline.Pipeline(
            pipeline_name=self.pipeline_name,
            pipeline_root="pipeline_root",
            components=[
                example_gen, statistics_gen, schema_gen,
                transform, model_resolver, trainer, evaluator, pusher
            ],
            enable_cache=True
        )
    
    def should_retrain(self, drift_score: float, accuracy: float):
        """Determinar si se debe reentrenar"""
        drift_threshold = 0.1
        accuracy_threshold = 0.85
        
        return drift_score > drift_threshold or accuracy < accuracy_threshold
```

#### 📊 Aplicación al Proyecto Integral
- **Automatizar completamente** el ciclo de vida de modelos
- **Validar rigurosamente** cada nueva versión
- **Implementar despliegue** gradual y seguro
- **Mantener trazabilidad** de todas las versiones

### **5. Explicabilidad y Cumplimiento**

#### ✅ Qué Hacer
- **Generar explicaciones** para cada predicción importante
- **Implementar monitoreo** de equidad y sesgos
- **Crear documentación** automática para auditorías
- **Mantener registros** completos y trazables

#### 🔧 Cómo Hacerlo
```python
# Ejemplo: Sistema de explicabilidad y cumplimiento
import shap
import pandas as pd
from datetime import datetime

class ExplainabilityManager:
    def __init__(self, model, feature_names):
        self.model = model
        self.feature_names = feature_names
        self.explainer = shap.DeepExplainer(model, shap.sample(feature_names, 100))
        self.explanations_log = []
    
    def explain_prediction(self, input_data: dict, prediction: float):
        """Generar explicación para una predicción"""
        input_array = self._preprocess_input(input_data)
        
        # Generar valores SHAP
        shap_values = self.explainer.shap_values(input_array)
        
        # Identificar características más influyentes
        feature_importance = sorted(
            zip(self.feature_names, shap_values[0]),
            key=lambda x: abs(x[1]),
            reverse=True
        )[:5]
        
        explanation = {
            'timestamp': datetime.now().isoformat(),
            'input_hash': hash(str(input_data)),
            'prediction': prediction,
            'shap_values': shap_values[0].tolist(),
            'top_features': feature_importance,
            'explanation_type': 'shap_deep'
        }
        
        # Registrar explicación
        self.explanations_log.append(explanation)
        
        return explanation
    
    def generate_audit_report(self, time_period: str = 'last_30_days'):
        """Generar reporte de auditoría"""
        recent_explanations = [
            exp for exp in self.explanations_log
            if self._is_in_time_period(exp['timestamp'], time_period)
        ]
        
        # Análisis de equidad
        fairness_metrics = self._calculate_fairness_metrics(recent_explanations)
        
        # Estadísticas de explicaciones
        explanation_stats = self._calculate_explanation_stats(recent_explanations)
        
        audit_report = {
            'period': time_period,
            'total_explanations': len(recent_explanations),
            'fairness_metrics': fairness_metrics,
            'explanation_stats': explanation_stats,
            'compliance_status': self._check_compliance(fairness_metrics),
            'generated_at': datetime.now().isoformat()
        }
        
        return audit_report
    
    def _calculate_fairness_metrics(self, explanations):
        """Calcular métricas de equidad"""
        # Implementar cálculo de métricas como:
        # - Demographic Parity
        # - Equal Opportunity
        # - Equalized Odds
        # - Disparate Impact
        
        return {
            'demographic_parity': 0.85,  # Ejemplo
            'equal_opportunity': 0.82,
            'equalized_odds': 0.80,
            'disparate_impact': 0.12
        }
    
    def _check_compliance(self, fairness_metrics):
        """Verificar cumplimiento regulatorio"""
        thresholds = {
            'demographic_parity': 0.8,
            'equal_opportunity': 0.8,
            'disparate_impact': 0.2
        }
        
        compliance_status = 'compliant'
        
        for metric, threshold in thresholds.items():
            if fairness_metrics.get(metric, 0) < threshold:
                compliance_status = 'non_compliant'
                break
        
        return compliance_status
```

#### 📊 Aplicación al Proyecto Integral
- **Explicar decisiones** del modelo de forma comprensible
- **Monitorear equidad** entre diferentes grupos demográficos
- **Generar documentación** automática para auditorías
- **Mantener trazabilidad** completa de decisiones

## 🎯 Aplicación a Proyectos Integrales

### **Metodología de Implementación**

#### **Paso 1: Definición del Marco Lógico**
```python
def define_monitoring_logical_framework(project_name):
    return {
        'project_name': project_name,
        'fin': {
            'objetivo': 'Mantener modelos de IA funcionando óptimamente en producción',
            'indicadores': ['Disponibilidad >99.9%', 'Drift detectado <24h', 'Reentrenamiento automático'],
            'verificacion': ['Uptime monitoring', 'Drift detection logs', 'Automated retraining reports'],
            'supuestos': ['Infraestructura estable', 'Equipo capacitado', 'Procesos definidos']
        },
        'propósito': {
            'objetivo': 'Implementar sistema completo de monitoreo y evaluación',
            'indicadores': ['Métricas avanzadas configuradas', 'Alertas funcionando', 'Dashboards operativos'],
            'verificacion': ['Métricas dashboard', 'Alert configurations', 'Dashboard screenshots'],
            'supuestos': ['Modelos disponibles', 'Herramientas instaladas', 'Acceso a sistemas']
        },
        'componentes': {
            'objetivo': 'Sistema de monitoreo funcional',
            'indicadores': ['Evaluación implementada', 'Drift detection activo', 'Reentrenamiento automático'],
            'verificacion': ['Evaluation scripts', 'Drift detection code', 'Retraining pipelines'],
            'supuestos': ['Datos de referencia', 'Sistemas de monitoreo', 'Pipeline tools']
        },
        'actividades': {
            'objetivo': 'Implementar sistema de monitoreo',
            'indicadores': ['Código desplegado', 'Tests pasados', 'Monitoreo activo'],
            'verificacion': ['Deploy logs', 'Test reports', 'Monitoring data'],
            'supuestos': ['Tiempo disponible', 'Permisos de acceso', 'Conocimientos técnicos']
        }
    }
```

## 📊 Métricas y Verificación

### **Dashboard de Seguimiento del Marco Lógico**

```python
class MonitoringDashboard:
    def create_dashboard(self, framework):
        """Crear dashboard para seguimiento del marco lógico"""
        
        dashboard = {
            'fin': self.create_fin_section(framework['fin']),
            'propósito': self.create_purpose_section(framework['propósito']),
            'componentes': self.create_components_section(framework['componentes']),
            'actividades': self.create_activities_section(framework['actividades'])
        }
        
        return dashboard
    
    def track_monitoring_metrics(self, framework):
        """Seguimiento de métricas de monitoreo"""
        
        progress = {}
        
        for level in ['fin', 'propósito', 'componentes', 'actividades']:
            indicators = framework[level]['indicadores']
            progress[level] = self.calculate_progress(indicators)
        
        return progress
```

## 🚀 Conclusión

La aplicación del marco lógico a proyectos de monitoreo de modelos proporciona:

- **Claridad estratégica** en objetivos y resultados esperados
- **Medición objetiva** del progreso y éxito
- **Verificación sistemática** de logros
- **Gestión de riesgos** mediante identificación de supuestos
- **Mejora continua** basada en evidencia

Al aplicar esta metodología, los estudiantes desarrollarán habilidades para:

- **Evaluar modelos** con métricas avanzadas y por subgrupos
- **Detectar drift** y responder proactivamente
- **Monitorear sistemas** en tiempo real con dashboards efectivos
- **Automatizar reentrenamiento** con pipelines robustos
- **Garantizar explicabilidad** y cumplimiento regulatorio

---

**Esta guía proporciona el marco metodológico para que los estudiantes apliquen sistemáticamente las mejores prácticas en todos sus proyectos de monitoreo y evaluación de modelos en producción.**
