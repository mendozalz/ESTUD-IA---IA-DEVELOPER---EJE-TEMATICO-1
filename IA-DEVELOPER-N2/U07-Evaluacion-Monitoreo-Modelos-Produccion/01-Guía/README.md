# Guía Conceptual: Evaluación y Monitoreo de Modelos en Producción

## 📖 Fundamentos Teóricos

### 1. Métricas Avanzadas para Evaluación

#### Métricas Clave en 2026

| Métrica | Fórmula | Cuándo Usar | Implementación en Python |
|----------|----------|---------------|-------------------------|
| Accuracy | (TP + TN) / (TP + TN + FP + FN) | Clasificación balanceada | `sklearn.metrics.accuracy_score(y_true, y_pred)` |
| Precision | TP / (TP + FP) | Minimizar falsos positivos (ej: spam, fraudes) | `sklearn.metrics.precision_score(y_true, y_pred)` |
| Recall (Sensitivity) | TP / (TP + FN) | Minimizar falsos negativos (ej: diagnóstico médico) | `sklearn.metrics.recall_score(y_true, y_pred)` |
| F1-Score | 2 * (Precision * Recall) / (Precision + Recall) | Clasificación desbalanceada | `sklearn.metrics.f1_score(y_true, y_pred)` |
| ROC-AUC | Área bajo la curva ROC (TPR vs. FPR) | Clasificación binaria | `sklearn.metrics.roc_auc_score(y_true, y_scores)` |
| PR-AUC | Área bajo la curva Precision-Recall | Clasificación desbalanceada | `sklearn.metrics.average_precision_score(y_true, y_scores)` |
| Log Loss | - (1/n) * Σ [y_i * log(p_i) + (1 - y_i) * log(1 - p_i)] | Clasificación probabilística | `sklearn.metrics.log_loss(y_true, y_probs)` |
| RMSE | √(1/n * Σ (y_i - ŷ_i)²) | Regresión | `sklearn.metrics.mean_squared_error(y_true, y_pred, squared=False)` |
| MAE | (1/n) * Σ |y_i - ŷ_i| | Regresión | `sklearn.metrics.mean_absolute_error(y_true, y_pred)` |
| R² | 1 - (Σ (y_i - ŷ_i)² / Σ (y_i - ȳ)²) | Regresión (proporción de varianza explicada) | `sklearn.metrics.r2_score(y_true, y_pred)` |

#### Ejemplo con ROC-AUC

```python
from sklearn.metrics import roc_auc_score, RocCurveDisplay
import matplotlib.pyplot as plt

y_true = [0, 1, 1, 0, 1]
y_scores = [0.1, 0.9, 0.8, 0.2, 0.75]  # Probabilidades predichas

auc = roc_auc_score(y_true, y_scores)
print(f"ROC-AUC: {auc:.4f}")

# Graficar curva ROC
RocCurveDisplay.from_predictions(y_true, y_scores)
plt.show()
```

### 2. Detección de Drift

#### Tipos de Drift en 2026

| Tipo de Drift | Descripción | Herramientas para Detección | Ejemplo de Código |
|----------------|-------------|---------------------------|------------------|
| Data Drift | Cambios en la distribución de los datos de entrada | Evidently AI, TFDV, Alibi Detect | ```python<br>from evidently.report import Report<br>from evidently.metrics import DatasetDriftMetric<br><br>report = Report(metrics=[DatasetDriftMetric()])<br>report.run(reference_data=ref_data, current_data=prod_data)<br>report.save_html("drift_report.html")``` |
| Concept Drift | Cambios en la relación entre entradas y salidas (el concepto objetivo cambia) | Alibi Detect, River | ```python<br>from alibi_detect import ConceptDrift<br>cd = ConceptDrift(X_ref, model)<br>preds = cd.predict(X)``` |
| Label Drift | Cambios en la distribución de las etiquetas | Evidently AI, TF Model Analysis | ```python<br>from tensorflow_model_analysis import model_eval_lib<br>eval_result = model_eval_lib.run_model_analysis(...)``` |
| Feature Drift | Cambios en una característica específica | Evidently AI, SHAP | ```python<br>from evidently.metrics import ColumnDriftMetric<br>report = Report(metrics=[ColumnDriftMetric(column_name='feature1')])``` |

#### Ejemplo con Evidently AI

```python
from evidently.report import Report
from evidently.metrics import DatasetDriftMetric, ColumnDriftMetric
from evidently.test_suite import TestSuite
from evidently.tests import TestNumberOfRowsWithMissingValues

# Detección de drift en el dataset completo
drift_report = Report(metrics=[
    DatasetDriftMetric(),
    ColumnDriftMetric(column_name='age'),
    ColumnDriftMetric(column_name='income')
])
drift_report.run(reference_data=ref_data, current_data=prod_data)
drift_report.save_html("drift_report.html")

# Tests automáticos
test_suite = TestSuite(tests=[
    TestNumberOfRowsWithMissingValues()
])
test_suite.run(reference_data=ref_data, current_data=prod_data)
test_suite.save_html("test_suite.html")
```

### 3. Monitoreo en Tiempo Real

#### Arquitectura de Monitoreo en 2026

```
[Modelo en Producción] → [Recolectar Métricas] → [Prometheus] → [Grafana] → [Alertas] → [Acciones (Reentrenamiento, Rollback)]
```

#### Componentes Clave

| Componente | Herramienta | Responsabilidad |
|-------------|-------------|-----------------|
| Recolectar Métricas | Prometheus, OpenTelemetry | Capturar métricas de rendimiento (latencia, accuracy, throughput) |
| Almacenar Métricas | Prometheus, InfluxDB | Base de datos time-series para métricas |
| Visualizar | Grafana | Dashboards para monitoreo en tiempo real |
| Alertas | Grafana, Alertmanager | Notificar cuando las métricas superan umbrales |
| Acciones | Airflow, MLflow | Reentrenar modelos, hacer rollback, escalar recursos |

#### Ejemplo con Prometheus y FastAPI

```python
from fastapi import FastAPI
from prometheus_client import make_asgi_app, Counter, Gauge, Histogram

app = FastAPI()
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

# Métricas
REQUEST_COUNT = Counter(
    'model_requests_total', 'Total model requests',
    ['model_name', 'status']
)
PREDICTION_LATENCY = Histogram(
    'prediction_latency_seconds', 'Prediction latency',
    ['model_name']
)
MODEL_ACCURACY = Gauge(
    'model_accuracy', 'Model accuracy',
    ['model_name']
)

@app.post("/predict")
async def predict(data: dict):
    start_time = time.time()
    try:
        prediction = model.predict(data['input'])
        REQUEST_COUNT.labels(model_name="fraud_detector", status="success").inc()
        return {"prediction": prediction.tolist()}
    except Exception as e:
        REQUEST_COUNT.labels(model_name="fraud_detector", status="error").inc()
        raise
    finally:
        PREDICTION_LATENCY.labels(model_name="fraud_detector").observe(time.time() - start_time)
```

### 4. Reentrenamiento Automático

#### Estrategias en 2026

| Estrategia | Descripción | Herramientas | Ventajas |
|-------------|-------------|---------------|-----------|
| Reentrenamiento Programado | Reentrenar el modelo en intervalos fijos (ej: cada semana) | Airflow, MLflow | Simple de implementar |
| Reentrenamiento por Drift | Reentrenar cuando se detecta drift significativo | Evidently + MLflow | Responde a cambios en los datos |
| Reentrenamiento por Rendimiento | Reentrenar cuando las métricas caen below un umbral | Prometheus + MLflow | Enfocado en el rendimiento del modelo |
| Online Learning | Actualizar el modelo con cada nueva muestra (sin batches) | River, TensorFlow | Adaptación continua |
| Shadow Deployment | Desplegar nueva versión junto a la actual y comparar métricas | Istio, TFX | Reduce riesgo de degradación |
| Canary Deployment | Desplegar nueva versión a un subconjunto de usuarios | Kubernetes, Istio | Pruebas en producción con bajo riesgo |

#### Ejemplo con MLflow y Airflow

```python
# DAG de Airflow para reentrenamiento (triggered por drift)
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime

def check_drift():
    # Usar Evidently para detectar drift
    report = Report(metrics=[DatasetDriftMetric()])
    report.run(reference_data=ref_data, current_data=prod_data)
    return report.metrics[0].result['drift_score'] > 0.1  # Umbral de drift

def retrain_model():
    # Reentrenar modelo con nuevos datos
    model = train_model(new_data)
    mlflow.log_model(model, "model")
    return model

with DAG(
    dag_id="model_retraining",
    start_date=datetime(2026, 1, 1),
    schedule_interval="@daily"
) as dag:
    check_drift_task = PythonOperator(
        task_id="check_drift",
        python_callable=check_drift
    )
    retrain_task = PythonOperator(
        task_id="retrain_model",
        python_callable=retrain_model
    )
    check_drift_task >> retrain_task
```

### 5. Explicabilidad y Cumplimiento

#### Herramientas en 2026

| Herramienta | Uso Principal | Ejemplo de Código |
|-------------|-----------------|------------------|
| SHAP | Explicar predicciones individuales | ```python<br>import shap<br>explainer = shap.Explainer(model)<br>shap_values = explainer(X_sample)<br>shap.plots.force(shap_values[0])``` |
| LIME | Explicaciones locales (para modelos no lineales) | ```python<br>from lime import lime_tabular<br>explainer = lime_tabular.LimeTabularExplainer(X_train)<br>exp = explainer.explain_instance(X_test[0], model.predict_proba)``` |
| Arize | Plataforma para observabilidad y explicabilidad | (API basada en web) |
| Fiddler | Monitoreo de modelos en producción con explicabilidad | (API basada en web) |
| TensorFlow Model Analysis | Evaluación de modelos con métricas por subgrupos | ```python<br>from tensorflow_model_analysis import model_eval_lib<br>eval_result = model_eval_lib.run_model_analysis(...)``` |

#### Ejemplo con SHAP para Explicabilidad

```python
import shap
import matplotlib.pyplot as plt

# Cargar modelo y datos de ejemplo
model = tf.keras.models.load_model('model.h5')
X_sample = x_test[:100]  # Muestra de datos de prueba

# Crear explainer
explainer = shap.DeepExplainer(model, X_sample[:10])
shap_values = explainer.shap_values(X_sample[:5])

# Visualizar
shap.initjs()
shap.force_plot(explainer.expected_value[0], shap_values[0][0], X_sample[0])
plt.savefig("shap_explanation.png")
```

### 6. A/B Testing y Shadow Deployment

#### Comparación de Estrategias

| Estrategia | Descripción | Implementación |
|-------------|-------------|----------------|
| A/B Testing | Comparar dos versiones del modelo en producción con un split de tráfico | ```python<br># Usar un router para dividir tráfico (ej: 50% a modelo A, 50% a modelo B)<br>if random.random() < 0.5:<br>    return model_a.predict(input)<br>else:<br>    return model_b.predict(input)``` |
| Shadow Deployment | Ejecutar la nueva versión en paralelo y comparar predicciones sin afectar usuarios | ```python<br># Ejecutar ambos modelos y comparar resultados<br>pred_a = model_a.predict(input)<br>pred_b = model_b.predict(input)<br>log_comparison(pred_a, pred_b)  # Comparar y registrar diferencias``` |
| Canary Deployment | Desplegar nueva versión a un pequeño subconjunto de usuarios | ```yaml<br># Kubernetes: Desplegar dos versiones con diferentes weights<br>spec:<br>  traffic:<br>  - revisionName: model-v1<br>    percent: 90<br>  - revisionName: model-v2<br>    percent: 10``` |

#### Ejemplo con Istio para A/B Testing

```yaml
apiVersion: networking.istio.io/v1alpha3
kind: VirtualService
metadata:
  name: model-route
spec:
  hosts:
  - model-service
  http:
  - route:
    - destination:
        host: model-service
        subset: v1
      weight: 50
    - destination:
        host: model-service
        subset: v2
      weight: 50
```

---

**Esta guía proporciona los fundamentos teóricos y ejemplos prácticos para evaluar, monitorear y mantener modelos de IA en producción con las mejores prácticas de 2026.**
