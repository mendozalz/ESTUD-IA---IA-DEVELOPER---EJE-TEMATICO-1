# Laboratorio 7.4: Reentrenamiento Automático con MLflow y TFX

## 🎯 Contexto
Un modelo de recomendación de productos en un e-commerce muestra signos de degradación. Se implementará un pipeline automático para detectar degradación en las métricas de recomendación, reentrenar el modelo con nuevos datos, y desplegar la nueva versión sin tiempo de inactividad.

## 🎯 Objetivos

- Detectar degradación en las métricas de recomendación
- Reentrenar el modelo con nuevos datos
- Desplegar la nueva versión sin tiempo de inactividad

## 📋 Marco Lógico del Laboratorio

| Componente | Objetivo | Indicadores | Medios de Verificación |
|------------|----------|-------------|------------------------|
| **Detección Degradación** | Monitorear métricas de recomendación | Métricas < umbral detectadas | Sistema de monitoreo activo |
| **Pipeline TFX** | Implementar reentrenamiento automático | Pipeline ejecutándose correctamente | Logs de TFX funcionando |
| **Gestión MLflow** | Tracking de experimentos y modelos | Modelos versionados en MLflow | MLflow UI accesible |
| **Despliegue Seguro** | Actualizar modelo sin downtime | Despliegue gradual funcionando | Canary/shadow deployment |

## 🛠️ Tecnologías Utilizadas

- **TensorFlow Extended (TFX)**: Pipelines end-to-end para ML
- **MLflow**: Gestión del ciclo de vida de modelos
- **Apache Airflow**: Orquestación de pipelines
- **Kubernetes**: Despliegue y orquestación de contenedores
- **TensorFlow**: Framework de ML para el modelo
- **Docker**: Contenerización de servicios

## 📁 Estructura del Laboratorio

```
Laboratorio 7.4 - Reentrenamiento Automático con MLflow y TFX/
├── README.md                           # Esta guía
├── requirements.txt                     # Dependencias Python
├── retraining_pipeline.py               # Pipeline de TFX
├── trainer_module.py                   # Módulo de entrenamiento
├── mlflow_tracking.py                  # Configuración de MLflow
├── kubernetes/                         # Configuración de K8s
│   ├── deployment.yaml                 # Despliegue del modelo
│   ├── service.yaml                    # Servicio del modelo
│   └── canary.yaml                    # Configuración de canary
├── data/                              # Datos de recomendación
│   ├── interactions/                   # Datos de interacciones usuario-item
│   ├── items/                         # Catálogo de productos
│   └── users/                         # Datos de usuarios
├── models/                            # Modelos entrenados
│   ├── baseline/                       # Modelo base
│   └── production/                     # Modelo en producción
└── outputs/                           # Resultados generados
    ├── mlruns/                        # Logs de MLflow
    ├── pipeline_root/                  # Salidas de TFX
    └── deployment_logs/                # Logs de despliegue
```

## 🚀 Implementación Paso a Paso

### Paso 1: Configuración del Entorno

```bash
pip install mlflow==3.0.0 tensorflow==2.15.0 tensorflow-data-validation==2.0.0 tfx==2.0.0 apache-airflow==2.7.0 kubernetes==28.1.0 pandas==2.0.0 numpy==1.24.0 scikit-learn==1.3.0
```

### Paso 2: Pipeline de TFX para Reentrenamiento

El script `retraining_pipeline.py` implementa un pipeline completo con TFX:

```python
from tfx.orchestration import pipeline
from tfx.components import (
    ExampleGen, StatisticsGen, SchemaGen, Transform,
    Trainer, Tuner, Evaluator, Pusher, CsvExampleGen
)
from tfx.proto import pusher_pb2, trainer_pb2
from tfx.dsl.components.common import resolver
from tfx.dsl.input_resolution.strategies.latest_blessed_model import LatestBlessedModelStrategy
from tfx.types import standard_artifacts
from tfx.orchestration.kubeflow import kubeflow_dag_runner

def create_pipeline():
    # 1. Ingestión de datos
    example_gen = CsvExampleGen(input_base='data/interactions')

    # 2. Estadísticas y esquema
    statistics_gen = StatisticsGen(examples=example_gen.outputs['examples'])
    schema_gen = SchemaGen(statistics=statistics_gen.outputs['statistics'])

    # 3. Transformación
    transform = Transform(
        examples=example_gen.outputs['examples'],
        schema=schema_gen.outputs['schema']
    )

    # 4. Entrenamiento
    trainer = Trainer(
        module_file='trainer_module.py',
        examples=transform.outputs['transformed_examples'],
        schema=schema_gen.outputs['schema'],
        transform_graph=transform.outputs['transform_graph'],
        train_args=trainer_pb2.TrainArgs(num_steps=10000),
        eval_args=trainer_pb2.EvalArgs(num_steps=5000)
    )

    # 5. Evaluación
    model_resolver = resolver.Resolver(
        strategy_class=LatestBlessedModelStrategy,
        model=standard_artifacts.Model,
        model_blessing=standard_artifacts.ModelBlessing
    ).with_id('Latest_blessed_model_resolver')

    evaluator = Evaluator(
        examples=example_gen.outputs['examples'],
        model=trainer.outputs['model'],
        baseline_model=model_resolver.outputs['model'],
        eval_config='eval_config.json'
    )

    # 6. Despliegue
    pusher = Pusher(
        model=trainer.outputs['model'],
        model_blessing=evaluator.outputs['blessing'],
        push_destination=pusher_pb2.PushDestination(
            filesystem=pusher_pb2.PushDestination.Filesystem(
                base_directory='serving_model'
            )
        )
    )

    return pipeline.Pipeline(
        pipeline_name="recommendation_retraining",
        pipeline_root="pipeline_root",
        components=[
            example_gen,
            statistics_gen,
            schema_gen,
            transform,
            trainer,
            model_resolver,
            evaluator,
            pusher
        ],
        enable_cache=True
    )

# Ejecutar pipeline
kubeflow_dag_runner.KubeflowDagRunner().run(create_pipeline())
```

### Paso 3: Módulo de Entrenamiento

El script `trainer_module.py` contiene la lógica de entrenamiento:

```python
import tensorflow as tf
from tfx.components.trainer.fn_args_utils import FnArgs
from tensorflow.keras.layers import Input, Dense, Embedding, Flatten, Concatenate
from tensorflow.keras.models import Model

def run_fn(fn_args: FnArgs):
    # Cargar datos transformados
    train_dataset = tf.data.experimental.make_batched_features_dataset(
        fn_args.transformed_examples,
        batch_size=32,
        shuffle=True
    )

    # Definir modelo de recomendación
    user_input = Input(shape=(1,), name='user_id')
    item_input = Input(shape=(1,), name='item_id')

    user_embedding = Embedding(1000, 32)(user_input)
    item_embedding = Embedding(500, 32)(item_input)

    user_vec = Flatten()(user_embedding)
    item_vec = Flatten()(item_embedding)

    concat = Concatenate()([user_vec, item_vec])
    dense = Dense(64, activation='relu')(concat)
    output = Dense(1, activation='sigmoid')(dense)

    model = Model(inputs=[user_input, item_input], outputs=output)
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    # Entrenar
    model.fit(train_dataset, epochs=10)

    # Guardar modelo
    model.save(fn_args.serving_model_dir)
```

### Paso 4: Configuración de MLflow

El script `mlflow_tracking.py` configura el tracking de experimentos:

```python
import mlflow
from mlflow.tracking import MlflowClient

# Iniciar experimento
mlflow.set_experiment("recommendation_retraining")

with mlflow.start_run():
    # Loguear parámetros
    mlflow.log_params({
        "batch_size": 32,
        "epochs": 10,
        "optimizer": "adam",
        "learning_rate": 0.001
    })

    # Loguear métricas (simuladas)
    mlflow.log_metrics({
        "train_accuracy": 0.95,
        "val_accuracy": 0.92,
        "precision": 0.88,
        "recall": 0.85,
        "auc": 0.93
    })

    # Loguear modelo
    mlflow.tensorflow.log_model(
        model,
        "model",
        registered_model_name="recommendation_model"
    )

    # Transición a producción si las métricas son buenas
    client = MlflowClient()
    client.transition_model_version_stage(
        name="recommendation_model",
        version=mlflow.active_run().info.run_id,
        stage="Production"
    )
```

### Paso 5: Despliegue en Kubernetes

Archivo `kubernetes/deployment.yaml`:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: recommendation-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: recommendation-api
  template:
    metadata:
      labels:
        app: recommendation-api
    spec:
      containers:
      - name: recommendation-api
        image: recommendation-model:latest
        ports:
        - containerPort: 8000
        env:
        - name: MODEL_PATH
          value: "/models/production"
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
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
```

### Paso 6: Configuración de Canary Deployment

Archivo `kubernetes/canary.yaml`:

```yaml
apiVersion: networking.istio.io/v1alpha3
kind: VirtualService
metadata:
  name: recommendation-route
spec:
  hosts:
  - recommendation-service
  http:
  - route:
    - destination:
        host: recommendation-service
        subset: v1
      weight: 90
    - destination:
        host: recommendation-service
        subset: v2
      weight: 10
---
apiVersion: networking.istio.io/v1alpha3
kind: DestinationRule
metadata:
  name: recommendation-destination
spec:
  host: recommendation-service
  subsets:
  - name: v1
    labels:
      version: v1
  - name: v2
    labels:
      version: v2
```

## 📊 Resultados Esperados

### Pipeline Automático Funcional
- **TFX Pipeline** ejecutándose periódicamente
- **MLflow Tracking** con experimentos registrados
- **Model Registry** con versiones controladas
- **Evaluación Automática** comparando nuevo vs baseline

### Despliegue Seguro
- **Canary Deployment** con 10% de tráfico al nuevo modelo
- **Monitoreo Continuo** de ambas versiones
- **Rollback Automático** si métricas empeoran
- **Zero Downtime** durante actualizaciones

### Métricas Monitoreadas
- **Accuracy**: Precisión del modelo
- **Precision/Recall**: Métricas de calidad
- **AUC**: Área bajo la curva ROC
- **Latency**: Tiempo de respuesta
- **Throughput**: Solicitudes por segundo

## 🔍 Análisis e Interpretación

### Estrategias de Reentrenamiento

1. **Programado**: Cada semana/mes independientemente
2. **Por Drift**: Cuando se detecta degradación
3. **Por Volumen**: Cuando hay suficientes datos nuevos
4. **Por Rendimiento**: Cuando las métricas caen below umbral

### Criterios de Evaluación

| Métrica | Umbral Mínimo | Acción |
|----------|----------------|--------|
| Accuracy | > 0.85 | Aceptar para producción |
| Precision | > 0.80 | Aceptar para producción |
| Recall | > 0.75 | Aceptar para producción |
| AUC | > 0.90 | Aceptar para producción |
| Latencia P95 | < 200ms | Aceptar para producción |

### Estrategias de Despliegue

1. **Blue-Green**: Cambio instantáneo con rollback rápido
2. **Canary**: Despliegue gradual con monitoreo
3. **Shadow**: Ejecución paralela sin afectar usuarios
4. **A/B Testing**: Comparación controlada de versiones

## 📌 Entregables del Laboratorio

| Entregable | Descripción | Formato |
|-------------|-------------|-----------|
| Pipeline de TFX | Reentrenamiento automático con TFX | retraining_pipeline.py |
| Módulo de Entrenamiento | Código para entrenar el modelo | trainer_module.py |
| Configuración de MLflow | Tracking de experimentos y modelos | mlflow_tracking.py |
| Configuración K8s | Despliegue en Kubernetes | kubernetes/ |
| Datos de Ejemplo | Interacciones usuario-item | data/ |
| Documentación | Guía para replicar el pipeline | README.md |

## 🎯 Criterios de Evaluación

- **Funcionalidad** (40%): Pipeline completo funcionando correctamente
- **Automatización** (25%): Reentrenamiento automático implementado
- **Despliegue Seguro** (20%): Canary/shadow deployment funcionando
- **Monitoreo** (15%): Métricas y tracking configurados

## 🚀 Extensión y Mejoras

### Mejoras Sugeridas
1. **Hyperparameter Tuning**: Optimización automática de hiperparámetros
2. **Feature Store**: Gestión centralizada de características
3. **Model Explainability**: SHAP/LIME para explicaciones
4. **Multi-Armedad**: Soporte para múltiples tipos de recomendación

### Aplicaciones en Producción
1. **Real-time Retraining**: Actualización con streaming data
2. **Federated Learning**: Entrenamiento distribuido
3. **Model Monitoring**: Monitoreo continuo de calidad
4. **Auto-scaling**: Escalado basado en carga

---

**Duración Estimada**: 12-15 horas  
**Nivel de Dificultad**: Avanzada  
**Prerrequisitos**: Conocimientos de ML pipelines, Kubernetes, MLOps
