# Guía Conceptual: Optimización de Modelos en 2026

## 📖 Fundamentos Teóricos

### 1. Hyperparameter Tuning Avanzado

Técnicas en 2026:

| Técnica | Descripción | Implementación | Casos de Uso |
|---------|-------------|----------------|--------------|
| Random Search | Búsqueda aleatoria en el espacio de hiperparámetros. | Optuna, Keras Tuner | Espacios de búsqueda grandes (ej: >10 hiperparámetros). |
| Bayesian Optimization | Modelos probabilísticos (Gaussian Processes) para encontrar el óptimo global. | Optuna, Scikit-optimize | Optimización costosa (ej: entrenamiento de LLM). |
| TPE (Tree-structured Parzen Estimator) | Optimización bayesiana con estimadores no paramétricos. | Optuna | Espacios de búsqueda complejos con dependencias. |
| Hyperband | Búsqueda adaptativa que descarta configuraciones pobres temprano. | Keras Tuner, Optuna | Ahorro de recursos en espacios grandes. |
| BOHB (Bayesian Optimization HyperBand) | Combina Bayesian Optimization y Hyperband. | Optuna | Equilibrio entre exploración y explotación. |
| Neural Architecture Search (NAS) | Búsqueda automática de la mejor arquitectura de red. | AutoKeras, Google NAS | Diseño automático de arquitecturas para tareas específicas. |
| Population-Based Training (PBT) | Optimización que ajusta hiperparámetros durante el entrenamiento. | Ray Tune | Entrenamiento de modelos en clusters de GPU. |
| Federated Hyperparameter Tuning | Tuning distribuido preservando privacidad de datos. | TensorFlow Federated + Optuna | Optimización con datos sensibles (ej: salud). |

#### Ejemplo con Optuna + TPE + Pruning:

```python
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import HyperbandPruner
from optuna.trial import TrialState

def objective(trial):
    # Hiperparámetros
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256])
    num_layers = trial.suggest_int("num_layers", 1, 5)
    dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5)
    l2_reg = trial.suggest_float("l2_reg", 1e-5, 1e-2, log=True)

    # Construir modelo
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=(784,)))
    for _ in range(num_layers):
        model.add(tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l2_reg)))
        model.add(tf.keras.layers.Dropout(dropout_rate))
    model.add(tf.keras.layers.Dense(10, activation='softmax'))

    # Compilar y entrenar
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # Callback para pruning
    pruning_callback = optuna.integration.TFKerasPruningCallback(trial, "val_accuracy")

    history = model.fit(
        x_train, y_train,
        batch_size=batch_size,
        epochs=50,
        validation_data=(x_val, y_val),
        callbacks=[pruning_callback],
        verbose=0
    )

    # Reportar métrica para pruning
    trial.report(history.history['val_accuracy'][-1], step=50)

    # Manejar pruning
    if trial.should_prune():
        raise optuna.TrialPruned()

    return history.history['val_accuracy'][-1]

# Configurar estudio con TPE y Hyperband
study = optuna.create_study(
    direction="maximize",
    sampler=TPESampler(n_startup_trials=10),
    pruner=HyperbandPruner(min_resource=1, reduction_factor=3)
)
study.optimize(objective, n_trials=100, timeout=3600)  # 1 hora de optimización

# Resultados
print("Mejor trial:")
trial = study.best_trial
print(f"  Accuracy: {trial.value:.4f}")
print(f"  Parámetros: {trial.params}")
```

### 2. Regularización y Normalización Avanzada

Técnicas en 2026:

| Técnica | Descripción | Implementación en TensorFlow/Keras | Cuándo Usar |
|---------|-------------|-----------------------------------|-------------|
| L1/L2 Regularization | Penaliza pesos grandes para evitar overfitting. | kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0.01, l2=0.01) | Modelos con muchos parámetros. |
| Dropout | Desactiva aleatoriamente neuronas durante el entrenamiento (p=0.2-0.5). | tf.keras.layers.Dropout(0.3) | Redes profundas (CNNs, RNNs). |
| SpatialDropout2D | Dropout aplicado a mapas de características (ej: CNNs). | tf.keras.layers.SpatialDropout2D(0.2) | CNNs para imágenes. |
| Batch Normalization | Normaliza activaciones entre capas para estabilizar el entrenamiento. | tf.keras.layers.BatchNormalization() | Redes profundas (evita vanishing gradients). |
| Layer Normalization | Normaliza activaciones dentro de una capa (alternativa a BatchNorm). | tf.keras.layers.LayerNormalization() | Transformers, RNNs. |
| Group Normalization | Normaliza por grupos de canales (útil para pequeños batch sizes). | tf.keras.layers.GroupNormalization(groups=8) | CNNs con batch sizes pequeños. |
| Weight Normalization | Normaliza pesos en lugar de activaciones. | tf.keras.layers.experimental.preprocessing.Normalization() | Redes muy profundas. |
| Spectral Normalization | Normaliza pesos por su norma espectral (estabilidad en GANs). | tf.keras.constraints.UnitNorm() + custom layer. | GANs y redes adversariales. |
| Stochastic Depth | Desactiva capas completas durante el entrenamiento (p=0.1-0.3). | Implementación custom con tf.keras.layers.Dropout en residual connections. | Redes residuales (ResNet, EfficientNet). |

#### Ejemplo con Regularización Avanzada:

```python
from tensorflow.keras import layers, models, regularizers, constraints

def build_regularized_model(input_shape, num_classes):
    model = models.Sequential()

    # Capa de entrada con normalización
    model.add(layers.Input(shape=input_shape))
    model.add(layers.experimental.preprocessing.Normalization())
    model.add(layers.Conv2D(32, (3, 3), activation='relu', kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4)))
    model.add(layers.BatchNormalization())
    model.add(layers.SpatialDropout2D(0.2))

    # Bloques residuales con stochastic depth
    for _ in range(3):
        model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
        model.add(layers.BatchNormalization())
        model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
        model.add(layers.BatchNormalization())
        # Stochastic depth (p=0.2)
        if np.random.random() < 0.2:
            model.add(layers.Lambda(lambda x: x * 0))

    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dense(128, activation='relu', kernel_constraint=constraints.UnitNorm()))
    model.add(layers.LayerNormalization())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(num_classes, activation='softmax'))

    return model

model = build_regularized_model((32, 32, 3), 10)
model.summary()
```

### 3. Optimización de Modelos para Producción

Técnicas en 2026:

| Técnica | Descripción | Herramienta | Impacto |
|---------|-------------|------------|---------|
| Post-Training Quantization | Reduce precisión de pesos (float32 → int8) sin reentrenamiento. | TensorFlow Lite, ONNX Runtime | Reduce tamaño del modelo en ~4x con pérdida mínima de precisión. |
| Quantization-Aware Training | Entrena el modelo con simulación de quantization para mejor precisión. | TensorFlow Model Optimization | Mejor precisión que post-training quantization. |
| Pruning | Elimina pesos no significativos (sparsity). | TensorFlow Model Optimization | Reduce tamaño del modelo en 50-90%. |
| Structured Pruning | Elimina neuronas o filtros completos (vs. pesos individuales). | TensorFlow Model Optimization | Mejor compatibilidad con hardware. |
| Distillation | Entrena un modelo pequeño ("student") usando un modelo grande ("teacher"). | Hugging Face DistilBERT | Mantiene 90-95% de la precisión con 50% menos parámetros. |
| Neural Architecture Search (NAS) | Busca automáticamente la mejor arquitectura para una tarea. | AutoKeras, Google NAS | Puede superar arquitecturas diseñadas manualmente. |
| ONNX Optimization | Optimiza modelos en formato ONNX para múltiples backends. | ONNX Runtime | Inferencia acelerada en CPU/GPU. |
| TensorRT Optimization | Optimiza modelos para GPUs NVIDIA (fusion de capas, kernel auto-tuning). | TensorRT | Latencia reducida en 10-100x en GPUs NVIDIA. |
| Hardware-Aware Optimization | Optimiza el modelo para hardware específico (ej: TPUs, NPUs). | TensorFlow Lite, Apache TVM | Mejor rendimiento en edge devices. |
| Dynamic Quantization | Aplica quantization dinámicamente en tiempo de ejecución. | PyTorch, ONNX Runtime | Flexibilidad para diferentes precisiones. |

#### Ejemplo de Quantization-Aware Training con TF:

```python
import tensorflow_model_optimization as tfmot

# Definir modelo
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Aplicar quantization-aware training
quantize_model = tfmot.quantization.keras.quantize_model

# `q_aware` significa que el modelo usará fake quantization durante el entrenamiento
q_aware_model = quantize_model(model)

# Compilar y entrenar
q_aware_model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Entrenar con datos de entrenamiento
q_aware_model.fit(x_train, y_train, epochs=5, validation_data=(x_val, y_val))

# Convertir a modelo TFLite con quantization
converter = tf.lite.TFLiteConverter.from_keras_model(q_aware_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
quantized_tflite_model = converter.convert()

# Guardar modelo quantizado
with open('quantized_model.tflite', 'wb') as f:
    f.write(quantized_tflite_model)
```

### 4. Neural Architecture Search (NAS)

Enfoques en 2026:

| Enfoque | Descripción | Herramienta | Ventajas |
|---------|-------------|------------|---------|
| AutoML | Búsqueda automática de arquitecturas y hiperparámetros. | AutoKeras, Google AutoML | Fácil de usar, buena para prototipos. |
| Diferenciable NAS | Optimiza la arquitectura usando gradientes. | DARTS, P-DARTS | Más eficiente que métodos basados en RL. |
| NAS basado en Refuerzo | Usa aprendizaje por refuerzo para explorar el espacio de arquitecturas. | ENAS, PNAS | Puede encontrar arquitecturas innovadoras. |
| NAS basado en Evolución | Usa algoritmos evolutivos para optimizar arquitecturas. | AmoebaNet, NSGA-Net | Buen equilibrio exploración/explotación. |
| Once-for-All (OFA) | Entrena una super-red una vez y muestra sub-redes óptimas para diferentes recursos. | OFA | Eficiencia en entrenamiento. |
| Hardware-Aware NAS | Optimiza arquitecturas para hardware específico (ej: latencia en móviles). | MnasNet, ProxylessNAS | Mejor rendimiento en edge devices. |
| Neural Predictor | Usa un modelo para predecir el rendimiento de arquitecturas. | NAO | Reduce costo computacional. |

#### Ejemplo con AutoKeras para Clasificación de Imágenes:

```python
import autokeras as ak

# Buscar la mejor arquitectura para CIFAR-10
clf = ak.ImageClassifier(
    max_trials=20,  # Número de arquitecturas a probar
    num_classes=10,
    objective='val_accuracy',
    directory='autokeras_cifar10',
    project_name='cifar10_classifier'
)

# Entrenar
history = clf.fit(
    x_train, y_train,
    validation_data=(x_val, y_val),
    epochs=50,
    callbacks=[
        ak.callbacks.EarlyStopping(patience=5),
        ak.callbacks.ModelCheckpoint('best_model.h5', save_best_only=True)
    ]
)

# Evaluar
loss, accuracy = clf.evaluate(x_test, y_test)
print(f"Test accuracy: {accuracy:.4f}")

# Exportar el mejor modelo
best_model = clf.export_model()
best_model.save('autokeras_best_model.h5', save_format='h5')
```

### 5. Evaluación de Modelos Optimizados

Métricas Clave en 2026:

| Métrica | Descripción | Implementación | Herramientas |
|---------|-------------|----------------|-------------|
| Accuracy | Proporción de predicciones correctas. | tf.keras.metrics.Accuracy() | TensorFlow, PyTorch |
| Precision/Recall | Métricas para clases desbalanceadas. | tf.keras.metrics.Precision(), Recall() | Scikit-learn, TensorFlow |
| F1-Score | Media armónica de precision y recall. | tf.keras.metrics.F1Score() | Scikit-learn |
| ROC-AUC | Área bajo la curva ROC (para clasificadores binarios). | tf.keras.metrics.AUC() | Scikit-learn |
| PR-AUC | Área bajo la curva Precision-Recall (mejor para clases desbalanceadas). | sklearn.metrics.average_precision_score() | Scikit-learn |
| Log Loss | Pérdida para clasificación probabilística. | tf.keras.metrics.BinaryCrossentropy(from_logits=True) | TensorFlow |
| RMSE | Raíz del error cuadrático medio (regresión). | tf.keras.metrics.RootMeanSquaredError() | TensorFlow |
| MAE | Error absoluto medio (regresión). | tf.keras.metrics.MeanAbsoluteError() | TensorFlow |
| R² | Proporción de varianza explicada (regresión). | tf.keras.metrics.R2Score() | TensorFlow 2.11+ |
| Latencia | Tiempo de inferencia por muestra. | timeit o tf.profiler | Python, TensorFlow |
| Throughput | Número de muestras procesadas por segundo. | Benchmark con batches de datos. | PyTorch, ONNX Runtime |
| FLOPs | Operaciones de punto flotante por segundo. | tf.profiler o thop (PyTorch) | TensorFlow, PyTorch |
| Tamaño del Modelo | Peso del modelo en MB. | os.path.getsize("model.h5") | Python |
| Sparsity | Proporción de pesos iguales a cero (después de pruning). | tf.nn.zero_fraction() | TensorFlow |
| Robustez | Resistencia a datos ruidosos o adversariales. | foolbox o cleverhans | Foolbox, CleverHans |
| Fairness | Equidad entre grupos demográficos. | fairlearn, AIF360 | Fairlearn, IBM AIF360 |
| Explicabilidad | Capacidad de explicar predicciones individuales. | shap, lime | SHAP, LIME |

#### Ejemplo de Benchmarking Completo:

```python
import time
import tensorflow as tf
import numpy as np
from thop import profile  # Para contar FLOPs (PyTorch)
import os

def benchmark_model(model, input_shape, num_samples=100, num_runs=10):
    # Generar datos de entrada aleatorios
    input_data = np.random.random((num_samples, *input_shape)).astype(np.float32)

    # Medir latencia
    latencies = []
    for _ in range(num_runs):
        start_time = time.time()
        _ = model.predict(input_data)
        latencies.append(time.time() - start_time)
    avg_latency = np.mean(latencies) * 1000  # ms

    # Medir throughput
    start_time = time.time()
    for _ in range(num_samples):
        _ = model.predict(input_data[0:1])  # Predicción individual
    throughput = num_samples / (time.time() - start_time)  # samples/second

    # Medir FLOPs (requiere PyTorch o conversión)
    # Para TensorFlow, podemos estimar:
    total_params = model.count_params()
    flops = 2 * total_params  # Estimación aproximada (cada parámetro participa en ~2 ops)

    # Tamaño del modelo
    model_size = os.path.getsize('model.h5') / (1024 * 1024)  # MB

    # Precisión
    _, accuracy = model.evaluate(x_test, y_test, verbose=0)

    return {
        'accuracy': accuracy,
        'latency_ms': avg_latency,
        'throughput_sps': throughput,
        'flops': flops,
        'model_size_mb': model_size,
        'params': total_params
    }

# Ejemplo de uso
model = tf.keras.models.load_model('model.h5')
metrics = benchmark_model(model, input_shape=(224, 224, 3))
print("Métricas del modelo:")
for k, v in metrics.items():
    print(f"{k}: {v:.2f}" if isinstance(v, float) else f"{k}: {v}")
```

### 6. Pipelines de Optimización con TFX y MLflow

Arquitectura de un pipeline de optimización en 2026:

```
[Datos Crudos] → [Preprocesamiento] → [Hyperparameter Tuning] → [Entrenamiento] → [Optimización (Pruning/Quantization)] → [Evaluación] → [Despliegue] → [Monitoreo]
```

#### Ejemplo con TFX + MLflow:

```python
from tfx.orchestration import pipeline
from tfx.components import (
    ExampleGen, StatisticsGen, SchemaGen, Transform,
    Tuner, Trainer, Evaluator, Pusher
)
from tfx.proto import pusher_pb2, trainer_pb2
from tfx.dsl.components.common import resolver
from tfx.types import standard_artifacts
import mlflow

def create_optimization_pipeline():
    # 1. Ingestión de datos
    example_gen = ExampleGen(input_base='data')

    # 2. Estadísticas y esquema
    statistics_gen = StatisticsGen(examples=example_gen.outputs['examples'])
    schema_gen = SchemaGen(statistics=statistics_gen.outputs['statistics'])

    # 3. Transformación
    transform = Transform(
        examples=example_gen.outputs['examples'],
        schema=schema_gen.outputs['schema']
    )

    # 4. Tuning de hiperparámetros con KerasTuner
    tuner = Tuner(
        module_file='tuner_module.py',
        examples=transform.outputs['transformed_examples'],
        schema=schema_gen.outputs['schema'],
        transform_graph=transform.outputs['transform_graph'],
        train_args=trainer_pb2.TrainArgs(num_steps=5000),
        eval_args=trainer_pb2.EvalArgs(num_steps=2000)
    )

    # 5. Entrenamiento con los mejores hiperparámetros
    trainer = Trainer(
        module_file='trainer_module.py',
        examples=transform.outputs['transformed_examples'],
        schema=schema_gen.outputs['schema'],
        transform_graph=transform.outputs['transform_graph'],
        hyperparameters=tuner.outputs['best_hyperparameters'],
        train_args=trainer_pb2.TrainArgs(num_steps=20000),
        eval_args=trainer_pb2.EvalArgs(num_steps=5000)
    )

    # 6. Optimización post-entrenamiento
    optimizer = Optimizer(
        model=trainer.outputs['model'],
        optimization_parameters={
            'pruning': {'initial_sparsity': 0.3, 'final_sparsity': 0.7},
            'quantization': {'scheme': 'post_training_dynamic_range'}
        }
    )

    # 7. Evaluación
    evaluator = Evaluator(
        examples=example_gen.outputs['examples'],
        model=optimizer.outputs['optimized_model'],
        baseline_model=None,
        eval_config='eval_config.json'
    )

    # 8. Despliegue y tracking con MLflow
    pusher = Pusher(
        model=optimizer.outputs['optimized_model'],
        push_destination=pusher_pb2.PushDestination(
            filesystem=pusher_pb2.PushDestination.Filesystem(
                base_directory='serving_model'
            )
        )
    )

    # Tracking con MLflow
    mlflow_tracker = MLflowTracker(
        model=optimizer.outputs['optimized_model'],
        eval_results=evaluator.outputs['evaluation']
    )

    return pipeline.Pipeline(
        pipeline_name="model_optimization_pipeline",
        pipeline_root="gs://optimization_pipeline/root",
        components=[
            example_gen,
            statistics_gen,
            schema_gen,
            transform,
            tuner,
            trainer,
            optimizer,
            evaluator,
            pusher,
            mlflow_tracker
        ],
        enable_cache=True
    )

# Componentes personalizados para MLflow
class MLflowTracker(tfx.components.BaseComponent):
    SPEC_CLASS = tfx.components.ComponentSpec(
        inputs={
            'model': tfx.dsl.components.InputArtifact[standard_artifacts.Model],
            'eval_results': tfx.dsl.components.InputArtifact[standard_artifacts.ModelEvaluation]
        }
    )

    def __init__(self, model, eval_results):
        super().__init__(spec=self.SPEC_CLASS)
        self.model = model
        self.eval_results = eval_results

    def execute(self):
        # Cargar modelo y métricas
        model = tf.keras.models.load_model(self.model.uri)
        with open(os.path.join(self.eval_results.uri, 'metrics.json')) as f:
            metrics = json.load(f)

        # Iniciar run de MLflow
        with mlflow.start_run():
            # Loguear parámetros (hiperparámetros del modelo)
            mlflow.log_params(metrics['hyperparameters'])

            # Loguear métricas
            mlflow.log_metrics({
                'accuracy': metrics['accuracy'],
                'loss': metrics['loss'],
                'auc': metrics['auc']
            })

            # Loguear modelo
            mlflow.tensorflow.log_model(model, "model")

            # Loguear artefactos adicionales
            mlflow.log_artifact(self.eval_results.uri, "evaluation")

        return {}
```

## 📚 Recursos Adicionales

### Documentación Oficial
- [Optuna Documentation](https://optuna.org/)
- [Keras Tuner Documentation](https://www.tensorflow.org/tutorials/keras/keras_tuner)
- [TensorFlow Model Optimization](https://www.tensorflow.org/model_optimization)
- [AutoKeras Documentation](https://autokeras.com/)

### Papers Fundamentales
- "DARTS: Differentiable Architecture Search" - Liu et al.
- "Optuna: A Next-generation Hyperparameter Optimization Framework" - Akiba et al.
- "EfficientNet: Rethinking Model Scaling" - Tan & Le
- "The Lottery Ticket Hypothesis" - Frankle & Carbin

---

**Última Actualización**: Febrero 2026  
**Versión**: 1.0  
**Duración Estimada**: 8 horas de estudio teórico
