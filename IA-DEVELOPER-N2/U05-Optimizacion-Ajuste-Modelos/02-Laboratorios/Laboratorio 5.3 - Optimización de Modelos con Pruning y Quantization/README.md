# Laboratorio 5.3: Optimización de Modelos con Pruning, Quantization y Distillation

## 🎯 Objetivos del Laboratorio

### Objetivo General
Optimizar un modelo de clasificación de imágenes para dispositivos móviles aplicando técnicas avanzadas de pruning, quantization y distillation para reducir tamaño y latencia manteniendo precisión.

### Objetivos Específicos
- Aplicar pruning estructurado para reducir el número de parámetros
- Implementar quantization-aware training para mantener precisión
- Desarrollar distillation para crear un modelo estudiante más pequeño
- Comparar trade-offs entre tamaño, precisión y latencia
- Optimizar para edge devices con restricciones computacionales

## 📋 Marco Lógico del Proyecto

| Componente | Objetivo | Indicadores | Medios de Verificación |
|------------|----------|-------------|------------------------|
| **Carga de Datos** | Preparar dataset CIFAR-10 | Dataset cargado y preprocesado | Scripts de carga y estadísticas |
| **Modelo Teacher** | Crear modelo base grande | Accuracy >92% en test | Modelo ResNet50V2 entrenado |
| **Pruning Estructurado** | Reducir tamaño del modelo | Reducción 50-70% en parámetros | Código de pruning y análisis |
| **Quantization-Aware** | Optimizar para inferencia | Modelo <5MB con <1% pérdida | Código QAT y conversión TFLite |
| **Distillation** | Crear modelo estudiante | 90-95% precisión del teacher | Implementación teacher-student |
| **Benchmarking** | Evaluar optimizaciones | Tabla comparativa completa | Suite de benchmarking |
| **Optimización Final** | Combinar todas las técnicas | Modelo optimizado final | Modelo final con métricas |

## 🛠️ Tecnologías y Herramientas

### Principales Tecnologías
- **TensorFlow 2.15**: Framework de deep learning
- **TensorFlow Model Optimization 0.7**: Pruning y quantization
- **TensorFlow Datasets 4.9**: Dataset CIFAR-10
- **ONNX Runtime 1.16**: Inferencia optimizada
- **TensorRT 8.6**: Optimización para GPUs NVIDIA

### Dependencias Adicionales
- **Thop**: Conteo de FLOPs (PyTorch)
- **NumPy**: Computación numérica
- **Matplotlib/Seaborn**: Visualización
- **Time**: Medición de latencia

## 📁 Estructura del Proyecto

```
Laboratorio 5.3 - Optimización de Modelos/
├── README.md                           # Guía del laboratorio
├── requirements.txt                    # Dependencias
├── src/
│   ├── load_cifar10.py                # Carga y preprocesamiento
│   ├── teacher_model.py               # Modelo teacher (ResNet50V2)
│   ├── model_pruning.py               # Pruning estructurado
│   ├── quantization_aware_training.py # QAT implementation
│   ├── model_distillation.py          # Teacher-student distillation
│   └── benchmark_models.py            # Benchmarking completo
├── models/
│   ├── teacher_model.h5               # Modelo teacher
│   ├── pruned_model.h5                # Modelo podado
│   ├── qat_model.h5                   # Modelo quantization-aware
│   ├── student_model.h5                # Modelo estudiante
│   └── final_optimized_model.tflite   # Modelo final optimizado
├── results/
│   ├── benchmark_results.csv          # Resultados de benchmarking
│   ├── model_comparisons/             # Gráficos comparativos
│   └── optimization_analysis/         # Análisis de optimización
└── docs/
    ├── methodology_report.md          # Reporte metodológico
    └── technical_documentation.md    # Documentación técnica
```

## 🔧 Implementación Detallada

### Fase 1: Carga y Preprocesamiento (CIFAR-10)

#### Dataset de Imágenes para Clasificación
```python
import tensorflow as tf
import tensorflow_datasets as tfds

def load_and_preprocess_cifar10():
    """
    Carga y preprocesa el dataset CIFAR-10
    """
    (train_ds, val_ds, test_ds), info = tfds.load(
        'cifar10',
        split=['train[:80%]', 'train[80%:90%]', 'train[90%:]'],
        shuffle_files=True,
        as_supervised=True,
        with_info=True
    )

    def preprocess(image, label):
        image = tf.image.resize(image, (32, 32))
        image = tf.cast(image, tf.float32) / 255.0
        return image, label

    train_ds = train_ds.map(preprocess).batch(128).prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.map(preprocess).batch(128).prefetch(tf.data.AUTOTUNE)
    test_ds = test_ds.map(preprocess).batch(128).prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds, test_ds, info
```

### Fase 2: Modelo Teacher (ResNet50V2)

#### Arquitectura Grande como Base
```python
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras import layers, models

def build_teacher_model(input_shape=(32, 32, 3), num_classes=10):
    """
    Construye el modelo teacher usando ResNet50V2
    """
    # Usar ResNet50V2 como teacher (transfer learning)
    base_model = ResNet50V2(
        include_top=False,
        weights='imagenet',
        input_shape=input_shape,
        pooling='avg'
    )
    base_model.trainable = False  # Congelar capas base inicialmente

    inputs = layers.Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model
```

### Fase 3: Pruning Estructurado

#### Reducción de Parámetros con TFMOT
```python
import tensorflow_model_optimization as tfmot

def apply_structured_pruning(model, initial_sparsity=0.30, final_sparsity=0.70):
    """
    Aplica pruning estructurado al modelo
    """
    # Definir pruning schedule
    pruning_params = {
        'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(
            initial_sparsity=initial_sparsity,
            final_sparsity=final_sparsity,
            begin_step=2000,
            end_step=6000
        )
    }

    # Aplicar pruning a las capas densas
    prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude

    pruned_model = tf.keras.Sequential()
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Dense) and layer.name != 'predictions':
            pruned_layer = prune_low_magnitude(layer, **pruning_params)
            pruned_model.add(pruned_layer)
        else:
            pruned_model.add(layer)

    return pruned_model
```

### Fase 4: Quantization-Aware Training

#### Optimización para Inferencia
```python
def apply_quantization_aware_training(model):
    """
    Aplica quantization-aware training
    """
    quantize_model = tfmot.quantization.keras.quantize_model

    # Quantizar solo las capas densas (no las convolucionales de ResNet)
    def apply_quantization_to_model(model):
        for layer in model.layers:
            if isinstance(layer, tf.keras.layers.Dense):
                layer._name = 'quant_' + layer.name  # Marcador para quantization

        # Aplicar QAT
        qat_model = quantize_model(model)
        qat_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        return qat_model

    return apply_quantization_to_model(model)
```

### Fase 5: Knowledge Distillation

#### Teacher-Student Architecture
```python
def build_student_model(input_shape=(32, 32, 3), num_classes=10):
    """
    Construye el modelo estudiante más pequeño
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

class DistillationLoss(tf.keras.losses.Loss):
    """
    Función de pérdida para distillation
    """
    def __init__(self, teacher_model, alpha=0.5, temperature=5.0):
        super().__init__()
        self.teacher_model = teacher_model
        self.alpha = alpha
        self.temperature = temperature

    def call(self, y_true, y_pred):
        # Pérdida estándar
        student_loss = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)

        # Pérdida de distillation
        teacher_pred = self.teacher_model(y_true[0])  # y_true[0] son las imágenes
        teacher_loss = tf.keras.losses.kullback_leibler_divergence(
            tf.nn.softmax(teacher_pred / self.temperature, axis=-1),
            tf.nn.softmax(y_pred / self.temperature, axis=-1)
        ) * (self.temperature ** 2)

        return self.alpha * student_loss + (1 - self.alpha) * teacher_loss
```

### Fase 6: Benchmarking Completo

#### Evaluación de Modelos Optimizados
```python
import time
import os
import numpy as np

def benchmark_model(model, input_shape, test_ds, model_name):
    """
    Evalúa rendimiento completo del modelo
    """
    # Tamaño del modelo
    if isinstance(model, tf.keras.Model):
        model_size = os.path.getsize(f'{model_name}.h5') / (1024 * 1024)  # MB
    else:  # TFLite
        model_size = os.path.getsize(f'{model_name}.tflite') / (1024 * 1024)

    # Latencia y throughput
    latencies = []
    for _ in range(100):
        start = time.time()
        if isinstance(model, tf.keras.Model):
            _ = model.predict(test_ds.take(1))
        else:
            # TFLite inference
            interpreter = tf.lite.Interpreter(model_content=model)
            interpreter.allocate_tensors()
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            
            input_data = test_ds.take(1).as_numpy_iterator().next()[0]
            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()
        latencies.append(time.time() - start)
    
    avg_latency = np.mean(latencies) * 1000  # ms

    # Medir throughput
    start = time.time()
    count = 0
    for batch in test_ds.take(10):
        if isinstance(model, tf.keras.Model):
            _ = model.predict(batch[0])
        else:
            interpreter.set_tensor(input_details[0]['index'], batch[0].numpy())
            interpreter.invoke()
        count += batch[0].shape[0]
    throughput = count / (time.time() - start)  # samples/second

    # Precisión
    if isinstance(model, tf.keras.Model):
        _, accuracy = model.evaluate(test_ds)
    else:
        # Evaluación TFLite
        correct = 0
        total = 0
        for batch in test_ds:
            images, labels = batch
            interpreter.set_tensor(input_details[0]['index'], images.numpy())
            interpreter.invoke()
            predictions = interpreter.get_tensor(output_details[0]['index'])
            correct += np.sum(np.argmax(predictions, axis=1) == labels.numpy())
            total += len(labels)
        accuracy = correct / total

    return {
        'model': model_name,
        'accuracy': accuracy,
        'latency_ms': avg_latency,
        'throughput_sps': throughput,
        'model_size_mb': model_size
    }
```

## 📊 Entregables del Laboratorio

### 1. Código Fuente Completo
- **load_cifar10.py**: Carga y preprocesamiento de CIFAR-10
- **teacher_model.py**: Modelo ResNet50V2 teacher
- **model_pruning.py**: Implementación de pruning estructurado
- **quantization_aware_training.py**: Optimización QAT
- **model_distillation.py**: Teacher-student distillation
- **benchmark_models.py**: Suite de benchmarking completo

### 2. Modelos Optimizados
- **teacher_model.h5**: Modelo base grande
- **pruned_model.h5**: Modelo podado
- **qat_model.h5**: Modelo quantization-aware
- **student_model.h5**: Modelo estudiante
- **final_optimized_model.tflite**: Modelo final optimizado

### 3. Resultados y Análisis
- **benchmark_results.csv**: Tabla comparativa completa
- **model_comparisons/**: Gráficos de trade-offs
- **optimization_analysis/**: Análisis detallado de optimizaciones

### 4. Documentación
- **methodology_report.md**: Reporte metodológico completo
- **technical_documentation.md**: Guías técnicas
- **README.md**: Instrucciones de uso

## 🎯 Criterios de Evaluación

### Componente Técnico (60%)
- **Implementación de Técnicas** (25%): Correcta aplicación de pruning, QAT, distillation
- **Optimización Completa** (20%): Modelo final optimizado con todas las técnicas
- **Benchmarking** (15%): Evaluación exhaustiva de rendimiento

### Componente Analítico (40%)
- **Análisis de Trade-offs** (20%): Balance entre tamaño, precisión, latencia
- **Documentación** (20%): Reportes completos y reproducibles

### Métricas de Éxito
- **Reducción de Tamaño**: >70% vs modelo original
- **Mantenimiento de Precisión**: >85% del modelo original
- **Reducción de Latencia**: >50% vs modelo original
- **Modelo Final**: <5MB con precisión >85%

## 🚀 Extensiones y Mejoras

### Opciones Avanzadas
1. **Hardware-Aware Optimization**: Optimización específica para dispositivos
2. **Neural Architecture Search**: Búsqueda automática de arquitecturas eficientes
3. **Mixed Precision Training**: Entrenamiento con precisión mixta
4. **Federated Learning**: Optimización preservando privacidad

### Aplicaciones en Producción
1. **Mobile Deployment**: Despliegue en dispositivos móviles
2. **Edge Computing**: Optimización para IoT y edge devices
3. **Cloud Optimization**: Optimización para servidores en la nube
4. **Real-time Inference**: Optimización para inferencia en tiempo real

## 📈 Análisis Esperado

### Trade-offs entre Técnicas

| Técnica | Reducción Tamaño | Pérdida Precisión | Reducción Latencia | Complejidad |
|---------|------------------|-------------------|-------------------|-------------|
| **Baseline** | 0% | 0% | 0% | Baja |
| **Pruning** | 50-70% | 1-3% | 20-40% | Media |
| **Quantization** | 75% | 2-5% | 40-60% | Media |
| **Distillation** | 60-80% | 5-10% | 30-50% | Alta |
| **Combinación** | 85-95% | 5-15% | 60-80% | Muy Alta |

### Insights Esperados
1. **Pruning**: Efectivo para reducir parámetros con mínima pérdida de precisión
2. **Quantization**: Mayor reducción de tamaño y latencia
3. **Distillation**: Balance óptimo entre tamaño y precisión
4. **Combinación**: Sinergia produce mejores resultados globales

---

**Duración Estimada**: 8-10 horas  
**Dificultad**: Avanzada  
**Prerrequisitos**: Conocimientos de deep learning y optimización de modelos
