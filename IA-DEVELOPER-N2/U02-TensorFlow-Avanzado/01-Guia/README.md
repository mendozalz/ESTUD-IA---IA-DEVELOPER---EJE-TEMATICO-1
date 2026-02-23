# U02 - Programación avanzada con TensorFlow

## Descripción breve

Construcción de modelos con TensorFlow/Keras a nivel avanzado: capas personalizadas, `tf.data`, callbacks, manejo de datasets complejos, y despliegue en producción para estudiantes de maestría.

## Objetivos de aprendizaje específicos

- Desarrollar modelos de IA para automatizar procesos empresariales.
- Implementar modelos con capas personalizadas y/o subclases (`tf.keras.Model`).
- Preparar pipelines de entrada eficientes con `tf.data` para entrenamiento escalable.
- Aplicar técnicas avanzadas de optimización y despliegue en producción.
- Integrar modelos con sistemas empresariales (APIs, bases de datos, IoT).

## Temas clave a cubrir

- API funcional vs. subclases, composición de modelos.
- `tf.data`: lectura, transformaciones, `batch`, `prefetch`.
- Capas personalizadas: implementación y serialización.
- Callbacks: EarlyStopping, ModelCheckpoint, TensorBoard.
- Optimización: Keras Tuner, mixed precision, entrenamiento distribuido.
- Exportación de modelos (SavedModel, TensorFlow Lite) para inferencia.
- Despliegue: TensorFlow Serving, FastAPI, Docker.

## Lecciones sugeridas

- Lección 1: Arquitecturas con Keras (multi-input, multi-output).
- Lección 2: `tf.data` para imágenes/texto/tabular.
- Lección 3: Capas/métricas personalizadas.
- Lección 4: Optimización y búsqueda de hiperparámetros.
- Lección 5: Despliegue y producción.
- Lección 6: Integración con sistemas empresariales.

## Programación Avanzada con TensorFlow: Construcción de Modelos con Capas Personalizadas y Manejo de Datasets Complejos

En este tema, profundizaremos en cómo construir modelos avanzados con TensorFlow, enfocándonos en capas personalizadas y el manejo de datasets complejos. Cubriremos desde los fundamentos teóricos hasta ejemplos prácticos, incluyendo cómo trabajar con datos estructurados, imágenes, texto y series temporales.

### 1. Introducción a Capas Personalizadas en TensorFlow

#### ¿Qué son las capas personalizadas?
Las capas personalizadas en TensorFlow permiten definir operaciones específicas que no están disponibles en las capas estándar de Keras. Esto es útil para:
- Implementar operaciones matemáticas personalizadas
- Crear arquitecturas innovadoras (ej. cápsulas, atención personalizada)
- Integrar lógica de negocio específica en el modelo

#### Cómo Crear una Capa Personalizada
Para crear una capa personalizada, debes heredar de `tf.keras.layers.Layer` y definir los métodos `build` y `call`.

**Ejemplo Básico: Capa Lineal Personalizada**
```python
import tensorflow as tf
from tensorflow.keras import layers

class LinearLayer(layers.Layer):
    def __init__(self, units=32):
        super(LinearLayer, self).__init__()
        self.units = units

    def build(self, input_shape):
        # Crear pesos entrenables
        self.w = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer='random_normal',
            trainable=True
        )
        self.b = self.add_weight(
            shape=(self.units,),
            initializer='zeros',
            trainable=True
        )

    def call(self, inputs):
        # Operación lineal: y = w*x + b
        return tf.matmul(inputs, self.w) + self.b

# Uso en un modelo
model = tf.keras.Sequential([
    LinearLayer(64),
    layers.Activation('relu'),
    layers.Dense(10, activation='softmax')
])
```

**Ejemplo Avanzado: Capa de Atención Personalizada**
```python
class AttentionLayer(layers.Layer):
    def __init__(self, units):
        super(AttentionLayer, self).__init__()
        self.units = units
        self.W1 = None
        self.W2 = None
        self.V = None

    def build(self, input_shape):
        self.W1 = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer='random_normal',
            trainable=True
        )
        self.W2 = self.add_weight(
            shape=(input_shape[1], self.units),
            initializer='random_normal',
            trainable=True
        )
        self.V = self.add_weight(
            shape=(self.units, 1),
            initializer='random_normal',
            trainable=True
        )

    def call(self, inputs):
        # inputs shape: (batch_size, seq_len, input_dim)
        score = tf.nn.tanh(tf.matmul(inputs, self.W1) + tf.matmul(inputs, self.W2, transpose_b=True))
        attention_weights = tf.nn.softmax(tf.matmul(score, self.V), axis=1)
        context = tf.reduce_sum(attention_weights * inputs, axis=1)
        return context

    def get_config(self):
        config = super().get_config()
        config.update({"units": self.units})
        return config
```

### 2. Manejo de Datasets Complejos

Los datasets complejos pueden incluir:
- Imágenes de alta resolución
- Secuencias temporales largas
- Datos estructurados con relaciones complejas (ej. grafos)
- Datos multimodales (ej. texto + imágenes)

TensorFlow proporciona herramientas para manejar estos datos de manera eficiente, como `tf.data.Dataset`.

#### Ejemplo 1: Carga de Imágenes con tf.data
```python
def load_and_preprocess_image(path, label):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [224, 224])
    image = tf.cast(image, tf.float32) / 255.0  # Normalizar a [0, 1]
    return image, label

# Crear un dataset a partir de una lista de rutas de imágenes
image_paths = ["path/to/image1.jpg", "path/to/image2.jpg", ...]
labels = [0, 1, ...]  # Etiquetas correspondientes
dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
dataset = dataset.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
dataset = dataset.batch(32).prefetch(tf.data.AUTOTUNE)
```

#### Ejemplo 2: Datos de Series Temporales con Ventanas Deslizantes
```python
def create_time_series_dataset(data, window_size, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices(data)
    dataset = dataset.window(window_size + 1, shift=1, drop_remainder=True)
    dataset = dataset.flat_map(lambda window: window.batch(window_size + 1))
    dataset = dataset.map(lambda window: (window[:-1], window[-1]))
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset

# Datos de ejemplo: serie temporal sintética
data = tf.range(1000, dtype=tf.float32)
dataset = create_time_series_dataset(data, window_size=10, batch_size=32)
```

#### Ejemplo 3: Datos de Grafos con Spektral
```python
import spektral as sp
import spektral.datasets as spd

# Cargar un dataset de grafos (ej. Cora)
dataset = spd.Cora()
A, X, y = dataset.adj, dataset.x, dataset.y  # Matriz de adyacencia, features, etiquetas

# Crear un dataset de TensorFlow
graph_dataset = tf.data.Dataset.from_tensor_slices((A, X, y))
graph_dataset = graph_dataset.batch(32).prefetch(tf.data.AUTOTUNE)
```

### 3. Optimización y Buenas Prácticas

#### Optimización del Rendimiento
```python
# Uso de tf.data con optimización
dataset = dataset.cache().prefetch(tf.data.AUTOTUNE)

# Entrenamiento distribuido
strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    model = build_model()  # Construir el modelo dentro del scope

# Mixed Precision
policy = tf.keras.mixed_precision.Policy('mixed_float16')
tf.keras.mixed_precision.set_global_policy(policy)
```

#### Buenas Prácticas para Capas Personalizadas
```python
# Inicialización de pesos con regularización
self.w = self.add_weight(
    shape=(input_shape[-1], self.units),
    initializer='glorot_uniform',
    trainable=True,
    regularizer=tf.keras.regularizers.l2(0.01)
)

# Serialización de capas personalizadas
def get_config(self):
    config = super().get_config()
    config.update({"units": self.units})
    return config

@classmethod
def from_config(cls, config):
    return cls(**config)
```

### 4. Casos de Uso Empresariales

#### Caso 1: Detección de Defectos en Manufactura
- **Sector**: Industria 4.0
- **Tecnologías**: CNN + tf.data + TensorFlow Lite
- **Objetivo**: Reducir defectos en línea de producción
- **ROI**: 15-20% reducción de desperdicios

#### Caso 2: Predicción de Demanda Energética
- **Sector**: Energía
- **Tecnologías**: LSTM + Keras Tuner + MQTT
- **Objetivo**: Optimizar generación de energía
- **ROI**: 10-15% reducción de costos

#### Caso 3: Detección de Fraudes Financieros
- **Sector**: Banca
- **Tecnologías**: GNN + Spektral + FastAPI
- **Objetivo**: Detectar transacciones fraudulentas
- **ROI**: 25% reducción de pérdidas por fraude

#### Caso 4: Análisis de Sentimientos en Marketing
- **Sector**: Marketing Digital
- **Tecnologías**: BERT + atención personalizada + TF Serving
- **Objetivo**: Monitorear sentimiento de marca
- **ROI**: 30% mejora en respuesta a crisis

#### Caso 5: Optimización de Rutas Logísticas
- **Sector**: Logística
- **Tecnologías**: GAT personalizado + OpenStreetMap + OR-Tools
- **Objetivo**: Optimizar rutas de entrega
- **ROI**: 10% reducción en costos de transporte

### 5. Evaluación y Métricas

#### Métricas por Tipo de Problema
| Tipo de Problema | Métricas Principales | Objetivos |
|------------------|---------------------|-----------|
| Clasificación | Accuracy, Precision, Recall, F1-Score | >90% accuracy |
| Regresión | MAE, RMSE, R² | MAE <5% del valor promedio |
| Detección de Anomalías | Precision, Recall, AUC-ROC | Precision >90%, Recall >85% |
| Series Temporales | MAE, MAPE, Forecast Bias | MAE <2% del valor promedio |

#### Métricas de Producción
| Métrica | Objetivo | Herramienta |
|---------|----------|------------|
| Latencia | <100ms por request | TensorFlow Serving |
| Throughput | >100 requests/segundo | Locust |
| Uso de Memoria | <2GB por modelo | TensorFlow Profiler |
| Disponibilidad | >99.9% uptime | Prometheus + Grafana |

### 6. Despliegue en Producción

#### Flujo de MLOps
```
[Datos] → [Entrenamiento] → [Validación] → [Exportación] → [Despliegue] → [Monitoreo]
    ↓           ↓           ↓           ↓           ↓           ↓
  tf.data    Keras Tuner   Model Analysis  SavedModel  TF Serving  TensorBoard
```

#### Ejemplo de Despliegue con Docker
```dockerfile
FROM tensorflow/serving
COPY model/ /models/model/
ENV MODEL_NAME=model
```

#### API con FastAPI
```python
from fastapi import FastAPI
import tensorflow as tf

app = FastAPI()
model = tf.keras.models.load_model('model.h5')

@app.post("/predict")
async def predict(data: dict):
    prediction = model.predict(data['input'])
    return {"prediction": prediction.tolist()}
```

### 7. Recursos Adicionales

#### Documentación
- [TensorFlow Official Guide](https://www.tensorflow.org/guide)
- [Keras Custom Layers](https://www.tensorflow.org/guide/keras/custom_layers_and_models)
- [tf.data Performance Guide](https://www.tensorflow.org/guide/data_performance)

#### Libros Recomendados
- "Deep Learning with Python" - François Chollet
- "Hands-On Machine Learning" - Aurélien Géron
- "Designing Machine Learning Systems" - Chip Huyen

#### Cursos Online
- TensorFlow Advanced Techniques Specialization (Coursera)
- MLOps Zoomcamp (DataTalksClub)
- CS224W: Machine Learning with Graphs (Stanford)

### 8. Proyecto Final Integrador

Los estudiantes deben implementar un sistema completo de ML que incluya:
1. **Preprocesamiento de datos** con tf.data
2. **Modelo con capas personalizadas**
3. **Optimización de hiperparámetros**
4. **Despliegue en producción**
5. **Monitoreo y mantenimiento**

#### Rúbrica de Evaluación
| Criterio | Peso | Métrica |
|----------|------|---------|
| Precisión del Modelo | 20% | >90% accuracy o equivalente |
| Optimización | 15% | Uso eficiente de recursos |
| Código y Documentación | 20% | Código limpio y bien documentado |
| Despliegue | 20% | Modelo funcional en producción |
| Integración | 15% | Conexión con sistemas externos |
| Creatividad | 10% | Solución innovadora |
