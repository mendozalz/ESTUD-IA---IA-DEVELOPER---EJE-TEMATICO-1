# U02 - Programación Avanzada con TensorFlow

## 📖 Descripción General

Esta unidad profundiza en los fundamentos teóricos y conceptuales de TensorFlow como framework de deep learning, explorando desde la arquitectura interna del framework hasta las técnicas avanzadas de optimización, despliegue e integración con sistemas empresariales. El enfoque es comprender el "porqué" detrás de cada técnica y su aplicación en escenarios reales de producción.

## 🎯 Objetivos de Aprendizaje Teóricos

### **Comprensión Profunda del Framework**
- **Arquitectura de TensorFlow**: Entender grafo computacional, eager execution, autógrafos
- **Sistema de Tipos**: Dominar tf.Tensor, tf.Variable, y sus operaciones
- **Memory Management**: Gestión de memoria GPU/CPU, optimización de recursos

### **Diseño Avanzado de Modelos**
- **API Funcional vs Subclases**: Cuándo y por qué usar cada enfoque
- **Capas Personalizadas**: Teoría detrás de implementación de layers custom
- **Mecanismos de Attention**: Fundamentos matemáticos y implementación

### **Optimización y Performance**
- **Teoría de Optimización**: Convergencia, learning rates, momentum
- **Parallel Computing**: Data/model/pipeline parallelism
- **Mixed Precision**: Teoría de computación de punto flotante

## 📚 Contenido Teórico Avanzado

### **Módulo 1: Fundamentos Internos de TensorFlow**

#### **1.1 Arquitectura del Grafo Computacional**
- **Computational Graphs**: Teoría de grafos en deep learning
- **Eager vs Graph Execution**: Trade-offs de performance y flexibilidad
- **Autógrafos**: tf.function y sus implicaciones de performance
- **GradientTape**: Teoría del cálculo automático de gradientes

#### **1.2 Sistema de Tipos y Operaciones**
- **Tensor Types**: Dense, sparse, ragged tensors - teoría y aplicaciones
- **Operaciones Tensoriales**: Broadcasting, slicing, advanced indexing
- **Memory Layout**: Row-major vs column-major, implicaciones de cache

#### **1.3 Device Management**
- **GPU Computing**: Arquitectura CUDA, kernels, memory hierarchy
- **TPU Architecture**: Matrix units, systolic arrays
- **Distributed Computing**: Parameter server vs all-reduce

### **Módulo 2: Arquitecturas Avanzadas de Modelos**

#### **2.1 API Funcional Profunda**
- **Model Composition**: Teoría de composición de funciones
- **Multiple Inputs/Outputs**: Arquitecturas multi-task learning
- **Shared Layers**: Transfer learning teórico
- **Residual Connections**: Teoría de gradient flow

#### **2.2 Subclases y Capas Personalizadas**
- **Layer Theory**: Forward/backward pass, state management
- **Custom Metrics**: Teoría de evaluación diferenciable
- **Custom Loss Functions**: Diseño de funciones de pérdida
- **Weight Initialization**: Teoría de inicialización (Xavier, He)

#### **2.3 Mecanismos de Attention**
- **Scaled Dot-Product Attention**: Fundamentos matemáticos
- **Multi-Head Attention**: Paralelización y representación
- **Self-Attention**: Teoría de representación contextual
- **Cross-Attention**: Aplicaciones en seq2seq

### **Módulo 3: Data Pipeline Avanzado**

#### **3.1 tf.data Profundo**
- **Pipeline Theory**: Prefetching, parallelism, caching
- **Data Augmentation**: Teoría de regularización por augmentación
- **Window Operations**: Series temporales y secuencias
- **Interoperability**: NumPy, Pandas, Apache Arrow

#### **3.2 Manejo de Datos Complejos**
- **Structured Data**: Feature columns, embeddings categóricos
- **Image Processing**: Convoluciones, augmentations, preprocessing
- **Text Processing**: Tokenization, embeddings, masking
- **Time Series**: Windowing, forecasting, anomaly detection

### **Módulo 4: Optimización y Training Avanzado**

#### **4.1 Teoría de Optimización**
- **Convex Optimization**: Fundamentos matemáticos
- **Non-Convex Optimization**: Landscape, saddle points
- **Adaptive Methods**: Adam, RMSprop - análisis teórico
- **Learning Rate Scheduling**: Cyclical, warmup, decay

#### **4.2 Regularización Avanzada**
- **Dropout Theory**: Approximate model averaging
- **Batch Normalization**: Internal covariate shift
- **Weight Decay**: L2 regularization en deep learning
- **Early Stopping**: Generalization bounds

#### **4.3 Distributed Training**
- **Data Parallelism**: Synchronous vs asynchronous
- **Model Parallelism**: Pipeline parallelism, model sharding
- **Gradient Compression**: Sparsification, quantization
- **Fault Tolerance**: Checkpointing, recovery

### **Módulo 5: Despliegue y Producción**

#### **5.1 Model Serving**
- **SavedModel Format**: Protocol buffers, signatures
- **TensorFlow Serving**: gRPC, REST APIs
- **Model Versioning**: A/B testing, canary deployments
- **Monitoring**: Performance metrics, drift detection

#### **5.2 Optimización para Inferencia**
- **TensorFlow Lite**: Quantization, pruning, delegate kernels
- **TensorFlow.js**: WebGL acceleration, browser optimization
- **ONNX Conversion**: Cross-framework compatibility
- **Edge Computing**: Mobile, embedded systems

#### **5.3 Integración Empresarial**
- **API Design**: REST vs GraphQL, streaming
- **Database Integration**: Feature stores, model registries
- **CI/CD for ML**: Automated testing, deployment pipelines
- **Security**: Model encryption, access control

## 🔬 Casos de Estudio Teóricos

### **Caso 1: Sistema de Recomendación a Gran Escala**
- **Two-Tower Architecture**: User/item embeddings
- **Negative Sampling**: Sampling strategies para implicit feedback
- **Real-time Inference**: Serving architecture
- **Cold Start Problem**: Métodos teóricos de solución

### **Caso 2: Procesamiento de Lenguaje Natural**
- **Transformer Architecture**: Self-attention mechanisms
- **Pre-training Objectives**: MLM, NSP, contrastive learning
- **Fine-tuning Strategies**: Adapter layers, prompt tuning
- **Multilingual Models**: Cross-lingual transfer learning

### **Caso 3: Computer Vision Avanzado**
- **Object Detection**: Anchor boxes, NMS, focal loss
- **Segmentation**: UNet, Mask R-CNN architectures
- **Video Analysis**: 3D convolutions, optical flow
- **Self-Supervised Learning**: Contrastive learning, BYOL

## 🏗️ Arquitectura de Sistemas TensorFlow

### **Design Patterns**
- **Model Registry**: Version control para modelos
- **Feature Store**: Centralized feature management
- **Experiment Tracking**: Reproducibility y comparación
- **Pipeline Orchestration**: Airflow, Kubeflow, Prefect

### **Performance Optimization**
- **Memory Profiling**: tf.profiler, memory leaks
- **GPU Utilization**: Kernel optimization, memory bandwidth
- **Latency Optimization**: Batching, model parallelism
- **Throughput Optimization**: Pipeline parallelism, async processing

## 📊 Métricas y Evaluación Teórica

### **Training Metrics**
- **Convergence Analysis**: Loss landscape, gradient norms
- **Generalization Gap**: Train/test performance analysis
- **Training Stability**: Gradient explosion/vanishing
- **Resource Utilization**: GPU memory, compute efficiency

### **Inference Metrics**
- **Latency Analysis**: P50, P95, P99 latencies
- **Throughput**: QPS, batch size optimization
- **Accuracy-Precision Trade-offs**: Quantization effects
- **Cost Analysis**: Compute costs, infrastructure costs

## 🎓 Evaluación del Aprendizaje Avanzado

### **Evaluación Teórica (40%)**
- **Análisis de Arquitecturas**: Comparación teórica de approaches
- **Optimization Analysis**: Convergencia y performance
- **Paper Critiques**: Análisis crítico de research papers
- **Mathematical Proofs**: Demostraciones de conceptos clave

### **Diseño de Sistemas (30%)**
- **System Architecture**: Diseño de sistemas completos
- **Performance Analysis**: Predicción teórica de performance
- **Scalability Design**: Diseño para grandes volúmenes
- **Integration Planning**: Planificación de integración empresarial

### **Implementación Avanzada (30%)**
- **Custom Layers**: Implementación de layers complejos
- **Optimization Experiments**: Comparación de optimizadores
- **Deployment Strategies**: Estrategias de despliegue
- **Performance Tuning**: Optimización de sistemas

## 📚 Recursos Teóricos Avanzados

### **Libros Especializados**
- **Deep Learning with TensorFlow** - A. Geron
- **Hands-On Machine Learning** - A. Geron
- **TensorFlow for Deep Learning** - B. Zhu

### **Papers Fundamentales**
- **Adam: A Method for Stochastic Optimization** - Kingma & Ba
- **Batch Normalization** - Ioffe & Szegedy
- **Attention Is All You Need** - Vaswani et al.

### **Documentación Técnica**
- **TensorFlow White Papers** - Google Research
- **XLA: Accelerated Linear Algebra** - TensorFlow compiler
- **TensorFlow Extended (TFX)** - Production ML platform

---

**Esta unidad proporciona el conocimiento teórico avanzado necesario para dominar TensorFlow en producción.**

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
