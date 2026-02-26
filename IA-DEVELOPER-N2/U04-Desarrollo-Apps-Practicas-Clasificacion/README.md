# Unidad 4: Desarrollo de Aplicaciones Prácticas de Clasificación (2026)
Título: "Clasificación Avanzada con CNNs, Transformers y Aplicaciones en Imágenes y Texto para 2026"

## 🎯 Objetivos de Aprendizaje
Al finalizar esta unidad, los estudiantes podrán:

- Diseñar y entrenar modelos de clasificación avanzada usando CNNs (para imágenes) y Transformers (para texto).
- Aplicar técnicas de transfer learning con modelos preentrenados (ej: EfficientNet, BERT).
- Optimizar modelos para despliegue en producción (quantization, pruning).
- Integrar modelos con aplicaciones reales (ej: clasificación de productos en retail, análisis de sentimiento en marketing).
- Evaluar modelos con métricas avanzadas (ROC-AUC, precision-recall curves).

## 📌 Contexto Tecnológico (2026)
En 2026, los modelos de clasificación avanzada se caracterizan por:

- Arquitecturas híbridas (CNNs + Transformers para tareas multimodales).
- Modelos preentrenados optimizados (ej: EfficientNetV3, ViT para imágenes; BERT, T5 para texto).
- Hardware especializado (GPUs NVIDIA H100, TPUs v5, aceleradores neuromórficos).
- Despliegue en edge (TensorFlow Lite para microcontroladores).
- Explicabilidad (SHAP, LIME para cumplir con regulaciones como GDPR).

### Herramientas clave en 2026:

| Herramienta | Versión | Uso Principal |
|-------------|---------|---------------|
| TensorFlow | 2.15 | Framework para construcción de modelos de clasificación. |
| Keras | 2.15 | API de alto nivel para TensorFlow. |
| EfficientNetV3 | - | Modelo preentrenado para clasificación de imágenes. |
| Vision Transformers (ViT) | - | Arquitectura basada en Transformers para visión por computadora. |
| BERT/T5 | - | Modelos preentrenados para clasificación de texto. |
| Hugging Face Transformers | 4.30 | Librería para modelos de lenguaje. |
| OpenCV | 4.9 | Procesamiento de imágenes. |
| Albumentations | 1.3 | Aumento de datos para imágenes. |
| TensorFlow Model Optimization | 2.15 | Optimización de modelos (pruning, quantization). |
| SHAP/LIME | 0.42/0.22 | Explicabilidad de modelos. |

## 🏗️ Estructura de la Unidad

### 01-Guía Conceptual: Clasificación Avanzada en 2026

#### 📖 Fundamentos Teóricos

**1. Clasificación con CNNs (Imágenes)**

Arquitecturas modernas:
- EfficientNetV3: Mejor relación precisión/eficiencia computacional.
- Vision Transformers (ViT): Atención aplicada a imágenes (supera a CNNs en algunos benchmarks).
- ResNeXt: Extensión de ResNet con conexiones "split-transform-merge".

**2. Clasificación con Transformers (Texto)**

Modelos preentrenados:
- BERT: Bidirectional Encoder Representations from Transformers.
- T5: Text-to-Text Transfer Transformer (versátil para múltiples tareas).
- RoBERTa: Optimización de BERT con más datos y entrenamiento prolongado.

**3. Técnicas de Transfer Learning**
- Fine-tuning: Ajustar un modelo preentrenado a un dataset específico.
- Feature Extraction: Usar el modelo preentrenado como extractor de características.
- Adaptadores (Adapters): Módulos ligeros para ajustar modelos sin reentrenar todos los pesos.

**4. Optimización de Modelos**

| Técnica | Descripción | Herramienta |
|----------|-------------|-------------|
| Quantization | Reducir precisión de pesos (ej: float32 → int8) | TensorFlow Model Optimization |
| Pruning | Eliminar pesos no significativos | TensorFlow Model Optimization |
| Distillation | Entrenar modelo pequeño usando uno grande | Hugging Face DistilBERT |
| ONNX | Formato abierto para modelos optimizados | ONNX Runtime |

**5. Evaluación de Modelos**

| Métrica | Descripción | Implementación en TensorFlow |
|---------|-------------|---------------------------|
| Accuracy | Proporción de predicciones correctas | tf.keras.metrics.Accuracy() |
| Precision/Recall | Métricas para clases desbalanceadas | tf.keras.metrics.Precision(), Recall() |
| ROC-AUC | Área bajo la curva ROC | tf.keras.metrics.AUC() |
| F1-Score | Media armónica de precision y recall | tf.keras.metrics.F1Score() |
| Confusion Matrix | Visualización de predicciones | tf.math.confusion_matrix |

### 02-Laboratorios Prácticos

#### 🔧 Laboratorio 4.1: Clasificación de Imágenes Médicas con EfficientNetV3 y ViT
**Contexto:**
Un hospital quiere clasificar imágenes de rayos X para detectar neumonía usando:
- EfficientNetV3 para eficiencia computacional.
- Vision Transformers (ViT) para precisión.
- Aumento de datos con Albumentations.

**1. Configuración del Entorno**
```bash
pip install tensorflow==2.15.0 albumentations==1.3.0 opencv-python==4.9.0
```

**2. Carga y Aumento de Datos**
```python
import albumentations as A
import cv2
import numpy as np

# Transformaciones de aumento
transform = A.Compose([
    A.RandomRotate90(),
    A.Flip(),
    A.RandomBrightnessContrast(p=0.2),
    A.Resize(224, 224)
])

def load_and_augment(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    augmented = transform(image=image)['image']
    return augmented / 255.0  # Normalizar
```

**3. Modelo Híbrido (EfficientNetV3 + ViT)**
```python
from tensorflow.keras.applications import EfficientNetV3Small
from tensorflow.keras import layers, models
import tensorflow as tf

# Cargar EfficientNetV3 (preentrenado)
efficientnet = EfficientNetV3Small(
    include_top=False,
    weights='imagenet',
    input_shape=(224, 224, 3)
)

# Bloque de atención (simplificado)
class AttentionBlock(layers.Layer):
    def __init__(self, units):
        super().__init__()
        self.query = layers.Dense(units)
        self.key = layers.Dense(units)
        self.value = layers.Dense(units)

    def call(self, inputs):
        q = self.query(inputs)
        k = self.key(inputs)
        v = self.value(inputs)
        scores = tf.matmul(q, k, transpose_b=True)
        attention = tf.nn.softmax(scores)
        return tf.matmul(attention, v)

# Modelo completo
inputs = layers.Input(shape=(224, 224, 3))
x = efficientnet(inputs)
x = layers.GlobalAveragePooling2D()(x)
x = AttentionBlock(128)(x)
outputs = layers.Dense(1, activation='sigmoid')(x)  # Binario: neumonía o no

model = models.Model(inputs=inputs, outputs=outputs)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
```

**4. Entrenamiento y Evaluación**
```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Generador de datos
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    'data/train',
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    subset='training'
)

val_generator = train_datagen.flow_from_directory(
    'data/train',
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    subset='validation'
)

# Entrenar
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=10,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(patience=3),
        tf.keras.callbacks.ModelCheckpoint('best_model.h5', save_best_only=True)
    ]
)
```

**5. Optimización para Edge (TensorFlow Lite)**
```python
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]  # Quantization
tflite_model = converter.convert()
with open('model.tflite', 'wb') as f:
    f.write(tflite_model)
```

**📌 Entregables del Laboratorio 4.1**

| Entregable | Descripción | Formato |
|-----------|-------------|---------|
| Cargador de Datos | Script para carga y aumento de imágenes. | data_loader.py |
| Modelo Híbrido | Implementación de EfficientNetV3 + ViT. | hybrid_model.py |
| Entrenamiento y Evaluación | Script para entrenar y evaluar el modelo. | train_eval.py |
| Modelo Optimizado | Versión quantizada para edge devices. | model.tflite |
| Documentación | Explicación del modelo y resultados. | README.md |

#### � Laboratorio 4.2: Clasificación de Texto con BERT y Adaptadores
**Contexto:**
Una empresa de marketing quiere clasificar reseñas de productos en positivas/negativas usando:
- BERT para precisión.
- Adaptadores (Adapters) para eficiencia.
- Explicabilidad con SHAP.

**1. Configuración del Entorno**
```bash
pip install transformers==4.30.0 torch==2.0.0 shap==0.42.0
```

**2. Carga y Preprocesamiento de Datos**
```python
from transformers import BertTokenizer
import pandas as pd

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def tokenize_texts(texts, max_length=128):
    return tokenizer(
        texts,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_tensors='tf'
    )

# Ejemplo
df = pd.read_csv('reviews.csv')
encodings = tokenize_texts(df['text'].tolist())
```

**3. Modelo con BERT y Adaptadores**
```python
from transformers import BertModel, AdapterConfig
import tensorflow as tf

# Cargar BERT
model = BertModel.from_pretrained('bert-base-uncased')

# Añadir adaptador para clasificación
adapter_config = AdapterConfig(
    mh_adapter=True,
    output_adapter=True,
    reduction_factor=16,
    non_linearity="relu"
)
model.add_adapter("classification", config=adapter_config)

# Capa de clasificación
output = tf.keras.layers.Dense(1, activation='sigmoid')(model.last_hidden_state[:, 0, :])
model = tf.keras.Model(inputs=model.input, outputs=output)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
```

**4. Explicabilidad con SHAP**
```python
import shap

explainer = shap.Explainer(model, masker=tokenize_texts([""]))
shap_values = explainer(["This product is amazing!", "Terrible experience."])
shap.plots.text(shap_values)
```

**5. Despliegue con FastAPI**
```python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class Review(BaseModel):
    text: str

@app.post("/predict")
def predict(review: Review):
    inputs = tokenize_texts([review.text])
    prediction = model.predict(inputs)
    return {"sentiment": "positive" if prediction[0][0] > 0.5 else "negative"}
```

**📌 Entregables del Laboratorio 4.2**

| Entregable | Descripción | Formato |
|-----------|-------------|---------|
| Cargador de Datos | Script para tokenización de textos. | data_loader.py |
| Modelo con Adaptadores | Implementación de BERT + Adaptadores. | bert_adapters.py |
| Explicabilidad | Visualización de importancia de tokens con SHAP. | explain_model.py |
| API de Predicciones | Servicio FastAPI para clasificación. | api.py |
| Documentación | Explicación del modelo y cómo desplegarlo. | README.md |

#### � Laboratorio 4.3: Clasificación Multimodal (Imagen + Texto) para Retail
**Contexto:**
Un retailer quiere clasificar productos usando:
- Imagen del producto (CNN).
- Descripción del producto (Transformer).
- Fusión multimodal para mayor precisión.

**1. Configuración del Entorno**
```bash
pip install tensorflow==2.15.0 transformers==4.30.0 opencv-python==4.9.0
```

**2. Modelo Multimodal**
```python
from tensorflow.keras.applications import EfficientNetB3
from transformers import TFDistilBertModel
import tensorflow as tf

# Modelo para imágenes
image_model = EfficientNetB3(
    include_top=False,
    weights='imagenet',
    input_shape=(300, 300, 3)
)

# Modelo para texto
text_model = TFDistilBertModel.from_pretrained('distilbert-base-uncased')

# Capa de fusión
image_features = tf.keras.layers.GlobalAveragePooling2D()(image_model.output)
text_features = text_model.output[:, 0, :]  # [CLS] token
combined = tf.keras.layers.Concatenate()([image_features, text_features])
output = tf.keras.layers.Dense(10, activation='softmax')(combined)  # 10 categorías

model = tf.keras.Model(
    inputs=[image_model.input, text_model.input],
    outputs=output
)
```

**3. Preprocesamiento de Datos**
```python
from transformers import DistilBertTokenizer
import cv2

image_tokenizer = lambda img_path: cv2.resize(cv2.imread(img_path), (300, 300)) / 255.0
text_tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

def preprocess_sample(image_path, text):
    return {
        'image': image_tokenizer(image_path),
        'text': text_tokenizer(text, return_tensors='tf', padding='max_length', max_length=128)
    }
```

**📌 Entregables del Laboratorio 4.3**

| Entregable | Descripción | Formato |
|-----------|-------------|---------|
| Modelo Multimodal | Implementación de CNN + Transformer. | multimodal_model.py |
| Preprocesamiento | Script para preparar datos de imagen y texto. | data_preprocessing.py |
| Entrenamiento | Script para entrenar el modelo. | train_multimodal.py |
| Documentación | Guía para replicar el modelo. | README.md |

#### 🔧 Laboratorio 4.4: Clasificación de Documentos Legales con Transformers y ONNX
**Contexto:**
Un bufete de abogados quiere clasificar documentos legales en categorías usando:
- Transformers (ej: Legal-BERT).
- ONNX para optimización.
- Despliegue en Azure ML.

**1. Configuración del Entorno**
```bash
pip install transformers==4.30.0 onnxruntime==1.16.0 torch==2.0.0
```

**2. Modelo con Legal-BERT**
```python
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("nlpaueb/legal-bert-base-uncased")
model = TFAutoModelForSequenceClassification.from_pretrained(
    "nlpaueb/legal-bert-base-uncased",
    num_labels=5  # 5 categorías legales
)
```

**📌 Entregables del Laboratorio 4.4**

| Entregable | Descripción | Formato |
|-----------|-------------|---------|
| Modelo Legal-BERT | Implementación con Transformers. | legal_bert.py |
| Exportación a ONNX | Conversión del modelo para optimización. | export_onnx.py |
| Despliegue en Azure ML | Script para desplegar el modelo en la nube. | azure_deploy.py |
| Documentación | Guía para replicar el despliegue. | README.md |

#### 🔧 Laboratorio 4.5: Clasificación de Productos en Retail con Vision Transformers (ViT) y Quantization
**Contexto:**
Un retailer quiere clasificar productos en estantes usando:
- Vision Transformers (ViT) para alta precisión.
- Quantization para despliegue en edge devices.
- TensorFlow Lite para inferencia en tiempo real.

**1. Configuración del Entorno**
```bash
pip install tensorflow==2.15.0 tensorflow-datasets==4.9.0
```

**2. Modelo ViT con Transfer Learning**
```python
import tensorflow as tf
import tensorflow_datasets as tfds
from vit_keras import vit

# Cargar ViT preentrenado
vit_model = vit.vit_b16(
    image_size=224,
    activation='softmax',
    pretrained=True,
    include_top=False,
    pretrained_top=False
)

# Añadir capa de clasificación
inputs = tf.keras.Input(shape=(224, 224, 3))
x = vit_model(inputs)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
outputs = tf.keras.layers.Dense(10, activation='softmax')(x)  # 10 categorías

model = tf.keras.Model(inputs=inputs, outputs=outputs)
```

**📌 Entregables del Laboratorio 4.5**

| Entregable | Descripción | Formato |
|-----------|-------------|---------|
| Modelo ViT | Implementación con Vision Transformers. | vit_model.py |
| Quantization | Script para optimizar el modelo. | quantize_model.py |
| Inferencia en Edge | Código para inferencia en dispositivos con recursos limitados. | edge_inference.py |
| Documentación | Guía para replicar el modelo y despliegue. | README.md |

### 03-Recursos y Documentación Oficial

#### 📚 Libros y Cursos Recomendados

| Recurso | Tipo | Enlace | Descripción |
|---------|-------|--------|-------------|
| Deep Learning with Python | Libro | Manning | Introducción práctica a deep learning con Keras. |
| Hands-On Machine Learning | Libro | O'Reilly | Cubre desde fundamentos hasta modelos avanzados. |
| Transformers for NLP | Curso | Hugging Face Course | Curso oficial de Hugging Face sobre Transformers. |
| Computer Vision with CNN | Curso | Coursera | Especialización en CNNs. |
| ONNX Tutorials | Documentación | ONNX Docs | Guías para optimización y despliegue de modelos. |
| TensorFlow Model Optimization | Guía | TF Model Optimization | Técnicas para modelos eficientes. |

#### 📖 Documentación de Herramientas

| Herramienta | Documentación Oficial | Uso en los Laboratorios |
|-------------|---------------------|------------------------|
| TensorFlow 2.15 | TensorFlow Docs | Todos los laboratorios. |
| Keras | Keras Guides | Laboratorios 4.1, 4.3, 4.5. |
| Hugging Face Transformers | HF Docs | Laboratorios 4.2, 4.4. |
| Albumentations | Albumentations Docs | Laboratorio 4.1. |
| ONNX | ONNX Tutorials | Laboratorio 4.4. |
| TensorFlow Lite | TFLite Guide | Laboratorios 4.1, 4.5. |
| SHAP | SHAP Docs | Laboratorio 4.2. |

### 04-Buenas Prácticas y Recomendaciones

#### 🔹 Para Clasificación de Imágenes

**Selección de Modelo:**
- Usa EfficientNetV3 para equilibrio entre precisión y eficiencia.
- Prueba Vision Transformers (ViT) si tienes suficientes datos (>1M imágenes).

**Aumento de Datos:**
- Aplica transformaciones que preserven la semántica.
- Usa Albumentations para transformaciones complejas.

**Optimización:**
- Quantization post-training para edge devices.
- Pruning para reducir tamaño del modelo.

#### 🔹 Para Clasificación de Texto

**Selección de Modelo:**
- BERT para tareas generales de NLP.
- Legal-BERT o BioBERT para dominios específicos.
- Adapters para ajustar modelos grandes sin reentrenar todos los pesos.

**Tokenización:**
- Usa truncation=True y padding='max_length'.
- Considera sentencepiece para tokenización subword eficiente.

**Explicabilidad:**
- SHAP para entender la importancia de tokens.
- LIME para explicaciones locales.

#### 🔹 Para Modelos Multimodales

**Fusión de Modalidades:**
- Usa concatenación o atención cruzada para combinar características.
- Ejemplo: `combined = tf.keras.layers.Concatenate()([image_features, text_features])`

**Preprocesamiento:**
- Normaliza imágenes a [0, 1] o [-1, 1] según el modelo base.
- Tokeniza texto con el tokenizador correspondiente.

#### 🔹 Para Despliegue

**Formato de Modelo:**
- TensorFlow Lite para edge devices.
- ONNX para interoperabilidad entre frameworks.
- SavedModel para despliegue con TensorFlow Serving.

**APIs:**
- FastAPI para APIs REST rápidas.
- gRPC para comunicación interna entre microservicios.
- Valida entradas con Pydantic (FastAPI).

**Monitoreo:**
- Usa Prometheus + Grafana para métricas en producción.
- Configura alertas para latencia > 500ms y errores 5xx.

### 📌 Proyecto Integrador: Sistema de Clasificación de Productos para E-Commerce

**Contexto:**
Un e-commerce quiere clasificar productos en imágenes de usuarios en 10 categorías usando un modelo multimodal (imagen + texto).

**📌 Arquitectura del Sistema:**
```
[Usuario sube imagen + descripción] → [API FastAPI] → [Modelo Multimodal] → [Base de Datos] → [Dashboard de Resultados]
```

**📌 Entregables del Proyecto Integrador:**

| Entregable | Descripción | Formato |
|-----------|-------------|---------|
| Modelo Multimodal | Implementación con EfficientNet + DistilBERT. | multimodal_ecommerce.py |
| Preprocesamiento | Script para preparar datos de imagen y texto. | preprocess_ecommerce.py |
| API con Explicabilidad | Servicio FastAPI con SHAP. | api_ecommerce.py |
| Dockerfile | Contenerización de la API. | Dockerfile |
| Configuración de Kubernetes | Despliegue escalable. | deployment.yaml |
| Monitoreo | Métricas con Prometheus. | prometheus.yml |
| Documentación | Guía completa para replicar el sistema. | README.md |

## 📊 Evaluación y Métricas

### 🎯 Criterios de Evaluación

#### **Componente Práctico (70%)**
- **Precisión del Modelo** (25%)
  - Accuracy, Precision, Recall, F1-Score
  - AUC-ROC para clasificación binaria
  - Confusion matrix y análisis de errores
- **Implementación Técnica** (25%)
  - Arquitectura del modelo
  - Optimización (quantization, pruning)
  - Código limpio y documentado
- **Despliegue y API** (20%)
  - API REST funcional
  - Dockerización
  - Monitoreo básico

#### **Componente Teórico (30%)**
- **Documentación** (15%)
  - README.md completo
  - Explicación del modelo
  - Guía de despliegue
- **Explicabilidad** (15%)
  - SHAP/LIME para interpretación
  - Visualizaciones
  - Análisis de errores

## 📈 Progreso y Entregables

### 📅 Cronograma Sugerido

| Semana | Laboratorio | Entregables |
|--------|-------------|-------------|
| 1 | 4.1 - Imágenes Médicas | Modelo híbrido + API |
| 2 | 4.2 - Texto con BERT | Modelo con adaptadores + SHAP |
| 3 | 4.3 - Multimodal Retail | Modelo multimodal + preprocesamiento |
| 4 | 4.4 - Documentos Legales | Modelo ONNX + despliegue Azure |
| 5 | 4.5 - Retail con ViT | Modelo quantizado + edge inference |
| 6-7 | Proyecto Integrador | Sistema completo de e-commerce |

### 🎯 Certificación

Al completar exitosamente todos los laboratorios y el proyecto integrador, recibirás:
- Certificado de "Especialista en Clasificación Avanzada con CNNs y Transformers"
- Portfolio de proyectos para GitHub
- Recomendación para roles de ML Engineer

## 📞 Soporte y Contacto

- **Foro de Discusión**: [Enlace al foro del curso]
- **Horas de Oficina**: Martes y Jueves 14:00-16:00 UTC
- **Email de Soporte**: ia-developer@ejemplo.com
- **Repositorio del Curso**: [GitHub del curso]

---

**Nota Importante**: Esta unidad está diseñada para ser completamente práctica. Todos los laboratorios incluyen código funcional y guías paso a paso para que puedas construir sistemas de clasificación listos para producción en 2026.
