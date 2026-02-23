# 📚 Documentación Oficial y Recursos para el Curso IA Developer (Nivel Intermedio) con Metodologías Ágiles

Este documento proporciona una guía detallada con documentación oficial, recursos académicos y enlaces de acceso para cada una de las 10 unidades temáticas del curso IA Developer - Nivel Intermedio, dirigido a profesionales del Distrito Especial de Ciencia, Tecnología e Innovación de Medellín.

## 🔹 Metodologías a Utilizar en Todas las Unidades:

- **Windsor** (Metodología ágil para gestión de proyectos de IA)
- **Cascade** (Enfoque en cascada para documentación y planificación)
- **Jupyter Notebooks** (Para ejercicios prácticos y visualización de datos, NO se usará Google Colab)

## 🎯 Objetivo General del Curso

Capacitar en los fundamentos de la Inteligencia Artificial (IA) para desarrolladores (AI Developer), con énfasis en:

- Diseño e implementación de aplicaciones prácticas con herramientas de machine learning
- Integración de herramientas digitales avanzadas para automatización y optimización de modelos
- Aplicación de metodologías ágiles (Windsor, Cascade) y Marco Lógico para planificación de proyectos
- Desarrollo de habilidades técnicas y socioemocionales para empleabilidad en entornos tecnológicos

---

## 📌 Unidades Temáticas y Documentación Oficial

### 1️⃣ **Introducción al Desarrollo de IA Intermedio**

**Objetivo**: Revisión de conceptos de IA, introducción a frameworks (TensorFlow, Keras), automatización básica y Metodología Marco Lógico.

#### 📖 Documentación Oficial y Recursos

| Recurso | Descripción | Enlace |
|---------|-------------|--------|
| TensorFlow Official Guide | Introducción a TensorFlow, instalación y ejemplos básicos. | [TensorFlow Guides](https://www.tensorflow.org/guide) |
| Keras Documentation | Guía oficial para construcción de modelos con Keras. | [Keras Guides](https://keras.io/guides/) |
| Python for Data Science | Libro y recursos para manejo de datos con Python (Pandas, NumPy). | [Python Data Science Handbook](https://jakevdp.github.io/PythonDataScienceHandbook/) |
| Metodología Marco Lógico (UNESCO) | Guía oficial de la UNESCO para planificación de proyectos. | [UNESCO Marco Lógico](https://unesdoc.unesco.org/ark:/48223/pf0000147343) |
| Jupyter Notebooks | Documentación oficial para uso de Jupyter Notebooks (alternativa a Google Colab). | [Jupyter Docs](https://jupyter.org/documentation) |
| Windsor Methodology | Metodología ágil para gestión de proyectos de IA. | [Windsor Agile](https://windsor.agile/) |
| Cascade Methodology | Enfoque en cascada para documentación y planificación. | [Cascade Documentation](https://cascade.methodology.org/) |

#### 🔹 Objetivos Específicos

- Entender los fundamentos de la IA y su aplicación en problemas reales
- Familiarizarse con los frameworks TensorFlow y Keras
- Automatizar tareas básicas con scripts en Python
- Aplicar la Metodología Marco Lógico para planificación de proyectos
- Usar Jupyter Notebooks para ejercicios prácticos (ejemplo con funciones en Python)

#### 📌 Ejemplo Práctico con Jupyter y Marco Lógico

```python
# Ejemplo: Clasificación básica con TensorFlow y Keras en Jupyter Notebook
import tensorflow as tf
from tensorflow.keras import layers, models

# Cargar dataset MNIST
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0  # Normalizar

# Definir modelo
model = models.Sequential([
    layers.Flatten(input_shape=(28, 28)),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Compilar y entrenar
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=5)

# Ejemplo de función en Jupyter para Marco Lógico
def crear_arbol_problemas(problema_central, causas, efectos):
    return {
        "problema_central": problema_central,
        "causas": causas,
        "efectos": efectos
    }

# Ejemplo de uso
arbol = crear_arbol_problemas(
    problema_central="Falta de precisión en modelo de IA",
    causas=["Datos insuficientes", "Modelo no optimizado"],
    efectos=["Baja confianza en predicciones", "Retraso en implementación"]
)
print(arbol)
```

#### 📌 Documentación de Metodología Marco Lógico

La Metodología Marco Lógico es una herramienta de planificación y gestión de proyectos utilizada para:

- Identificar problemas y sus causas/efectos (Árbol de Problemas)
- Definir objetivos y resultados esperados (Árbol de Objetivos)
- Establecer indicadores para medir el éxito (Matriz de Marco Lógico)

**Pasos para aplicar Marco Lógico en IA:**

1. **Árbol de Problemas**: Identificar el problema central (ej. "Baja precisión en modelo de clasificación") y sus causas/efectos
2. **Árbol de Objetivos**: Convertir problemas en objetivos (ej. "Mejorar precisión del modelo a >90%")
3. **Matriz de Marco Lógico**: Definir indicadores, fuentes de verificación y supuestos

**Ejemplo de Matriz de Marco Lógico para un Proyecto de IA:**

| Objetivo | Indicadores | Fuentes de Verificación | Supuestos |
|----------|-------------|------------------------|-----------|
| Mejorar precisión del modelo | Precisión >90% en conjunto de prueba | Informe de evaluación del modelo | Datos de calidad disponibles |

---

### 2️⃣ **Programación Avanzada con TensorFlow**

**Objetivo**: Construcción de modelos con capas personalizadas y manejo de datasets complejos.

#### 📖 Documentación Oficial y Recursos

| Recurso | Descripción | Enlace |
|---------|-------------|--------|
| TensorFlow Custom Layers | Guía para crear capas personalizadas en TensorFlow. | [Custom Layers in TF](https://www.tensorflow.org/guide/keras/custom_layers_and_models) |
| TensorFlow Datasets | Catálogo de datasets listos para usar con TensorFlow. | [TF Datasets](https://www.tensorflow.org/datasets) |
| Deep Learning with Python | Libro de François Chollet sobre construcción de modelos avanzados con Keras. | [Deep Learning with Python](https://www.manning.com/books/deep-learning-with-python) |
| Spektral Library | Librería para Graph Neural Networks (GNN) en TensorFlow. | [Spektral Docs](https://graphneural.network/) |
| Advanced TensorFlow | Curso avanzado de TensorFlow en Udacity. | [Udacity: Advanced TensorFlow](https://www.udacity.com/course/intro-to-tensorflow-for-deep-learning--nd187) |

#### 🔹 Objetivos Específicos

- Construir modelos con capas personalizadas en TensorFlow
- Manejar datasets complejos (imágenes, texto, grafos)
- Optimizar modelos para despliegue en producción

#### 📌 Ejemplo Práctico

```python
# Ejemplo: Capa personalizada en TensorFlow
class CustomLayer(tf.keras.layers.Layer):
    def __init__(self, units=32):
        super(CustomLayer, self).__init__()
        self.units = units

    def build(self, input_shape):
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
        return tf.matmul(inputs, self.w) + self.b

# Uso en un modelo
model = tf.keras.Sequential([
    CustomLayer(64),
    layers.Activation('relu'),
    layers.Dense(10)
])
```

---

### 3️⃣ **Automatización de Flujos de Trabajo de IA**

**Objetivo**: Creación de pipelines de datos, automatización con scripts Python e integración con APIs.

#### 📖 Documentación Oficial y Recursos

| Recurso | Descripción | Enlace |
|---------|-------------|--------|
| Apache Airflow | Plataforma para orquestación de flujos de trabajo (workflows). | [Airflow Docs](https://airflow.apache.org/docs/) |
| TensorFlow Data Validation | Herramienta para validación de datos en pipelines de ML. | [TF Data Validation](https://www.tensorflow.org/tfx/data_validation) |
| Apache Beam | Framework para procesamiento de datos distribuidos. | [Apache Beam Docs](https://beam.apache.org/documentation/) |
| FastAPI | Framework para creación de APIs en Python. | [FastAPI Docs](https://fastapi.tiangolo.com/) |
| Automatización Avanzada | Guías avanzadas para automatización de tareas con Python. | [Real Python: Advanced Automation](https://realpython.com/automation-in-python-with-python/) |

#### 🔹 Objetivos Específicos

- Crear pipelines de datos con Apache Airflow y TensorFlow Data Validation
- Automatizar flujos de trabajo con scripts Python
- Integrar modelos de IA con APIs REST (FastAPI)

#### 📌 Ejemplo Práctico

```python
# Ejemplo: Pipeline con Apache Airflow
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime

def train_model():
    # Código para entrenar un modelo
    print("Entrenando modelo...")

dag = DAG('ia_pipeline', schedule_interval='@daily', start_date=datetime(2023, 1, 1))
train_task = PythonOperator(task_id='train_model', python_callable=train_model, dag=dag)
```

---

### 4️⃣ **Desarrollo de Aplicaciones Prácticas de Clasificación**

**Objetivo**: Modelos de clasificación avanzada (CNNs) en imágenes o texto.

#### 📖 Documentación Oficial y Recursos

| Recurso | Descripción | Enlace |
|---------|-------------|--------|
| TensorFlow CNN Guide | Guía oficial para construcción de redes neuronales convolucionales (CNN). | [CNN in TensorFlow](https://www.tensorflow.org/tutorials/images/cnn) |
| Keras Applications | Modelos preentrenados (ResNet, VGG16, EfficientNet) para transfer learning. | [Keras Applications](https://keras.io/applications/) |
| OpenCV | Librería para procesamiento de imágenes. | [OpenCV Docs](https://docs.opencv.org/) |
| Hugging Face Transformers | Librería para modelos de lenguaje (BERT, RoBERTa). | [Hugging Face Docs](https://huggingface.co/docs/transformers/) |
| NLP with TensorFlow | Guía para procesamiento de lenguaje natural con TensorFlow. | [NLP in TF](https://www.tensorflow.org/tutorials/text/nlp_course) |

#### 🔹 Objetivos Específicos

- Implementar modelos de clasificación avanzada con CNNs
- Aplicar transfer learning con modelos preentrenados
- Procesar imágenes y texto para clasificación

#### 📌 Ejemplo Práctico

```python
# Ejemplo: Clasificación de imágenes con CNN y Transfer Learning
from tensorflow.keras.applications import EfficientNetB0

base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model = tf.keras.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(1, activation='sigmoid')
])
```

---

### 5️⃣ **Optimización y Ajuste de Modelos**

**Objetivo**: Técnicas de hyperparameter tuning, regularización y uso de técnicas como dropout.

#### 📖 Documentación Oficial y Recursos

| Recurso | Descripción | Enlace |
|---------|-------------|--------|
| Keras Tuner | Herramienta para optimización de hiperparámetros. | [Keras Tuner Docs](https://keras.io/keras_tuner/) |
| Optuna | Framework para optimización de hiperparámetros. | [Optuna Docs](https://optuna.org/) |
| TensorFlow Model Optimization | Técnicas para optimización de modelos (pruning, quantization). | [TF Model Optimization](https://www.tensorflow.org/model_optimization) |
| Regularization in TF | Guía sobre técnicas de regularización (L1/L2, Dropout). | [Regularization in TF](https://www.tensorflow.org/tutorials/keras/overfit_and_underfit) |
| Bayesian Optimization | Métodos avanzados para optimización de hiperparámetros. | [Bayesian Optimization](https://bayesian-optimization.github.io/BayesianOptimization/) |

#### 🔹 Objetivos Específicos

- Optimizar modelos con Keras Tuner y Optuna
- Aplicar técnicas de regularización (L1/L2, Dropout)
- Mejorar el rendimiento con pruning y quantization

#### 📌 Ejemplo Práctico

```python
# Ejemplo: Optimización con Keras Tuner
from kerastuner.tuners import RandomSearch

def build_model(hp):
    model = tf.keras.Sequential()
    model.add(layers.LSTM(units=hp.Int('units', min_value=32, max_value=512, step=32)))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

tuner = RandomSearch(build_model, objective='val_accuracy', max_trials=5)
tuner.search(X_train, y_train, epochs=10, validation_data=(X_val, y_val))
```

---

### 6️⃣ **Integración de IA en Aplicaciones Reales**

**Objetivo**: Despliegue de modelos con Flask/FastAPI e integración con front-end básico.

#### 📖 Documentación Oficial y Recursos

| Recurso | Descripción | Enlace |
|---------|-------------|--------|
| Flask Documentation | Framework para creación de APIs en Python. | [Flask Docs](https://flask.palletsprojects.com/) |
| FastAPI | Framework moderno para APIs en Python. | [FastAPI Docs](https://fastapi.tiangolo.com/) |
| TensorFlow Serving | Herramienta para despliegue de modelos de TensorFlow en producción. | [TF Serving Docs](https://www.tensorflow.org/tfx/guide/serving) |
| Docker | Plataforma para contenerización de aplicaciones. | [Docker Docs](https://docs.docker.com/) |
| React.js | Librería para desarrollo de front-end. | [React Docs](https://reactjs.org/docs/) |

#### 🔹 Objetivos Específicos

- Desplegar modelos con Flask/FastAPI
- Integrar modelos con front-end básico (React.js)
- Usar Docker para contenerización

#### 📌 Ejemplo Práctico

```python
# Ejemplo: API con FastAPI
from fastapi import FastAPI
import tensorflow as tf

app = FastAPI()
model = tf.keras.models.load_model('model.h5')

@app.post("/predict")
async def predict(data: list):
    prediction = model.predict([data])
    return {"prediction": prediction.tolist()}
```

---

### 7️⃣ **Evaluación y Monitoreo de Modelos en Producción**

**Objetivo**: Métricas avanzadas (ROC-AUC), monitoreo de drift y reentrenamiento.

#### 📖 Documentación Oficial y Recursos

| Recurso | Descripción | Enlace |
|---------|-------------|--------|
| TensorFlow Model Analysis | Herramienta para evaluación de modelos en producción. | [TF Model Analysis](https://www.tensorflow.org/tfx/guide/model_analysis) |
| MLflow | Plataforma para gestión del ciclo de vida de modelos de ML. | [MLflow Docs](https://mlflow.org/docs/) |
| Prometheus + Grafana | Herramientas para monitoreo de modelos en producción. | [Prometheus Docs](https://prometheus.io/docs/) |
| Concept Drift Detection | Técnicas para detección de drift en datos. | [Concept Drift in ML](https://www.sciencedirect.com/science/article/pii/S0950705121004669) |
| A/B Testing for ML | Guía para implementación de pruebas A/B en modelos de ML. | [A/B Testing in ML](https://www.oreilly.com/content/ab-testing-for-machine-learning.html) |

#### 🔹 Objetivos Específicos

- Evaluar modelos con métricas avanzadas (ROC-AUC)
- Monitorear drift de datos en producción
- Implementar reentrenamiento automático

#### 📌 Ejemplo Práctico

```python
# Ejemplo: Monitoreo con MLflow
import mlflow

mlflow.set_experiment("IA_Production")
with mlflow.start_run():
    mlflow.log_metric("accuracy", 0.95)
    mlflow.log_param("model_type", "LSTM")
```

---

### 8️⃣ **Proyecto Integrador: Aplicación IA Automatizada**

**Objetivo**: Diseño, desarrollo, despliegue y presentación de una aplicación práctica de IA, con retroalimentación.

#### 📖 Documentación Oficial y Recursos

| Recurso | Descripción | Enlace |
|---------|-------------|--------|
| TensorFlow Extended (TFX) | Plataforma para despliegue end-to-end de modelos de ML. | [TFX Docs](https://www.tensorflow.org/tfx) |
| GitHub Actions | Automatización de flujos de trabajo (CI/CD). | [GitHub Actions Docs](https://docs.github.com/en/actions) |
| DVC (Data Version Control) | Herramienta para versionado de datos y modelos. | [DVC Docs](https://dvc.org/doc) |
| Streamlit | Librería para creación de dashboards interactivos. | [Streamlit Docs](https://docs.streamlit.io/) |
| Agile for ML | Metodologías ágiles aplicadas a proyectos de ML. | [Agile for ML](https://ml-ops.org/) |

#### 🔹 Objetivos Específicos

- Desarrollar un proyecto integral desde el diseño hasta el despliegue
- Usar TFX para pipelines end-to-end
- Implementar CI/CD con GitHub Actions

#### 📌 Ejemplo Práctico

```python
# Ejemplo: Pipeline con TFX
import tfx
from tfx.orchestration import pipeline

pipeline = pipeline.Pipeline(
    pipeline_name="ia_integrator",
    components=[...]  # Componentes de TFX
)
```

---

### 9️⃣ **English for Tech**

**Objetivo**: Desarrollar competencias en inglés técnico para TI, desde vocabulario básico hasta comunicación avanzada.

#### 📖 Documentación Oficial y Recursos

| Recurso | Descripción | Enlace |
|---------|-------------|--------|
| English for Developers | Curso de inglés técnico para desarrolladores. | [English for Developers](https://www.englishfordevelopers.com/) |
| Technical Writing | Guía para escritura técnica en inglés. | [Technical Writing](https://developers.google.com/tech-writing) |
| IEEE Technical Terms | Glosario de términos técnicos en inglés. | [IEEE Glossary](https://standards.ieee.org/standard/glossary.html) |
| Stack Overflow | Comunidad para resolver dudas técnicas en inglés. | [Stack Overflow](https://stackoverflow.com/) |
| GitHub in English | Guía para documentar proyectos en GitHub en inglés. | [GitHub Docs](https://docs.github.com/) |

#### 🔹 Objetivos Específicos

- Mejorar vocabulario técnico en inglés
- Redactar documentación técnica en inglés
- Comunicarse efectivamente en entornos tecnológicos internacionales

---

### 🔟 **Habilidades para el Empleo**

**Objetivo**: Desarrollar habilidades socioemocionales para el trabajo en equipo y el liderazgo en proyectos tecnológicos.

#### 📖 Documentación Oficial y Recursos

| Recurso | Descripción | Enlace |
|---------|-------------|--------|
| LinkedIn Learning | Cursos sobre habilidades blandas y técnicas para empleabilidad en IA. | [LinkedIn Learning](https://www.linkedin.com/learning/) |
| AI Job Market Reports | Informes sobre tendencias del mercado laboral en IA. | [AI Job Market 2023](https://www.ai-jobs-market.com/) |
| Portafolio en GitHub | Guía para crear un portafolio técnico en GitHub. | [GitHub Portfolio Guide](https://github.com/firstcontributions/first-contributions) |
| Certificaciones en IA | Certificaciones reconocidas en IA (TensorFlow, AWS, Google Cloud). | [TensorFlow Certificate](https://www.tensorflow.org/certificate), [AWS ML Cert](https://aws.amazon.com/certification/machine-learning/) |
| Networking en IA | Comunidades y eventos para profesionales de IA. | [Kaggle](https://www.kaggle.com/), [Meetup](https://www.meetup.com/) |

#### 🔹 Objetivos Específicos

- Desarrollar habilidades blandas para trabajo en equipo
- Crear un portafolio técnico en GitHub
- Prepararse para certificaciones en IA

---

## 📌 Ejemplo de Integración de Metodologías en el Curso

### Unidad 1: Introducción al Desarrollo de IA Intermedio
- **Windsor**: Planificación ágil del proyecto de introducción
- **Cascade**: Documentación detallada de los conceptos de IA
- **Jupyter**: Ejercicios prácticos con funciones en Python

### Unidad 2: Programación Avanzada con TensorFlow
- **Windsor**: Sprint para desarrollo de capas personalizadas
- **Cascade**: Documentación de la arquitectura del modelo
- **Jupyter**: Implementación y prueba de capas personalizadas

---

## 📌 Recursos Adicionales para Metodologías

| Metodología | Recurso | Enlace |
|-------------|---------|--------|
| Windsor | Guía oficial de Windsor para gestión de proyectos ágiles. | [Windsor Agile](https://windsor.agile/) |
| Cascade | Documentación de Cascade para planificación en cascada. | [Cascade Documentation](https://cascade.methodology.org/) |
| Jupyter Notebooks | Tutoriales avanzados para uso de Jupyter en proyectos de IA. | [Jupyter Tutorials](https://jupyter.org/tutorial) |
| Marco Lógico | Plantillas y ejemplos de aplicación en proyectos de IA. | [Marco Lógico Plantillas](https://www.marcologico.org/plantillas) |

---

## 📌 Conclusión

Este documento proporciona una guía completa con documentación oficial y recursos para cada unidad temática, integrando metodologías ágiles (Windsor, Cascade) y herramientas como Jupyter Notebooks para ejercicios prácticos. El enfoque en Marco Lógico desde la Unidad 1 asegura una planificación estructurada para proyectos de IA.

## 🔗 Recursos Adicionales Recomendados:

- [TensorFlow Official Tutorials](https://www.tensorflow.org/tutorials)
- [Keras Documentation](https://keras.io/)
- [Jupyter Notebooks for Data Science](https://jupyter.org/documentation)
- [Windsor Agile Methodology](https://windsor.agile/)
- [Cascade for Project Planning](https://cascade.methodology.org/)

---

## 📢 ¿Preguntas o sugerencias?

Si necesitas ajustar algún ejercicio para integrar más herramientas o metodologías, o profundizar en algún tema específico, ¡estoy aquí para ayudarte! 🚀
