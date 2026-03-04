# Laboratorio 2.3: Redes Neuronales Avanzadas y Aplicaciones Prácticas

## 🎯 **Objetivo General**
Profundizar en 12 tipos de redes neuronales mediante 3 proyectos integradores que combinen capas personalizadas con automatización de procesos empresariales. Cada proyecto usará 4 tipos de redes neuronales diferentes, integrando TensorFlow/Keras, OpenCV, y librerías de procesamiento de datos.

---

## 📊 **Tipos de Redes Neuronales a Cubrir**

| Tipo de Red | Aplicación Típica | Librerías Clave | Ejemplo en Proyectos |
|-------------|-------------------|-----------------|---------------------|
| 1. Redes Neuronales Feedforward (Dense) | Clasificación, regresión | TensorFlow, Keras | Clasificación de productos defectuosos |
| 2. Redes Convolucionales (CNN) | Visión por computadora | TensorFlow, OpenCV | Detección de objetos en almacenes |
| 3. Redes Recurrentes (RNN/LSTM) | Series temporales, texto | TensorFlow, NLTK | Predicción de demanda en retail |
| 4. Redes de Atención (Transformers) | Procesamiento de lenguaje natural | TensorFlow, Hugging Face | Chatbots para servicio al cliente |
| 5. Redes Generativas (GANs) | Generación de datos sintéticos | TensorFlow, Keras | Generación de imágenes de productos |
| 6. Redes de Grafos (GNN) | Datos en grafos | TensorFlow, PyTorch Geometric | Análisis de redes de distribución |
| 7. Redes de Memoria (Memory Networks) | Respuesta a preguntas complejas | TensorFlow, Keras | Asistentes virtuales con contexto |
| 8. Redes Neuro-Simbólicas | Integración de lógica y aprendizaje | TensorFlow, Pyke | Sistemas expertos en diagnóstico médico |
| 9. Redes Capsulares (CapsNet) | Reconocimiento de poses y relaciones | TensorFlow | Detección de posturas en seguridad laboral |
| 10. Redes de Difusión (Diffusion Models) | Generación de imágenes de alta calidad | TensorFlow, Hugging Face | Diseño de productos virtuales |
| 11. Redes Spiking | Computación neuromórfica | Nengo, TensorFlow | Procesamiento eficiente en edge devices |
| 12. Redes Híbridas | Combinación de múltiples tipos | TensorFlow, Keras | Sistemas de recomendación multimodal |

---

## 🏭 **Proyecto 1: Automatización en Logística (Detección de Daños en Paquetes)**

### **Contexto**
Una empresa de logística quiere automatizar la detección de paquetes dañados en su centro de distribución usando visión por computadora y clasificación automática.

### **Redes Utilizadas**
- **CNN** para detección de daños en imágenes
- **RNN** para analizar secuencias de imágenes (paquetes en cinta transportadora)
- **GANs** para generar imágenes sintéticas de paquetes dañados
- **Redes Capsulares** para detectar orientación y tipo de daño

### **Estructura del Proyecto**
```
proyecto1_logistica/
├── data/
│   ├── raw/                  # Imágenes reales de paquetes
│   ├── processed/            # Imágenes procesadas
│   └── synthetic/            # Imágenes generadas por GANs
├── models/
│   ├── cnn_model.h5          # Modelo CNN
│   ├── rnn_model.h5          # Modelo RNN
│   ├── gan_generator.h5      # Generador GAN
│   └── capsnet_model.h5      # Modelo CapsNet
├── scripts/
│   ├── preprocess.py         # Preprocesamiento de imágenes
│   ├── train_cnn.py          # Entrenamiento CNN
│   ├── train_rnn.py          # Entrenamiento RNN
│   ├── train_gan.py           # Entrenamiento GAN
│   ├── train_capsnet.py      # Entrenamiento CapsNet
│   └── predict.py            # Script para predicciones
├── notebooks/
│   ├── exploracion.ipynb      # Análisis exploratorio
│   ├── cnn.ipynb              # Notebook CNN
│   ├── rnn.ipynb              # Notebook RNN
│   ├── gan.ipynb              # Notebook GAN
│   └── capsnet.ipynb           # Notebook CapsNet
├── app/
│   ├── main.py                # API FastAPI
│   ├── static/                # Archivos estáticos
│   └── templates/             # Plantillas HTML
└── README.md
```

### **Implementación Principal**

#### **1. Red Neuronal Convolucional (CNN)**
```python
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers

def build_cnn_model(input_shape=(224, 224, 3)):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model
```

#### **2. Red Neuronal Recurrente (RNN/LSTM) para Secuencias**
```python
def build_rnn_model(sequence_length=5, image_shape=(224, 224, 3)):
    model = models.Sequential([
        layers.TimeDistributed(
            models.Sequential([
                layers.Conv2D(32, (3, 3), activation='relu'),
                layers.MaxPooling2D((2, 2)),
                layers.Conv2D(64, (3, 3), activation='relu'),
                layers.MaxPooling2D((2, 2)),
                layers.Flatten()
            ]),
            input_shape=(sequence_length,) + image_shape
        ),
        layers.LSTM(64),
        layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model
```

#### **3. Red Generativa Adversarial (GAN)**
```python
def build_generator(latent_dim=100):
    model = models.Sequential([
        layers.Dense(256, input_dim=latent_dim),
        layers.LeakyReLU(alpha=0.2),
        layers.BatchNormalization(),
        layers.Dense(512),
        layers.LeakyReLU(alpha=0.2),
        layers.BatchNormalization(),
        layers.Dense(64 * 64 * 3, activation='tanh'),
        layers.Reshape((64, 64, 3))
    ])
    return model

def build_discriminator(input_shape=(64, 64, 3)):
    model = models.Sequential([
        layers.Flatten(input_shape=input_shape),
        layers.Dense(512),
        layers.LeakyReLU(alpha=0.2),
        layers.Dense(256),
        layers.LeakyReLU(alpha=0.2),
        layers.Dense(1, activation='sigmoid')
    ])
    return model
```

#### **4. Red Capsular (CapsNet)**
```python
class CapsuleLayer(layers.Layer):
    def __init__(self, num_capsule, dim_capsule, routings=3):
        super().__init__()
        self.num_capsule = num_capsule
        self.dim_capsule = dim_capsule
        self.routings = routings
        
    def build(self, input_shape):
        input_dim_capsule = input_shape[-1]
        self.W = self.add_weight(
            shape=[input_dim_capsule, self.num_capsule * self.dim_capsule],
            initializer='glorot_uniform',
            name='W'
        )
        
    def call(self, inputs):
        input_capsules = inputs
        input_capsules = tf.expand_dims(input_capsules, -1)
        input_capsules = tf.expand_dims(input_capsules, -1)
        
        inputs_hat = tf.reduce_sum(
            tf.matmul(self.W, input_capsules),
            axis=-2
        )
        
        # Dynamic routing algorithm
        b = tf.zeros(shape=[tf.shape(inputs_hat)[0], self.num_capsule, 1])
        
        for i in range(self.routings):
            c = tf.nn.softmax(b, axis=1)
            outputs = squash(tf.reduce_sum(c * inputs_hat, axis=1))
            
            if i < self.routings - 1:
                b += tf.reduce_sum(inputs_hat * outputs, axis=-1)
                
        return outputs

def squash(vector_tensor):
    vector_squared_norm = tf.reduce_sum(tf.square(vector_tensor), axis=-2, keepdims=True)
    factor = vector_squared_norm / (1 + vector_squared_norm) / tf.sqrt(vector_squared_norm + 1e-8)
    return factor * vector_tensor
```

---

## 🏥 **Proyecto 2: Automatización en Salud (Diagnóstico de Enfermedades en Radiografías)**

### **Contexto**
Un hospital quiere automatizar el diagnóstico de neumonía en radiografías de tórax usando múltiples tipos de redes neuronales.

### **Redes Utilizadas**
- **CNN** para clasificación de imágenes
- **Transformers** para analizar informes médicos asociados
- **GANs** para generar radiografías sintéticas
- **Redes de Grafos (GNN)** para analizar relaciones entre pacientes

### **Estructura del Proyecto**
```
proyecto2_salud/
├── data/
│   ├── raw/                  # Radiografías reales
│   ├── processed/            # Radiografías procesadas
│   └── reports/              # Informes médicos (texto)
├── models/
│   ├── cnn_pneumonia.h5      # Modelo CNN
│   ├── transformer_reports.h5 # Modelo Transformer
│   ├── gan_rx.h5            # Generador GAN
│   └── gnn_patients.h5      # Modelo GNN
├── scripts/
│   ├── preprocess_rx.py
│   ├── train_cnn.py
│   ├── train_transformer.py
│   ├── train_gan.py
│   ├── train_gnn.py
│   └── predict.py
├── notebooks/
│   ├── cnn.ipynb
│   ├── transformer.ipynb
│   ├── gan.ipynb
│   └── gnn.ipynb
└── README.md
```

### **Implementación Principal**

#### **1. CNN para Clasificación de Radiografías**
```python
def build_pneumonia_cnn(input_shape=(224, 224, 1)):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC()]
    )
    return model
```

#### **2. Transformers para Análisis de Informes Médicos**
```python
from transformers import TFDistilBertForSequenceClassification, DistilBertTokenizer

class MedicalReportAnalyzer:
    def __init__(self):
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.model = TFDistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased')
        
    def analyze_report(self, text):
        inputs = self.tokenizer(text, return_tensors="tf", truncation=True, padding=True)
        outputs = self.model(inputs)
        prediction = tf.argmax(outputs.logits, axis=1).numpy()[0]
        confidence = tf.nn.softmax(outputs.logits)[0][prediction]
        
        return {
            'diagnosis': 'pneumonia' if prediction == 1 else 'normal',
            'confidence': float(confidence)
        }
```

#### **3. Red de Grafos para Análisis de Pacientes**
```python
import tensorflow as tf
from spektral.layers import GraphConv

class PatientGNN(tf.keras.Model):
    def __init__(self, hidden_units=32):
        super().__init__()
        self.conv1 = GraphConv(hidden_units, activation='relu')
        self.conv2 = GraphConv(16, activation='relu')
        self.dense = tf.keras.layers.Dense(1, activation='sigmoid')

    def call(self, inputs):
        x, a = inputs
        x = self.conv1([x, a])
        x = self.conv2([x, a])
        x = tf.reduce_mean(x, axis=1)  # Global pooling
        return self.dense(x)
```

---

## 🛍️ **Proyecto 3: Automatización en Retail (Recomendación de Productos Personalizada)**

### **Contexto**
Una cadena de retail quiere personalizar recomendaciones de productos usando múltiples tipos de redes neuronales.

### **Redes Utilizadas**
- **Redes Híbridas (CNN + RNN)** para analizar imágenes y descripciones
- **Transformers** para entender reseñas de clientes
- **Redes de Memoria** para mantener contexto en conversaciones
- **Redes Neuro-Simbólicas** para combinar reglas de negocio

### **Estructura del Proyecto**
```
proyecto3_retail/
├── data/
│   ├── products/              # Imágenes y descripciones de productos
│   ├── reviews/                # Reseñas de clientes
│   └── transactions/           # Historial de compras
├── models/
│   ├── hybrid_model.h5         # Modelo híbrido (CNN + RNN)
│   ├── transformer_reviews.h5  # Modelo Transformer
│   ├── memory_network.h5       # Red de Memoria
│   └── neuro_symbolic_model.h5 # Modelo neuro-simbólico
├── scripts/
│   ├── preprocess.py
│   ├── train_hybrid.py
│   ├── train_transformer.py
│   ├── train_memory.py
│   ├── train_neuro_symbolic.py
│   └── recommend.py
├── app/
│   ├── main.py                 # API FastAPI
│   └── static/                 # Archivos estáticos
└── README.md
```

### **Implementación Principal**

#### **1. Red Híbrida (CNN + RNN)**
```python
def build_hybrid_model(image_shape=(224, 224, 3), vocab_size=10000, max_seq_length=100):
    # Rama de imágenes (CNN)
    image_input = layers.Input(shape=image_shape)
    x = layers.Conv2D(32, (3, 3), activation='relu')(image_input)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Flatten()(x)
    image_features = layers.Dense(128, activation='relu')(x)
    
    # Rama de texto (RNN)
    text_input = layers.Input(shape=(max_seq_length,))
    y = layers.Embedding(vocab_size, 128)(text_input)
    y = layers.LSTM(64)(y)
    text_features = layers.Dense(128, activation='relu')(y)
    
    # Combinar características
    combined = layers.Concatenate()([image_features, text_features])
    z = layers.Dense(256, activation='relu')(combined)
    z = layers.Dropout(0.5)(z)
    output = layers.Dense(num_classes, activation='softmax')(z)
    
    model = models.Model(inputs=[image_input, text_input], outputs=output)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model
```

#### **2. Red de Memoria para Contexto**
```python
class MemoryNetwork(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, memory_size=100):
        super().__init__()
        self.embedding = layers.Embedding(vocab_size, embedding_dim)
        self.memory = layers.Embedding(memory_size, embedding_dim)
        self.lstm = layers.LSTM(64)
        self.attention = layers.Dense(1, activation='softmax')
        self.output_layer = layers.Dense(num_classes, activation='softmax')

    def call(self, inputs):
        # inputs: [batch_size, sequence_length]
        embedded = self.embedding(inputs)  # [batch_size, seq_len, embed_dim]
        
        # Procesar con LSTM
        lstm_out = self.lstm(embedded)  # [batch_size, 64]
        
        # Aplicar atención sobre memoria
        memory_keys = self.memory.embeddings  # [memory_size, embed_dim]
        attention_scores = tf.matmul(
            tf.expand_dims(lstm_out, 1),  # [batch_size, 1, 64]
            memory_keys,                    # [memory_size, embed_dim]
            transpose_b=True
        )  # [batch_size, 1, memory_size]
        
        attention_weights = tf.nn.softmax(attention_scores, axis=-1)
        memory_output = tf.reduce_sum(
            attention_weights * tf.expand_dims(memory_keys, 0), 
            axis=1
        )  # [batch_size, embed_dim]
        
        # Combinar LSTM y memoria
        combined = tf.concat([lstm_out, memory_output], axis=-1)
        output = self.output_layer(combined)
        
        return output
```

#### **3. Red Neuro-Simbólica**
```python
class NeuroSymbolicModel(tf.keras.Model):
    def __init__(self, num_classes):
        super().__init__()
        self.dense1 = layers.Dense(64, activation='relu')
        self.dense2 = layers.Dense(32, activation='relu')
        self.output = layers.Dense(num_classes, activation='softmax')
        
        # Reglas simbólicas codificadas
        self.rules = {
            'young_tech_lover': lambda features: features[0] < 0.3 and features[1] > 0.7,
            'budget_conscious': lambda features: features[2] > 0.8,
            'premium_seeker': lambda features: features[3] > 0.6
        }

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        
        # Aplicar reglas simbólicas
        rule_weights = tf.ones_like(x)
        
        for rule_name, rule_func in self.rules.items():
            if rule_func(inputs):
                # Ajustar pesos según regla
                if rule_name == 'young_tech_lover':
                    rule_weights = rule_weights * tf.constant([1.5, 1.0, 0.8, 1.0])
                elif rule_name == 'budget_conscious':
                    rule_weights = rule_weights * tf.constant([0.8, 1.5, 1.0, 0.7])
                elif rule_name == 'premium_seeker':
                    rule_weights = rule_weights * tf.constant([1.2, 0.8, 1.5, 1.3])
        
        x = x * rule_weights
        return self.output(x)
```

---

## 📋 **Guía para Estudiantes: Cómo Elegir un Proyecto**

### **1. Identificar el Contexto**
- **Logística**: Automatización de almacenes, detección de daños, optimización de rutas
- **Salud**: Diagnóstico por imágenes, análisis de historiales médicos, predicción de brotes
- **Retail**: Recomendación de productos, análisis de reseñas, gestión de inventarios

### **2. Seleccionar Tipos de Redes**
- **CNN**: Siempre que haya imágenes (radiografías, productos, paquetes)
- **RNN/Transformers**: Para datos secuenciales (texto, series temporales)
- **GANs**: Si necesitas generar datos sintéticos (imágenes, texto)
- **GNN**: Si los datos son relaciones (redes sociales, pacientes en contacto)

### **3. Definir Métricas de Éxito**
- **Precisión/Recall**: Para clasificación
- **Silhouette Score**: Para clustering
- **Recompensa Acumulada**: Para aprendizaje por refuerzo
- **ROI**: Para proyectos empresariales

---

## 📊 **Evaluación de los Proyectos**

| Criterio | Ponderación | Descripción |
|----------|-------------|-------------|
| Originalidad | 20% | Creatividad en la aplicación y contexto elegido |
| Implementación Técnica | 30% | Correcta implementación de los 4 tipos de redes neuronales |
| Integración de Modelos | 20% | Cómo se combinan los modelos en un sistema coherente |
| Documentación | 15% | Claridad en README, comentarios en código y explicaciones |
| Impacto Potencial | 15% | Justificación de la necesidad de automatización y métricas |

---

## 🚀 **Recursos Adicionales**

### **Datasets Recomendados**
- **ChestX-ray8**: 112,120 radiografías de tórax para diagnóstico médico
- **MovieLens**: 100,000 ratings de películas para sistemas de recomendación
- **Kaggle Datasets**: Diversos datasets para diferentes aplicaciones

### **Librerías y Frameworks**
- **TensorFlow/Keras**: Para implementación de redes neuronales
- **Hugging Face**: Modelos preentrenados de Transformers
- **Spektral**: Redes neuronales de grafos en TensorFlow
- **OpenCV**: Procesamiento de imágenes
- **FastAPI**: Creación de APIs para despliegue

### **Documentación y Tutoriales**
- **Deep Learning Book**: Ian Goodfellow et al.
- **TensorFlow Tutorials**: Tutoriales oficiales
- **Papers with Code**: Implementaciones de papers recientes

---

## 🎯 **Conclusión**

Este laboratorio proporciona herramientas avanzadas para que los estudiantes:

1. **Domine 12 tipos de redes neuronales** y sus aplicaciones prácticas
2. **Integre múltiples modelos** en un solo sistema coherente
3. **Aplique IA a problemas reales** en logística, salud y retail
4. **Desarrolle proyectos completos** desde preprocesamiento hasta despliegue

### **Próximos Pasos**
- **Unidad 3**: Optimización de modelos y despliegue en producción
- **Unidad 4**: Integración con sistemas empresariales
- **Proyecto Integrador**: Combinar todo lo aprendido en un sistema end-to-end

---

**Nota para el Docente**: Los proyectos pueden adaptarse a cualquier contexto específico. Se enfoca en la aplicación práctica y evaluación continua usando las métricas propuestas.
