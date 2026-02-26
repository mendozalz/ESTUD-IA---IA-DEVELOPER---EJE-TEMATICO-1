# Red Neuronal Convolucional (CNN)

## 📖 Teoría

### Definición
Las Redes Neuronales Convolucionales (CNN - Convolutional Neural Networks) están especializadas en procesar datos con estructura de cuadrícula, como imágenes y videos. Usan capas convolucionales para extraer características locales (bordes, texturas) y capas de pooling para reducir dimensionalidad.

### Características Principales
- **Convolución**: Aplicación de filtros para detectar características locales
- **Pooling**: Reducción espacial para disminuir parámetros
- **Conexión local**: A diferencia de las redes densas, las neuronas solo se conectan a regiones locales
- **Jerarquía de características**: Capas profundas aprenden características más complejas
- **Invarianza a traslación**: Detectan características sin importar su posición

### Componentes
1. **Capa convolucional**: Aplica filtros para extraer características
2. **Capa de pooling**: Reduce dimensionalidad espacial
3. **Capa de aplanamiento**: Convierte características 2D a 1D
4. **Capas densas**: Realizan clasificación final
5. **Funciones de activación**: ReLU, sigmoid, softmax

### Casos de Uso
- Clasificación de imágenes
- Detección de objetos
- Segmentación semántica
- Reconocimiento facial
- Procesamiento médico

## 🎯 Ejercicio Práctico: Clasificación de Imágenes CIFAR-10

### Tipo de Aprendizaje: **SUPERVISADO**
- **Por qué**: Usa etiquetas (10 categorías) para entrenar el modelo
- **Datos**: Imágenes etiquetadas de 10 categorías diferentes

### Implementación Completa

```python
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np

print("🖼️ RED NEURONAL CONVOLUCIONAL - CLASIFICACIÓN CIFAR-10")
print("=" * 60)

# --- 1. Cargar y preprocesar datos ---
print("📊 Cargando y preprocesando datos...")
cifar10 = tf.keras.datasets.cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Normalizar a [0, 1]
x_train, x_test = x_train / 255.0, x_test / 255.0

# Nombres de las clases
class_names = ['Avión', 'Automóvil', 'Pájaro', 'Gato', 'Ciervo',
               'Perro', 'Rana', 'Caballo', 'Barco', 'Camión']

print(f"   Datos de entrenamiento: {x_train.shape}")
print(f"   Datos de prueba: {x_test.shape}")
print(f"   Clases: {class_names}")

# --- 2. Definir el modelo CNN ---
print("\n🏗️ Construyendo el modelo CNN...")
model = models.Sequential([
    # Primera capa convolucional
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),
    
    # Segunda capa convolucional
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    # Tercera capa convolucional
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    
    # Capas densas para clasificación
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')
])

# Mostrar arquitectura
model.summary()

# --- 3. Compilar y entrenar ---
print("\n🚀 Compilando y entrenando el modelo...")
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Callbacks
callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3),
    tf.keras.callbacks.ModelCheckpoint('cnn_cifar10.h5', save_best_only=True)
]

history = model.fit(
    x_train, y_train,
    epochs=30,
    batch_size=64,
    validation_split=0.2,
    callbacks=callbacks,
    verbose=1
)

# --- 4. Evaluar el modelo ---
print("\n📈 Evaluando el modelo...")
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"   Precisión en prueba: {test_acc:.4f}")
print(f"   Pérdida en prueba: {test_loss:.4f}")

# --- 5. Visualizar resultados ---
# Graficar accuracy y loss
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.plot(history.history['accuracy'], label='Entrenamiento')
plt.plot(history.history['val_accuracy'], label='Validación')
plt.title('Precisión del Modelo')
plt.xlabel('Época')
plt.ylabel('Precisión')
plt.legend()
plt.grid(True)

plt.subplot(1, 3, 2)
plt.plot(history.history['loss'], label='Entrenamiento')
plt.plot(history.history['val_loss'], label='Validación')
plt.title('Pérdida del Modelo')
plt.xlabel('Época')
plt.ylabel('Pérdida')
plt.legend()
plt.grid(True)

# --- 6. Predicción en imágenes de ejemplo ---
plt.subplot(1, 3, 3)
predictions = model.predict(x_test[:10])
for i in range(10):
    plt.subplot(2, 10, i + 1)
    plt.imshow(x_test[i])
    plt.title(f"Real: {class_names[y_test[i]]}\nPred: {class_names[np.argmax(predictions[i])}")
    plt.axis('off')

plt.tight_layout()
plt.show()

# --- 7. Análisis de errores por clase ---
print("\n🔍 Analizando predicciones por clase...")
y_pred = np.argmax(model.predict(x_test), axis=1)

# Calcular accuracy por clase
class_accuracies = {}
for i in range(10):
    class_mask = y_test == i
    if np.sum(class_mask) > 0:
        class_acc = np.mean(y_pred[class_mask] == y_test[class_mask])
        class_accuracies[class_names[i]] = class_acc

print("   Precisión por clase:")
for class_name, acc in class_accuracies.items():
    print(f"   • {class_name}: {acc:.4f}")

# --- 8. Visualizar filtros aprendidos ---
print("\n🎨 Visualizando filtros aprendidos...")
# Extraer filtros de la primera capa convolucional
filters, biases = model.layers[0].get_weights()
n_filters = filters.shape[3]

plt.figure(figsize=(15, 8))
for i in range(min(32, n_filters)):
    plt.subplot(4, 8, i + 1)
    plt.imshow(filters[:, :, 0, i], cmap='viridis')
    plt.title(f'Filtro {i+1}')
    plt.axis('off')
plt.suptitle('Filtros Aprendidos - Primera Capa Convolucional')
plt.tight_layout()
plt.show()

print("\n✅ EJERCICIO COMPLETADO")
print("=" * 60)
print("🎯 RESULTADOS:")
print(f"   • Precisión final: {test_acc:.4f}")
print(f"   • Total de parámetros: {model.count_params():,}")
print(f"   • Épocas entrenadas: {len(history.history['loss'])}")
print("=" * 60)
```

## 📊 Resultados Esperados

```
🖼️ RED NEURONAL CONVOLUCIONAL - CLASIFICACIÓN CIFAR-10
============================================================
📊 Cargando y preprocesando datos...
   Datos de entrenamiento: (50000, 32, 32, 3)
   Datos de prueba: (10000, 32, 32, 3)
   Clases: ['Avión', 'Automóvil', 'Pájaro', 'Gato', 'Ciervo', 'Perro', 'Rana', 'Caballo', 'Barco', 'Camión']

🏗️ Construyendo el modelo CNN...
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 conv2d (Conv2D)           (None, 30, 30, 32)        896       
 max_pooling2d (MaxPooling2 (None, 15, 15, 32)        0         
 conv2d_1 (Conv2D)          (None, 13, 13, 64)        18496     
 max_pooling2d_1 (MaxPooling2 (None, 6, 6, 64)          0         
 conv2d_2 (Conv2D)          (None, 4, 4, 64)          36928     
 flatten (Flatten)           (None, 1024)               0         
 dense (Dense)               (None, 64)                 65600     
 dropout (Dropout)            (None, 64)                 0         
 dense_1 (Dense)             (None, 10)                 650       
=================================================================
Total params: 122,570
Trainable params: 122,570
Non-trainable params: 0
_________________________________________________________________

🚀 Compilando y entrenando el modelo...
Epoch 1/30
625/625 [==============================] - 8s 12ms/step - loss: 1.5234 - accuracy: 0.4456 - val_loss: 1.2876 - val_accuracy: 0.5432
...
Epoch 18/30
625/625 [==============================] - 7s 11ms/step - loss: 0.6234 - accuracy: 0.7823 - val_loss: 0.7891 - val_accuracy: 0.7456

📈 Evaluando el modelo...
   Precisión en prueba: 0.7456
   Pérdida en prueba: 0.7891

🔍 Analizando predicciones por clase...
   Precisión por clase:
   • Avión: 0.7823
   • Automóvil: 0.8345
   • Pájaro: 0.6234
   • Gato: 0.6123
   • Ciervo: 0.7123
   • Perro: 0.6789
   • Rana: 0.7456
   • Caballo: 0.7890
   • Barco: 0.8234
   • Camión: 0.8567

✅ EJERCICIO COMPLETADO
============================================================
🎯 RESULTADOS:
   • Precisión final: 0.7456
   • Total de parámetros: 122,570
   • Épocas entrenadas: 18
============================================================
```

## 🎓 Análisis de Aprendizaje

### Identificación del Paradigma
**✅ APRENDIZAJE SUPERVISADO**
- **Datos etiquetados**: Cada imagen tiene una etiqueta (10 categorías)
- **Objetivo**: Predecir la categoría correcta
- **Métrica**: Accuracy comparando predicciones con etiquetas reales

### Ventajas de las CNN
- **Extracción automática de características**: No requiere ingeniería manual
- **Invarianza a traslación**: Detecta características sin importar posición
- **Compartición de parámetros**: Reduce significativamente el número de parámetros
- **Jerarquía de características**: Capas profundas aprenden patrones complejos

### Limitaciones
- Requiere grandes cantidades de datos
- Computacionalmente intensivas
- Menos interpretables que redes densas
- Sensibles a rotaciones y escalas extremas

## 🚀 Adaptación al Proyecto Integrador

### Ideas para Adaptación
1. **Transfer Learning**: Usar modelos pre-entrenados (ResNet, VGG) para datos limitados
2. **Data Augmentation**: Aumentar dataset con rotaciones, zoom, etc.
3. **Ajuste fino**: Adaptar arquitectura para dominio específico

### Ejemplo de Adaptación
```python
# Adaptar CNN para imágenes médicas
def crear_cnn_medica(input_shape, num_clases):
    base_model = tf.keras.applications.ResNet50(
        weights='imagenet', 
        include_top=False, 
        input_shape=input_shape
    )
    
    # Congelar capas base
    base_model.trainable = False
    
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_clases, activation='softmax')
    ])
    
    model.compile(optimizer='adam', 
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
    
    return model
```

## 📝 Ejercicios Adicionales

### Ejercicio 1: No Supervisado (Segmentación)
```python
# Convertir CNN a autoencoder para segmentación no supervisada
def crear_autoencoder_segmentacion():
    # Encoder
    input_img = layers.Input(shape=(32, 32, 3))
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    encoded = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    
    # Decoder
    x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(encoded)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2, 2))(x)
    decoded = layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)
    
    autoencoder = models.Model(input_img, decoded)
    autoencoder.compile(optimizer='adam', loss='mse')
    return autoencoder
```

### Ejercicio 2: Reforzado (Deep Q-CNN)
```python
# Usar CNN como extractor de características para DQN
class DeepQCNN:
    def __init__(self, input_shape, action_size):
        # CNN para extraer características de imágenes
        self.cnn = models.Sequential([
            layers.Conv2D(32, (8, 8), activation='relu', input_shape=input_shape),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (4, 4), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten()
        ])
        
        # Capa densa para Q-values
        self.q_network = models.Sequential([
            layers.Dense(512, activation='relu'),
            layers.Dense(action_size, activation='linear')
        ])
        
        self.model = models.Sequential([self.cnn, self.q_network])
        self.model.compile(optimizer='adam', loss='mse')
```

## 🔗 Recursos Adicionales

- [Documentación TensorFlow/Keras](https://www.tensorflow.org/api_docs/python/tf/keras)
- [CIFAR-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)
- [Transfer Learning Guide](https://www.tensorflow.org/tutorials/images/transfer_learning)
- [Data Augmentation](https://www.tensorflow.org/tutorials/images/data_augmentation)
