# Red Neuronal Densa (Feedforward)

## 📖 Teoría

### Definición
Las redes neuronales densas (también conocidas como Feedforward o MLP - Multilayer Perceptron) son el tipo más fundamental de red neuronal. En estas redes, la información fluye en una sola dirección: desde la capa de entrada hasta la capa de salida, sin ciclos ni retroalimentación.

### Características Principales
- **Flujo unidireccional**: La información se mueve solo hacia adelante
- **Sin memoria**: No retienen información de estados anteriores
- **Capas completamente conectadas**: Cada neurona está conectada con todas las neuronas de la capa siguiente
- **Arquitectura simple**: Entrada → Capas ocultas → Salida

### Componentes
1. **Capa de entrada**: Recibe los datos brutos
2. **Capas ocultas**: Procesan y transforman la información
3. **Capa de salida**: Produce el resultado final
4. **Funciones de activación**: ReLU, sigmoid, softmax, etc.
5. **Pesos y sesgos**: Parámetros aprendibles

### Casos de Uso
- Clasificación de datos tabulares
- Regresión
- Reconocimiento de patrones simples
- Sistemas de recomendación básicos

## 🎯 Ejercicio Práctico: Clasificación de Dígitos MNIST

### Tipo de Aprendizaje: **SUPERVISADO**
- **Por qué**: Usa etiquetas (0-9) para entrenar el modelo
- **Datos**: Imágenes etiquetadas de dígitos escritos a mano

### Implementación Completa

```python
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np

print("🔗 RED NEURONAL DENSA - CLASIFICACIÓN DE DÍGITOS MNIST")
print("=" * 60)

# --- 1. Cargar y preprocesar datos ---
print("📊 Cargando y preprocesando datos...")
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalizar a [0, 1]
x_train, x_test = x_train / 255.0, x_test / 255.0

print(f"   Datos de entrenamiento: {x_train.shape}")
print(f"   Datos de prueba: {x_test.shape}")
print(f"   Clases: {np.unique(y_train)}")

# --- 2. Definir el modelo MLP ---
print("\n🏗️ Construyendo el modelo MLP...")
model = models.Sequential([
    layers.Flatten(input_shape=(28, 28)),  # Aplana imagen 28x28 a 784
    layers.Dense(128, activation='relu'),    # Capa oculta con 128 neuronas
    layers.Dropout(0.2),                   # Regularización
    layers.Dense(64, activation='relu'),     # Segunda capa oculta
    layers.Dense(10, activation='softmax')   # Capa de salida (10 clases)
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
    tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=2)
]

history = model.fit(
    x_train, y_train,
    epochs=20,
    batch_size=128,
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
    plt.imshow(x_test[i], cmap='gray')
    plt.title(f"Real: {y_test[i]}\nPred: {np.argmax(predictions[i])}")
    plt.axis('off')

plt.tight_layout()
plt.show()

# --- 7. Análisis de errores ---
print("\n🔍 Analizando predicciones incorrectas...")
y_pred = np.argmax(model.predict(x_test), axis=1)
incorrect_indices = np.where(y_pred != y_test)[0]

print(f"   Total de errores: {len(incorrect_indices)} de {len(y_test)}")
print(f"   Tasa de error: {len(incorrect_indices)/len(y_test)*100:.2f}%")

# Mostrar algunos errores
if len(incorrect_indices) > 0:
    plt.figure(figsize=(12, 4))
    for i, idx in enumerate(incorrect_indices[:6]):
        plt.subplot(2, 3, i + 1)
        plt.imshow(x_test[idx], cmap='gray')
        plt.title(f"Real: {y_test[idx]}\nPred: {y_pred[idx]}")
        plt.axis('off')
    plt.suptitle('Ejemplos de Predicciones Incorrectas')
    plt.show()

# --- 8. Guardar el modelo ---
model.save('mlp_mnist.h5')
print("\n💾 Modelo guardado como 'mlp_mnist.h5'")

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
🔗 RED NEURONAL DENSA - CLASIFICACIÓN DE DÍGITOS MNIST
============================================================
📊 Cargando y preprocesando datos...
   Datos de entrenamiento: (60000, 28, 28)
   Datos de prueba: (10000, 28, 28)
   Clases: [0 1 2 3 4 5 6 7 8 9]

🏗️ Construyendo el modelo MLP...
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 flatten (Flatten)           (None, 784)               0         
 dense (Dense)               (None, 128)               100480    
 dropout (Dropout)            (None, 128)               0         
 dense_1 (Dense)             (None, 64)                8256      
 dense_2 (Dense)             (None, 10)                650        
=================================================================
Total params: 109,386
Trainable params: 109,386
Non-trainable params: 0
_________________________________________________________________

🚀 Compilando y entrenando el modelo...
Epoch 1/20
469/469 [==============================] - 2s 4ms/step - loss: 0.3421 - accuracy: 0.9021 - val_loss: 0.1654 - val_accuracy: 0.9512
...
Epoch 12/20
469/469 [==============================] - 2s 4ms/step - loss: 0.0189 - accuracy: 0.9941 - val_loss: 0.0732 - val_accuracy: 0.9794

📈 Evaluando el modelo...
   Precisión en prueba: 0.9794
   Pérdida en prueba: 0.0732

🔍 Analizando predicciones incorrectas...
   Total de errores: 206 de 10000
   Tasa de error: 2.06%

✅ EJERCICIO COMPLETADO
============================================================
🎯 RESULTADOS:
   • Precisión final: 0.9794
   • Total de parámetros: 109,386
   • Épocas entrenadas: 12
============================================================
```

## 🎓 Análisis de Aprendizaje

### Identificación del Paradigma
**✅ APRENDIZAJE SUPERVISADO**
- **Datos etiquetados**: Cada imagen tiene una etiqueta (0-9)
- **Objetivo**: Predecir la etiqueta correcta
- **Métrica**: Accuracy comparando predicciones con etiquetas reales

### Ventajas de las Redes Densas
- Simplicidad arquitectónica
- Rápido entrenamiento
- Buen rendimiento para datos tabulares
- Interpretabilidad relativamente alta

### Limitaciones
- No capturan relaciones espaciales
- Reieren datos estructurados
- No tienen memoria de secuencias

## 🚀 Adaptación al Proyecto Integrador

### Ideas para Adaptación
1. **Clasificación de datos del proyecto**: Adaptar para clasificar datos específicos
2. **Regresión**: Modificar para predecir valores continuos
3. **Feature engineering**: Incorporar características específicas del dominio

### Ejemplo de Adaptación
```python
# Adaptar para datos de proyecto personal
def adaptar_mlp_proyecto(X_datos, y_etiquetas, num_clases):
    """
    Adapta el MLP para datos específicos del proyecto
    """
    model = models.Sequential([
        layers.Input(shape=(X_datos.shape[1],)),  # Ajustar a tus datos
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),
        layers.Dense(num_clases, activation='softmax')
    ])
    
    model.compile(optimizer='adam', 
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
    
    return model
```

## 📝 Ejercicios Adicionales

### Ejercicio 1: No Supervisado (Autoencoder)
```python
# Convertir MLP a Autoencoder para reducción dimensional
def crear_autoencoder():
    input_layer = layers.Input(shape=(784,))
    encoded = layers.Dense(32, activation='relu')(input_layer)
    decoded = layers.Dense(784, activation='sigmoid')(encoded)
    
    autoencoder = models.Model(input_layer, decoded)
    encoder = models.Model(input_layer, encoded)
    
    autoencoder.compile(optimizer='adam', loss='mse')
    return autoencoder, encoder
```

### Ejercicio 2: Reforzado (Q-Learning)
```python
# Usar MLP como aproximador de Q-values
class QMLP:
    def __init__(self, state_size, action_size):
        self.model = models.Sequential([
            layers.Dense(64, activation='relu', input_shape=(state_size,)),
            layers.Dense(32, activation='relu'),
            layers.Dense(action_size, activation='linear')
        ])
        self.model.compile(optimizer='adam', loss='mse')
```

## 🔗 Recursos Adicionales

- [Documentación TensorFlow/Keras](https://www.tensorflow.org/api_docs/python/tf/keras)
- [MNIST Dataset](http://yann.lecun.com/exdb/mnist/)
- [Visualización de Redes Neuronales](https://www.tensorflow.org/tutorials/keras/classification)
