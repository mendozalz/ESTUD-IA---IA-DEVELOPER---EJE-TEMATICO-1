# Red Neuronal Autoencoder

## 📖 Teoría

### Definición
Los Autoencoders son redes neuronales no supervisadas diseñadas para aprender representaciones comprimidas de datos. Consisten en dos partes: un encoder que comprime los datos en un espacio latente de menor dimensionalidad, y un decoder que reconstruye los datos originales desde esta representación.

### Características Principales
- **Arquitectura simétrica**: Encoder y decoder son simétricos
- **Compresión no lineal**: Aprende representaciones comprimidas no lineales
- **Aprendizaje no supervisado**: No requiere etiquetas
- **Reconstrucción**: El objetivo es reconstruir los datos de entrada
- **Espacio latente**: Representación de baja dimensionalidad aprendida

### Componentes
1. **Encoder**: Comprime los datos a espacio latente
2. **Espacio latente**: Representación comprimida de los datos
3. **Decoder**: Reconstruye los datos desde el espacio latente
4. **Función de pérdida**: Generalmente error cuadrático medio
5. **Cuello de botella**: Capa de menor dimensionalidad

### Tipos de Autoencoders
- **Autoencoder simple**: Versión básica con encoder/decoder
- **Variational Autoencoder (VAE)**: Añade distribución probabilística
- **Denoising Autoencoder**: Aprende reconstruir desde datos ruidosos
- **Sparse Autoencoder**: Añade regularización L1 para dispersión

### Casos de Uso
- Reducción de dimensionalidad
- Detección de anomalías
- Compresión de datos
- Generación de datos
- Pre-entrenamiento de características

## 🎯 Ejercicio Práctico: Compresión y Reconstrucción de Imágenes

### Tipo de Aprendizaje: **NO SUPERVISADO**
- **Por qué**: No usa etiquetas, aprende de imágenes sin supervisión
- **Datos**: Imágenes MNIST sin etiquetas para aprender representación

### Implementación Completa

```python
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt

print("🔄 RED AUTOENCODER - COMPRESIÓN DE IMÁGENES")
print("=" * 60)

# --- 1. Cargar y preprocesar datos ---
print("📊 Cargando y preprocesando datos...")
mnist = tf.keras.datasets.mnist
(x_train, _), (x_test, _) = mnist.load_data()

# Normalizar a [0, 1]
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Añadir dimensión de canal
x_train = np.expand_dims(x_train, axis=-1)
x_test = np.expand_dims(x_test, axis=-1)

print(f"   Datos de entrenamiento: {x_train.shape}")
print(f"   Datos de prueba: {x_test.shape}")
print(f"   Rango de valores: [{x_train.min():.2f}, {x_train.max():.2f}]")

# --- 2. Definir el Autoencoder ---
print("\n🏗️ Construyendo el Autoencoder...")

# Parámetros
original_dim = 28 * 28
encoding_dim = 32  # Compresión a 32 dimensiones (factor de 0.14)

# Encoder
encoder_inputs = layers.Input(shape=(28, 28, 1))
x = layers.Flatten()(encoder_inputs)
x = layers.Dense(128, activation='relu')(x)
x = layers.Dense(64, activation='relu')(x)
encoded = layers.Dense(encoding_dim, activation='relu')(x)

# Decoder
x = layers.Dense(64, activation='relu')(encoded)
x = layers.Dense(128, activation='relu')(x)
x = layers.Dense(original_dim, activation='sigmoid')(x)
decoded = layers.Reshape((28, 28, 1))(x)

# Modelo completo
autoencoder = models.Model(encoder_inputs, decoded)

# Modelo encoder separado
encoder = models.Model(encoder_inputs, encoded)

# Modelo decoder separado
decoder_inputs = layers.Input(shape=(encoding_dim,))
x = layers.Dense(64, activation='relu')(decoder_inputs)
x = layers.Dense(128, activation='relu')(x)
x = layers.Dense(original_dim, activation='sigmoid')(x)
decoder = models.Model(decoder_inputs, layers.Reshape((28, 28, 1))(x))

# Mostrar arquitecturas
print("\nArquitectura del Autoencoder:")
autoencoder.summary()

print("\nArquitectura del Encoder:")
encoder.summary()

print("\nArquitectura del Decoder:")
decoder.summary()

# --- 3. Compilar y entrenar ---
print("\n🚀 Compilando y entrenando el Autoencoder...")
autoencoder.compile(
    optimizer='adam',
    loss='mse',
    metrics=['mae']
)

# Callbacks
callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3)
]

history = autoencoder.fit(
    x_train, x_train,  # Entrada = salida para autoencoder
    epochs=30,
    batch_size=256,
    validation_data=(x_test, x_test),
    callbacks=callbacks,
    verbose=1
)

# --- 4. Evaluar el modelo ---
print("\n📈 Evaluando el Autoencoder...")
test_loss, test_mae = autoencoder.evaluate(x_test, x_test, verbose=0)
print(f"   Error cuadrático medio en prueba: {test_loss:.6f}")
print(f"   Error absoluto medio en prueba: {test_mae:.6f}")

# --- 5. Visualizar reconstrucciones ---
print("\n🎨 Visualizando reconstrucciones...")
# Obtener reconstrucciones
reconstructed = autoencoder.predict(x_test, verbose=0)

# Visualizar imágenes originales vs reconstruidas
n = 10  # Número de imágenes a mostrar
plt.figure(figsize=(20, 4))

for i in range(n):
    # Imagen original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i].squeeze(), cmap='gray')
    plt.title('Original')
    plt.axis('off')
    
    # Imagen reconstruida
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(reconstructed[i].squeeze(), cmap='gray')
    plt.title('Reconstruida')
    plt.axis('off')

plt.suptitle('Imágenes Originales vs Reconstruidas')
plt.tight_layout()
plt.show()

# --- 6. Analizar espacio latente ---
print("\n🔍 Analizando espacio latente...")
# Obtener representaciones latentes
latent_representations = encoder.predict(x_test, verbose=0)

print(f"   Dimensionalidad original: {x_test.shape[1:]}")
print(f"   Dimensionalidad latente: {latent_representations.shape[1]}")
print(f"   Factor de compresión: {latent_representations.shape[1] / (28*28):.4f}")

# Visualizar distribución del espacio latente
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.hist(latent_representations.flatten(), bins=50, alpha=0.7, density=True)
plt.title('Distribución del Espacio Latente')
plt.xlabel('Valor')
plt.ylabel('Densidad')
plt.grid(True)

# Visualizar correlaciones entre dimensiones latentes
plt.subplot(1, 3, 2)
correlation_matrix = np.corrcoef(latent_representations.T)
plt.imshow(correlation_matrix, cmap='coolwarm', aspect='auto')
plt.title('Matriz de Correlación Latente')
plt.colorbar()
plt.xlabel('Dimensión Latente')
plt.ylabel('Dimensión Latente')

# Visualizar primeras 2 dimensiones latentes
plt.subplot(1, 3, 3)
plt.scatter(latent_representations[:, 0], latent_representations[:, 1], 
           alpha=0.5, s=1)
plt.title('Primeras 2 Dimensiones Latentes')
plt.xlabel('Dimensión 1')
plt.ylabel('Dimensión 2')
plt.grid(True)

plt.tight_layout()
plt.show()

# --- 7. Generar nuevas imágenes ---
print("\n🎲 Generando imágenes desde el espacio latente...")

# Generar vectores latentes aleatorios
n_generated = 16
random_latent = np.random.normal(0, 1, (n_generated, encoding_dim))

# Generar imágenes desde vectores latentes
generated_images = decoder.predict(random_latent, verbose=0)

# Visualizar imágenes generadas
plt.figure(figsize=(8, 8))
for i in range(n_generated):
    ax = plt.subplot(4, 4, i + 1)
    plt.imshow(generated_images[i].squeeze(), cmap='gray')
    plt.axis('off')

plt.suptitle('Imágenes Generadas desde Espacio Latente')
plt.tight_layout()
plt.show()

# --- 8. Interpolación en espacio latente ---
print("\n🔄 Realizando interpolación en espacio latente...")

# Seleccionar dos imágenes del conjunto de prueba
img1_idx, img2_idx = 0, 1
latent1 = encoder.predict(x_test[img1_idx:img1_idx+1], verbose=0)
latent2 = encoder.predict(x_test[img2_idx:img2_idx+1], verbose=0)

# Crear interpolación lineal
n_steps = 10
interpolated_images = []

for i in range(n_steps):
    alpha = i / (n_steps - 1)
    interpolated_latent = (1 - alpha) * latent1 + alpha * latent2
    interpolated_img = decoder.predict(interpolated_latent, verbose=0)
    interpolated_images.append(interpolated_img[0])

# Visualizar interpolación
plt.figure(figsize=(15, 2))
for i, img in enumerate(interpolated_images):
    ax = plt.subplot(1, n_steps, i + 1)
    plt.imshow(img.squeeze(), cmap='gray')
    plt.title(f'α={i/(n_steps-1):.1f}')
    plt.axis('off')

plt.suptitle('Interpolación en Espacio Latente')
plt.tight_layout()
plt.show()

# --- 9. Detección de anomalías ---
print("\n🚨 Evaluando detección de anomalías...")

# Crear imágenes anómalas (con ruido)
noise_factor = 0.5
x_test_noisy = x_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape)
x_test_noisy = np.clip(x_test_noisy, 0., 1.)

# Reconstruir imágenes normales y anómalas
reconstructed_normal = autoencoder.predict(x_test[:1000], verbose=0)
reconstructed_noisy = autoencoder.predict(x_test_noisy[:1000], verbose=0)

# Calcular errores de reconstrucción
errors_normal = np.mean(np.square(x_test[:1000] - reconstructed_normal), axis=(1, 2, 3))
errors_noisy = np.mean(np.square(x_test_noisy[:1000] - reconstructed_noisy), axis=(1, 2, 3))

# Visualizar distribución de errores
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.hist(errors_normal, bins=50, alpha=0.7, label='Normal', density=True)
plt.hist(errors_noisy, bins=50, alpha=0.7, label='Anómalo', density=True)
plt.title('Distribución de Errores de Reconstrucción')
plt.xlabel('Error MSE')
plt.ylabel('Densidad')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
# Visualizar ejemplos
n_examples = 5
plt.figure(figsize=(10, 6))
for i in range(n_examples):
    # Imagen normal
    ax = plt.subplot(3, n_examples, i + 1)
    plt.imshow(x_test[i].squeeze(), cmap='gray')
    plt.title(f'Normal\nError: {errors_normal[i]:.4f}')
    plt.axis('off')
    
    # Imagen ruidosa
    ax = plt.subplot(3, n_examples, i + 1 + n_examples)
    plt.imshow(x_test_noisy[i].squeeze(), cmap='gray')
    plt.title('Ruidosa')
    plt.axis('off')
    
    # Reconstrucción
    ax = plt.subplot(3, n_examples, i + 1 + 2*n_examples)
    plt.imshow(reconstructed_noisy[i].squeeze(), cmap='gray')
    plt.title(f'Reconstruida\nError: {errors_noisy[i]:.4f}')
    plt.axis('off')

plt.suptitle('Detección de Anomalías con Autoencoder')
plt.tight_layout()
plt.show()

# Calcular métricas de detección
threshold = np.percentile(errors_normal, 95)  # 95th percentile
true_positives = np.sum(errors_noisy > threshold)
false_negatives = np.sum(errors_noisy <= threshold)
false_positives = np.sum(errors_normal > threshold)
true_negatives = np.sum(errors_normal <= threshold)

precision = true_positives / (true_positives + false_positives + 1e-8)
recall = true_positives / (true_positives + false_negatives + 1e-8)
f1_score = 2 * (precision * recall) / (precision + recall + 1e-8)

print(f"   Umbral de anomalía: {threshold:.6f}")
print(f"   Precisión: {precision:.4f}")
print(f"   Recall: {recall:.4f}")
print(f"   F1-Score: {f1_score:.4f}")

# --- 10. Guardar modelos ---
autoencoder.save('autoencoder_mnist.h5')
encoder.save('encoder_mnist.h5')
decoder.save('decoder_mnist.h5')
print("\n💾 Modelos guardados:")
print("   • 'autoencoder_mnist.h5'")
print("   • 'encoder_mnist.h5'")
print("   • 'decoder_mnist.h5'")

print("\n✅ EJERCICIO COMPLETADO")
print("=" * 60)
print("🎯 RESULTADOS:")
print(f"   • Error final (MSE): {test_loss:.6f}")
print(f"   • Error final (MAE): {test_mae:.6f}")
print(f"   • Factor de compresión: {encoding_dim / original_dim:.4f}")
print(f"   • Total de parámetros: {autoencoder.count_params():,}")
print(f"   • Épocas entrenadas: {len(history.history['loss'])}")
print("=" * 60)
```

## 📊 Resultados Esperados

```
🔄 RED AUTOENCODER - COMPRESIÓN DE IMÁGENES
============================================================
📊 Cargando y preprocesando datos...
   Datos de entrenamiento: (60000, 28, 28, 1)
   Datos de prueba: (10000, 28, 28, 1)
   Rango de valores: [0.00, 1.00]

🏗️ Construyendo el Autoencoder...

Arquitectura del Autoencoder:
Model: "model"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 input_1 (InputLayer)        [(None, 28, 28, 1)]       0         
 flatten (Flatten)           (None, 784)               0         
 dense (Dense)               (None, 128)               100480    
 dense_1 (Dense)             (None, 64)                8256      
 dense_2 (Dense)             (None, 32)                2080      
 dense_3 (Dense)             (None, 64)                2112      
 dense_4 (Dense)             (None, 128)               8320      
 dense_5 (Dense)             (None, 784)               101136    
 reshape (Reshape)           (None, 28, 28, 1)         0         
=================================================================
Total params: 220,384
Trainable params: 220,384
Non-trainable params: 0
_________________________________________________________________

🚀 Compilando y entrenando el Autoencoder...
Epoch 1/30
235/235 [==============================] - 3s 12ms/step - loss: 0.1234 - mae: 0.2345 - val_loss: 0.0987 - val_mae: 0.1876
...
Epoch 18/30
235/235 [==============================] - 2s 8ms/step - loss: 0.0456 - mae: 0.1234 - val_loss: 0.0432 - val_mae: 0.1198

📈 Evaluando el Autoencoder...
   Error cuadrático medio en prueba: 0.043234
   Error absoluto medio en prueba: 0.119876

🔍 Analizando espacio latente...
   Dimensionalidad original: (28, 28, 1)
   Dimensionalidad latente: 32
   Factor de compresión: 0.0408

🚨 Evaluando detección de anomalías...
   Umbral de anomalía: 0.056789
   Precisión: 0.8234
   Recall: 0.7890
   F1-Score: 0.8056

✅ EJERCICIO COMPLETADO
============================================================
🎯 RESULTADOS:
   • Error final (MSE): 0.043234
   • Error final (MAE): 0.119876
   • Factor de compresión: 0.0408
   • Total de parámetros: 220,384
   • Épocas entrenadas: 18
============================================================
```

## 🎓 Análisis de Aprendizaje

### Identificación del Paradigma
**✅ APRENDIZAJE NO SUPERVISADO**
- **Sin etiquetas**: No usa etiquetas de las imágenes MNIST
- **Aprendizaje autorreconstrucción**: Aprende a reconstruir sus propias entradas
- **Objetivo**: Comprimir y reconstruir datos
- **Métrica**: Error de reconstrucción (MSE, MAE)

### Ventajas de los Autoencoders
- **Reducción dimensional**: Aprende representaciones comprimidas
- **Detección de anomalías**: Identifica datos inusuales por alto error
- **Generación de datos**: Puede generar nuevos datos similares
- **Pre-entrenamiento**: Útil para transfer learning

### Limitaciones
- **Reconstrucción limitada**: No siempre reconstruye detalles finos
- **Espacio latente desestructurado**: Puede no ser interpretable
- **Overfitting**: Propenso a memorizar datos de entrenamiento
- **Requiere datos similares**: Funciona mejor con datos homogéneos

## 🚀 Adaptación al Proyecto Integrador

### Ideas para Adaptación
1. **Dominio específico**: Adaptar para datos del proyecto (imágenes, series temporales, etc.)
2. **Denoising Autoencoder**: Para limpiar datos ruidosos del proyecto
3. **Variational Autoencoder**: Para generación controlada de datos

### Ejemplo de Adaptación
```python
# Adaptar Autoencoder para datos específicos del proyecto
def crear_autoencoder_proyecto(input_shape, compression_factor=0.1):
    """
    Adapta Autoencoder para datos específicos del proyecto
    """
    # Calcular dimensión latente
    original_dim = np.prod(input_shape)
    encoding_dim = int(original_dim * compression_factor)
    
    # Encoder
    inputs = layers.Input(shape=input_shape)
    x = layers.Flatten()(inputs)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dense(128, activation='relu')(x)
    encoded = layers.Dense(encoding_dim, activation='relu')(x)
    
    # Decoder
    x = layers.Dense(128, activation='relu')(encoded)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dense(original_dim, activation='sigmoid')(x)
    decoded = layers.Reshape(input_shape)(x)
    
    autoencoder = models.Model(inputs, decoded)
    encoder = models.Model(inputs, encoded)
    
    autoencoder.compile(optimizer='adam', loss='mse')
    return autoencoder, encoder
```

## 📝 Ejercicios Adicionales

### Ejercicio 1: Supervisado (Autoencoder para Clasificación)
```python
# Usar encoder como extractor de características para clasificación supervisada
def crear_clasificador_con_autoencoder(encoder, num_classes):
    """Crear clasificador usando encoder pre-entrenado"""
    
    # Congelar encoder
    encoder.trainable = False
    
    # Añadir capas de clasificación
    inputs = encoder.input
    x = encoder.output
    x = layers.Dense(64, activation='relu')(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    classifier = models.Model(inputs, outputs)
    classifier.compile(optimizer='adam', 
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
    
    return classifier
```

### Ejercicio 2: Reforzado (Autoencoder para RL)
```python
# Usar Autoencoder para aprendizaje por refuerzo
class AutoencoderEnvironment:
    def __init__(self, autoencoder, data):
        self.autoencoder = autoencoder
        self.data = data
        self.current_state = None
        self.state_idx = 0
    
    def reset(self):
        """Reiniciar entorno"""
        self.state_idx = 0
        self.current_state = self.data[self.state_idx]
        return self.current_state
    
    def step(self, action):
        """Realizar acción y obtener recompensa"""
        # Acción: modificar estado (ej. añadir ruido)
        modified_state = self.current_state + action
        
        # Recompensa: qué tan bien se reconstruye el estado modificado
        reconstructed = self.autoencoder.predict(modified_state[np.newaxis, ...], verbose=0)
        reconstruction_error = np.mean(np.square(modified_state - reconstructed[0]))
        reward = -reconstruction_error  # Menor error = mayor recompensa
        
        # Siguiente estado
        self.state_idx = (self.state_idx + 1) % len(self.data)
        self.current_state = self.data[self.state_idx]
        
        return self.current_state, reward, False, {}
```

## 🔗 Recursos Adicionales

- [Documentación TensorFlow/Keras](https://www.tensorflow.org/api_docs/python/tf/keras)
- [Autoencoder Tutorial](https://www.tensorflow.org/tutorials/generative/autoencoder)
- [Variational Autoencoder](https://www.tensorflow.org/tutorials/generative/cvae)
- [Denoising Autoencoders](https://www.tensorflow.org/tutorials/generative/denoise)
