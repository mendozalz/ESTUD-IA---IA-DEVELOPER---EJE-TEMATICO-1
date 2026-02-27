# Red Neuronal Residual Network (ResNet)

## 📖 Teoría

### Definición
Las Redes Residuales (ResNet - Residual Networks) son arquitecturas de deep learning introducidas en 2015 que解决了 el problema de degradación en redes profundas mediante el uso de "conexiones residuales" o "skip connections".

### Características Principales
- **Conexiones residuales**: Saltan capas para preservar información
- **Bloques residuales**: Unidades básicas con caminos alternativos
- **Deep learning profundo**: Permite entrenar redes muy profundas (100+ capas)
- **Mitigación de vanishing gradients**: Las conexiones ayudan al flujo de gradientes
- **Identity mapping**: Permite que capas aprendan funciones de identidad

### Componentes
1. **Skip Connections**: Conexiones que saltan capas
2. **Bloques Residuales**: Unidades con caminos principal y residual
3. **Batch Normalization**: Normalización para estabilizar entrenamiento
4. **Bottleneck Layers**: Reducción dimensional para eficiencia
5. **Global Average Pooling**: Reemplaza capas densas finales

### Arquitecturas ResNet
- **ResNet-18/34**: Versiones ligeras para prototipado
- **ResNet-50/101/152**: Versiones profundas para producción
- **ResNeXt**: Extension con cardinalidad
- **Wide ResNet**: Versiones más anchas que profundas

### Ventajas sobre redes convencionales
- **Entrenamiento más profundo**: Permite redes de 100+ capas
- **Mejor convergencia**: Más estable y rápida
- **Menos overfitting**: Regularización implícita
- **Transfer learning excel**: Base para muchos modelos pre-entrenados

### Casos de Uso
- Clasificación de imágenes
- Detección de objetos
- Segmentación semántica
- Transfer learning
- Reconocimiento facial

## 🎯 Ejercicio Práctico: Clasificación de Imágenes CIFAR-100

### Tipo de Aprendizaje: **SUPERVISADO**
- **Por qué**: Usa imágenes etiquetadas con 100 categorías
- **Datos**: Dataset CIFAR-100 con etiquetas de clase

### Implementación Completa

```python
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt

print("🔀 RED RESNET - CLASIFICACIÓN CIFAR-100")
print("=" * 60)

# --- 1. Cargar y preprocesar datos ---
print("📊 Cargando y preprocesando datos...")
cifar100 = tf.keras.datasets.cifar100
(x_train, y_train), (x_test, y_test) = cifar100.load_data()

# Normalizar a [0, 1]
x_train, x_test = x_train / 255.0, x_test / 255.0

# Aplanar etiquetas
y_train, y_test = y_train.flatten(), y_test.flatten()

# Data augmentation
datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    zoom_range=0.1
)

datagen.fit(x_train)

print(f"   Datos de entrenamiento: {x_train.shape}")
print(f"   Datos de prueba: {x_test.shape}")
print(f"   Clases: {len(np.unique(y_train))}")

# --- 2. Definir bloque residual ---
print("\n🏗️ Construyendo bloques residuales...")

def residual_block(x, filters, kernel_size=3, stride=1):
    """Bloque residual básico"""
    # Camino principal
    shortcut = x
    
    # Primera convolución
    x = layers.Conv2D(filters, kernel_size, strides=stride, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    # Segunda convolución
    x = layers.Conv2D(filters, kernel_size, padding='same')(x)
    x = layers.BatchNormalization()(x)
    
    # Ajustar shortcut si es necesario
    if stride != 1 or shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, 1, strides=stride, padding='same')(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)
    
    # Conexión residual
    x = layers.Add()([x, shortcut])
    x = layers.Activation('relu')(x)
    
    return x

def bottleneck_block(x, filters, stride=1):
    """Bloque bottleneck para ResNet-50+"""
    shortcut = x
    
    # 1x1 convolución reducción
    x = layers.Conv2D(filters // 4, 1, strides=stride, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    # 3x3 convolución
    x = layers.Conv2D(filters // 4, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    # 1x1 convolución expansión
    x = layers.Conv2D(filters, 1, padding='same')(x)
    x = layers.BatchNormalization()(x)
    
    # Ajustar shortcut
    if stride != 1 or shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, 1, strides=stride, padding='same')(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)
    
    # Conexión residual
    x = layers.Add()([x, shortcut])
    x = layers.Activation('relu')(x)
    
    return x

# --- 3. Construir ResNet ---
print("Construyendo arquitectura ResNet...")

def build_resnet(input_shape=(32, 32, 3), num_classes=100, depth=34):
    """Construir ResNet según profundidad"""
    inputs = layers.Input(shape=input_shape)
    
    # Capa inicial
    x = layers.Conv2D(64, 3, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    # Configuración según profundidad
    if depth == 18:
        # ResNet-18: 2 bloques por etapa
        x = residual_block(x, 64)
        x = residual_block(x, 64)
        x = residual_block(x, 128, stride=2)
        x = residual_block(x, 128)
        x = residual_block(x, 256, stride=2)
        x = residual_block(x, 256)
        x = residual_block(x, 512, stride=2)
        x = residual_block(x, 512)
        
    elif depth == 34:
        # ResNet-34: 3-4-6-3 bloques por etapa
        for _ in range(3):
            x = residual_block(x, 64)
        for _ in range(4):
            x = residual_block(x, 128, stride=2 if _ == 0 else 1)
        for _ in range(6):
            x = residual_block(x, 256, stride=2 if _ == 0 else 1)
        for _ in range(3):
            x = residual_block(x, 512, stride=2 if _ == 0 else 1)
            
    elif depth == 50:
        # ResNet-50: 3-4-6-3 bloques bottleneck
        for _ in range(3):
            x = bottleneck_block(x, 64)
        for _ in range(4):
            x = bottleneck_block(x, 128, stride=2 if _ == 0 else 1)
        for _ in range(6):
            x = bottleneck_block(x, 256, stride=2 if _ == 0 else 1)
        for _ in range(3):
            x = bottleneck_block(x, 512, stride=2 if _ == 0 else 1)
    
    # Capa final
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(num_classes, activation='softmax')(x)
    
    model = models.Model(inputs, x)
    return model

# Crear modelo ResNet-34
model = build_resnet(depth=34)

# Mostrar arquitectura
model.summary()

# --- 4. Compilar y entrenar ---
print("\n🚀 Compilando y entrenando el modelo...")
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Callbacks
callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=8),
    tf.keras.callbacks.ModelCheckpoint('resnet_cifar100.h5', save_best_only=True)
]

# Entrenar con data augmentation
history = model.fit(
    datagen.flow(x_train, y_train, batch_size=64),
    epochs=100,
    validation_data=(x_test, y_test),
    callbacks=callbacks,
    verbose=1
)

# --- 5. Evaluar el modelo ---
print("\n📈 Evaluando el modelo...")
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"   Precisión en prueba: {test_acc:.4f}")
print(f"   Pérdida en prueba: {test_loss:.4f}")

# --- 6. Análisis de características aprendidas ---
print("\n🔍 Analizando características aprendidas...")

# Extraer características de capas intermedias
feature_extractor = models.Model(
    inputs=model.input,
    outputs=model.layers[-3].output  # Antes de la capa final
)

# Obtener características para algunas imágenes
sample_images = x_test[:100]
features = feature_extractor.predict(sample_images, verbose=0)

print(f"   Dimensionalidad de características: {features.shape}")

# Visualizar distribución de características
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.hist(features.flatten(), bins=50, alpha=0.7, density=True)
plt.title('Distribución de Características')
plt.xlabel('Valor')
plt.ylabel('Densidad')
plt.grid(True)

# --- 7. Comparación con red convencional ---
print("\n⚖️ Comparando ResNet vs CNN convencional...")

# Crear CNN convencional equivalente
def build_conventional_cnn(input_shape=(32, 32, 3), num_classes=100):
    model = models.Sequential([
        layers.Conv2D(64, 3, activation='relu', padding='same', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.Conv2D(64, 3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2),
        
        layers.Conv2D(128, 3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(128, 3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2),
        
        layers.Conv2D(256, 3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(256, 3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2),
        
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

# Entrenar CNN convencional por menos épocas
cnn_model = build_conventional_cnn()
cnn_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

print("   Entrenando CNN convencional...")
cnn_history = cnn_model.fit(
    datagen.flow(x_train, y_train, batch_size=64),
    epochs=30,
    validation_data=(x_test, y_test),
    verbose=0
)

# Comparar rendimiento
cnn_loss, cnn_acc = cnn_model.evaluate(x_test, y_test, verbose=0)

print(f"   ResNet-34 - Precisión: {test_acc:.4f}, Parámetros: {model.count_params():,}")
print(f"   CNN Convencional - Precisión: {cnn_acc:.4f}, Parámetros: {cnn_model.count_params():,}")

# --- 8. Visualizar entrenamiento ---
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.plot(history.history['accuracy'], label='ResNet Entrenamiento')
plt.plot(history.history['val_accuracy'], label='ResNet Validación')
plt.title('Precisión ResNet durante Entrenamiento')
plt.xlabel('Época')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

plt.subplot(1, 3, 2)
plt.plot(history.history['loss'], label='ResNet Entrenamiento')
plt.plot(history.history['val_loss'], label='ResNet Validación')
plt.title('Pérdida ResNet durante Entrenamiento')
plt.xlabel('Época')
plt.ylabel('Sparse Categorical Crossentropy')
plt.legend()
plt.grid(True)

# --- 9. Análisis de gradientes ---
plt.subplot(1, 3, 3)
# Simular análisis de flujo de gradientes
epochs_trained = len(history.history['loss'])
gradient_flow = np.exp(-np.linspace(0, 3, epochs_trained))  # Simulación

plt.plot(gradient_flow, label='ResNet (con skip connections)')
plt.plot(np.exp(-np.linspace(0, 5, epochs_trained)), label='CNN (sin skip connections)')
plt.title('Flujo de Gradientes (Simulado)')
plt.xlabel('Época')
plt.ylabel('Magnitud de Gradiente')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# --- 10. Visualizar predicciones ---
print("\n🎨 Visualizando predicciones...")
# Obtener predicciones
predictions = model.predict(x_test[:20], verbose=0)
predicted_classes = np.argmax(predictions, axis=1)

# Visualizar algunas predicciones
plt.figure(figsize=(15, 8))
for i in range(20):
    plt.subplot(4, 5, i + 1)
    plt.imshow(x_test[i])
    plt.title(f'Real: {y_test[i]}\nPred: {predicted_classes[i]}')
    plt.axis('off')

plt.suptitle('Predicciones de ResNet en CIFAR-100')
plt.tight_layout()
plt.show()

# --- 11. Análisis de errores ---
print("\n🔍 Analizando predicciones incorrectas...")
y_pred = np.argmax(model.predict(x_test, verbose=0), axis=1)
incorrect_indices = np.where(y_pred != y_test)[0]

print(f"   Total de errores: {len(incorrect_indices)} de {len(y_test)}")
print(f"   Tasa de error: {len(incorrect_indices)/len(y_test)*100:.2f}%")

# Mostrar algunos errores
if len(incorrect_indices) > 0:
    plt.figure(figsize=(12, 6))
    for i, idx in enumerate(incorrect_indices[:8]):
        plt.subplot(2, 4, i + 1)
        plt.imshow(x_test[idx])
        plt.title(f'Real: {y_test[idx]}\nPred: {y_pred[idx]}')
        plt.axis('off')
    plt.suptitle('Ejemplos de Predicciones Incorrectas')
    plt.tight_layout()
    plt.show()

# --- 12. Guardar el modelo ---
model.save('resnet_cifar100_final.h5')
print("\n💾 Modelo guardado como 'resnet_cifar100_final.h5'")

print("\n✅ EJERCICIO COMPLETADO")
print("=" * 60)
print("🎯 RESULTADOS:")
print(f"   • Precisión final: {test_acc:.4f}")
print(f"   • Pérdida final: {test_loss:.4f}")
print(f"   • Total de parámetros: {model.count_params():,}")
print(f"   • Épocas entrenadas: {len(history.history['loss'])}")
print(f"   • Ventaja sobre CNN: {(test_acc - cnn_acc)*100:.2f}%")
print("=" * 60)
```

## 📊 Resultados Esperados

```
🔀 RED RESNET - CLASIFICACIÓN CIFAR-100
============================================================
📊 Cargando y preprocesando datos...
   Datos de entrenamiento: (50000, 32, 32, 3)
   Datos de prueba: (10000, 32, 32, 3)
   Clases: 100

🏗️ Construyendo bloques residuales...
Construyendo arquitectura ResNet...
Model: "model"
__________________________________________________________________________________________________
 Layer (type)                    Output Shape         Param #     
=================================================================
 input_1 (InputLayer)            [(None, 32, 32, 3)]  0         
 conv2d (Conv2D)                 (None, 32, 32, 64)    1792       
 batch_normalization (BatchNorm (None, 32, 32, 64)    256       
 activation (Activation)        (None, 32, 32, 64)    0         
 ... (múltiples capas residuales) ...
 global_average_pooling2d (Glob (None, 64)           0         
 dense (Dense)                   (None, 100)           6500       
=================================================================
Total params: 21,285,644
Trainable params: 21,283,588
Non-trainable params: 2,056
__________________________________________________________________________________________________

🚀 Compilando y entrenando el modelo...
Epoch 1/100
782/782 [==============================] - 45s 58ms/step - loss: 4.1234 - accuracy: 0.0456 - val_loss: 3.8765 - val_accuracy: 0.1234
...
Epoch 67/100
782/782 [==============================] - 38s 49ms/step - loss: 0.8765 - accuracy: 0.7234 - val_loss: 1.2345 - val_accuracy: 0.6543

📈 Evaluando el modelo...
   Precisión en prueba: 0.6543
   Pérdida en prueba: 1.2345

⚖️ Comparando ResNet vs CNN convencional...
   Entrenando CNN convencional...
   ResNet-34 - Precisión: 0.6543, Parámetros: 21,285,644
   CNN Convencional - Precisión: 0.5432, Parámetros: 15,234,567

✅ EJERCICIO COMPLETADO
============================================================
🎯 RESULTADOS:
   • Precisión final: 0.6543
   • Pérdida final: 1.2345
   • Total de parámetros: 21,285,644
   • Épocas entrenadas: 67
   • Ventaja sobre CNN: 11.11%
============================================================
```

## 🎓 Análisis de Aprendizaje

### Identificación del Paradigma
**✅ APRENDIZAJE SUPERVISADO**
- **Datos etiquetados**: Usa imágenes CIFAR-100 con 100 categorías
- **Objetivo**: Clasificar imágenes en categorías correctas
- **Métrica**: Accuracy comparando predicciones con etiquetas reales

### Ventajas de ResNet
- **Deep learning profundo**: Permite entrenar redes muy profundas
- **Mejor convergencia**: Más estable y rápida que redes convencionales
- **Skip connections**: Mitigan el problema de vanishing gradients
- **Transfer learning**: Excelente base para modelos pre-entrenados

### Limitaciones
- **Complejidad computacional**: Requiere más recursos que redes simples
- **Memoria**: Las conexiones residuales usan más memoria
- **Overfitting**: Puede sobreajustar en datasets pequeños
- **Diseño**: Requiere diseño cuidadoso de la arquitectura

## 🚀 Adaptación al Proyecto Integrador

### Ideas para Adaptación
1. **Transfer learning**: Usar ResNet pre-entrenado para dominio específico
2. **Fine-tuning**: Adaptar últimas capas para proyecto particular
3. **Multi-task**: Extender para múltiples tareas simultáneas

### Ejemplo de Adaptación
```python
# Adaptar ResNet para transfer learning
def crear_resnet_transfer(num_classes, input_shape=(224, 224, 3)):
    """
    Adapta ResNet pre-entrenada para transfer learning
    """
    # Cargar ResNet pre-entrenada
    base_model = tf.keras.applications.ResNet50(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )
    
    # Congelar capas base
    base_model.trainable = False
    
    # Añadir capas personalizadas
    inputs = layers.Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(256, activation='relu')(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = models.Model(inputs, outputs)
    
    # Compilar con learning rate bajo
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model
```

## 📝 Ejercicios Adicionales

### Ejercicio 1: No Supervisado (ResNet Autoencoder)
```python
# Usar ResNet como encoder para autoencoder no supervisado
def crear_resnet_autoencoder(input_shape=(32, 32, 3)):
    """Crear autoencoder usando ResNet como encoder"""
    
    # Encoder (ResNet sin capa final)
    encoder_inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(64, 3, padding='same')(encoder_inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    # Bloques residuales comprimen información
    for _ in range(3):
        x = residual_block(x, 128, stride=2)
    
    # Espacio latente
    latent = layers.GlobalAveragePooling2D()(x)
    latent = layers.Dense(128, activation='relu')(latent)
    
    # Decoder
    decoder_inputs = layers.Input(shape=(128,))
    x = layers.Dense(8 * 8 * 256, activation='relu')(decoder_inputs)
    x = layers.Reshape((8, 8, 256))(x)
    
    # Upsampling y convoluciones
    x = layers.Conv2DTranspose(128, 3, strides=2, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    x = layers.Conv2DTranspose(64, 3, strides=2, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    outputs = layers.Conv2D(3, 3, activation='sigmoid', padding='same')(x)
    
    encoder = models.Model(encoder_inputs, latent)
    decoder = models.Model(decoder_inputs, outputs)
    autoencoder = models.Model(encoder_inputs, decoder(encoder(encoder_inputs)))
    
    autoencoder.compile(optimizer='adam', loss='mse')
    return autoencoder, encoder, decoder
```

### Ejercicio 2: Reforzado (ResNet-DQN)
```python
# Usar ResNet como extractor de características para DQN
class ResNetDQN:
    def __init__(self, input_shape, num_actions):
        # ResNet como extractor de características
        self.feature_extractor = models.Sequential([
            layers.Conv2D(32, 3, activation='relu', input_shape=input_shape),
            layers.ResidualBlock(64),
            layers.ResidualBlock(128),
            layers.GlobalAveragePooling2D()
        ])
        
        # Cabeza DQN
        self.q_network = models.Sequential([
            layers.Dense(512, activation='relu'),
            layers.Dense(256, activation='relu'),
            layers.Dense(num_actions)
        ])
        
        self.model = models.Sequential([self.feature_extractor, self.q_network])
        self.target_model = models.clone_model(self.model)
        
        self.model.compile(optimizer='adam', loss='mse')
    
    def get_q_values(self, state):
        """Obtener Q-values para un estado"""
        return self.model.predict(state[np.newaxis, ...], verbose=0)[0]
    
    def update_target_network(self):
        """Actualizar red target"""
        self.target_model.set_weights(self.model.get_weights())
```

## 🔗 Recursos Adicionales

- [Documentación TensorFlow/Keras](https://www.tensorflow.org/api_docs/python/tf/keras)
- [ResNet Paper Original](https://arxiv.org/abs/1512.03385)
- [Transfer Learning Guide](https://www.tensorflow.org/tutorials/images/transfer_learning)
- [Keras Applications](https://www.tensorflow.org/api_docs/python/tf/keras/applications)
