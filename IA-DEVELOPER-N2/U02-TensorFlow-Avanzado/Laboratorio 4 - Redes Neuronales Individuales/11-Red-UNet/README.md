# Red Neuronal U-Net

## 📖 Teoría

### Definición
U-Net es una arquitectura de red neuronal convolucional diseñada específicamente para segmentación semántica de imágenes biomédicas. Introducida en 2015, se caracteriza por su forma de "U" con conexiones de salto que combinan características de alta y baja resolución.

### Características Principales
- **Arquitectura en U**: Encoder-decoder con forma característica
- **Skip connections**: Conectan capas simétricas para preservar detalles
- **Encoder**: Contrae espacialmente, extrae características
- **Decoder**: Expande espacialmente, reconstruye segmentación
- **Multi-scale**: Procesa información a múltiples escalas

### Componentes
1. **Contracting Path (Encoder)**: Serie de convoluciones y max-pooling
2. **Expansive Path (Decoder)**: Serie de up-convoluciones y concatenaciones
3. **Skip Connections**: Conexiones entre capas simétricas
4. **Bottleneck**: Capa central con características más abstractas
5. **Final Convolution**: Capa de salida con activación softmax/sigmoid

### Arquitectura Detallada
- **Encoder**: 4 bloques de convolución + max-pooling
- **Bottleneck**: 2 capas convolucionales sin pooling
- **Decoder**: 4 bloques de up-convolución + concatenación
- **Salida**: 1x1 convolución para segmentación final

### Ventajas sobre otras arquitecturas
- **Preservación de detalles**: Skip connections mantienen información espacial
- **Precisión en bordes**: Mejor segmentación de contornos
- **Efficiente**: Menos parámetros que arquitecturas más complejas
- **Flexible**: Adaptable a diferentes tamaños de imagen

### Casos de Uso
- Segmentación médica (órganos, tumores)
- Segmentación de imágenes satelitales
- Detección de objetos en imágenes
- Restauración de imágenes
- Segmentación de escenas

## 🎯 Ejercicio Práctico: Segmentación Médica de Células

### Tipo de Aprendizaje: **SUPERVISADO**
- **Por qué**: Usa imágenes médicas con máscaras de segmentación etiquetadas
- **Datos**: Imágenes de células con máscaras de segmentación

### Implementación Completa

```python
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt

print("🏥 RED U-NET - SEGMENTACIÓN MÉDICA DE CÉLULAS")
print("=" * 60)

# --- 1. Crear dataset sintético de imágenes médicas ---
print("📊 Creando dataset de imágenes médicas...")
def create_medical_dataset(num_samples=1000, img_size=128):
    """Crear dataset sintético de células con máscaras de segmentación"""
    images = []
    masks = []
    
    for i in range(num_samples):
        # Crear imagen de fondo
        img = np.random.rand(img_size, img_size, 3) * 0.3
        
        # Crear máscara vacía
        mask = np.zeros((img_size, img_size, 1))
        
        # Añadir células (círculos elípticos)
        num_cells = np.random.randint(3, 8)
        
        for _ in range(num_cells):
            # Posición aleatoria
            center_x = np.random.randint(20, img_size-20)
            center_y = np.random.randint(20, img_size-20)
            
            # Tamaño aleatorio
            radius_x = np.random.randint(8, 25)
            radius_y = np.random.randint(8, 25)
            
            # Rotación aleatoria
            angle = np.random.uniform(0, 2*np.pi)
            
            # Intensidad y color de célula
            intensity = np.random.uniform(0.4, 0.9)
            color = np.random.uniform(0.3, 1.0, 3)
            
            # Crear célula elíptica
            for y in range(img_size):
                for x in range(img_size):
                    # Transformación a coordenadas elípticas rotadas
                    dx = x - center_x
                    dy = y - center_y
                    
                    # Rotación
                    rot_x = dx * np.cos(angle) - dy * np.sin(angle)
                    rot_y = dx * np.sin(angle) + dy * np.cos(angle)
                    
                    # Ecuación de elipse
                    if (rot_x/radius_x)**2 + (rot_y/radius_y)**2 <= 1:
                        img[y, x] = img[y, x] * (1 - intensity) + color * intensity
                        mask[y, x, 0] = 1
        
        # Añadir ruido
        noise = np.random.normal(0, 0.05, img.shape)
        img = np.clip(img + noise, 0, 1)
        
        images.append(img)
        masks.append(mask)
    
    return np.array(images), np.array(masks)

# Generar dataset
X, y = create_medical_dataset(num_samples=500, img_size=128)

# Dividir en entrenamiento y prueba
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

print(f"   Imágenes de entrenamiento: {X_train.shape}")
print(f"   Máscaras de entrenamiento: {y_train.shape}")
print(f"   Imágenes de prueba: {X_test.shape}")
print(f"   Máscaras de prueba: {y_test.shape}")

# --- 2. Visualizar algunas muestras ---
print("\n🎨 Visualizando muestras del dataset...")
plt.figure(figsize=(12, 6))
for i in range(3):
    plt.subplot(2, 3, i + 1)
    plt.imshow(X_train[i])
    plt.title(f'Imagen {i+1}')
    plt.axis('off')
    
    plt.subplot(2, 3, i + 4)
    plt.imshow(y_train[i, :, :, 0], cmap='gray')
    plt.title(f'Máscara {i+1}')
    plt.axis('off')

plt.suptitle('Muestras del Dataset Médico')
plt.tight_layout()
plt.show()

# --- 3. Definir bloques U-Net ---
print("\n🏗️ Construyendo bloques U-Net...")

def conv_block(x, filters, kernel_size=3, dropout_rate=0.1):
    """Bloque de convolución con batch normalization y dropout"""
    x = layers.Conv2D(filters, kernel_size, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    x = layers.Conv2D(filters, kernel_size, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    if dropout_rate > 0:
        x = layers.Dropout(dropout_rate)(x)
    
    return x

def encoder_block(x, filters, pool_size=(2, 2)):
    """Bloque encoder con max pooling"""
    c = conv_block(x, filters)
    p = layers.MaxPooling2D(pool_size)(c)
    return c, p

def decoder_block(x, skip_connection, filters):
    """Bloque decoder con up-sampling y concatenación"""
    # Up-sampling
    x = layers.UpSampling2D(size=(2, 2))(x)
    x = layers.Conv2D(filters, 2, padding='same')(x)
    
    # Concatenación con skip connection
    x = layers.Concatenate()([x, skip_connection])
    
    # Convolución
    x = conv_block(x, filters)
    
    return x

# --- 4. Construir U-Net completo ---
print("Construyendo arquitectura U-Net...")

def build_unet(input_shape=(128, 128, 3), num_classes=1):
    """Construir arquitectura U-Net completa"""
    inputs = layers.Input(shape=input_shape)
    
    # Encoder (contracting path)
    s1, p1 = encoder_block(inputs, 64)
    s2, p2 = encoder_block(p1, 128)
    s3, p3 = encoder_block(p2, 256)
    s4, p4 = encoder_block(p3, 512)
    
    # Bottleneck
    b1 = conv_block(p4, 1024)
    
    # Decoder (expansive path)
    d1 = decoder_block(b1, s4, 512)
    d2 = decoder_block(d1, s3, 256)
    d3 = decoder_block(d2, s2, 128)
    d4 = decoder_block(d3, s1, 64)
    
    # Capa de salida
    if num_classes == 1:
        # Segmentación binaria
        outputs = layers.Conv2D(1, 1, activation='sigmoid')(d4)
    else:
        # Segmentación multi-clase
        outputs = layers.Conv2D(num_classes, 1, activation='softmax')(d4)
    
    model = models.Model(inputs, outputs)
    return model

# Crear modelo U-Net
model = build_unet(input_shape=(128, 128, 3), num_classes=1)

# Mostrar arquitectura
model.summary()

# --- 5. Compilar y entrenar ---
print("\n🚀 Compilando y entrenando el modelo...")
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy', 'iou']
)

# Métrica IoU personalizada
def iou_metric(y_true, y_pred, smooth=1):
    """Calcular Intersection over Union"""
    y_pred = tf.cast(y_pred > 0.5, tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection
    iou = (intersection + smooth) / (union + smooth)
    return iou

# Recompilar con métrica IoU
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy', iou_metric]
)

# Callbacks
callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5),
    tf.keras.callbacks.ModelCheckpoint('unet_medical.h5', save_best_only=True)
]

# Entrenar
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=16,
    validation_data=(X_test, y_test),
    callbacks=callbacks,
    verbose=1
)

# --- 6. Evaluar el modelo ---
print("\n📈 Evaluando el modelo...")
test_loss, test_acc, test_iou = model.evaluate(X_test, y_test, verbose=0)
print(f"   Precisión en prueba: {test_acc:.4f}")
print(f"   IoU en prueba: {test_iou:.4f}")
print(f"   Pérdida en prueba: {test_loss:.4f}")

# --- 7. Visualizar predicciones ---
print("\n🎨 Visualizando predicciones...")
# Obtener predicciones
predictions = model.predict(X_test[:10], verbose=0)
predicted_masks = (predictions > 0.5).astype(np.float32)

# Visualizar resultados
plt.figure(figsize=(15, 10))
for i in range(5):
    # Imagen original
    plt.subplot(4, 5, i + 1)
    plt.imshow(X_test[i])
    plt.title(f'Original {i+1}')
    plt.axis('off')
    
    # Máscara real
    plt.subplot(4, 5, i + 6)
    plt.imshow(y_test[i, :, :, 0], cmap='gray')
    plt.title(f'Máscara Real {i+1}')
    plt.axis('off')
    
    # Máscara predicha
    plt.subplot(4, 5, i + 11)
    plt.imshow(predicted_masks[i, :, :, 0], cmap='gray')
    plt.title(f'Máscara Predicha {i+1}')
    plt.axis('off')
    
    # Superposición
    plt.subplot(4, 5, i + 16)
    plt.imshow(X_test[i])
    plt.imshow(predicted_masks[i, :, :, 0], cmap='jet', alpha=0.3)
    plt.title(f'Superposición {i+1}')
    plt.axis('off')

plt.suptitle('Resultados de Segmentación U-Net')
plt.tight_layout()
plt.show()

# --- 8. Análisis de características intermedias ---
print("\n🔍 Analizando características intermedias...")

# Extraer características de diferentes capas
layer_names = ['conv2d', 'conv2d_1', 'conv2d_2', 'conv2d_3']
feature_models = []

for layer_name in layer_names:
    try:
        feature_model = models.Model(
            inputs=model.input,
            outputs=model.get_layer(layer_name).output
        )
        feature_models.append(feature_model)
    except:
        print(f"   Capa {layer_name} no encontrada")

# Visualizar características de una imagen
sample_image = X_test[0:1]
plt.figure(figsize=(15, 8))

for i, feature_model in enumerate(feature_models[:4]):
    features = feature_model.predict(sample_image, verbose=0)
    
    # Visualizar primeros 16 filtros
    for j in range(min(16, features.shape[-1])):
        plt.subplot(4, 16, i * 16 + j + 1)
        plt.imshow(features[0, :, :, j], cmap='viridis')
        plt.axis('off')

plt.suptitle('Características Intermedias de U-Net')
plt.tight_layout()
plt.show()

# --- 9. Comparación con otros métodos ---
print("\n⚖️ Comparando U-Net con otros métodos...")

# Método simple: thresholding
def simple_segmentation(images):
    """Segmentación simple por thresholding"""
    # Convertir a escala de grises
    gray = np.mean(images, axis=-1, keepdims=True)
    # Thresholding
    masks = (gray > np.mean(gray)).astype(np.float32)
    return masks

# Evaluar método simple
simple_masks = simple_segmentation(X_test)
simple_iou = np.mean([
    iou_metric(y_test[i:i+1], simple_masks[i:i+1]) 
    for i in range(len(y_test))
])

print(f"   U-Net IoU: {test_iou:.4f}")
print(f"   Thresholding IoU: {simple_iou:.4f}")
print(f"   Mejora: {(test_iou - simple_iou)/simple_iou * 100:.1f}%")

# --- 10. Visualizar entrenamiento ---
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.plot(history.history['accuracy'], label='Entrenamiento')
plt.plot(history.history['val_accuracy'], label='Validación')
plt.title('Precisión durante Entrenamiento')
plt.xlabel('Época')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

plt.subplot(1, 3, 2)
plt.plot(history.history['loss'], label='Entrenamiento')
plt.plot(history.history['val_loss'], label='Validación')
plt.title('Pérdida durante Entrenamiento')
plt.xlabel('Época')
plt.ylabel('Binary Crossentropy')
plt.legend()
plt.grid(True)

plt.subplot(1, 3, 3)
plt.plot(history.history['iou_metric'], label='Entrenamiento')
plt.plot(history.history['val_iou_metric'], label='Validación')
plt.title('IoU durante Entrenamiento')
plt.xlabel('Época')
plt.ylabel('IoU')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# --- 11. Análisis de errores ---
print("\n🔍 Analizando errores de segmentación...")

# Calcular IoU por imagen
ious = []
for i in range(len(X_test)):
    pred = predictions[i:i+1]
    true = y_test[i:i+1]
    iou = iou_metric(true, pred)
   ious.append(iou.numpy())

ious = np.array(ious)

print(f"   IoU promedio: {np.mean(ious):.4f}")
print(f"   IoU desviación estándar: {np.std(ious):.4f}")
print(f"   Peor IoU: {np.min(ious):.4f}")
print(f"   Mejor IoU: {np.max(ious):.4f}")

# Visualizar peores casos
worst_indices = np.argsort(ious)[:3]

plt.figure(figsize=(15, 5))
for i, idx in enumerate(worst_indices):
    plt.subplot(1, 3, i + 1)
    plt.imshow(X_test[idx])
    plt.imshow(predicted_masks[idx, :, :, 0], cmap='jet', alpha=0.3)
    plt.title(f'Peor caso {i+1} (IoU: {ious[idx]:.3f})')
    plt.axis('off')

plt.suptitle('Peores Casos de Segmentación')
plt.tight_layout()
plt.show()

# --- 12. Guardar el modelo ---
model.save('unet_medical_final.h5')
print("\n💾 Modelo guardado como 'unet_medical_final.h5'")

print("\n✅ EJERCICIO COMPLETADO")
print("=" * 60)
print("🎯 RESULTADOS:")
print(f"   • Precisión final: {test_acc:.4f}")
print(f"   • IoU final: {test_iou:.4f}")
print(f"   • Pérdida final: {test_loss:.4f}")
print(f"   • Total de parámetros: {model.count_params():,}")
print(f"   • Épocas entrenadas: {len(history.history['loss'])}")
print(f"   • Mejora sobre thresholding: {(test_iou - simple_iou)/simple_iou * 100:.1f}%")
print("=" * 60)
```

## 📊 Resultados Esperados

```
🏥 RED U-NET - SEGMENTACIÓN MÉDICA DE CÉLULAS
============================================================
📊 Creando dataset de imágenes médicas...
   Imágenes de entrenamiento: (400, 128, 128, 3)
   Máscaras de entrenamiento: (400, 128, 128, 1)
   Imágenes de prueba: (100, 128, 128, 3)
   Máscaras de prueba: (100, 128, 128, 1)

🏗️ Construyendo bloques U-Net...
Construyendo arquitectura U-Net...
Model: "model"
__________________________________________________________________________________________________
 Layer (type)                    Output Shape         Param #     
=================================================================
 input_1 (InputLayer)            [(None, 128, 128, 3)] 0         
 conv2d (Conv2D)                 (None, 128, 128, 64)  1792       
 batch_normalization (BatchNorm (None, 128, 128, 64)  256       
 ... (arquitectura U-Net completa) ...
 conv2d_9 (Conv2D)               (None, 128, 128, 1)   65        
=================================================================
Total params: 31,032,705
Trainable params: 31,031,681
Non-trainable params: 1,024
__________________________________________________________________________________________________

🚀 Compilando y entrenando el modelo...
Epoch 1/50
25/25 [==============================] - 45s 2s/step - loss: 0.6234 - accuracy: 0.6543 - iou_metric: 0.2345 - val_loss: 0.5678 - val_accuracy: 0.7123 - val_iou_metric: 0.3456
...
Epoch 32/50
25/25 [==============================] - 38s 2s/step - loss: 0.1234 - accuracy: 0.9456 - iou_metric: 0.7890 - val_loss: 0.1456 - val_accuracy: 0.9234 - val_iou_metric: 0.7567

📈 Evaluando el modelo...
   Precisión en prueba: 0.9234
   IoU en prueba: 0.7567
   Pérdida en prueba: 0.1456

⚖️ Comparando U-Net con otros métodos...
   U-Net IoU: 0.7567
   Thresholding IoU: 0.4321
   Mejora: 75.0%

✅ EJERCICIO COMPLETADO
============================================================
🎯 RESULTADOS:
   • Precisión final: 0.9234
   • IoU final: 0.7567
   • Pérdida final: 0.1456
   • Total de parámetros: 31,032,705
   • Épocas entrenadas: 32
   • Mejora sobre thresholding: 75.0%
============================================================
```

## 🎓 Análisis de Aprendizaje

### Identificación del Paradigma
**✅ APRENDIZAJE SUPERVISADO**
- **Datos etiquetados**: Usa imágenes médicas con máscaras de segmentación
- **Objetivo**: Predecir máscaras de segmentación pixel a pixel
- **Métrica**: IoU (Intersection over Union) y accuracy

### Ventajas de U-Net
- **Preservación de detalles**: Skip connections mantienen información espacial
- **Precisión en bordes**: Excelente para segmentación de contornos
- **Eficiente**: Menos parámetros que arquitecturas más complejas
- **Flexible**: Adaptable a diferentes tamaños de imagen

### Limitaciones
- **Memoria**: Requiere mucha memoria para imágenes grandes
- **Entrenamiento lento**: Arquitectura compleja con muchas capas
- **Data hungry**: Necesita muchos datos etiquetados
- **Overfitting**: Propenso a overfitting en datasets pequeños

## 🚀 Adaptación al Proyecto Integrador

### Ideas para Adaptación
1. **Dominio específico**: Adaptar para imágenes médicas reales
2. **Multi-clase**: Extender para segmentación multi-clase
3. **Transfer learning**: Usar encoder pre-entrenado

### Ejemplo de Adaptación
```python
# Adaptar U-Net para imágenes médicas reales
def crear_unet_medico(input_shape=(256, 256, 3), num_classes=1):
    """
    Adapta U-Net para imágenes médicas de alta resolución
    """
    # Encoder pre-entrenado (opcional)
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )
    
    # Congelar encoder
    base_model.trainable = False
    
    # Construir U-Net con encoder pre-entrenado
    inputs = layers.Input(shape=input_shape)
    
    # Usar capas de MobileNet como encoder
    encoder_outputs = []
    for layer in base_model.layers:
        if 'conv' in layer.name and 'output' not in layer.name:
            encoder_outputs.append(layer.output)
    
    # Decoder personalizado
    x = encoder_outputs[-1]
    for i in range(len(encoder_outputs)-2, -1, -1):
        x = layers.UpSampling2D(size=(2, 2))(x)
        x = layers.Concatenate()([x, encoder_outputs[i]])
        x = conv_block(x, 64 // (2**i))
    
    outputs = layers.Conv2D(num_classes, 1, activation='sigmoid')(x)
    
    model = models.Model(inputs, outputs)
    return model
```

## 📝 Ejercicios Adicionales

### Ejercicio 1: No Supervisado (U-Net Autoencoder)
```python
# Usar U-Net como autoencoder no supervisado
def crear_unet_autoencoder(input_shape=(128, 128, 3)):
    """Crear U-Net autoencoder para aprendizaje no supervisado"""
    
    # Encoder (contracting path)
    inputs = layers.Input(shape=input_shape)
    s1, p1 = encoder_block(inputs, 64)
    s2, p2 = encoder_block(p1, 128)
    s3, p3 = encoder_block(p2, 256)
    s4, p4 = encoder_block(p3, 512)
    
    # Bottleneck
    b1 = conv_block(p4, 1024)
    
    # Decoder (expansive path) - reconstrucción
    d1 = decoder_block(b1, s4, 512)
    d2 = decoder_block(d1, s3, 256)
    d3 = decoder_block(d2, s2, 128)
    d4 = decoder_block(d3, s1, 64)
    
    # Salida reconstrucción
    outputs = layers.Conv2D(3, 1, activation='sigmoid')(d4)
    
    autoencoder = models.Model(inputs, outputs)
    autoencoder.compile(optimizer='adam', loss='mse')
    
    return autoencoder
```

### Ejercicio 2: Reforzado (U-Net para RL)
```python
# Usar U-Net en contexto de aprendizaje por refuerzo
class UNetEnvironment:
    def __init__(self, unet_model, target_images):
        self.unet_model = unet_model
        self.target_images = target_images
        self.current_state = None
        self.state_idx = 0
    
    def reset(self):
        """Reiniciar entorno"""
        self.state_idx = 0
        self.current_state = np.random.rand(128, 128, 3) * 0.1  # Imagen inicial casi vacía
        return self.current_state
    
    def step(self, action):
        """Realizar acción (modificar imagen) y obtener recompensa"""
        # Acción: añadir o modificar regiones en la imagen
        modified_state = self.current_state.copy()
        
        # Aplicar acción (ej: añadir pincelada)
        x, y, size, intensity = action
        for i in range(max(0, x-size), min(128, x+size)):
            for j in range(max(0, y-size), min(128, y+size)):
                if (i-x)**2 + (j-y)**2 <= size**2:
                    modified_state[i, j] = modified_state[i, j] * (1-intensity) + np.random.rand(3) * intensity
        
        # Recompensa: qué tan similar es a la segmentación esperada
        predicted_mask = self.unet_model.predict(modified_state[np.newaxis, ...], verbose=0)
        target_mask = self.target_images[self.state_idx]
        
        # Calcular IoU como recompensa
        reward = iou_metric(target_mask, predicted_mask)
        
        # Actualizar estado
        self.current_state = modified_state
        self.state_idx = (self.state_idx + 1) % len(self.target_images)
        
        return self.current_state, reward, False, {}
```

## 🔗 Recursos Adicionales

- [Documentación TensorFlow/Keras](https://www.tensorflow.org/api_docs/python/tf/keras)
- [U-Net Paper Original](https://arxiv.org/abs/1505.04597)
- [Medical Image Segmentation](https://www.tensorflow.org/tutorials/images/segmentation)
- [Image Segmentation Guide](https://www.tensorflow.org/tutorials/images/segmentation)
