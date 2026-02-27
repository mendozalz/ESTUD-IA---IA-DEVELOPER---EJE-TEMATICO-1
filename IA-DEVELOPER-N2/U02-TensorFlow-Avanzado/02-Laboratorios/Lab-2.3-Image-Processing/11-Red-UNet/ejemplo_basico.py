import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt

print("RED U-NET - SEGMENTACION MEDICA DE CELULAS")
print("=" * 60)

# --- 1. Crear dataset sintético de imágenes médicas ---
print("Creando dataset de imágenes médicas...")
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
print("\nVisualizando muestras del dataset...")
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

plt.suptitle('Muestras del Dataset Medico')
plt.tight_layout()
plt.show()

# --- 3. Definir bloques U-Net ---
print("\nConstruyendo bloques U-Net...")

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
print("\nCompilando y entrenando el modelo...")

# Métrica IoU personalizada
def iou_metric(y_true, y_pred, smooth=1):
    """Calcular Intersection over Union"""
    y_pred = tf.cast(y_pred > 0.5, tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection
    iou = (intersection + smooth) / (union + smooth)
    return iou

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
    epochs=30,  # Reducido para demostracion
    batch_size=16,
    validation_data=(X_test, y_test),
    callbacks=callbacks,
    verbose=1
)

# --- 6. Evaluar el modelo ---
print("\nEvaluando el modelo...")
test_loss, test_acc, test_iou = model.evaluate(X_test, y_test, verbose=0)
print(f"   Precisión en prueba: {test_acc:.4f}")
print(f"   IoU en prueba: {test_iou:.4f}")
print(f"   Pérdida en prueba: {test_loss:.4f}")

# --- 7. Visualizar predicciones ---
print("\nVisualizando predicciones...")
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

# --- 8. Comparación con otros métodos ---
print("\nComparando U-Net con otros métodos...")

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

# --- 9. Visualizar entrenamiento ---
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.plot(history.history['accuracy'], label='Entrenamiento')
plt.plot(history.history['val_accuracy'], label='Validación')
plt.title('Precision durante Entrenamiento')
plt.xlabel('Epoca')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

plt.subplot(1, 3, 2)
plt.plot(history.history['loss'], label='Entrenamiento')
plt.plot(history.history['val_loss'], label='Validación')
plt.title('Perdida durante Entrenamiento')
plt.xlabel('Epoca')
plt.ylabel('Binary Crossentropy')
plt.legend()
plt.grid(True)

plt.subplot(1, 3, 3)
plt.plot(history.history['iou_metric'], label='Entrenamiento')
plt.plot(history.history['val_iou_metric'], label='Validación')
plt.title('IoU durante Entrenamiento')
plt.xlabel('Epoca')
plt.ylabel('IoU')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# --- 10. Guardar el modelo ---
model.save('unet_medical_final.h5')
print("\nModelo guardado como 'unet_medical_final.h5'")

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
