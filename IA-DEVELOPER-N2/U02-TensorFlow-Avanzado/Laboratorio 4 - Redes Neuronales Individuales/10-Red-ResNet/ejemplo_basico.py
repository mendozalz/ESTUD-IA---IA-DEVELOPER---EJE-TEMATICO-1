import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt

print("RED RESNET - CLASIFICACION CIFAR-100")
print("=" * 60)

# --- 1. Cargar y preprocesar datos ---
print("Cargando y preprocesando datos...")
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
print("\nConstruyendo bloques residuales...")

def residual_block(x, filters, kernel_size=3, stride=1):
    """Bloque residual basico"""
    # Camino principal
    shortcut = x
    
    # Primera convolucion
    x = layers.Conv2D(filters, kernel_size, strides=stride, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    # Segunda convolucion
    x = layers.Conv2D(filters, kernel_size, padding='same')(x)
    x = layers.BatchNormalization()(x)
    
    # Ajustar shortcut si es necesario
    if stride != 1 or shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, 1, strides=stride, padding='same')(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)
    
    # Conexion residual
    x = layers.Add()([x, shortcut])
    x = layers.Activation('relu')(x)
    
    return x

def bottleneck_block(x, filters, stride=1):
    """Bloque bottleneck para ResNet-50+"""
    shortcut = x
    
    # 1x1 convolucion reduccion
    x = layers.Conv2D(filters // 4, 1, strides=stride, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    # 3x3 convolucion
    x = layers.Conv2D(filters // 4, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    # 1x1 convolucion expansion
    x = layers.Conv2D(filters, 1, padding='same')(x)
    x = layers.BatchNormalization()(x)
    
    # Ajustar shortcut
    if stride != 1 or shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, 1, strides=stride, padding='same')(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)
    
    # Conexion residual
    x = layers.Add()([x, shortcut])
    x = layers.Activation('relu')(x)
    
    return x

# --- 3. Construir ResNet ---
print("Construyendo arquitectura ResNet...")

def build_resnet(input_shape=(32, 32, 3), num_classes=100, depth=18):
    """Construir ResNet segun profundidad"""
    inputs = layers.Input(shape=input_shape)
    
    # Capa inicial
    x = layers.Conv2D(64, 3, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    # Configuracion segun profundidad
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

# Crear modelo ResNet-18 (mas ligero para demostracion)
model = build_resnet(depth=18)

# Mostrar arquitectura
model.summary()

# --- 4. Compilar y entrenar ---
print("\nCompilando y entrenando el modelo...")
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Callbacks
callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5),
    tf.keras.callbacks.ModelCheckpoint('resnet_cifar100.h5', save_best_only=True)
]

# Entrenar con data augmentation (reducir épocas para demostracion)
history = model.fit(
    datagen.flow(x_train, y_train, batch_size=64),
    epochs=30,  # Reducido para demostracion
    validation_data=(x_test, y_test),
    callbacks=callbacks,
    verbose=1
)

# --- 5. Evaluar el modelo ---
print("\nEvaluando el modelo...")
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"   Precisión en prueba: {test_acc:.4f}")
print(f"   Pérdida en prueba: {test_loss:.4f}")

# --- 6. Análisis de características aprendidas ---
print("\nAnalizando características aprendidas...")

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
plt.title('Distribucion de Caracteristicas')
plt.xlabel('Valor')
plt.ylabel('Densidad')
plt.grid(True)

# --- 7. Comparación con red convencional ---
print("\nComparando ResNet vs CNN convencional...")

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
    epochs=15,  # Reducido para comparacion
    validation_data=(x_test, y_test),
    verbose=0
)

# Comparar rendimiento
cnn_loss, cnn_acc = cnn_model.evaluate(x_test, y_test, verbose=0)

print(f"   ResNet-18 - Precisión: {test_acc:.4f}, Parámetros: {model.count_params():,}")
print(f"   CNN Convencional - Precisión: {cnn_acc:.4f}, Parámetros: {cnn_model.count_params():,}")

# --- 8. Visualizar entrenamiento ---
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.plot(history.history['accuracy'], label='ResNet Entrenamiento')
plt.plot(history.history['val_accuracy'], label='ResNet Validación')
plt.title('Precision ResNet durante Entrenamiento')
plt.xlabel('Epoca')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

plt.subplot(1, 3, 2)
plt.plot(history.history['loss'], label='ResNet Entrenamiento')
plt.plot(history.history['val_loss'], label='ResNet Validación')
plt.title('Perdida ResNet durante Entrenamiento')
plt.xlabel('Epoca')
plt.ylabel('Sparse Categorical Crossentropy')
plt.legend()
plt.grid(True)

# --- 9. Análisis de gradientes ---
plt.subplot(1, 3, 3)
# Simular análisis de flujo de gradientes
epochs_trained = len(history.history['loss'])
gradient_flow = np.exp(-np.linspace(0, 3, epochs_trained))  # Simulacion

plt.plot(gradient_flow, label='ResNet (con skip connections)')
plt.plot(np.exp(-np.linspace(0, 5, epochs_trained)), label='CNN (sin skip connections)')
plt.title('Flujo de Gradientes (Simulado)')
plt.xlabel('Epoca')
plt.ylabel('Magnitud de Gradiente')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# --- 10. Visualizar predicciones ---
print("\nVisualizando predicciones...")
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
print("\nAnalizando predicciones incorrectas...")
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
print("\nModelo guardado como 'resnet_cifar100_final.h5'")

print("\n✅ EJERCICIO COMPLETADO")
print("=" * 60)
print("🎯 RESULTADOS:")
print(f"   • Precisión final: {test_acc:.4f}")
print(f"   • Pérdida final: {test_loss:.4f}")
print(f"   • Total de parámetros: {model.count_params():,}")
print(f"   • Épocas entrenadas: {len(history.history['loss'])}")
print(f"   • Ventaja sobre CNN: {(test_acc - cnn_acc)*100:.2f}%")
print("=" * 60)
