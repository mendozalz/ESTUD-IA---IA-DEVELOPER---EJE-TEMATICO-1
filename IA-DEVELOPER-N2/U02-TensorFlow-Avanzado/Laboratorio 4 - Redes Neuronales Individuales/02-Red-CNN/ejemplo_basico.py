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
