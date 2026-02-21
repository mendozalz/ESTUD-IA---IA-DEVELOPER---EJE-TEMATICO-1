import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np

print("RED NEURONAL DENSA - CLASIFICACION DE DIGITOS MNIST")
print("=" * 60)

# --- 1. Cargar y preprocesar datos ---
print("Cargando y preprocesando datos...")
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalizar a [0, 1]
x_train, x_test = x_train / 255.0, x_test / 255.0

print(f"   Datos de entrenamiento: {x_train.shape}")
print(f"   Datos de prueba: {x_test.shape}")
print(f"   Clases: {np.unique(y_train)}")

# --- 2. Definir el modelo MLP ---
print("Construyendo el modelo MLP...")
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
print("Compilando y entrenando el modelo...")
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
print("Evaluando el modelo...")
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
    plt.title(f"Real: {y_train[i]}\nPred: {np.argmax(predictions[i])}")
    plt.axis('off')

plt.tight_layout()
plt.show()

# --- 7. Análisis de errores ---
print("Generando predicciones y visualizaciones...")
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
    plt.suptitle("Analizando predicciones incorrectas...")
    plt.show()

# --- 8. Guardar el modelo ---
model.save('mlp_mnist.h5')
print("\n💾 Modelo guardado como 'mlp_mnist.h5'")

print("✅ EJERCICIO COMPLETADO")
print("=" * 60)
print("🎯 RESULTADOS:")
print(f"   • Precisión final: {test_acc:.4f}")
print(f"   • Total de parámetros: {model.count_params():,}")
print(f"   • Épocas entrenadas: {len(history.history['loss'])}")
print("=" * 60)
