import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt

print("RED AUTOENCODER - COMPRESION DE IMAGENES")
print("=" * 60)

# --- 1. Cargar y preprocesar datos ---
print("Cargando y preprocesando datos...")
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
print("\nConstruyendo el Autoencoder...")

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
print("\nCompilando y entrenando el Autoencoder...")
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
print("\nEvaluando el Autoencoder...")
test_loss, test_mae = autoencoder.evaluate(x_test, x_test, verbose=0)
print(f"   Error cuadrático medio en prueba: {test_loss:.6f}")
print(f"   Error absoluto medio en prueba: {test_mae:.6f}")

# --- 5. Visualizar reconstrucciones ---
print("\nVisualizando reconstrucciones...")
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

plt.suptitle('Imagenes Originales vs Reconstruidas')
plt.tight_layout()
plt.show()

# --- 6. Analizar espacio latente ---
print("\nAnalizando espacio latente...")
# Obtener representaciones latentes
latent_representations = encoder.predict(x_test, verbose=0)

print(f"   Dimensionalidad original: {x_test.shape[1:]}")
print(f"   Dimensionalidad latente: {latent_representations.shape[1]}")
print(f"   Factor de compresión: {latent_representations.shape[1] / (28*28):.4f}")

# Visualizar distribución del espacio latente
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.hist(latent_representations.flatten(), bins=50, alpha=0.7, density=True)
plt.title('Distribucion del Espacio Latente')
plt.xlabel('Valor')
plt.ylabel('Densidad')
plt.grid(True)

# Visualizar correlaciones entre dimensiones latentes
plt.subplot(1, 3, 2)
correlation_matrix = np.corrcoef(latent_representations.T)
plt.imshow(correlation_matrix, cmap='coolwarm', aspect='auto')
plt.title('Matriz de Correlacion Latente')
plt.colorbar()
plt.xlabel('Dimension Latente')
plt.ylabel('Dimension Latente')

# Visualizar primeras 2 dimensiones latentes
plt.subplot(1, 3, 3)
plt.scatter(latent_representations[:, 0], latent_representations[:, 1], 
           alpha=0.5, s=1)
plt.title('Primeras 2 Dimensiones Latentes')
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.grid(True)

plt.tight_layout()
plt.show()

# --- 7. Generar nuevas imágenes ---
print("\nGenerando imagenes desde el espacio latente...")

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

plt.suptitle('Imagenes Generadas desde Espacio Latente')
plt.tight_layout()
plt.show()

# --- 8. Interpolación en espacio latente ---
print("\nRealizando interpolacion en espacio latente...")

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

plt.suptitle('Interpolacion en Espacio Latente')
plt.tight_layout()
plt.show()

# --- 9. Detección de anomalías ---
print("\nEvaluando deteccion de anomalias...")

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
plt.hist(errors_noisy, bins=50, alpha=0.7, label='Anomalo', density=True)
plt.title('Distribucion de Errores de Reconstruccion')
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

plt.suptitle('Deteccion de Anomalias con Autoencoder')
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
print("\nModelos guardados:")
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
