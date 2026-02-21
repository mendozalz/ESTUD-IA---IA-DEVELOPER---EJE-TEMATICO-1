import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt
import time

print("RED GRU - ANALISIS DE SENTIMIENTOS EN TIEMPO REAL")
print("=" * 60)

# --- 1. Crear dataset de ejemplo ---
print("Creando dataset de analisis de sentimientos...")
# Dataset de reseñas de películas con sentimientos
reviews = [
    "Esta película es increíblemente buena",
    "No me gustó nada, fue terrible",
    "Actuación mediocre, guion débil",
    "Me encantó, muy recomendada",
    "Aburrida y predecible",
    "Dirección excelente, photography increíble",
    "Perdí el tiempo, no la vean",
    "Obra maestra del cine moderno",
    "Regular, podría ser mejor",
    "Sorprendente e innovadora",
    "Los efectos especiales son impresionantes",
    "Historia confusa y mal actuada",
    "Comedia tierna y conmovedora",
    "Terror predecible y sin sustos",
    "Drama profundo y emocionante",
    "Acción emocionante y bien coreografiada",
    "Documental informativo y bien hecho",
    "Animación infantil divertida",
    "Thriller tenso y misterioso",
    "Romance cliché y aburrido"
]

sentiments = [1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0]  # 1=positivo, 0=negativo

# --- 2. Preprocesamiento de texto ---
print("Preprocesando texto...")
# Tokenización
tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=1000, oov_token="<OOV>")
tokenizer.fit_on_texts(reviews)

# Convertir a secuencias
sequences = tokenizer.texts_to_sequences(reviews)
padded_sequences = tf.keras.preprocessing.sequence.pad_sequences(
    sequences, maxlen=15, padding='post', truncating='post'
)

# Convertir a arrays
X = np.array(padded_sequences)
y = np.array(sentiments)

print(f"   Vocabulario: {len(tokenizer.word_index)} palabras")
print(f"   Secuencias procesadas: {X.shape}")
print(f"   Sentimientos positivos: {np.sum(y)}")
print(f"   Sentimientos negativos: {np.sum(len(y)-y)}")

# --- 3. Definir el modelo GRU ---
print("\nConstruyendo el modelo GRU...")
model = models.Sequential([
    layers.Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=64, input_length=15),
    layers.GRU(64, return_sequences=True, dropout=0.2, recurrent_dropout=0.2),
    layers.GRU(32, dropout=0.2, recurrent_dropout=0.2),
    layers.Dense(16, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')
])

# Mostrar arquitectura
model.summary()

# --- 4. Compilar y entrenar ---
print("\nCompilando y entrenando el modelo...")
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Callbacks
callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5)
]

history = model.fit(
    X, y,
    epochs=50,
    batch_size=4,
    validation_split=0.3,
    callbacks=callbacks,
    verbose=1
)

# --- 5. Evaluar el modelo ---
print("\nEvaluando el modelo...")
loss, accuracy = model.evaluate(X, y, verbose=0)
print(f"   Precisión final: {accuracy:.4f}")
print(f"   Pérdida final: {loss:.4f}")

# --- 6. Función de predicción en tiempo real ---
def predict_sentiment_gru(text, tokenizer, model, maxlen=15):
    """Predecir sentimiento usando GRU"""
    # Preprocesar texto
    sequence = tokenizer.texts_to_sequences([text])
    padded = tf.keras.preprocessing.sequence.pad_sequences(
        sequence, maxlen=maxlen, padding='post', truncating='post'
    )
    
    # Predecir
    prediction = model.predict(padded, verbose=0)[0][0]
    sentiment = "Positivo" if prediction > 0.5 else "Negativo"
    confidence = prediction if prediction > 0.5 else 1 - prediction
    
    return sentiment, confidence

# --- 7. Evaluar con ejemplos ---
print("\nEvaluando predicciones...")
test_reviews = [
    "Esta película es fantástica",
    "No me gustó para nada",
    "Actuación regular pero dirección buena",
    "Experiencia cinematográfica increíble",
    "Guion confuso y actuación pobre"
]

for review in test_reviews:
    sentiment, confidence = predict_sentiment_gru(review, tokenizer, model)
    print(f"   Review: '{review}'")
    print(f"   Sentimiento: {sentiment} (confianza: {confidence:.3f})")
    print()

# --- 8. Análisis de atención en GRU ---
print("Analizando comportamiento del GRU...")

# Extraer salidas intermedias para análisis
intermediate_model = models.Model(
    inputs=model.input,
    outputs=[layer.output for layer in model.layers if 'gru' in layer.name]
)

# Obtener salidas de capas GRU
gru_outputs = intermediate_model.predict(X[:5], verbose=0)

# Visualizar activaciones de GRU
plt.figure(figsize=(15, 5))

for i, (layer_name, output) in enumerate(zip(['GRU_1', 'GRU_2'], gru_outputs)):
    plt.subplot(1, 2, i + 1)
    
    # Promediar activaciones por timestep
    avg_activations = np.mean(output, axis=-1)
    
    for seq_idx in range(min(3, output.shape[0])):
        plt.plot(avg_activations[seq_idx], label=f'Secuencia {seq_idx+1}', alpha=0.7)
    
    plt.title(f'Activaciones {layer_name}')
    plt.xlabel('Timestep')
    plt.ylabel('Activacion Promedio')
    plt.legend()
    plt.grid(True)

plt.suptitle('Analisis de Activaciones de Capas GRU')
plt.tight_layout()
plt.show()

# --- 9. Comparación con LSTM ---
print("\nComparando GRU vs LSTM...")
# Crear modelo LSTM equivalente para comparación
lstm_model = models.Sequential([
    layers.Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=64, input_length=15),
    layers.LSTM(64, return_sequences=True, dropout=0.2, recurrent_dropout=0.2),
    layers.LSTM(32, dropout=0.2, recurrent_dropout=0.2),
    layers.Dense(16, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')
])

lstm_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Entrenar LSTM por menos épocas para comparación
print("   Entrenando modelo LSTM equivalente...")
lstm_history = lstm_model.fit(
    X, y,
    epochs=20,
    batch_size=4,
    validation_split=0.3,
    verbose=0
)

# Comparar rendimiento
lstm_loss, lstm_accuracy = lstm_model.evaluate(X, y, verbose=0)

print(f"   GRU - Precisión: {accuracy:.4f}, Parámetros: {model.count_params():,}")
print(f"   LSTM - Precisión: {lstm_accuracy:.4f}, Parámetros: {lstm_model.count_params():,}")

# --- 10. Visualizar entrenamiento ---
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='GRU Entrenamiento')
plt.plot(history.history['val_accuracy'], label='GRU Validación')
plt.title('Precision GRU durante Entrenamiento')
plt.xlabel('Epoca')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='GRU Entrenamiento')
plt.plot(history.history['val_loss'], label='GRU Validación')
plt.title('Perdida GRU durante Entrenamiento')
plt.xlabel('Epoca')
plt.ylabel('Binary Crossentropy')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# --- 11. Análisis de eficiencia ---
print("\nAnalizando eficiencia computacional...")

# Medir tiempo de inferencia GRU
start_time = time.time()
for _ in range(100):
    model.predict(X[:1], verbose=0)
gru_time = time.time() - start_time

# Medir tiempo de inferencia LSTM
start_time = time.time()
for _ in range(100):
    lstm_model.predict(X[:1], verbose=0)
lstm_time = time.time() - start_time

print(f"   Tiempo inferencia GRU (100 predicciones): {gru_time:.3f}s")
print(f"   Tiempo inferencia LSTM (100 predicciones): {lstm_time:.3f}s")
print(f"   GRU es {lstm_time/gru_time:.2f}x más rápido que LSTM")

# --- 12. Guardar el modelo ---
model.save('gru_sentiment.h5')
print("\nModelo guardado como 'gru_sentiment.h5'")

print("\n✅ EJERCICIO COMPLETADO")
print("=" * 60)
print("🎯 RESULTADOS:")
print(f"   • Precisión final: {accuracy:.4f}")
print(f"   • Pérdida final: {loss:.4f}")
print(f"   • Total de parámetros: {model.count_params():,}")
print(f"   • Épocas entrenadas: {len(history.history['loss'])}")
print(f"   • Ventaja de velocidad vs LSTM: {lstm_time/gru_time:.2f}x")
print("=" * 60)
