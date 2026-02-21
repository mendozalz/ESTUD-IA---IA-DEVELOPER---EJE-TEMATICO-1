import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt

print("RED TRANSFORMERS - CLASIFICACION DE TEXTO")
print("=" * 60)

# --- 1. Crear dataset de ejemplo ---
print("Creando dataset de clasificacion de texto...")
# Dataset simplificado de reseñas de películas
texts = [
    "Esta película es excelente, me encantó",
    "No me gustó, fue muy aburrida",
    "El acting fue increíble, muy recomendada",
    "Perdí mi tiempo, no la vean",
    "Me sorprendió positivamente, gran historia",
    "El guion es débil, no tiene sentido",
    "Los efectos visuales son asombrosos",
    "La dirección es mediocre, decepcionante",
    "Una obra maestra del cine moderno",
    "No vale la pena, muy predecible",
    "Actuaciones memorables y emotivas",
    "El ritmo es lento, me dormí",
    "Innovadora y original, muy creativa",
    "Cliché y repetitiva, nada nuevo",
    "Recomendada para toda la familia"
]

labels = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]  # 1=positivo, 0=negativo

# --- 2. Preprocesamiento de texto ---
print("Preprocesando texto...")
# Tokenización simple
tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=1000, oov_token="<OOV>")
tokenizer.fit_on_texts(texts)

# Convertir texto a secuencias
sequences = tokenizer.texts_to_sequences(texts)
padded_sequences = tf.keras.preprocessing.sequence.pad_sequences(
    sequences, maxlen=20, padding='post', truncating='post'
)

# Convertir a numpy arrays
X = np.array(padded_sequences)
y = np.array(labels)

print(f"   Vocabulario: {len(tokenizer.word_index)} palabras")
print(f"   Secuencias procesadas: {X.shape}")
print(f"   Etiquetas: {np.bincount(y)}")

# --- 3. Definir el modelo Transformer ---
print("\nConstruyendo el modelo Transformer...")

class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = models.Sequential([
            layers.Dense(ff_dim, activation="relu"),
            layers.Dense(embed_dim),
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super(TokenAndPositionEmbedding, self).__init__()
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions

# Parámetros del modelo
embed_dim = 32
num_heads = 2
ff_dim = 32
maxlen = 20
vocab_size = len(tokenizer.word_index) + 1

# Construir modelo
inputs = layers.Input(shape=(maxlen,))
embedding_layer = TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim)
x = embedding_layer(inputs)
transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
x = transformer_block(x)
x = layers.GlobalAveragePooling1D()(x)
x = layers.Dropout(0.1)(x)
x = layers.Dense(20, activation="relu")(x)
x = layers.Dropout(0.1)(x)
outputs = layers.Dense(1, activation="sigmoid")(x)

model = models.Model(inputs=inputs, outputs=outputs)

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

# --- 6. Función de predicción ---
def predict_sentiment(text, tokenizer, model, maxlen=20):
    """Predecir sentimiento de un texto"""
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
test_texts = [
    "Esta película es fantástica",
    "No me gustó para nada",
    "Actuación regular, podría ser mejor",
    "Experiencia cinematográfica increíble"
]

for text in test_texts:
    sentiment, confidence = predict_sentiment(text, tokenizer, model)
    print(f"   Texto: '{text}'")
    print(f"   Sentimiento: {sentiment} (confianza: {confidence:.3f})")
    print()

# --- 8. Visualizar atención (conceptual) ---
print("Visualizando pesos de atención (conceptual)...")

# Extraer pesos de atención de la primera capa
attention_weights = None
for layer in model.layers:
    if hasattr(layer, 'att'):
        # Simular pesos de atención para visualización
        attention_weights = np.random.rand(4, maxlen, maxlen)  # 4 heads
        break

if attention_weights is not None:
    plt.figure(figsize=(15, 5))
    
    for i in range(min(4, attention_weights.shape[0])):
        plt.subplot(1, 4, i + 1)
        plt.imshow(attention_weights[i], cmap='Blues', aspect='auto')
        plt.title(f'Head {i+1} Attention')
        plt.xlabel('Key Position')
        plt.ylabel('Query Position')
        plt.colorbar()
    
    plt.suptitle('Pesos de Atención Multi-Head (Conceptual)')
    plt.tight_layout()
    plt.show()

# --- 9. Visualizar entrenamiento ---
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Entrenamiento')
plt.plot(history.history['val_accuracy'], label='Validación')
plt.title('Precisión durante Entrenamiento')
plt.xlabel('Época')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Entrenamiento')
plt.plot(history.history['val_loss'], label='Validación')
plt.title('Pérdida durante Entrenamiento')
plt.xlabel('Época')
plt.ylabel('Binary Crossentropy')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# --- 10. Guardar el modelo ---
model.save('transformer_sentiment.h5')
print("\nModelo guardado como 'transformer_sentiment.h5'")

print("\n✅ EJERCICIO COMPLETADO")
print("=" * 60)
print("🎯 RESULTADOS:")
print(f"   • Precisión final: {accuracy:.4f}")
print(f"   • Pérdida final: {loss:.4f}")
print(f"   • Total de parámetros: {model.count_params():,}")
print(f"   • Épocas entrenadas: {len(history.history['loss'])}")
print("=" * 60)
