import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt

print("🧠 RED LSTM - TRADUCCIÓN AUTOMÁTICA SIMPLIFICADA")
print("=" * 60)

# --- 1. Crear dataset de ejemplo ---
print("📊 Creando dataset de traducción...")
# Dataset simplificado de inglés a español
english_sentences = [
    "hello world", "how are you", "good morning", "thank you", "see you later",
    "i love you", "this is good", "where are you", "what time is it", "nice to meet you",
    "the weather is nice", "i am hungry", "let me help", "can you help", "i am learning"
]

spanish_translations = [
    "hola mundo", "como estas", "buenos dias", "gracias", "hasta luego",
    "te quiero", "esto es bueno", "donde estas", "que hora es", "gusto en conocerte",
    "el clima esta bueno", "tengo hambre", "dejame ayudar", "puedes ayudar", "estoy aprendiendo"
]

# Crear vocabulario
vocab = set()
for sent in english_sentences + spanish_translations:
    vocab.update(sent.lower().split())

vocab = sorted(list(vocab))
word_to_idx = {word: idx + 1 for idx, word in enumerate(vocab)}  # +1 para padding
idx_to_word = {idx: word for word, idx in word_to_idx.items()}

vocab_size = len(vocab) + 1  # +1 para padding
max_length = 10  # Máxima longitud de oración

print(f"   Vocabulario: {vocab_size} palabras")
print(f"   Oraciones de entrenamiento: {len(english_sentences)}")

# --- 2. Preprocesar datos ---
def encode_sentences(sentences):
    """Codificar oraciones como secuencias de índices"""
    encoded = []
    for sent in sentences:
        words = sent.lower().split()
        encoded.append([word_to_idx.get(word, 0) for word in words[:max_length]])
    return np.array(encoded)

# Codificar oraciones
encoder_input = encode_sentences(english_sentences)
decoder_input = encode_sentences(spanish_translations)
decoder_output = np.zeros((len(spanish_translations), max_length, vocab_size))

# Crear one-hot para decoder output
for i, sent in enumerate(spanish_translations):
    words = sent.lower().split()
    for j, word in enumerate(words[:max_length]):
        if word in word_to_idx:
            decoder_output[i, j, word_to_idx[word]] = 1.0

print(f"   Formato entrada encoder: {encoder_input.shape}")
print(f"   Formato entrada decoder: {decoder_input.shape}")
print(f"   Formato salida decoder: {decoder_output.shape}")

# --- 3. Definir el modelo LSTM ---
print("\n🏗️ Construyendo el modelo LSTM...")
# Encoder
encoder_inputs = layers.Input(shape=(max_length,))
encoder_embedding = layers.Embedding(vocab_size, 128)(encoder_inputs)
encoder_lstm = layers.LSTM(256, return_state=True)
_, state_h, state_c = encoder_lstm(encoder_embedding)
encoder_states = [state_h, state_c]

# Decoder
decoder_inputs = layers.Input(shape=(max_length,))
decoder_embedding = layers.Embedding(vocab_size, 128)(decoder_inputs)
decoder_lstm = layers.LSTM(256, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=encoder_states)
decoder_dense = layers.Dense(vocab_size, activation='softmax')(decoder_outputs)

# Modelo completo
model = models.Model([encoder_inputs, decoder_inputs], decoder_outputs)

# Mostrar arquitectura
model.summary()

# --- 4. Compilar y entrenar ---
print("\n🚀 Compilando y entrenando el modelo...")
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Callbacks
callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5)
]

history = model.fit(
    [encoder_input, decoder_input],
    decoder_output,
    epochs=100,
    batch_size=8,
    validation_split=0.2,
    callbacks=callbacks,
    verbose=1
)

# --- 5. Función de traducción ---
def translate_sentence(sentence, model, word_to_idx, idx_to_word, max_length=10):
    """Traducir una oración usando el modelo entrenado"""
    # Preprocesar entrada
    words = sentence.lower().split()
    input_seq = np.array([[word_to_idx.get(word, 0) for word in words[:max_length]]])
    
    # Inicializar secuencia de salida (con token de inicio)
    output_seq = np.zeros((1, max_length))
    output_seq[0, 0] = word_to_idx.get('<start>', 1)  # Token de inicio simplificado
    
    # Traducción paso a paso
    translated_words = []
    for i in range(max_length - 1):
        predictions = model.predict([input_seq, output_seq], verbose=0)
        predicted_idx = np.argmax(predictions[0, i])
        
        if predicted_idx == 0:  # Token de padding
            break
            
        if predicted_idx in idx_to_word:
            translated_words.append(idx_to_word[predicted_idx])
            output_seq[0, i + 1] = predicted_idx
        else:
            break
    
    return ' '.join(translated_words)

# --- 6. Evaluar con ejemplos ---
print("\n📈 Evaluando traducciones...")
test_sentences = [
    "hello my friend",
    "the weather is good", 
    "i am learning spanish",
    "thank you very much"
]

for i, sentence in enumerate(test_sentences):
    translation = translate_sentence(sentence, model, word_to_idx, idx_to_word)
    print(f"   {i+1}. Inglés: '{sentence}'")
    print(f"      Español: '{translation}'")
    print()

# --- 7. Visualizar entrenamiento ---
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.plot(history.history['loss'], label='Entrenamiento')
plt.plot(history.history['val_loss'], label='Validación')
plt.title('Pérdida durante Entrenamiento')
plt.xlabel('Época')
plt.ylabel('Categorical Crossentropy')
plt.legend()
plt.grid(True)

plt.subplot(1, 3, 2)
plt.plot(history.history['accuracy'], label='Entrenamiento')
plt.plot(history.history['val_accuracy'], label='Validación')
plt.title('Precisión durante Entrenamiento')
plt.xlabel('Época')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

# --- 8. Análisis de atención (visualización conceptual) ---
plt.subplot(1, 3, 3)
# Simular pesos de atención (conceptual)
epochs_trained = len(history.history['loss'])
attention_weights = np.random.rand(10, 10)  # Simulación conceptual

plt.imshow(attention_weights, cmap='Blues', aspect='auto')
plt.title('Pesos de Atención (Conceptual)')
plt.xlabel('Posición de Origen')
plt.ylabel('Posición de Destino')
plt.colorbar()

plt.tight_layout()
plt.show()

# --- 9. Guardar el modelo ---
model.save('lstm_translation.h5')
print("\n💾 Modelo guardado como 'lstm_translation.h5'")

print("\n✅ EJERCICIO COMPLETADO")
print("=" * 60)
print("🎯 RESULTADOS:")
print(f"   • Pérdida final: {history.history['loss'][-1]:.4f}")
print(f"   • Precisión final: {history.history['accuracy'][-1]:.4f}")
print(f"   • Total de parámetros: {model.count_params():,}")
print(f"   • Épocas entrenadas: {len(history.history['loss'])}")
print("=" * 60)
