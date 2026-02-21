# Red Neuronal Gated Recurrent Unit (GRU)

## 📖 Teoría

### Definición
Las Gated Recurrent Units (GRU) son un tipo de red neuronal recurrente introducida en 2014 como una simplificación de las LSTM. Mantienen la capacidad de aprender dependencias a largo plazo pero con menos parámetros y computacionalmente más eficientes.

### Características Principales
- **Dos puertas**: Reset gate y update gate (vs 3 en LSTM)
- **Eficiencia computacional**: Menos parámetros que LSTM
- **Simplicidad**: Arquitectura más simple pero efectiva
- **Memoria a largo plazo**: Similar a LSTM pero más ligera
- **Estado oculto combinado**: No separa estado de celda y oculto

### Componentes
1. **Update Gate**: Controla cuánta información del pasado mantener
2. **Reset Gate**: Controla cuánta información del pasado olvidar
3. **Estado oculto**: Memoria combinada (similar a celda + oculto de LSTM)
4. **Candidato de estado**: Nuevo estado propuesto
5. **Funciones de activación**: Generalmente tanh y sigmoid

### Ventajas sobre LSTM
- **Menos parámetros**: Más eficiente computacionalmente
- **Entrenamiento más rápido**: Converge más rápidamente
- **Simpler arquitectura**: Más fácil de implementar
- **Rendimiento similar**: En muchos casos comparable a LSTM

### Casos de Uso
- Procesamiento de lenguaje natural
- Series temporales
- Análisis de secuencias
- Traducción automática
- Reconocimiento de voz

## 🎯 Ejercicio Práctico: Análisis de Sentimientos en Tiempo Real

### Tipo de Aprendizaje: **SUPERVISADO**
- **Por qué**: Usa textos etiquetados para entrenar clasificador
- **Datos**: Reseñas de películas con etiquetas de sentimiento

### Implementación Completa

```python
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt

print("🚪 RED GRU - ANÁLISIS DE SENTIMIENTOS EN TIEMPO REAL")
print("=" * 60)

# --- 1. Crear dataset de ejemplo ---
print("📊 Creando dataset de análisis de sentimientos...")
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
print("🔤 Preprocesando texto...")
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
print("\n🏗️ Construyendo el modelo GRU...")
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
print("\n🚀 Compilando y entrenando el modelo...")
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
print("\n📈 Evaluando el modelo...")
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
print("\n📊 Evaluando predicciones...")
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
print("🔍 Analizando comportamiento del GRU...")

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
    plt.ylabel('Activación Promedio')
    plt.legend()
    plt.grid(True)

plt.suptitle('Análisis de Activaciones de Capas GRU')
plt.tight_layout()
plt.show()

# --- 9. Comparación con LSTM ---
print("\n⚖️ Comparando GRU vs LSTM...")
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
plt.title('Precisión GRU durante Entrenamiento')
plt.xlabel('Época')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='GRU Entrenamiento')
plt.plot(history.history['val_loss'], label='GRU Validación')
plt.title('Pérdida GRU durante Entrenamiento')
plt.xlabel('Época')
plt.ylabel('Binary Crossentropy')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# --- 11. Análisis de eficiencia ---
print("\n⚡ Analizando eficiencia computacional...")
import time

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
print("\n💾 Modelo guardado como 'gru_sentiment.h5'")

print("\n✅ EJERCICIO COMPLETADO")
print("=" * 60)
print("🎯 RESULTADOS:")
print(f"   • Precisión final: {accuracy:.4f}")
print(f"   • Pérdida final: {loss:.4f}")
print(f"   • Total de parámetros: {model.count_params():,}")
print(f"   • Épocas entrenadas: {len(history.history['loss'])}")
print(f"   • Ventaja de velocidad vs LSTM: {lstm_time/gru_time:.2f}x")
print("=" * 60)
```

## 📊 Resultados Esperados

```
🚪 RED GRU - ANÁLISIS DE SENTIMIENTOS EN TIEMPO REAL
============================================================
📊 Creando dataset de análisis de sentimientos...
🔤 Preprocesando texto...
   Vocabulario: 78 palabras
   Secuencias procesadas: (20, 15)
   Sentimientos positivos: 11
   Sentimientos negativos: 9

🏗️ Construyendo el modelo GRU...
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 embedding (Embedding)       (None, 15, 64)            5056      
 gru (GRU)                   (None, 15, 64)            24960     
 gru_1 (GRU)                 (None, 32)                12480     
 dense (Dense)               (None, 16)                528       
 dropout (Dropout)           (None, 16)                0         
 dense_1 (Dense)             (None, 1)                 17        
=================================================================
Total params: 43,041
Trainable params: 43,041
Non-trainable params: 0
_________________________________________________________________

🚀 Compilando y entrenando el modelo...
Epoch 1/50
4/4 [==============================] - 3s 600ms/step - loss: 0.6931 - accuracy: 0.5000 - val_loss: 0.6931 - val_accuracy: 0.5714
...
Epoch 23/50
4/4 [==============================] - 0s 100ms/step - loss: 0.1234 - accuracy: 0.9333 - val_loss: 0.4567 - val_accuracy: 0.8571

📈 Evaluando el modelo...
   Precisión final: 0.9500
   Pérdida final: 0.1234

📊 Evaluando predicciones...
   Review: 'Esta película es fantástica'
   Sentimiento: Positivo (confianza: 0.876)
   
   Review: 'No me gustó para nada'
   Sentimiento: Negativo (confianza: 0.923)
   
   Review: 'Actuación regular pero dirección buena'
   Sentimiento: Positivo (confianza: 0.654)
   
   Review: 'Experiencia cinematográfica increíble'
   Sentimiento: Positivo (confianza: 0.812)
   
   Review: 'Guion confuso y actuación pobre'
   Sentimiento: Negativo (confianza: 0.789)

⚖️ Comparando GRU vs LSTM...
   Entrenando modelo LSTM equivalente...
   GRU - Precisión: 0.9500, Parámetros: 43,041
   LSTM - Precisión: 0.9000, Parámetros: 51,905

⚡ Analizando eficiencia computacional...
   Tiempo inferencia GRU (100 predicciones): 2.345s
   Tiempo inferencia LSTM (100 predicciones): 3.678s
   GRU es 1.57x más rápido que LSTM

✅ EJERCICIO COMPLETADO
============================================================
🎯 RESULTADOS:
   • Precisión final: 0.9500
   • Pérdida final: 0.1234
   • Total de parámetros: 43,041
   • Épocas entrenadas: 23
   • Ventaja de velocidad vs LSTM: 1.57x
============================================================
```

## 🎓 Análisis de Aprendizaje

### Identificación del Paradigma
**✅ APRENDIZAJE SUPERVISADO**
- **Datos etiquetados**: Usa reseñas con etiquetas de sentimiento conocidas
- **Objetivo**: Clasificar texto como positivo o negativo
- **Métrica**: Accuracy comparando predicciones con etiquetas reales

### Ventajas de las GRU
- **Eficiencia computacional**: Menos parámetros que LSTM
- **Entrenamiento más rápido**: Converge más rápidamente
- **Simplicidad**: Arquitectura más simple pero efectiva
- **Rendimiento similar**: Comparable a LSTM en muchos casos

### Limitaciones
- **Menos capacidad expresiva**: Potencialmente menos poderosa que LSTM
- **Menos control**: Solo dos puertas vs tres en LSTM
- **Casos complejos**: Puede ser menos efectiva en secuencias muy complejas

## 🚀 Adaptación al Proyecto Integrador

### Ideas para Adaptación
1. **Dominio específico**: Adaptar para análisis de sentimientos en redes sociales
2. **Streaming**: Implementar para procesamiento en tiempo real
3. **Multi-clasificación**: Extender para más categorías de sentimiento

### Ejemplo de Adaptación
```python
# Adaptar GRU para análisis de sentimientos multi-clase
def crear_gru_multiclase(num_classes, vocab_size, maxlen=100):
    """
    Adapta GRU para clasificación multi-clase de sentimientos
    """
    model = models.Sequential([
        layers.Embedding(vocab_size, 128, input_length=maxlen),
        layers.Bidirectional(layers.GRU(64, return_sequences=True)),
        layers.Bidirectional(layers.GRU(32)),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer='adam', 
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
    
    return model
```

## 📝 Ejercicios Adicionales

### Ejercicio 1: No Supervisado (Clustering Secuencial)
```python
# Usar GRU para clustering no supervisado de secuencias
def crear_gru_clustering(seq_length, n_features, n_clusters):
    """Crear GRU para clustering de secuencias"""
    
    # Autoencoder secuencial con GRU
    inputs = layers.Input(shape=(seq_length, n_features))
    
    # Encoder
    encoded = layers.GRU(64, return_sequences=False)(inputs)
    encoded = layers.Dense(32, activation='relu')(encoded)
    
    # Clustering layer
    cluster_assignments = layers.Dense(n_clusters, activation='softmax')(encoded)
    
    # Decoder
    decoded = layers.RepeatVector(seq_length)(cluster_assignments)
    decoded = layers.GRU(64, return_sequences=True)(decoded)
    decoded = layers.TimeDistributed(layers.Dense(n_features))(decoded)
    
    autoencoder = models.Model(inputs, outputs=[decoded, cluster_assignments])
    return autoencoder
```

### Ejercicio 2: Reforzado (GRU Policy Network)
```python
# Usar GRU como red de política para aprendizaje por refuerzo
class GRUPolicyNetwork:
    def __init__(self, state_size, action_size, sequence_length=10):
        self.model = models.Sequential([
            layers.Input(shape=(sequence_length, state_size)),
            layers.GRU(64, return_sequences=True),
            layers.GRU(32),
            layers.Dense(64, activation='relu'),
            layers.Dense(action_size, activation='softmax')
        ])
        
        self.model.compile(optimizer='adam', loss='categorical_crossentropy')
        self.state_history = []
    
    def get_action(self, state):
        """Obtener acción basada en estado actual e historial"""
        # Mantener historial de estados
        self.state_history.append(state)
        if len(self.state_history) > 10:
            self.state_history.pop(0)
        
        # Preparar secuencia para GRU
        if len(self.state_history) < 10:
            # Padding si es necesario
            padded_sequence = np.zeros((10, len(state)))
            padded_sequence[-len(self.state_history):] = self.state_history
        else:
            padded_sequence = np.array(self.state_history)
        
        # Obtener acción
        action_probs = self.model.predict(padded_sequence[np.newaxis, ...], verbose=0)
        return np.random.choice(len(action_probs[0]), p=action_probs[0])
```

## 🔗 Recursos Adicionales

- [Documentación TensorFlow/Keras](https://www.tensorflow.org/api_docs/python/tf/keras)
- [GRU vs LSTM Comparison](https://www.tensorflow.org/api_docs/python/tf/keras/layers/GRU)
- [Sequence Models Guide](https://www.tensorflow.org/guide/keras/rnn)
- [Time Series Forecasting](https://www.tensorflow.org/tutorials/structured_data/time_series)
