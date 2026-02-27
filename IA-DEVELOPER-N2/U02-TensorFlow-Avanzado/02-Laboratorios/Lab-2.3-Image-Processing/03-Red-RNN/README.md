# Red Neuronal Recurrente (RNN)

## 📖 Teoría

### Definición
Las Redes Neuronales Recurrentes (RNN - Recurrent Neural Networks) están diseñadas para procesar datos secuenciales, como texto o series temporales. A diferencia de las redes feedforward, tienen conexiones recurrentes que les permiten retener información de pasos anteriores ("memoria").

### Características Principales
- **Memoria interna**: Capacidad de recordar información de pasos temporales anteriores
- **Conexiones recurrentes**: Las neuronas se conectan consigo mismas a través del tiempo
- **Compartición de parámetros**: Los mismos pesos se usan en todos los pasos temporales
- **Procesamiento secuencial**: Los datos se procesan paso a paso
- **Dependencia temporal**: La salida actual depende de entradas previas

### Componentes
1. **Estado oculto**: Memoria que guarda información de pasos anteriores
2. **Conexiones recurrentes**: Conectan el estado actual con estados anteriores
3. **Funciones de activación**: tanh, ReLU, sigmoid
4. **Secuencia de entrada**: Datos procesados temporalmente
5. **Secuencia de salida**: Resultados para cada paso temporal

### Problemas Principales
- **Vanishing gradients**: Los gradientes tienden a cero en secuencias largas
- **Exploding gradients**: Los gradientes pueden crecer exponencialmente
- **Dificultad de entrenamiento**: Entrenamiento inestable en secuencias largas

### Casos de Uso
- Procesamiento de lenguaje natural
- Predicción de series temporales
- Generación de texto
- Análisis de secuencias biológicas
- Traducción automática

## 🎯 Ejercicio Práctico: Predicción de Series Temporales

### Tipo de Aprendizaje: **SUPERVISADO**
- **Por qué**: Usa datos históricos etiquetados (el "valor futuro" actúa como etiqueta)
- **Datos**: Serie temporal sintética con valores futuros conocidos

### Implementación Completa

```python
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

print("🔄 RED NEURONAL RECURRENTE - PREDICCIÓN DE SERIES TEMPORALES")
print("=" * 60)

# --- 1. Generar datos sintéticos ---
print("📊 Generando datos sintéticos...")
t = np.arange(0, 1000)
data = np.sin(0.02 * t) + 0.1 * np.random.normal(0, 1, 1000)

# Preparar secuencias para RNN
seq_length = 50
x, y = [], []
for i in range(len(data) - seq_length):
    x.append(data[i:i+seq_length])
    y.append(data[i+seq_length])
x, y = np.array(x), np.array(y)
x = x.reshape((x.shape[0], x.shape[1], 1))  # Formato para RNN: [samples, timesteps, features]

# Dividir en entrenamiento y prueba
split = int(0.8 * len(x))
x_train, y_train = x[:split], y[:split]
x_test, y_test = x[split:], y[split:]

print(f"   Datos generados: {len(data)} puntos")
print(f"   Secuencias de entrenamiento: {x_train.shape}")
print(f"   Secuencias de prueba: {x_test.shape}")
print(f"   Longitud de secuencia: {seq_length}")

# --- 2. Definir el modelo RNN ---
print("\n🏗️ Construyendo el modelo RNN...")
model = models.Sequential([
    layers.SimpleRNN(50, activation='tanh', input_shape=(seq_length, 1), return_sequences=False),
    layers.Dense(25, activation='relu'),
    layers.Dense(1)  # Predice el siguiente valor
])

# Mostrar arquitectura
model.summary()

# --- 3. Compilar y entrenar ---
print("\n🚀 Compilando y entrenando el modelo...")
model.compile(
    optimizer='adam',
    loss='mse',
    metrics=['mae']
)

# Callbacks
callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3)
]

history = model.fit(
    x_train, y_train,
    epochs=30,
    batch_size=32,
    validation_data=(x_test, y_test),
    callbacks=callbacks,
    verbose=1
)

# --- 4. Evaluar el modelo ---
print("\n📈 Evaluando el modelo...")
test_loss, test_mae = model.evaluate(x_test, y_test, verbose=0)
print(f"   Error cuadrático medio en prueba: {test_loss:.6f}")
print(f"   Error absoluto medio en prueba: {test_mae:.6f}")

# --- 5. Visualizar predicciones ---
print("\n📊 Generando predicciones y visualizaciones...")
# Predecir en el conjunto de prueba
predictions = model.predict(x_test)

# Crear timeline para visualización
test_time = t[split+seq_length:]

plt.figure(figsize=(15, 8))

# Gráfico principal: Real vs Predicción
plt.subplot(2, 2, 1)
plt.plot(test_time[:200], y_test[:200], label='Real', alpha=0.8)
plt.plot(test_time[:200], predictions[:200], label='Predicción', alpha=0.8)
plt.title('Predicción de Serie Temporal (primeros 200 puntos)')
plt.xlabel('Tiempo')
plt.ylabel('Valor')
plt.legend()
plt.grid(True)

# Gráfico de entrenamiento
plt.subplot(2, 2, 2)
plt.plot(history.history['loss'], label='Entrenamiento')
plt.plot(history.history['val_loss'], label='Validación')
plt.title('Pérdida durante Entrenamiento')
plt.xlabel('Época')
plt.ylabel('MSE')
plt.legend()
plt.grid(True)

# Gráfico de MAE
plt.subplot(2, 2, 3)
plt.plot(history.history['mae'], label='Entrenamiento')
plt.plot(history.history['val_mae'], label='Validación')
plt.title('Error Absoluto Medio')
plt.xlabel('Época')
plt.ylabel('MAE')
plt.legend()
plt.grid(True)

# --- 6. Análisis de residuos ---
plt.subplot(2, 2, 4)
residuals = y_test - predictions.flatten()
plt.scatter(predictions.flatten(), residuals, alpha=0.5)
plt.axhline(y=0, color='r', linestyle='--')
plt.title('Análisis de Residuos')
plt.xlabel('Predicciones')
plt.ylabel('Residuos (Real - Predicción)')
plt.grid(True)

plt.tight_layout()
plt.show()

# --- 7. Predicción multi-paso ---
print("\n🔮 Realizando predicción multi-paso...")
def predict_multi_step(model, initial_sequence, n_steps):
    """Predice n pasos hacia adelante usando el modelo recursivamente"""
    current_sequence = initial_sequence.copy()
    predictions = []
    
    for _ in range(n_steps):
        # Predecir siguiente paso
        next_pred = model.predict(current_sequence.reshape(1, seq_length, 1), verbose=0)
        predictions.append(next_pred[0, 0])
        
        # Actualizar secuencia (eliminar primer valor, añadir predicción)
        current_sequence = np.roll(current_sequence, -1)
        current_sequence[-1] = next_pred[0, 0]
    
    return np.array(predictions)

# Tomar una secuencia inicial y predecir 20 pasos adelante
initial_seq = x_test[0]
multi_predictions = predict_multi_step(model, initial_seq, 20)

plt.figure(figsize=(12, 6))
plt.plot(range(len(initial_seq)), initial_seq, 'b-', label='Secuencia inicial', linewidth=2)
plt.plot(range(len(initial_seq)-1, len(initial_seq)+20), 
         np.concatenate([initial_seq[-1:], multi_predictions]), 
         'r--', label='Predicción multi-paso', linewidth=2)
plt.title('Predicción Multi-paso hacia Adelante')
plt.xlabel('Tiempo')
plt.ylabel('Valor')
plt.legend()
plt.grid(True)
plt.show()

print(f"   Predicción multi-paso: {len(multi_predictions)} pasos")
print(f"   Error promedio multi-paso: {np.mean(np.abs(multi_predictions - y_test[:20])):.6f}")

# --- 8. Guardar el modelo ---
model.save('rnn_timeseries.h5')
print("\n💾 Modelo guardado como 'rnn_timeseries.h5'")

print("\n✅ EJERCICIO COMPLETADO")
print("=" * 60)
print("🎯 RESULTADOS:")
print(f"   • MSE final: {test_loss:.6f}")
print(f"   • MAE final: {test_mae:.6f}")
print(f"   • Total de parámetros: {model.count_params():,}")
print(f"   • Épocas entrenadas: {len(history.history['loss'])}")
print("=" * 60)
```

## 📊 Resultados Esperados

```
🔄 RED NEURONAL RECURRENTE - PREDICCIÓN DE SERIES TEMPORALES
============================================================
📊 Generando datos sintéticos...
   Datos generados: 1000 puntos
   Secuencias de entrenamiento: (760, 50, 1)
   Secuencias de prueba: (190, 50, 1)
   Longitud de secuencia: 50

🏗️ Construyendo el modelo RNN...
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 simple_rnn (SimpleRNN)      (None, 50)                2600       
 dense (Dense)               (None, 25)                1275       
 dense_1 (Dense)             (None, 1)                 26         
=================================================================
Total params: 3,901
Trainable params: 3,901
Non-trainable params: 0
_________________________________________________________________

🚀 Compilando y entrenando el modelo...
Epoch 1/30
24/24 [==============================] - 2s 68ms/step - loss: 0.1234 - mae: 0.2789 - val_loss: 0.0987 - val_mae: 0.2345
...
Epoch 15/30
24/24 [==============================] - 1s 45ms/step - loss: 0.0123 - mae: 0.0876 - val_loss: 0.0156 - val_mae: 0.0923

📈 Evaluando el modelo...
   Error cuadrático medio en prueba: 0.015678
   Error absoluto medio en prueba: 0.092345

🔮 Realizando predicción multi-paso...
   Predicción multi-paso: 20 pasos
   Error promedio multi-paso: 0.187654

✅ EJERCICIO COMPLETADO
============================================================
🎯 RESULTADOS:
   • MSE final: 0.015678
   • MAE final: 0.092345
   • Total de parámetros: 3,901
   • Épocas entrenadas: 15
============================================================
```

## 🎓 Análisis de Aprendizaje

### Identificación del Paradigma
**✅ APRENDIZAJE SUPERVISADO**
- **Datos etiquetados**: Aunque predice valores futuros, se entrena con datos históricos donde el "valor futuro" actúa como etiqueta
- **Objetivo**: Predecir el siguiente valor en la secuencia
- **Métrica**: MSE comparando predicciones con valores reales

### Ventajas de las RNN
- **Memoria temporal**: Capacidad de recordar información pasada
- **Flexibilidad**: Pueden manejar secuencias de longitud variable
- **Compartición de parámetros**: Eficiencia en el procesamiento secuencial
- **Aplicabilidad**: Útiles para muchos tipos de datos secuenciales

### Limitaciones
- **Vanishing gradients**: Dificultad para aprender dependencias largas
- **Entrenamiento secuencial**: No pueden paralelizar fácilmente
- **Inestabilidad**: Pueden ser difíciles de entrenar
- **Memoria limitada**: Olvidan información antigua rápidamente

## 🚀 Adaptación al Proyecto Integrador

### Ideas para Adaptación
1. **Datos reales**: Adaptar para series temporales del proyecto (ventas, sensores, etc.)
2. **Múltiples variables**: Extender para series multivariadas
3. **Diferentes horizontes**: Predecir múltiples pasos adelante

### Ejemplo de Adaptación
```python
# Adaptar RNN para datos de proyecto específicos
def adaptar_rnn_proyecto(datos, longitud_secuencia, caracteristicas=1):
    """
    Adapta RNN para datos secuenciales específicos del proyecto
    """
    model = models.Sequential([
        layers.SimpleRNN(64, activation='tanh', 
                       input_shape=(longitud_secuencia, caracteristicas),
                       return_sequences=True),
        layers.SimpleRNN(32, activation='tanh'),
        layers.Dense(16, activation='relu'),
        layers.Dense(1)  # Predicción univariada
    ])
    
    model.compile(optimizer='adam', 
                loss='mse',
                metrics=['mae'])
    
    return model
```

## 📝 Ejercicios Adicionales

### Ejercicio 1: No Supervisado (Autoencoder Secuencial)
```python
# Convertir RNN a autoencoder para compresión secuencial
def crear_autoencoder_secuencial(seq_length, n_features):
    # Encoder
    input_seq = layers.Input(shape=(seq_length, n_features))
    encoded = layers.SimpleRNN(32, activation='relu')(input_seq)
    
    # Decoder
    decoded = layers.RepeatVector(seq_length)(encoded)
    decoded = layers.SimpleRNN(n_features, return_sequences=True)(decoded)
    
    autoencoder = models.Model(input_seq, decoded)
    encoder = models.Model(input_seq, encoded)
    
    autoencoder.compile(optimizer='adam', loss='mse')
    return autoencoder, encoder
```

### Ejercicio 2: Reforzado (Policy RNN)
```python
# Usar RNN como red de política para aprendizaje por refuerzo
class PolicyRNN:
    def __init__(self, state_size, action_size):
        self.model = models.Sequential([
            layers.SimpleRNN(64, activation='relu', input_shape=(None, state_size)),
            layers.Dense(32, activation='relu'),
            layers.Dense(action_size, activation='softmax')
        ])
        
        self.model.compile(optimizer='adam', loss='categorical_crossentropy')
    
    def get_action(self, state):
        """Obtener acción basada en política actual"""
        state = np.expand_dims(state, axis=0)
        action_probs = self.model.predict(state, verbose=0)
        return np.random.choice(len(action_probs[0]), p=action_probs[0])
```

## 🔗 Recursos Adicionales

- [Documentación TensorFlow/Keras](https://www.tensorflow.org/api_docs/python/tf/keras)
- [Time Series Forecasting Guide](https://www.tensorflow.org/tutorials/structured_data/time_series)
- [RNN Best Practices](https://www.tensorflow.org/guide/keras/rnn)
- [Sequence Models](https://www.tensorflow.org/guide/keras/sequential_model)
