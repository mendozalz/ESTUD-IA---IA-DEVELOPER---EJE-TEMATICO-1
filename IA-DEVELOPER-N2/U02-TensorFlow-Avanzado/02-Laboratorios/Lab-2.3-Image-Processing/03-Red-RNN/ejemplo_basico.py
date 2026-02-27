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

# Gráfico de residuos
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

# --- 6. Predicción multi-paso ---
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

# --- 7. Guardar el modelo ---
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
