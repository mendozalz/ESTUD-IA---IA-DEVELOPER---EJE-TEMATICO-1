# Ejercicio 2: Datos de Series Temporales con Ventanas Deslizantes

## ⚡ Caso de Uso: Predicción de Consumo Energético en Planta Industrial

### Contexto Empresarial
En el sector energético, predecir el consumo futuro permite optimizar la generación, reducir costos operativos y mejorar la eficiencia del sistema. Las plantas industriales consumen grandes cantidades de energía y necesitan predicciones precisas para planificar la producción.

### Problema a Resolver
- **Predicción**: Estimar consumo energético para las próximas 24-48 horas
- **Optimización**: Ajustar generación y compra de energía
- **Costos**: Reducir picos de consumo y tarifas por demanda
- **ROI**: Reducción del 10-15% en costos energéticos

## 🎯 Objetivos de Aprendizaje

1. **Manejo de series temporales**:
   - Crear ventanas deslizantes con `tf.data.Dataset`
   - Normalización y manejo de datos faltantes

2. **Modelos secuenciales avanzados**:
   - Uso de LSTM, GRU y Transformers para series temporales
   - Optimización con Keras Tuner

3. **Integración con sistemas IoT**:
   - Simular datos de sensores y desplegar el modelo en un broker MQTT
   - Monitoreo en tiempo real

## 📊 Dataset

### Opción 1: Dataset Público - Industrial Energy Consumption
- **Descripción**: Consumo energético de plantas industriales
- **Descarga**: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/individual+household+electric+power+consumption)
- **Características**: Mediciones cada minuto, múltiples variables

### Opción 2: Datos Sintéticos (para desarrollo)
- **Generación**: Simular patrones de consumo industrial
- **Ventajas**: Control sobre patrones estacionales y ruido

## 🛠️ Implementación

### Paso 1: Configuración del Entorno
```python
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.tuners import RandomSearch
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import json
import time
```

### Paso 2: Generación de Datos Sintéticos
```python
def generate_energy_data(n=10000, noise_level=0.1):
    """Generar datos sintéticos de consumo energético industrial"""
    time = np.arange(n)
    
    # Patrones diarios (ciclos de 24 horas)
    daily_pattern = 100 * np.sin(2 * np.pi * time / (24 * 60)) + 200
    
    # Patrones semanales (ciclos de 7 días)
    weekly_pattern = 50 * np.sin(2 * np.pi * time / (7 * 24 * 60))
    
    # Tendencia estacional
    seasonal_trend = 0.05 * time
    
    # Patrones de producción (turnos de trabajo)
    production_pattern = np.zeros(n)
    for i in range(n):
        hour = (i // 60) % 24
        if 6 <= hour < 14:  # Turno matutino
            production_pattern[i] = 80
        elif 14 <= hour < 22:  # Turno vespertino
            production_pattern[i] = 120
        else:  # Turno nocturno
            production_pattern[i] = 40
    
    # Eventos aleatorios (mantenimiento, paradas)
    events = np.zeros(n)
    for _ in range(np.random.randint(5, 15)):
        start = np.random.randint(0, n - 1440)  # Evento de hasta 24 horas
        duration = np.random.randint(60, 1440)
        events[start:start+duration] = np.random.uniform(-100, -50)
    
    # Combinar todos los patrones
    base_consumption = daily_pattern + weekly_pattern + seasonal_trend + production_pattern + events
    
    # Añadir ruido
    noise = np.random.normal(0, noise_level * np.std(base_consumption), n)
    
    # Asegurar valores positivos
    consumption = np.maximum(base_consumption + noise, 10)
    
    return consumption
```

### Paso 3: Creación de Dataset con Ventanas Deslizantes
```python
def create_time_series_dataset(data, window_size=24, horizon=1, batch_size=32):
    """
    Crear dataset con ventanas deslizantes para series temporales
    
    Args:
        data: Array numpy con datos de series temporales
        window_size: Tamaño de la ventana de entrada
        horizon: Horizonte de predicción (pasos futuros)
        batch_size: Tamaño del batch
    """
    dataset = tf.data.Dataset.from_tensor_slices(data)
    
    # Crear ventanas deslizantes
    dataset = dataset.window(window_size + horizon, shift=1, drop_remainder=True)
    dataset = dataset.flat_map(lambda window: window.batch(window_size + horizon))
    
    # Separar características (X) y etiquetas (y)
    dataset = dataset.map(lambda window: (window[:-horizon], window[-horizon:]))
    
    # Batch y prefetch
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset

def prepare_multivariate_data(data, feature_cols, target_col, window_size=24, batch_size=32):
    """
    Preparar datos multivariados para series temporales
    """
    # Normalizar datos
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)
    
    # Separar características y objetivo
    X = scaled_data[:, feature_cols]
    y = scaled_data[:, target_col]
    
    # Crear datasets
    X_dataset = create_time_series_dataset(X, window_size, batch_size=batch_size)
    y_dataset = create_time_series_dataset(y, window_size, batch_size=batch_size)
    
    # Combinar datasets
    dataset = tf.data.Dataset.zip((X_dataset, y_dataset))
    
    return dataset, scaler
```

### Paso 4: Construcción de Modelos Avanzados
```python
def build_lstm_model(window_size, num_features=1, units=[64, 32]):
    """Construir modelo LSTM para series temporales"""
    model = models.Sequential()
    
    # Primera capa LSTM
    model.add(layers.LSTM(
        units[0],
        input_shape=(window_size, num_features),
        return_sequences=len(units) > 1,
        dropout=0.2,
        recurrent_dropout=0.2
    ))
    
    # Capas LSTM adicionales
    for i, unit in enumerate(units[1:], 1):
        return_sequences = i < len(units) - 1
        model.add(layers.LSTM(
            unit,
            return_sequences=return_sequences,
            dropout=0.2,
            recurrent_dropout=0.2
        ))
    
    # Capas densas
    model.add(layers.Dense(16, activation='relu'))
    model.add(layers.Dropout(0.1))
    model.add(layers.Dense(1))  # Predicción de un valor
    
    return model

def build_transformer_model(window_size, num_features=1, d_model=64, num_heads=4):
    """Construir modelo Transformer para series temporales"""
    inputs = layers.Input(shape=(window_size, num_features))
    
    # Positional encoding
    positions = tf.range(start=0, limit=window_size, delta=1)
    position_embedding = layers.Embedding(input_dim=window_size, output_dim=d_model)(positions)
    
    # Proyección de entrada
    x = layers.Dense(d_model)(inputs)
    x = x + position_embedding
    
    # Multi-head attention
    attention_output = layers.MultiHeadAttention(
        num_heads=num_heads,
        key_dim=d_model // num_heads
    )(x, x)
    
    # Add & Norm
    x = layers.Add()([x, attention_output])
    x = layers.LayerNormalization()(x)
    
    # Feed-forward
    ffn_output = layers.Dense(d_model * 4, activation='relu')(x)
    ffn_output = layers.Dense(d_model)(ffn_output)
    
    # Add & Norm
    x = layers.Add()([x, ffn_output])
    x = layers.LayerNormalization()(x)
    
    # Global average pooling y salida
    x = layers.GlobalAveragePooling1D()(x)
    outputs = layers.Dense(1)(x)
    
    model = models.Model(inputs=inputs, outputs=outputs)
    return model
```

### Paso 5: Optimización con Keras Tuner
```python
def build_model_for_tuning(hp, window_size, num_features=1):
    """Construir modelo para optimización de hiperparámetros"""
    model = models.Sequential()
    
    # Hiperparámetros a optimizar
    num_layers = hp.Int('num_layers', 1, 3)
    dropout_rate = hp.Float('dropout_rate', 0.1, 0.5, step=0.1)
    learning_rate = hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4])
    
    # Primera capa LSTM
    model.add(layers.LSTM(
        hp.Int('units_0', min_value=32, max_value=256, step=32),
        input_shape=(window_size, num_features),
        return_sequences=num_layers > 1,
        dropout=dropout_rate
    ))
    
    # Capas LSTM adicionales
    for i in range(1, num_layers):
        return_sequences = i < num_layers - 1
        model.add(layers.LSTM(
            hp.Int(f'units_{i}', min_value=32, max_value=256, step=32),
            return_sequences=return_sequences,
            dropout=dropout_rate
        ))
    
    # Capas densas
    model.add(layers.Dense(hp.Int('dense_units', 16, 128, step=16), activation='relu'))
    model.add(layers.Dropout(dropout_rate))
    model.add(layers.Dense(1))
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='mse',
        metrics=['mae', 'mape']
    )
    
    return model

def optimize_hyperparameters(train_dataset, val_dataset, window_size, num_features=1):
    """Optimizar hiperparámetros con Keras Tuner"""
    
    tuner = RandomSearch(
        lambda hp: build_model_for_tuning(hp, window_size, num_features),
        objective='val_loss',
        max_trials=10,
        executions_per_trial=2,
        directory='tuner_results',
        project_name='energy_prediction'
    )
    
    tuner.search(
        train_dataset,
        validation_data=val_dataset,
        epochs=20,
        callbacks=[tf.keras.callbacks.EarlyStopping(patience=5)]
    )
    
    best_model = tuner.get_best_models(num_models=1)[0]
    best_hyperparameters = tuner.get_best_hyperparameters(num_trials=1)[0]
    
    return best_model, best_hyperparameters
```

### Paso 6: Integración con MQTT
```python
class EnergyPredictionSystem:
    """Sistema de predicción energética con MQTT"""
    
    def __init__(self, model_path, scaler_path):
        self.model = tf.keras.models.load_model(model_path)
        self.scaler = joblib.load(scaler_path)
        self.window_size = 24
        self.data_buffer = []
        
    def setup_mqtt_client(self, broker_host='localhost', port=1883):
        """Configurar cliente MQTT"""
        import paho.mqtt.client as mqtt
        
        self.client = mqtt.Client()
        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message
        
        self.client.connect(broker_host, port, 60)
        
        # Suscribirse a topic de datos de sensores
        self.client.subscribe("energy/sensors")
        
        return self.client
    
    def on_connect(self, client, userdata, flags, rc):
        """Callback de conexión MQTT"""
        print(f"Conectado al broker MQTT con resultado: {rc}")
    
    def on_message(self, client, userdata, msg):
        """Callback de recepción de mensajes MQTT"""
        try:
            data = json.loads(msg.payload.decode())
            energy_value = data['energy']
            timestamp = data['timestamp']
            
            # Agregar al buffer
            self.data_buffer.append(energy_value)
            
            # Mantener tamaño del buffer
            if len(self.data_buffer) > self.window_size:
                self.data_buffer.pop(0)
            
            # Realizar predicción si tenemos suficientes datos
            if len(self.data_buffer) == self.window_size:
                prediction = self.predict_next_consumption()
                
                # Publicar predicción
                prediction_msg = {
                    'timestamp': timestamp,
                    'predicted_energy': float(prediction),
                    'prediction_horizon': '1 hour'
                }
                
                client.publish("energy/predictions", json.dumps(prediction_msg))
                print(f"Predicción publicada: {prediction:.2f} kWh")
                
        except Exception as e:
            print(f"Error procesando mensaje: {e}")
    
    def predict_next_consumption(self):
        """Predecir siguiente consumo energético"""
        if len(self.data_buffer) < self.window_size:
            return None
        
        # Preparar datos
        recent_data = np.array(self.data_buffer[-self.window_size:])
        recent_data = recent_data.reshape(1, self.window_size, 1)
        
        # Normalizar
        recent_data_scaled = self.scaler.transform(recent_data.reshape(-1, 1)).reshape(1, self.window_size, 1)
        
        # Realizar predicción
        prediction_scaled = self.model.predict(recent_data_scaled, verbose=0)
        
        # Desnormalizar
        prediction = self.scaler.inverse_transform(prediction_scaled)[0][0]
        
        return prediction
    
    def start_monitoring(self):
        """Iniciar monitoreo continuo"""
        self.client.loop_forever()
```

### Paso 7: Sistema de Alertas
```python
class EnergyAlertSystem:
    """Sistema de alertas para consumo energético"""
    
    def __init__(self, threshold_multiplier=1.5):
        self.threshold_multiplier = threshold_multiplier
        self.baseline_consumption = 0
        self.alert_history = []
        
    def calculate_baseline(self, historical_data):
        """Calcular consumo base"""
        self.baseline_consumption = np.mean(historical_data)
        return self.baseline_consumption
    
    def check_alerts(self, current_consumption, predicted_consumption):
        """Verificar condiciones de alerta"""
        alerts = []
        
        # Alerta por consumo excesivo
        if current_consumption > self.baseline_consumption * self.threshold_multiplier:
            alerts.append({
                'type': 'high_consumption',
                'severity': 'warning',
                'message': f'Consumo actual ({current_consumption:.2f}) excede el umbral',
                'timestamp': time.time()
            })
        
        # Alerta por predicción de pico
        if predicted_consumption > self.baseline_consumption * self.threshold_multiplier:
            alerts.append({
                'type': 'predicted_peak',
                'severity': 'info',
                'message': f'Se predice un pico de consumo ({predicted_consumption:.2f})',
                'timestamp': time.time()
            })
        
        # Alerta por desviación inusual
        deviation = abs(predicted_consumption - current_consumption) / current_consumption
        if deviation > 0.3:  # 30% de desviación
            alerts.append({
                'type': 'unusual_deviation',
                'severity': 'warning',
                'message': f'Desviación inusual entre actual y predicho: {deviation:.2%}',
                'timestamp': time.time()
            })
        
        return alerts
    
    def send_alert(self, alert):
        """Enviar alerta (simulado)"""
        print(f"🚨 ALERTA [{alert['severity'].upper()}]: {alert['message']}")
        
        # Aquí se podría integrar con sistemas reales:
        # - Email (smtplib)
        # - Slack (webhook)
        # - SMS (Twilio)
        # - Sistema de gestión (API REST)
```

## 🧪 Ejecución Completa

```python
def main():
    """Función principal para ejecutar el ejercicio completo"""
    
    print("⚡ EJERCICIO 2: PREDICCIÓN DE CONSUMO ENERGÉTICO")
    print("=" * 60)
    
    # 1. Generar datos sintéticos
    print("📊 Generando datos de consumo energético...")
    energy_data = generate_energy_data(n=10000, noise_level=0.1)
    
    # Visualizar datos
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    plt.plot(energy_data[:1440])  # Primer día
    plt.title('Consumo Energético - Primer Día')
    plt.xlabel('Minutos')
    plt.ylabel('Consumo (kWh)')
    
    plt.subplot(1, 2, 2)
    plt.plot(energy_data[:10080])  # Primera semana
    plt.title('Consumo Energético - Primera Semana')
    plt.xlabel('Minutos')
    plt.ylabel('Consumo (kWh)')
    
    plt.tight_layout()
    plt.show()
    
    # 2. Preparar datasets
    print("\n🔄 Preparando datasets con ventanas deslizantes...")
    window_size = 24  # 24 horas de datos históricos
    
    # Normalizar datos
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(energy_data.reshape(-1, 1)).flatten()
    
    # Crear datasets
    train_size = int(0.8 * len(scaled_data))
    train_data = scaled_data[:train_size]
    val_data = scaled_data[train_size:]
    
    train_dataset = create_time_series_dataset(train_data, window_size, batch_size=32)
    val_dataset = create_time_series_dataset(val_data, window_size, batch_size=32)
    
    print(f"   Datos de entrenamiento: {len(train_data)}")
    print(f"   Datos de validación: {len(val_data)}")
    print(f"   Tamaño de ventana: {window_size}")
    
    # 3. Optimizar hiperparámetros
    print("\n🔍 Optimizando hiperparámetros...")
    best_model, best_hyperparams = optimize_hyperparameters(
        train_dataset, val_dataset, window_size
    )
    
    print("   Mejores hiperparámetros:")
    for param, value in best_hyperparams.values.items():
        print(f"     {param}: {value}")
    
    # 4. Entrenamiento final
    print("\n🚀 Entrenando modelo final...")
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5),
        tf.keras.callbacks.ModelCheckpoint('best_energy_model.h5', save_best_only=True)
    ]
    
    history = best_model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=50,
        callbacks=callbacks
    )
    
    # 5. Evaluación
    print("\n📈 Evaluando modelo...")
    test_loss, test_mae, test_mape = best_model.evaluate(val_dataset)
    print(f"   MAE: {test_mae:.4f}")
    print(f"   MAPE: {test_mape:.4f}%")
    
    # 6. Predicciones y visualización
    print("\n🎯 Generando predicciones...")
    
    # Obtener predicciones
    predictions = []
    actuals = []
    
    for batch_x, batch_y in val_dataset.take(10):
        pred = best_model.predict(batch_x, verbose=0)
        predictions.extend(pred.flatten())
        actuals.extend(batch_y.numpy().flatten())
    
    # Desnormalizar
    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()
    actuals = scaler.inverse_transform(np.array(actuals).reshape(-1, 1)).flatten()
    
    # Visualizar predicciones
    plt.figure(figsize=(12, 6))
    plt.plot(actuals[:100], label='Consumo Real', alpha=0.7)
    plt.plot(predictions[:100], label='Predicción', alpha=0.7)
    plt.title('Predicción de Consumo Energético')
    plt.xlabel('Tiempo')
    plt.ylabel('Consumo (kWh)')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # 7. Sistema de alertas
    print("\n🚨 Configurando sistema de alertas...")
    alert_system = EnergyAlertSystem(threshold_multiplier=1.3)
    baseline = alert_system.calculate_baseline(energy_data[-1440:])  # Último día
    print(f"   Consumo base: {baseline:.2f} kWh")
    
    # Probar sistema de alertas
    test_current = 350.0
    test_predicted = 420.0
    alerts = alert_system.check_alerts(test_current, test_predicted)
    
    for alert in alerts:
        alert_system.send_alert(alert)
    
    # 8. Guardar modelo y scaler
    print("\n💾 Guardando modelo y scaler...")
    best_model.save('energy_prediction_model.h5')
    import joblib
    joblib.dump(scaler, 'energy_scaler.pkl')
    
    print("\n✅ EJERCICIO COMPLETADO")
    print("=" * 60)
    print("🎯 RESULTADOS:")
    print(f"   • MAE final: {test_mae:.4f}")
    print(f"   • MAPE final: {test_mape:.4f}%")
    print(f"   • Modelo guardado: energy_prediction_model.h5")
    print(f"   • ROI estimado: 10-15% reducción en costos energéticos")
    print("=" * 60)

if __name__ == "__main__":
    main()
```

## 📊 Métricas de Evaluación

### Métricas de Predicción
- **MAE (Mean Absolute Error)**: Error absoluto medio
- **RMSE (Root Mean Square Error)**: Raíz del error cuadrático medio
- **MAPE (Mean Absolute Percentage Error)**: Error porcentual absoluto medio
- **Forecast Bias**: Sesgo sistemático de las predicciones

### Métricas de Producción
- **Latencia**: Tiempo de predicción (<50ms)
- **Throughput**: Predicciones por segundo (>100/s)
- **Disponibilidad**: Uptime del sistema (>99.5%)

## 🚀 Desafíos Adicionales

### Desafío 1: Modelos Ensemble
- Combinar múltiples modelos (LSTM + Transformer)
- Implementar voting o stacking
- Optimizar pesos del ensemble

### Desafío 2: Predicción Multi-step
- Extender predicción a múltiples pasos futuros
- Usar seq2seq o direct forecasting
- Evaluar precisión a diferentes horizontes

### Desafío 3: Integración Real
- Conectar con sensores IoT reales
- Implementar dashboard con Grafana
- Añadir sistema de notificaciones

## 📝 Entrega Requerida

1. **Código fuente** completo y funcional
2. **Informe técnico** con:
   - Análisis exploratorio de datos
   - Arquitectura del modelo
   - Resultados experimentales
3. **Prototipo MQTT** funcionando
4. **Dashboard** de monitoreo en tiempo real

## 🔗 Recursos Adicionales

- [Time Series Forecasting Guide](https://www.tensorflow.org/tutorials/structured_data/time_series)
- [Keras Tuner Documentation](https://keras.io/keras_tuner/)
- [MQTT Protocol](https://mqtt.org/)
- [Energy Forecasting Papers](https://arxiv.org/list/cs.LG/recent)
