"""
Laboratorio 5.4 - Modelos Manuales para Comparación
===================================================

Este script implementa modelos manuales tradicionales para
comparar con los resultados de Neural Architecture Search.
"""

import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Any, Tuple, List
import time
from sklearn.metrics import mean_absolute_error, mean_squared_error

class ManualTimeSeriesModels:
    """
    Clase para implementar modelos manuales de series temporales
    """
    
    def __init__(self, input_shape: Tuple[int, int]):
        """
        Inicializa los modelos manuales
        
        Args:
            input_shape: Forma de entrada (timesteps, features)
        """
        self.input_shape = input_shape
        self.models = {}
        self.histories = {}
        
    def create_lstm_model(self, units: List[int] = [64, 32],
                         dropout_rate: float = 0.2,
                         dense_units: List[int] = [16]) -> tf.keras.Model:
        """
        Crea modelo LSTM para series temporales
        
        Args:
            units: Unidades de capas LSTM
            dropout_rate: Tasa de dropout
            dense_units: Unidades de capas densas
            
        Returns:
            Modelo LSTM
        """
        print("🔧 Creando modelo LSTM...")
        
        model = tf.keras.Sequential()
        
        # Primera capa LSTM
        model.add(tf.keras.layers.LSTM(
            units[0], 
            return_sequences=True if len(units) > 1 else False,
            input_shape=self.input_shape
        ))
        model.add(tf.keras.layers.Dropout(dropout_rate))
        
        # Capas LSTM adicionales
        for i, unit in enumerate(units[1:], 1):
            return_sequences = i < len(units) - 1
            model.add(tf.keras.layers.LSTM(unit, return_sequences=return_sequences))
            model.add(tf.keras.layers.Dropout(dropout_rate))
        
        # Capas densas
        for unit in dense_units:
            model.add(tf.keras.layers.Dense(unit, activation='relu'))
            model.add(tf.keras.layers.Dropout(dropout_rate))
        
        # Capa de salida
        model.add(tf.keras.layers.Dense(1))
        
        # Compilar
        model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )
        
        self.models['lstm'] = model
        
        print(f"✅ Modelo LSTM creado:")
        model.summary()
        
        return model
    
    def create_gru_model(self, units: List[int] = [64, 32],
                        dropout_rate: float = 0.2,
                        dense_units: List[int] = [16]) -> tf.keras.Model:
        """
        Crea modelo GRU para series temporales
        
        Args:
            units: Unidades de capas GRU
            dropout_rate: Tasa de dropout
            dense_units: Unidades de capas densas
            
        Returns:
            Modelo GRU
        """
        print("🔧 Creando modelo GRU...")
        
        model = tf.keras.Sequential()
        
        # Primera capa GRU
        model.add(tf.keras.layers.GRU(
            units[0], 
            return_sequences=True if len(units) > 1 else False,
            input_shape=self.input_shape
        ))
        model.add(tf.keras.layers.Dropout(dropout_rate))
        
        # Capas GRU adicionales
        for i, unit in enumerate(units[1:], 1):
            return_sequences = i < len(units) - 1
            model.add(tf.keras.layers.GRU(unit, return_sequences=return_sequences))
            model.add(tf.keras.layers.Dropout(dropout_rate))
        
        # Capas densas
        for unit in dense_units:
            model.add(tf.keras.layers.Dense(unit, activation='relu'))
            model.add(tf.keras.layers.Dropout(dropout_rate))
        
        # Capa de salida
        model.add(tf.keras.layers.Dense(1))
        
        # Compilar
        model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )
        
        self.models['gru'] = model
        
        print(f"✅ Modelo GRU creado:")
        model.summary()
        
        return model
    
    def create_cnn_model(self, filters: List[int] = [64, 32, 16],
                        kernel_sizes: List[int] = [3, 3, 3],
                        dropout_rate: float = 0.2) -> tf.keras.Model:
        """
        Crea modelo CNN para series temporales
        
        Args:
            filters: Filtros de capas convolucionales
            kernel_sizes: Tamaños de kernel
            dropout_rate: Tasa de dropout
            
        Returns:
            Modelo CNN
        """
        print("🔧 Creando modelo CNN...")
        
        model = tf.keras.Sequential()
        
        # Capas convolucionales
        for i, (filter_count, kernel_size) in enumerate(zip(filters, kernel_sizes)):
            if i == 0:
                model.add(tf.keras.layers.Conv1D(
                    filter_count, kernel_size, activation='relu',
                    input_shape=self.input_shape
                ))
            else:
                model.add(tf.keras.layers.Conv1D(
                    filter_count, kernel_size, activation='relu'
                ))
            
            model.add(tf.keras.layers.MaxPooling1D(2))
            model.add(tf.keras.layers.Dropout(dropout_rate))
        
        # Aplanar y capas densas
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(64, activation='relu'))
        model.add(tf.keras.layers.Dropout(dropout_rate))
        model.add(tf.keras.layers.Dense(32, activation='relu'))
        model.add(tf.keras.layers.Dropout(dropout_rate))
        
        # Capa de salida
        model.add(tf.keras.layers.Dense(1))
        
        # Compilar
        model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )
        
        self.models['cnn'] = model
        
        print(f"✅ Modelo CNN creado:")
        model.summary()
        
        return model
    
    def create_bidirectional_lstm_model(self, units: List[int] = [64, 32],
                                      dropout_rate: float = 0.2) -> tf.keras.Model:
        """
        Crea modelo LSTM bidireccional
        
        Args:
            units: Unidades de capas LSTM
            dropout_rate: Tasa de dropout
            
        Returns:
            Modelo LSTM bidireccional
        """
        print("🔧 Creando modelo LSTM Bidireccional...")
        
        model = tf.keras.Sequential()
        
        # Primera capa LSTM bidireccional
        model.add(tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(units[0], return_sequences=True),
            input_shape=self.input_shape
        ))
        model.add(tf.keras.layers.Dropout(dropout_rate))
        
        # Segunda capa LSTM bidireccional
        model.add(tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(units[1])
        ))
        model.add(tf.keras.layers.Dropout(dropout_rate))
        
        # Capas densas
        model.add(tf.keras.layers.Dense(32, activation='relu'))
        model.add(tf.keras.layers.Dropout(dropout_rate))
        
        # Capa de salida
        model.add(tf.keras.layers.Dense(1))
        
        # Compilar
        model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )
        
        self.models['bilstm'] = model
        
        print(f"✅ Modelo LSTM Bidireccional creado:")
        model.summary()
        
        return model
    
    def create_attention_model(self, units: int = 64) -> tf.keras.Model:
        """
        Crea modelo con mecanismo de atención
        
        Args:
            units: Unidades de la capa LSTM
            
        Returns:
            Modelo con atención
        """
        print("🔧 Creando modelo con Atención...")
        
        # Input
        inputs = tf.keras.Input(shape=self.input_shape)
        
        # Capa LSTM
        lstm_out = tf.keras.layers.LSTM(units, return_sequences=True)(inputs)
        
        # Mecanismo de atención
        attention = tf.keras.layers.Dense(1, activation='tanh')(lstm_out)
        attention = tf.keras.layers.Flatten()(attention)
        attention = tf.keras.layers.Activation('softmax')(attention)
        attention = tf.keras.layers.RepeatVector(units)(attention)
        attention = tf.keras.layers.Permute([2, 1])(attention)
        
        # Aplicar atención
        attended = tf.keras.layers.Multiply()([lstm_out, attention])
        attended = tf.keras.layers.Lambda(lambda x: tf.reduce_sum(x, axis=1))(attended)
        
        # Capas densas
        x = tf.keras.layers.Dense(32, activation='relu')(attended)
        x = tf.keras.layers.Dropout(0.2)(x)
        outputs = tf.keras.layers.Dense(1)(x)
        
        # Crear modelo
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        
        # Compilar
        model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )
        
        self.models['attention'] = model
        
        print(f"✅ Modelo con Atención creado:")
        model.summary()
        
        return model
    
    def create_transformer_model(self, d_model: int = 64,
                                num_heads: int = 4,
                                dropout_rate: float = 0.1) -> tf.keras.Model:
        """
        Crea modelo Transformer para series temporales
        
        Args:
            d_model: Dimensión del modelo
            num_heads: Número de cabezas de atención
            dropout_rate: Tasa de dropout
            
        Returns:
            Modelo Transformer
        """
        print("🔧 Creando modelo Transformer...")
        
        # Input
        inputs = tf.keras.Input(shape=self.input_shape)
        
        # Positional encoding (simplificada)
        x = inputs
        for i in range(d_model // self.input_shape[1]):
            x = tf.keras.layers.Dense(d_model // self.input_shape[1], activation='relu')(x)
        
        # Multi-head self attention
        attention_layer = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=d_model // num_heads
        )
        attention_output = attention_layer(x, x)
        attention_output = tf.keras.layers.Dropout(dropout_rate)(attention_output)
        
        # Add & Norm
        x = tf.keras.layers.Add()([x, attention_output])
        x = tf.keras.layers.LayerNormalization()(x)
        
        # Feed forward
        ff_output = tf.keras.layers.Dense(d_model, activation='relu')(x)
        ff_output = tf.keras.layers.Dense(self.input_shape[1])(ff_output)
        ff_output = tf.keras.layers.Dropout(dropout_rate)(ff_output)
        
        # Add & Norm
        x = tf.keras.layers.Add()([x, ff_output])
        x = tf.keras.layers.LayerNormalization()(x)
        
        # Global average pooling
        x = tf.keras.layers.GlobalAveragePooling1D()(x)
        
        # Capas densas
        x = tf.keras.layers.Dense(32, activation='relu')(x)
        x = tf.keras.layers.Dropout(dropout_rate)(x)
        outputs = tf.keras.layers.Dense(1)(x)
        
        # Crear modelo
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        
        # Compilar
        model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )
        
        self.models['transformer'] = model
        
        print(f"✅ Modelo Transformer creado:")
        model.summary()
        
        return model
    
    def create_ensemble_model(self) -> tf.keras.Model:
        """
        Crea modelo ensemble combinando varios modelos
        
        Returns:
            Modelo ensemble
        """
        print("🔧 Creando modelo Ensemble...")
        
        # Inputs
        inputs = tf.keras.Input(shape=self.input_shape)
        
        # Crear modelos base
        lstm_out = self.models['lstm'](inputs)
        gru_out = self.models['gru'](inputs)
        cnn_out = self.models['cnn'](inputs)
        
        # Combinar salidas
        combined = tf.keras.layers.Concatenate()([lstm_out, gru_out, cnn_out])
        
        # Capas densas del ensemble
        x = tf.keras.layers.Dense(32, activation='relu')(combined)
        x = tf.keras.layers.Dropout(0.2)(x)
        x = tf.keras.layers.Dense(16, activation='relu')(x)
        outputs = tf.keras.layers.Dense(1)(x)
        
        # Crear modelo
        ensemble_model = tf.keras.Model(inputs=inputs, outputs=outputs)
        
        # Compilar
        ensemble_model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )
        
        self.models['ensemble'] = ensemble_model
        
        print(f"✅ Modelo Ensemble creado:")
        ensemble_model.summary()
        
        return ensemble_model
    
    def train_model(self, model_name: str,
                   X_train: np.ndarray, y_train: np.ndarray,
                   X_val: np.ndarray, y_val: np.ndarray,
                   epochs: int = 50,
                   batch_size: int = 32,
                   patience: int = 10) -> Dict[str, Any]:
        """
        Entrena un modelo específico
        
        Args:
            model_name: Nombre del modelo a entrenar
            X_train, y_train: Datos de entrenamiento
            X_val, y_val: Datos de validación
            epochs: Número de épocas
            batch_size: Tamaño de batch
            patience: Paciencia para early stopping
            
        Returns:
            Resultados del entrenamiento
        """
        print(f"🎓 Entrenando modelo: {model_name}")
        
        if model_name not in self.models:
            raise ValueError(f"Modelo {model_name} no encontrado")
        
        model = self.models[model_name]
        
        # Callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=patience,
                restore_best_weights=True,
                verbose=1
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        # Entrenar
        start_time = time.time()
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        training_time = time.time() - start_time
        
        # Guardar historial
        self.histories[model_name] = history.history
        
        results = {
            'model_name': model_name,
            'training_time': training_time,
            'history': history.history,
            'epochs_trained': len(history.history['loss'])
        }
        
        print(f"✅ Entrenamiento completado en {training_time:.2f} segundos")
        
        return results
    
    def evaluate_model(self, model_name: str,
                      X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """
        Evalúa un modelo específico
        
        Args:
            model_name: Nombre del modelo a evaluar
            X_test, y_test: Datos de prueba
            
        Returns:
            Resultados de evaluación
        """
        print(f"🧪 Evaluando modelo: {model_name}")
        
        if model_name not in self.models:
            raise ValueError(f"Modelo {model_name} no encontrado")
        
        model = self.models[model_name]
        
        # Evaluar
        start_time = time.time()
        test_metrics = model.evaluate(X_test, y_test, verbose=0)
        evaluation_time = time.time() - start_time
        
        # Predicciones
        predictions = model.predict(X_test, verbose=0)
        
        # Calcular métricas adicionales
        mae = mean_absolute_error(y_test, predictions)
        mse = mean_squared_error(y_test, predictions)
        rmse = np.sqrt(mse)
        
        # MAPE (Mean Absolute Percentage Error)
        mape = np.mean(np.abs((y_test - predictions) / np.maximum(y_test, 1e-8))) * 100
        
        results = {
            'model_name': model_name,
            'test_loss': test_metrics[0],
            'test_mae': test_metrics[1],
            'predictions': predictions,
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'mape': mape,
            'evaluation_time': evaluation_time
        }
        
        print(f"✅ Evaluación completada:")
        print(f"   - MAE: {mae:.4f}")
        print(f"   - RMSE: {rmse:.4f}")
        print(f"   - MAPE: {mape:.2f}%")
        
        return results
    
    def train_all_models(self, X_train: np.ndarray, y_train: np.ndarray,
                        X_val: np.ndarray, y_val: np.ndarray,
                        epochs: int = 50) -> Dict[str, Any]:
        """
        Entrena todos los modelos disponibles
        
        Args:
            X_train, y_train: Datos de entrenamiento
            X_val, y_val: Datos de validación
            epochs: Número de épocas
            
        Returns:
            Resultados de todos los entrenamientos
        """
        print("🎓 Entrenando todos los modelos...")
        
        training_results = {}
        
        for model_name in self.models.keys():
            try:
                results = self.train_model(
                    model_name, X_train, y_train, X_val, y_val, epochs
                )
                training_results[model_name] = results
            except Exception as e:
                print(f"⚠️ Error entrenando {model_name}: {e}")
                training_results[model_name] = {'error': str(e)}
        
        return training_results
    
    def evaluate_all_models(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """
        Evalúa todos los modelos disponibles
        
        Args:
            X_test, y_test: Datos de prueba
            
        Returns:
            Resultados de todas las evaluaciones
        """
        print("🧪 Evaluando todos los modelos...")
        
        evaluation_results = {}
        
        for model_name in self.models.keys():
            try:
                results = self.evaluate_model(model_name, X_test, y_test)
                evaluation_results[model_name] = results
            except Exception as e:
                print(f"⚠️ Error evaluando {model_name}: {e}")
                evaluation_results[model_name] = {'error': str(e)}
        
        return evaluation_results
    
    def compare_models(self, evaluation_results: Dict[str, Any]) -> pd.DataFrame:
        """
        Compara todos los modelos evaluados
        
        Args:
            evaluation_results: Resultados de evaluación
            
        Returns:
            DataFrame con comparación
        """
        print("📊 Comparando modelos...")
        
        comparison_data = []
        
        for model_name, results in evaluation_results.items():
            if 'error' in results:
                continue
            
            comparison_data.append({
                'Model': model_name,
                'Test Loss': results['test_loss'],
                'MAE': results['mae'],
                'RMSE': results['rmse'],
                'MAPE (%)': results['mape'],
                'Evaluation Time (s)': results['evaluation_time']
            })
        
        df = pd.DataFrame(comparison_data)
        df = df.sort_values('MAE')  # Ordenar por MAE (menor es mejor)
        
        print("✅ Comparación completada:")
        print(df.to_string(index=False))
        
        return df
    
    def plot_model_comparison(self, evaluation_results: Dict[str, Any]):
        """
        Visualiza comparación de modelos
        
        Args:
            evaluation_results: Resultados de evaluación
        """
        print("📊 Visualizando comparación de modelos...")
        
        # Filtrar modelos con errores
        valid_results = {k: v for k, v in evaluation_results.items() if 'error' not in v}
        
        if not valid_results:
            print("⚠️ No hay resultados válidos para visualizar")
            return
        
        model_names = list(valid_results.keys())
        metrics = ['mae', 'rmse', 'mape']
        metric_labels = ['MAE', 'RMSE', 'MAPE (%)']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # MAE
        mae_values = [valid_results[name]['mae'] for name in model_names]
        axes[0, 0].bar(model_names, mae_values, color='skyblue')
        axes[0, 0].set_title('Mean Absolute Error (MAE)')
        axes[0, 0].set_ylabel('MAE')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # RMSE
        rmse_values = [valid_results[name]['rmse'] for name in model_names]
        axes[0, 1].bar(model_names, rmse_values, color='lightcoral')
        axes[0, 1].set_title('Root Mean Square Error (RMSE)')
        axes[0, 1].set_ylabel('RMSE')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # MAPE
        mape_values = [valid_results[name]['mape'] for name in model_names]
        axes[1, 0].bar(model_names, mape_values, color='lightgreen')
        axes[1, 0].set_title('Mean Absolute Percentage Error (MAPE)')
        axes[1, 0].set_ylabel('MAPE (%)')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Tiempo de evaluación
        time_values = [valid_results[name]['evaluation_time'] for name in model_names]
        axes[1, 1].bar(model_names, time_values, color='gold')
        axes[1, 1].set_title('Evaluation Time')
        axes[1, 1].set_ylabel('Time (seconds)')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()
    
    def plot_predictions(self, model_name: str,
                       X_test: np.ndarray, y_test: np.ndarray,
                       num_samples: int = 100):
        """
        Visualiza predicciones de un modelo
        
        Args:
            model_name: Nombre del modelo
            X_test, y_test: Datos de prueba
            num_samples: Número de muestras a visualizar
        """
        print(f"📊 Visualizando predicciones de {model_name}...")
        
        if model_name not in self.models:
            print(f"⚠️ Modelo {model_name} no encontrado")
            return
        
        model = self.models[model_name]
        predictions = model.predict(X_test[:num_samples], verbose=0)
        
        plt.figure(figsize=(12, 6))
        
        # Valores reales vs predichos
        x_axis = range(num_samples)
        plt.plot(x_axis, y_test[:num_samples], 'b-', label='Real', alpha=0.7)
        plt.plot(x_axis, predictions, 'r-', label='Predicción', alpha=0.7)
        
        plt.title(f'Predicciones del Modelo {model_name}')
        plt.xlabel('Muestra')
        plt.ylabel('Valor')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def save_models(self, directory: str = 'manual_models'):
        """
        Guarda todos los modelos entrenados
        
        Args:
            directory: Directorio donde guardar los modelos
        """
        import os
        
        print(f"💾 Guardando modelos en {directory}/...")
        
        os.makedirs(directory, exist_ok=True)
        
        for model_name, model in self.models.items():
            filepath = os.path.join(directory, f'{model_name}.h5')
            model.save(filepath)
            print(f"   - {model_name} guardado en {filepath}")
        
        print("✅ Modelos guardados exitosamente")


def main():
    """
    Función principal para probar modelos manuales
    """
    print("🚀 Iniciando prueba de modelos manuales")
    
    # Generar datos de prueba
    from generate_time_series import TimeSeriesGenerator
    
    # Crear datos sintéticos
    generator = TimeSeriesGenerator(num_samples=5000, start_date='2020-01-01')
    dataset = generator.generate_complete_dataset()
    
    # Usar secuencias de 6 horas
    X = dataset['sequences']['6h']['X']
    y = dataset['sequences']['6h']['y']
    
    # Para regresión, usar solo el primer valor de y
    if len(y.shape) > 2:
        y = y[:, 0, 0]  # Primera hora de predicción, primera feature
    elif len(y.shape) > 1:
        y = y[:, 0]  # Primera hora de predicción
    
    # Dividir datos
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Validación
    val_idx = int(len(X_train) * 0.8)
    X_val = X_train[val_idx:]
    y_val = y_train[val_idx:]
    X_train = X_train[:val_idx]
    y_train = y_train[:val_idx]
    
    # Crear modelos manuales
    manual_models = ManualTimeSeriesModels(input_shape=X_train.shape[1:])
    
    # Crear diferentes modelos
    manual_models.create_lstm_model()
    manual_models.create_gru_model()
    manual_models.create_cnn_model()
    manual_models.create_bidirectional_lstm_model()
    manual_models.create_attention_model()
    
    # Entrenar todos los modelos
    training_results = manual_models.train_all_models(
        X_train, y_train, X_val, y_val, epochs=20
    )
    
    # Evaluar todos los modelos
    evaluation_results = manual_models.evaluate_all_models(X_test, y_test)
    
    # Comparar modelos
    comparison_df = manual_models.compare_models(evaluation_results)
    
    # Visualizar resultados
    manual_models.plot_model_comparison(evaluation_results)
    
    # Visualizar predicciones del mejor modelo
    best_model = comparison_df.iloc[0]['Model']
    manual_models.plot_predictions(best_model, X_test, y_test)
    
    # Guardar modelos
    manual_models.save_models()
    
    print("✅ Prueba de modelos manuales completada exitosamente")


if __name__ == "__main__":
    main()
