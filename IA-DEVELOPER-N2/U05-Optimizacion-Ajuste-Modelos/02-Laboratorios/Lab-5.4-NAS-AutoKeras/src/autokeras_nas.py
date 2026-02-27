"""
Laboratorio 5.4 - Neural Architecture Search con AutoKeras
==========================================================

Este script implementa Neural Architecture Search (NAS) usando
AutoKeras para encontrar automáticamente la mejor arquitectura.
"""

import tensorflow as tf
import autokeras as ak
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Any, Tuple, List
import time
import json
from sklearn.model_selection import train_test_split

class AutoKerasNAS:
    """
    Clase para implementar Neural Architecture Search con AutoKeras
    """
    
    def __init__(self, max_trials: int = 100, 
                 objective: str = 'val_loss',
                 directory: str = 'nas_results'):
        """
        Inicializa el NAS con AutoKeras
        
        Args:
            max_trials: Número máximo de trials (arquitecturas a probar)
            objective: Objetivo de optimización
            directory: Directorio para guardar resultados
        """
        self.max_trials = max_trials
        self.objective = objective
        self.directory = directory
        self.model = None
        self.search_history = []
        self.best_model = None
        
    def prepare_data(self, X: np.ndarray, y: np.ndarray,
                    test_size: float = 0.2,
                    validation_split: float = 0.2) -> Dict[str, np.ndarray]:
        """
        Prepara los datos para NAS
        
        Args:
            X: Features de entrada
            y: Labels de salida
            test_size: Proporción de datos de prueba
            validation_split: Proporción de validación
            
        Returns:
            Diccionario con datasets divididos
        """
        print("📊 Preparando datos para NAS...")
        
        # Asegurar shapes correctos
        if len(X.shape) == 2:
            # Para series temporales, agregar dimensión de features
            X = X.reshape(X.shape[0], X.shape[1], 1)
        
        # Dividir datos
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        # Dividir entrenamiento en train/validation
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=validation_split, random_state=42
        )
        
        data_splits = {
            'X_train': X_train,
            'X_val': X_val,
            'X_test': X_test,
            'y_train': y_train,
            'y_val': y_val,
            'y_test': y_test
        }
        
        print(f"✅ Datos preparados:")
        print(f"   - Train: {X_train.shape}")
        print(f"   - Validation: {X_val.shape}")
        print(f"   - Test: {X_test.shape}")
        
        return data_splits
    
    def create_text_regression_model(self) -> ak.StructuredDataRegressor:
        """
        Crea modelo para datos estructurados (regresión)
        
        Returns:
            Modelo AutoKeras para regresión
        """
        print("🏗️ Creando modelo de regresión estructurada...")
        
        # Definir input
        input_node = ak.StructuredDataInput()
        
        # Bloque de procesamiento
        processed = ak.StructuredDataBlock()(input_node)
        
        # Bloque de regresión
        output_node = ak.RegressionHead()(processed)
        
        # Crear modelo
        model = ak.StructuredDataRegressor(
            inputs=input_node,
            outputs=output_node,
            max_trials=self.max_trials,
            objective=self.objective,
            directory=self.directory,
            overwrite=True,
            seed=42
        )
        
        return model
    
    def create_time_series_model(self, lookback: int = 24,
                               forecast: int = 6) -> ak.TimeseriesForecaster:
        """
        Crea modelo para series temporales
        
        Args:
            lookback: Ventana de tiempo para lookback
            forecast: Horizonte de predicción
            
        Returns:
            Modelo AutoKeras para series temporales
        """
        print(f"🕒 Creando modelo de series temporales (lookback={lookback}, forecast={forecast})...")
        
        # Crear modelo de series temporales
        model = ak.TimeseriesForecaster(
            lookback=lookback,
            forecast=forecast,
            max_trials=self.max_trials,
            objective=self.objective,
            directory=self.directory,
            overwrite=True,
            seed=42
        )
        
        return model
    
    def create_image_model(self, input_shape: Tuple[int, int, int]) -> ak.ImageClassifier:
        """
        Crea modelo para datos de imágenes
        
        Args:
            input_shape: Forma de entrada de imágenes
            
        Returns:
            Modelo AutoKeras para clasificación de imágenes
        """
        print(f"🖼️ Creando modelo de imágenes (shape={input_shape})...")
        
        # Crear modelo de imágenes
        model = ak.ImageClassifier(
            input_shape=input_shape,
            max_trials=self.max_trials,
            objective=self.objective,
            directory=self.directory,
            overwrite=True,
            seed=42
        )
        
        return model
    
    def create_text_model(self, max_tokens: int = 10000) -> ak.TextClassifier:
        """
        Crea modelo para datos de texto
        
        Args:
            max_tokens: Número máximo de tokens
            
        Returns:
            Modelo AutoKeras para clasificación de texto
        """
        print(f"📝 Creando modelo de texto (max_tokens={max_tokens})...")
        
        # Crear modelo de texto
        model = ak.TextClassifier(
            max_tokens=max_tokens,
            max_trials=self.max_trials,
            objective=self.objective,
            directory=self.directory,
            overwrite=True,
            seed=42
        )
        
        return model
    
    def create_hybrid_model(self, input_shapes: Dict[str, Any]) -> ak.AutoModel:
        """
        Crea modelo híbrido que combina múltiples tipos de datos
        
        Args:
            input_shapes: Diccionario con formas de entrada por tipo
            
        Returns:
            Modelo AutoKeras híbrido
        """
        print("🔀 Creando modelo híbrido...")
        
        # Definir inputs según tipos disponibles
        inputs = []
        
        if 'image' in input_shapes:
            img_input = ak.ImageInput(shape=input_shapes['image'])
            img_processed = ak.ImageBlock()(img_input)
            inputs.append(img_processed)
        
        if 'structured' in input_shapes:
            struct_input = ak.StructuredDataInput()
            struct_processed = ak.StructuredDataBlock()(struct_input)
            inputs.append(struct_processed)
        
        if 'text' in input_shapes:
            text_input = ak.TextInput()
            text_processed = ak.TextBlock()(text_input)
            inputs.append(text_processed)
        
        if 'timeseries' in input_shapes:
            ts_input = ak.TimeseriesInput()
            ts_processed = ak.TimeseriesBlock()(ts_input)
            inputs.append(ts_processed)
        
        # Combinar outputs
        if len(inputs) == 1:
            merged = inputs[0]
        else:
            merged = ak.Merge()(inputs)
        
        # Agregar capas densas
        dense = ak.DenseBlock()(merged)
        output = ak.RegressionHead()(dense)
        
        # Crear modelo
        model = ak.AutoModel(
            inputs=inputs,
            outputs=output,
            max_trials=self.max_trials,
            objective=self.objective,
            directory=self.directory,
            overwrite=True,
            seed=42
        )
        
        return model
    
    def search_architecture(self, model_type: str,
                           data_splits: Dict[str, np.ndarray],
                           **kwargs) -> Dict[str, Any]:
        """
        Ejecuta la búsqueda de arquitectura
        
        Args:
            model_type: Tipo de modelo ('timeseries', 'image', 'text', 'structured', 'hybrid')
            data_splits: Datos divididos
            **kwargs: Argumentos adicionales según tipo de modelo
            
        Returns:
            Resultados de la búsqueda
        """
        print(f"🔍 Iniciando búsqueda de arquitectura para modelo: {model_type}")
        
        start_time = time.time()
        
        # Crear modelo según tipo
        if model_type == 'timeseries':
            model = self.create_time_series_model(**kwargs)
            X_train = data_splits['X_train']
            y_train = data_splits['y_train']
            X_val = data_splits['X_val']
            y_val = data_splits['y_val']
            
        elif model_type == 'image':
            model = self.create_image_model(**kwargs)
            X_train = data_splits['X_train']
            y_train = data_splits['y_train']
            X_val = data_splits['X_val']
            y_val = data_splits['y_val']
            
        elif model_type == 'text':
            model = self.create_text_model(**kwargs)
            X_train = data_splits['X_train']
            y_train = data_splits['y_train']
            X_val = data_splits['X_val']
            y_val = data_splits['y_val']
            
        elif model_type == 'structured':
            model = self.create_text_regression_model()
            # Para datos estructurados, necesitamos formato diferente
            X_train = data_splits['X_train']
            y_train = data_splits['y_train']
            X_val = data_splits['X_val']
            y_val = data_splits['y_val']
            
        elif model_type == 'hybrid':
            model = self.create_hybrid_model(**kwargs)
            X_train = data_splits['X_train']
            y_train = data_splits['y_train']
            X_val = data_splits['X_val']
            y_val = data_splits['y_val']
            
        else:
            raise ValueError(f"Tipo de modelo no soportado: {model_type}")
        
        # Callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7
            )
        ]
        
        # Ejecutar búsqueda
        print(f"🚀 Ejecutando {self.max_trials} trials...")
        
        history = model.fit(
            x=X_train,
            y=y_train,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=1
        )
        
        search_time = time.time() - start_time
        
        # Evaluar mejor modelo
        print("📊 Evaluando mejor modelo...")
        test_metrics = model.evaluate(data_splits['X_test'], data_splits['y_test'])
        
        # Exportar mejor modelo
        best_model = model.export_model()
        
        # Obtener resumen de la búsqueda
        search_summary = self._get_search_summary(model)
        
        results = {
            'model_type': model_type,
            'best_model': best_model,
            'search_time': search_time,
            'test_metrics': test_metrics,
            'search_summary': search_summary,
            'training_history': history.history if hasattr(history, 'history') else None
        }
        
        print(f"✅ Búsqueda completada en {search_time:.2f} segundos")
        print(f"   - Mejor métrica: {test_metrics[1]:.4f}")
        print(f"   - Trials ejecutados: {search_summary.get('total_trials', 'N/A')}")
        
        self.model = model
        self.best_model = best_model
        
        return results
    
    def _get_search_summary(self, model) -> Dict[str, Any]:
        """
        Obtiene resumen de la búsqueda de arquitectura
        
        Args:
            model: Modelo AutoKeras entrenado
            
        Returns:
            Resumen de la búsqueda
        """
        try:
            # Intentar obtener información del tuner
            tuner = model.tuner
            
            summary = {
                'total_trials': len(tuner.oracle.get_best_trials(num_trials=self.max_trials)),
                'best_trial_score': tuner.oracle.get_best_trials(num_trials=1)[0].score,
                'search_space_size': len(tuner.oracle.hyperparameters.space)
            }
            
            # Obtener información de los mejores trials
            best_trials = tuner.oracle.get_best_trials(num_trials=5)
            summary['top_5_trials'] = [
                {
                    'trial_id': trial.trial_id,
                    'score': trial.score,
                    'hyperparameters': trial.hyperparameters.values
                }
                for trial in best_trials
            ]
            
        except Exception as e:
            print(f"⚠️ No se pudo obtener resumen detallado: {e}")
            summary = {'status': 'summary_not_available'}
        
        return summary
    
    def compare_with_manual_models(self, data_splits: Dict[str, np.ndarray],
                                 results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compara el modelo encontrado con modelos manuales
        
        Args:
            data_splits: Datos divididos
            results: Resultados del NAS
            
        Returns:
            Comparación con modelos manuales
        """
        print("🔬 Comparando con modelos manuales...")
        
        manual_results = {}
        
        # Modelo LSTM manual
        lstm_result = self._create_manual_lstm(data_splits)
        manual_results['lstm_manual'] = lstm_result
        
        # Modelo CNN manual
        cnn_result = self._create_manual_cnn(data_splits)
        manual_results['cnn_manual'] = cnn_result
        
        # Modelo Dense simple
        dense_result = self._create_manual_dense(data_splits)
        manual_results['dense_manual'] = dense_result
        
        # Crear comparación
        comparison = {
            'nas_model': {
                'test_loss': results['test_metrics'][0],
                'test_metric': results['test_metrics'][1],
                'search_time': results['search_time']
            },
            'manual_models': manual_results,
            'improvement': {
                'vs_lstm': ((manual_results['lstm_manual']['test_metric'] - results['test_metrics'][1]) / manual_results['lstm_manual']['test_metric']) * 100,
                'vs_cnn': ((manual_results['cnn_manual']['test_metric'] - results['test_metrics'][1]) / manual_results['cnn_manual']['test_metric']) * 100,
                'vs_dense': ((manual_results['dense_manual']['test_metric'] - results['test_metrics'][1]) / manual_results['dense_manual']['test_metric']) * 100
            }
        }
        
        print("✅ Comparación completada:")
        for model_name, improvement in comparison['improvement'].items():
            print(f"   - NAS vs {model_name}: {improvement:.2f}% de mejora")
        
        return comparison
    
    def _create_manual_lstm(self, data_splits: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Crea y entrena modelo LSTM manual"""
        print("🔧 Creando modelo LSTM manual...")
        
        model = tf.keras.Sequential([
            tf.keras.layers.LSTM(64, return_sequences=True, input_shape=data_splits['X_train'].shape[1:]),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.LSTM(32),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(data_splits['y_train'].shape[1] if len(data_splits['y_train'].shape) > 1 else 1)
        ])
        
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        
        # Entrenar
        history = model.fit(
            data_splits['X_train'], data_splits['y_train'],
            validation_data=(data_splits['X_val'], data_splits['y_val']),
            epochs=50, batch_size=32, verbose=0
        )
        
        # Evaluar
        test_metrics = model.evaluate(data_splits['X_test'], data_splits['y_test'], verbose=0)
        
        return {'test_loss': test_metrics[0], 'test_metric': -test_metrics[1]}  # Negativo para MSE
    
    def _create_manual_cnn(self, data_splits: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Crea y entrena modelo CNN manual"""
        print("🔧 Creando modelo CNN manual...")
        
        model = tf.keras.Sequential([
            tf.keras.layers.Conv1D(64, 3, activation='relu', input_shape=data_splits['X_train'].shape[1:]),
            tf.keras.layers.MaxPooling1D(2),
            tf.keras.layers.Conv1D(32, 3, activation='relu'),
            tf.keras.layers.MaxPooling1D(2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(data_splits['y_train'].shape[1] if len(data_splits['y_train'].shape) > 1 else 1)
        ])
        
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        
        # Entrenar
        history = model.fit(
            data_splits['X_train'], data_splits['y_train'],
            validation_data=(data_splits['X_val'], data_splits['y_val']),
            epochs=50, batch_size=32, verbose=0
        )
        
        # Evaluar
        test_metrics = model.evaluate(data_splits['X_test'], data_splits['y_test'], verbose=0)
        
        return {'test_loss': test_metrics[0], 'test_metric': -test_metrics[1]}
    
    def _create_manual_dense(self, data_splits: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Crea y entrena modelo Dense manual"""
        print("🔧 Creando modelo Dense manual...")
        
        # Aplanar datos si son secuenciales
        X_train_flat = data_splits['X_train'].reshape(data_splits['X_train'].shape[0], -1)
        X_val_flat = data_splits['X_val'].reshape(data_splits['X_val'].shape[0], -1)
        X_test_flat = data_splits['X_test'].reshape(data_splits['X_test'].shape[0], -1)
        
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu', input_shape=(X_train_flat.shape[1],)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(data_splits['y_train'].shape[1] if len(data_splits['y_train'].shape) > 1 else 1)
        ])
        
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        
        # Entrenar
        history = model.fit(
            X_train_flat, data_splits['y_train'],
            validation_data=(X_val_flat, data_splits['y_val']),
            epochs=50, batch_size=32, verbose=0
        )
        
        # Evaluar
        test_metrics = model.evaluate(X_test_flat, data_splits['y_test'], verbose=0)
        
        return {'test_loss': test_metrics[0], 'test_metric': -test_metrics[1]}
    
    def analyze_best_architecture(self) -> Dict[str, Any]:
        """
        Analiza la arquitectura del mejor modelo encontrado
        
        Returns:
            Análisis detallado de la arquitectura
        """
        print("🔍 Analizando mejor arquitectura encontrada...")
        
        if self.best_model is None:
            print("⚠️ No hay modelo para analizar")
            return {}
        
        # Obtener summary del modelo
        model_summary = []
        self.best_model.summary(print_fn=lambda x: model_summary.append(x))
        
        # Contar parámetros
        total_params = self.best_model.count_params()
        
        # Analizar capas
        layer_types = {}
        for layer in self.best_model.layers:
            layer_type = type(layer).__name__
            layer_types[layer_type] = layer_types.get(layer_type, 0) + 1
        
        # Calcular complejidad estimada
        complexity_score = self._calculate_complexity_score(self.best_model)
        
        analysis = {
            'total_parameters': total_params,
            'layer_types': layer_types,
            'complexity_score': complexity_score,
            'model_summary': '\n'.join(model_summary),
            'num_layers': len(self.best_model.layers)
        }
        
        print("✅ Análisis completado:")
        print(f"   - Parámetros totales: {total_params:,}")
        print(f"   - Número de capas: {analysis['num_layers']}")
        print(f"   - Score de complejidad: {complexity_score:.2f}")
        
        return analysis
    
    def _calculate_complexity_score(self, model: tf.keras.Model) -> float:
        """
        Calcula un score de complejidad para el modelo
        
        Args:
            model: Modelo a analizar
            
        Returns:
            Score de complejidad (0-100)
        """
        # Factores de complejidad
        params_factor = min(model.count_params() / 1_000_000, 1.0) * 40  # 40 puntos max
        layers_factor = min(len(model.layers) / 50, 1.0) * 30  # 30 puntos max
        
        # Tipos de capa complejos
        complex_layers = ['LSTM', 'GRU', 'Conv', 'Attention', 'MultiHeadAttention']
        complex_count = sum(1 for layer in model.layers 
                          if any(complex in type(layer).__name__ for complex in complex_layers))
        complexity_factor = min(complex_count / 10, 1.0) * 30  # 30 puntos max
        
        return params_factor + layers_factor + complexity_factor
    
    def visualize_search_results(self, results: Dict[str, Any],
                                comparison: Dict[str, Any]):
        """
        Visualiza resultados de la búsqueda
        
        Args:
            results: Resultados del NAS
            comparison: Comparación con modelos manuales
        """
        print("📊 Visualizando resultados...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Comparación de métricas
        models = ['NAS'] + list(comparison['manual_models'].keys())
        metrics = [results['test_metrics'][1]] + [comparison['manual_models'][model]['test_metric'] 
                                                 for model in comparison['manual_models'].keys()]
        
        bars = axes[0, 0].bar(models, metrics, color=['green', 'blue', 'orange', 'red'])
        axes[0, 0].set_title('Comparación de Métricas')
        axes[0, 0].set_ylabel('Métrica (menor es mejor)')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Añadir valores en barras
        for bar, metric in zip(bars, metrics):
            axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                           f'{metric:.4f}', ha='center', va='bottom')
        
        # 2. Tiempo de búsqueda vs entrenamiento manual
        search_time = results['search_time']
        manual_times = [300, 300, 300]  # Estimado 5 minutos por modelo manual
        
        times = [search_time] + manual_times
        time_labels = ['NAS Search'] + [f'{model} Train' for model in comparison['manual_models'].keys()]
        
        axes[0, 1].bar(time_labels, times, color=['green', 'blue', 'orange', 'red'])
        axes[0, 1].set_title('Tiempo de Entrenamiento/Búsqueda')
        axes[0, 1].set_ylabel('Tiempo (segundos)')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # 3. Mejora porcentual
        improvements = list(comparison['improvement'].values())
        improvement_labels = [key.replace('vs_', '') for key in comparison['improvement'].keys()]
        
        colors = ['green' if imp > 0 else 'red' for imp in improvements]
        axes[1, 0].bar(improvement_labels, improvements, color=colors, alpha=0.7)
        axes[1, 0].set_title('Mejora de NAS vs Modelos Manuales')
        axes[1, 0].set_ylabel('Mejora (%)')
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # 4. Resumen de búsqueda (si disponible)
        if 'search_summary' in results and 'top_5_trials' in results['search_summary']:
            top_trials = results['search_summary']['top_5_trials'][:5]
            trial_ids = [f"Trial {i+1}" for i in range(len(top_trials))]
            trial_scores = [trial['score'] for trial in top_trials]
            
            axes[1, 1].plot(trial_ids, trial_scores, 'o-', color='purple')
            axes[1, 1].set_title('Top 5 Trials')
            axes[1, 1].set_ylabel('Score')
            axes[1, 1].tick_params(axis='x', rotation=45)
            axes[1, 1].grid(True, alpha=0.3)
        else:
            axes[1, 1].text(0.5, 0.5, 'Resumen de búsqueda\nno disponible', 
                           ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('Resumen de Búsqueda')
        
        plt.tight_layout()
        plt.show()
    
    def save_results(self, results: Dict[str, Any],
                    comparison: Dict[str, Any],
                    analysis: Dict[str, Any],
                    filepath: str = 'nas_results.json'):
        """
        Guarda resultados completos
        
        Args:
            results: Resultados del NAS
            comparison: Comparación con modelos manuales
            analysis: Análisis de arquitectura
            filepath: Ruta del archivo JSON
        """
        print(f"💾 Guardando resultados en {filepath}...")
        
        complete_results = {
            'nas_results': results,
            'comparison': comparison,
            'architecture_analysis': analysis,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Convertir objetos no serializables
        def make_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: make_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [make_serializable(item) for item in obj]
            else:
                return str(obj)
        
        serializable_results = make_serializable(complete_results)
        
        with open(filepath, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print("✅ Resultados guardados exitosamente")


def main():
    """
    Función principal para probar AutoKeras NAS
    """
    print("🚀 Iniciando Neural Architecture Search con AutoKeras")
    
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
    
    # Crear NAS
    nas = AutoKerasNAS(max_trials=20, objective='val_loss')  # Reducido para prueba
    
    # Preparar datos
    data_splits = nas.prepare_data(X, y)
    
    # Ejecutar búsqueda
    results = nas.search_architecture(
        model_type='timeseries',
        data_splits=data_splits,
        lookback=24,
        forecast=6
    )
    
    # Comparar con modelos manuales
    comparison = nas.compare_with_manual_models(data_splits, results)
    
    # Analizar arquitectura
    analysis = nas.analyze_best_architecture()
    
    # Visualizar resultados
    nas.visualize_search_results(results, comparison)
    
    # Guardar resultados
    nas.save_results(results, comparison, analysis)
    
    print("✅ Neural Architecture Search completado exitosamente")


if __name__ == "__main__":
    main()
