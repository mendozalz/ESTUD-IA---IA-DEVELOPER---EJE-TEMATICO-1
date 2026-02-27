"""
Laboratorio 5.3 - Quantization Optimization
===========================================

Este script implementa técnicas de quantization para optimizar
el modelo para inferencia en dispositivos edge.
"""

import tensorflow as tf
import tensorflow_model_optimization as tfmot
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, Tuple, List
import time
import tempfile
import os

class QuantizationOptimizer:
    """
    Clase para aplicar técnicas de quantization al modelo
    """
    
    def __init__(self, base_model: tf.keras.Model):
        """
        Inicializa el optimizador de quantization
        
        Args:
            base_model: Modelo base a optimizar
        """
        self.base_model = base_model
        self.quantized_model = None
        self.tflite_model = None
        self.quantization_history = None
        
    def apply_quantization_aware_training(self, train_dataset: tf.data.Dataset,
                                        val_dataset: tf.data.Dataset,
                                        epochs: int = 5) -> tf.keras.Model:
        """
        Aplica quantization-aware training
        
        Args:
            train_dataset: Dataset de entrenamiento
            val_dataset: Dataset de validación
            epochs: Número de épocas de QAT
            
        Returns:
            Modelo con quantization-aware training
        """
        print("🎯 Aplicando Quantization-Aware Training (QAT)...")
        
        # Crear modelo QAT
        def apply_quantization(layer):
            if isinstance(layer, tf.keras.layers.Dense) or isinstance(layer, tf.keras.layers.Conv2D):
                return tfmot.quantization.keras.quantize_annotate_model(layer)
            return layer
        
        # Anotar modelo para quantization
        annotated_model = tf.keras.models.clone_model(
            self.base_model,
            clone_function=apply_quantization
        )
        
        # Aplicar quantization-aware training
        self.quantized_model = tfmot.quantization.keras.quantize_model(annotated_model)
        
        # Compilar modelo
        self.quantized_model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print("✅ Modelo QAT creado:")
        self.quantized_model.summary()
        
        # Entrenar con QAT
        print("🎓 Entrenando con Quantization-Aware Training...")
        
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=3,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=2,
                min_lr=1e-6
            )
        ]
        
        start_time = time.time()
        history = self.quantized_model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )
        training_time = time.time() - start_time
        
        self.quantization_history = history.history
        
        print(f"✅ QAT completado en {training_time:.2f} segundos")
        
        return self.quantized_model
    
    def convert_to_tflite(self, representative_dataset: tf.data.Dataset,
                          quantization_type: str = 'dynamic') -> bytes:
        """
        Convierte el modelo a TensorFlow Lite
        
        Args:
            representative_dataset: Dataset representativo para calibración
            quantization_type: Tipo de quantization ('dynamic', 'int8', 'float16')
            
        Returns:
            Modelo TFLite en bytes
        """
        print(f"🔄 Convirtiendo a TFLite (quantization: {quantization_type})...")
        
        # Usar modelo cuantizado si existe, sino el base
        model_to_convert = self.quantized_model if self.quantized_model is not None else self.base_model
        
        # Preparar representative dataset para int8 quantization
        def representative_data_gen():
            for images, _ in representative_dataset.take(100):
                yield [images.numpy()]
        
        # Converter options según tipo
        converter = tf.lite.TFLiteConverter.from_keras_model(model_to_convert)
        
        if quantization_type == 'dynamic':
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
        elif quantization_type == 'int8':
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.representative_dataset = representative_data_gen
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.inference_input_type = tf.uint8
            converter.inference_output_type = tf.uint8
        elif quantization_type == 'float16':
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_types = [tf.float16]
        
        # Convertir
        start_time = time.time()
        self.tflite_model = converter.convert()
        conversion_time = time.time() - start_time
        
        print(f"✅ Conversión a TFLite completada en {conversion_time:.2f} segundos")
        
        return self.tflite_model
    
    def save_tflite_model(self, filepath: str, quantization_type: str = 'dynamic'):
        """
        Guarda el modelo TFLite
        
        Args:
            filepath: Ruta donde guardar el modelo
            quantization_type: Tipo de quantization para el nombre
        """
        if self.tflite_model is None:
            print("⚠️ No hay modelo TFLite para guardar")
            return
        
        full_path = f"{filepath}_{quantization_type}.tflite"
        print(f"💾 Guardando modelo TFLite en {full_path}...")
        
        with open(full_path, 'wb') as f:
            f.write(self.tflite_model)
        
        # Obtener tamaño
        size_mb = len(self.tflite_model) / (1024 * 1024)
        print(f"✅ Modelo guardado: {size_mb:.2f} MB")
    
    def evaluate_tflite_model(self, test_dataset: tf.data.Dataset,
                             quantization_type: str = 'dynamic') -> Dict[str, Any]:
        """
        Evalúa el modelo TFLite
        
        Args:
            test_dataset: Dataset de prueba
            quantization_type: Tipo de quantization
            
        Returns:
            Diccionario con métricas de evaluación
        """
        print(f"🧪 Evaluando modelo TFLite ({quantization_type})...")
        
        if self.tflite_model is None:
            print("⚠️ No hay modelo TFLite para evaluar")
            return {}
        
        # Cargar intérprete TFLite
        interpreter = tf.lite.Interpreter(model_content=self.tflite_model)
        interpreter.allocate_tensors()
        
        # Obtener detalles de input/output
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        # Evaluar
        correct_predictions = 0
        total_samples = 0
        all_predictions = []
        all_labels = []
        
        for batch_images, batch_labels in test_dataset:
            batch_size = batch_images.shape[0]
            
            for i in range(batch_size):
                # Preparar input
                input_data = np.expand_dims(batch_images[i], axis=0).astype(np.float32)
                
                # Ajustar input según tipo de quantization
                if quantization_type == 'int8':
                    input_scale, input_zero_point = input_details[0]['quantization']
                    input_data = input_data / input_scale + input_zero_point
                    input_data = input_data.astype(np.uint8)
                
                # Ejecutar inferencia
                interpreter.set_tensor(input_details[0]['index'], input_data)
                interpreter.invoke()
                
                # Obtener output
                output_data = interpreter.get_tensor(output_details[0]['index'])
                
                # Procesar output según tipo de quantization
                if quantization_type == 'int8':
                    output_scale, output_zero_point = output_details[0]['quantization']
                    output_data = output_data.astype(np.float32)
                    output_data = (output_data - output_zero_point) * output_scale
                
                # Calcular predicción
                predicted_class = np.argmax(output_data[0])
                true_class = np.argmax(batch_labels[i])
                
                if predicted_class == true_class:
                    correct_predictions += 1
                
                total_samples += 1
                all_predictions.append(output_data[0])
                all_labels.append(batch_labels[i])
        
        # Calcular métricas
        accuracy = correct_predictions / total_samples
        
        results = {
            'accuracy': accuracy,
            'correct_predictions': correct_predictions,
            'total_samples': total_samples,
            'quantization_type': quantization_type,
            'model_size_mb': len(self.tflite_model) / (1024 * 1024)
        }
        
        print(f"✅ Evaluación TFLite completada:")
        print(f"   - Accuracy: {accuracy:.4f}")
        print(f"   - Correctas: {correct_predictions}/{total_samples}")
        print(f"   - Tamaño: {results['model_size_mb']:.2f} MB")
        
        return results
    
    def compare_quantization_methods(self, test_dataset: tf.data.Dataset,
                                   representative_dataset: tf.data.Dataset) -> Dict[str, Any]:
        """
        Compara diferentes métodos de quantization
        
        Args:
            test_dataset: Dataset de prueba
            representative_dataset: Dataset representativo
            
        Returns:
            Diccionario con comparación de métodos
        """
        print("📊 Comparando métodos de quantization...")
        
        # Evaluar modelo base
        print("🔍 Evaluando modelo base...")
        base_metrics = self.base_model.evaluate(test_dataset, verbose=1)
        
        # Evaluar modelo QAT
        qat_metrics = None
        if self.quantized_model is not None:
            print("🔍 Evaluando modelo QAT...")
            qat_metrics = self.quantized_model.evaluate(test_dataset, verbose=1)
        
        # Evaluar diferentes tipos de TFLite
        tflite_results = {}
        quantization_types = ['dynamic', 'float16', 'int8']
        
        for q_type in quantization_types:
            print(f"🔍 Evaluando TFLite {q_type}...")
            self.convert_to_tflite(representative_dataset, q_type)
            tflite_results[q_type] = self.evaluate_tflite_model(test_dataset, q_type)
        
        # Calcular tamaños
        def get_model_size(model):
            if model is None:
                return 0
            with tempfile.NamedTemporaryFile() as tmp:
                model.save(tmp.name, save_format='h5')
                size = tmp.tell()
            return size / (1024 * 1024)  # MB
        
        base_size = get_model_size(self.base_model)
        qat_size = get_model_size(self.quantized_model) if self.quantized_model else 0
        
        comparison = {
            'base_model': {
                'accuracy': base_metrics[1],
                'loss': base_metrics[0],
                'size_mb': base_size
            },
            'qat_model': {
                'accuracy': qat_metrics[1] if qat_metrics else None,
                'loss': qat_metrics[0] if qat_metrics else None,
                'size_mb': qat_size
            },
            'tflite_models': tflite_results,
            'size_reductions': {}
        }
        
        # Calcular reducciones de tamaño
        for q_type, results in tflite_results.items():
            reduction = (base_size - results['model_size_mb']) / base_size * 100
            comparison['size_reductions'][q_type] = reduction
        
        print("✅ Comparación completada:")
        for q_type, reduction in comparison['size_reductions'].items():
            print(f"   - Reducción tamaño {q_type}: {reduction:.1f}%")
        
        return comparison
    
    def benchmark_inference_speed(self, test_dataset: tf.data.Dataset,
                                 num_samples: int = 100) -> Dict[str, Any]:
        """
        Benchmark de velocidad de inferencia
        
        Args:
            test_dataset: Dataset de prueba
            num_samples: Número de muestras para benchmark
            
        Returns:
            Diccionario con resultados de benchmark
        """
        print("⚡ Benchmark de velocidad de inferencia...")
        
        results = {}
        
        # Benchmark modelo base
        print("🔍 Benchmark modelo base...")
        base_times = self._benchmark_model(self.base_model, test_dataset, num_samples)
        results['base_model'] = base_times
        
        # Benchmark modelo QAT
        if self.quantized_model is not None:
            print("🔍 Benchmark modelo QAT...")
            qat_times = self._benchmark_model(self.quantized_model, test_dataset, num_samples)
            results['qat_model'] = qat_times
        
        # Benchmark modelos TFLite
        tflite_types = ['dynamic', 'float16', 'int8']
        for q_type in tflite_types:
            print(f"🔍 Benchmark TFLite {q_type}...")
            # Convertir si no existe
            if self.tflite_model is None or q_type != 'dynamic':
                # Necesitamos representative dataset, usar test_dataset como aproximación
                self.convert_to_tflite(test_dataset, q_type)
            
            tflite_times = self._benchmark_tflite(test_dataset, num_samples)
            results[f'tflite_{q_type}'] = tflite_times
        
        # Calcular speedups
        base_avg_time = results['base_model']['avg_time']
        for model_type, times in results.items():
            if model_type != 'base_model':
                speedup = base_avg_time / times['avg_time']
                results[model_type]['speedup'] = speedup
        
        print("✅ Benchmark completado:")
        for model_type, times in results.items():
            speedup = times.get('speedup', 1.0)
            print(f"   - {model_type}: {times['avg_time']:.4f}s (speedup: {speedup:.2f}x)")
        
        return results
    
    def _benchmark_model(self, model: tf.keras.Model, dataset: tf.data.Dataset,
                        num_samples: int) -> Dict[str, float]:
        """
        Benchmark interno para modelos Keras
        
        Args:
            model: Modelo a evaluar
            dataset: Dataset de prueba
            num_samples: Número de muestras
            
        Returns:
            Diccionario con tiempos
        """
        times = []
        samples_processed = 0
        
        for batch_images, _ in dataset:
            batch_size = batch_images.shape[0]
            
            for i in range(min(batch_size, num_samples - samples_processed)):
                start_time = time.time()
                
                # Inferencia
                input_data = np.expand_dims(batch_images[i], axis=0)
                _ = model.predict(input_data, verbose=0)
                
                end_time = time.time()
                times.append(end_time - start_time)
                
                samples_processed += 1
                if samples_processed >= num_samples:
                    break
            
            if samples_processed >= num_samples:
                break
        
        return {
            'avg_time': np.mean(times),
            'min_time': np.min(times),
            'max_time': np.max(times),
            'std_time': np.std(times)
        }
    
    def _benchmark_tflite(self, dataset: tf.data.Dataset, num_samples: int) -> Dict[str, float]:
        """
        Benchmark interno para modelos TFLite
        
        Args:
            dataset: Dataset de prueba
            num_samples: Número de muestras
            
        Returns:
            Diccionario con tiempos
        """
        if self.tflite_model is None:
            return {}
        
        interpreter = tf.lite.Interpreter(model_content=self.tflite_model)
        interpreter.allocate_tensors()
        
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        times = []
        samples_processed = 0
        
        for batch_images, _ in dataset:
            batch_size = batch_images.shape[0]
            
            for i in range(min(batch_size, num_samples - samples_processed)):
                start_time = time.time()
                
                # Preparar input
                input_data = np.expand_dims(batch_images[i], axis=0).astype(np.float32)
                
                # Ejecutar inferencia
                interpreter.set_tensor(input_details[0]['index'], input_data)
                interpreter.invoke()
                
                # Obtener output
                _ = interpreter.get_tensor(output_details[0]['index'])
                
                end_time = time.time()
                times.append(end_time - start_time)
                
                samples_processed += 1
                if samples_processed >= num_samples:
                    break
            
            if samples_processed >= num_samples:
                break
        
        return {
            'avg_time': np.mean(times),
            'min_time': np.min(times),
            'max_time': np.max(times),
            'std_time': np.std(times)
        }
    
    def plot_quantization_comparison(self, comparison_results: Dict[str, Any]):
        """
        Visualiza comparación de métodos de quantization
        
        Args:
            comparison_results: Resultados de comparación
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Accuracy comparison
        models = ['Base', 'QAT'] + [f'TFLite {k}' for k in comparison_results['tflite_models'].keys()]
        accuracies = [
            comparison_results['base_model']['accuracy'],
            comparison_results['qat_model']['accuracy'] if comparison_results['qat_model']['accuracy'] else 0
        ] + [v['accuracy'] for v in comparison_results['tflite_models'].values()]
        
        axes[0, 0].bar(models, accuracies, color=['blue', 'green', 'orange', 'red', 'purple'])
        axes[0, 0].set_title('Accuracy por Método')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Size comparison
        sizes = [
            comparison_results['base_model']['size_mb'],
            comparison_results['qat_model']['size_mb'] if comparison_results['qat_model']['size_mb'] > 0 else 0
        ] + [v['model_size_mb'] for v in comparison_results['tflite_models'].values()]
        
        axes[0, 1].bar(models, sizes, color=['blue', 'green', 'orange', 'red', 'purple'])
        axes[0, 1].set_title('Tamaño por Método')
        axes[0, 1].set_ylabel('Tamaño (MB)')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Size reduction
        reductions = [0] + list(comparison_results['size_reductions'].values())
        tflite_models = list(comparison_results['size_reductions'].keys())
        
        axes[1, 0].bar(tflite_models, reductions, color=['orange', 'red', 'purple'])
        axes[1, 0].set_title('Reducción de Tamaño (vs Base)')
        axes[1, 0].set_ylabel('Reducción (%)')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Accuracy vs Size scatter
        for i, model in enumerate(models):
            if i < 2:  # Base y QAT
                axes[1, 1].scatter(sizes[i], accuracies[i], s=100, c='blue', label=model)
            else:
                axes[1, 1].scatter(sizes[i], accuracies[i], s=100, c='red', label=model)
        
        axes[1, 1].set_xlabel('Tamaño (MB)')
        axes[1, 1].set_ylabel('Accuracy')
        axes[1, 1].set_title('Accuracy vs Tamaño')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def plot_inference_speed(self, benchmark_results: Dict[str, Any]):
        """
        Visualiza benchmark de velocidad de inferencia
        
        Args:
            benchmark_results: Resultados de benchmark
        """
        models = list(benchmark_results.keys())
        avg_times = [v['avg_time'] for v in benchmark_results.values()]
        speedups = [v.get('speedup', 1.0) for v in benchmark_results.values()]
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Tiempo promedio
        bars = axes[0].bar(models, avg_times, color=['blue', 'green', 'orange', 'red', 'purple'])
        axes[0].set_title('Tiempo Promedio de Inferencia')
        axes[0].set_ylabel('Tiempo (segundos)')
        axes[0].tick_params(axis='x', rotation=45)
        
        # Añadir valores en las barras
        for bar, time_val in zip(bars, avg_times):
            axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0001,
                        f'{time_val:.4f}s', ha='center', va='bottom')
        
        # Speedup
        bars = axes[1].bar(models, speedups, color=['blue', 'green', 'orange', 'red', 'purple'])
        axes[1].set_title('Speedup vs Modelo Base')
        axes[1].set_ylabel('Speedup (x)')
        axes[1].tick_params(axis='x', rotation=45)
        axes[1].axhline(y=1, color='black', linestyle='--', alpha=0.7)
        
        # Añadir valores en las barras
        for bar, speedup in zip(bars, speedups):
            axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                        f'{speedup:.2f}x', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()


def main():
    """
    Función principal para probar quantization
    """
    print("🚀 Iniciando prueba de quantization optimization")
    
    # Cargar modelo base
    from teacher_model import TeacherModel
    from load_cifar10 import CIFAR10DataLoader
    
    # Crear datos
    loader = CIFAR10DataLoader(batch_size=32)
    datasets = loader.create_datasets()
    
    # Crear modelo teacher simplificado
    teacher = TeacherModel()
    teacher.build_model()
    teacher.compile_model()
    
    # Entrenar brevemente
    teacher.train_model(
        datasets['train_dataset'],
        datasets['val_dataset'],
        epochs=2,
        fine_tune_epochs=1
    )
    
    # Aplicar quantization
    quantizer = QuantizationOptimizer(teacher.model)
    quantized_model = quantizer.apply_quantization_aware_training(
        datasets['train_dataset'],
        datasets['val_dataset'],
        epochs=3
    )
    
    # Comparar métodos
    comparison = quantizer.compare_quantization_methods(
        datasets['test_dataset'],
        datasets['train_dataset']
    )
    
    # Visualizar comparación
    quantizer.plot_quantization_comparison(comparison)
    
    # Benchmark de velocidad
    benchmark = quantizer.benchmark_inference_speed(datasets['test_dataset'])
    quantizer.plot_inference_speed(benchmark)
    
    # Guardar modelos
    quantizer.save_tflite_model('quantized_model', 'dynamic')
    quantizer.save_tflite_model('quantized_model', 'int8')
    quantizer.save_tflite_model('quantized_model', 'float16')
    
    print("✅ Prueba de quantization completada exitosamente")


if __name__ == "__main__":
    main()
