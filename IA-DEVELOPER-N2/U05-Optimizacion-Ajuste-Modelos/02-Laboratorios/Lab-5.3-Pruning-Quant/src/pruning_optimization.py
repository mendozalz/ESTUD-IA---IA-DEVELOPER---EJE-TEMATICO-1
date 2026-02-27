"""
Laboratorio 5.3 - Pruning Optimization
=======================================

Este script implementa técnicas de pruning para reducir el tamaño
y complejidad del modelo manteniendo la precisión.
"""

import tensorflow as tf
import tensorflow_model_optimization as tfmot
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, Tuple, List
import time
import tempfile

class PruningOptimizer:
    """
    Clase para aplicar técnicas de pruning al modelo
    """
    
    def __init__(self, base_model: tf.keras.Model):
        """
        Inicializa el optimizador de pruning
        
        Args:
            base_model: Modelo base a optimizar
        """
        self.base_model = base_model
        self.pruned_model = None
        self.pruning_history = None
        
    def apply_structured_pruning(self, target_sparsity: float = 0.5,
                                 begin_step: int = 1000,
                                 end_step: int = 10000) -> tf.keras.Model:
        """
        Aplica pruning estructurado al modelo
        
        Args:
            target_sparsity: Esparsidad objetivo (0.5 = 50% de parámetros eliminados)
            begin_step: Paso inicial de pruning
            end_step: Paso final de pruning
            
        Returns:
            Modelo con pruning aplicado
        """
        print(f"✂️ Aplicando pruning estructurado (target_sparsity={target_sparsity})...")
        
        # Crear modelo con pruning
        pruning_params = {
            'pruning_schedule': tfmot.sparsity.keras.ConstantSparsity(
                target_sparsity=target_sparsity,
                begin_step=begin_step,
                end_step=end_step
            ),
            'pruning_policy': tfmot.sparsity.keras.PolynomialDecay(
                initial_sparsity=0.0,
                final_sparsity=target_sparsity,
                begin_step=begin_step,
                end_step=end_step,
                frequency=100
            )
        }
        
        # Aplicar pruning solo a capas convolucionales y densas
        def apply_pruning_to_dense(layer):
            if isinstance(layer, tf.keras.layers.Dense) or isinstance(layer, tf.keras.layers.Conv2D):
                return tfmot.sparsity.keras.prune_low_magnitude(layer, **pruning_params)
            return layer
        
        self.pruned_model = tf.keras.models.clone_model(
            self.base_model,
            clone_function=apply_pruning_to_dense
        )
        
        # Recompilar modelo
        self.pruned_model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print(f"✅ Modelo con pruning creado:")
        self.pruned_model.summary()
        
        return self.pruned_model
    
    def fine_tune_pruned_model(self, train_dataset: tf.data.Dataset,
                              val_dataset: tf.data.Dataset,
                              epochs: int = 10) -> Dict[str, Any]:
        """
        Fine-tuning del modelo con pruning
        
        Args:
            train_dataset: Dataset de entrenamiento
            val_dataset: Dataset de validación
            epochs: Número de épocas de fine-tuning
            
        Returns:
            Diccionario con resultados del fine-tuning
        """
        print("🎓 Fine-tuning del modelo con pruning...")
        
        # Callbacks específicos para pruning
        callbacks = [
            tfmot.sparsity.keras.UpdatePruningStep(),
            tfmot.sparsity.keras.PruningSummaries(log_dir='pruning_logs'),
            tf.keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=5,
                restore_best_weights=True
            )
        ]
        
        # Fine-tuning
        start_time = time.time()
        history = self.pruned_model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )
        fine_tune_time = time.time() - start_time
        
        self.pruning_history = history.history
        
        print(f"✅ Fine-tuning completado en {fine_tune_time:.2f} segundos")
        
        return {
            'history': history.history,
            'fine_tune_time': fine_tune_time
        }
    
    def strip_pruning(self) -> tf.keras.Model:
        """
        Remueve los wrappers de pruning para obtener el modelo final
        
        Returns:
            Modelo optimizado sin wrappers de pruning
        """
        print("🔧 Removiendo wrappers de pruning...")
        
        if self.pruned_model is None:
            print("⚠️ No hay modelo con pruning para procesar")
            return None
        
        # Strip pruning
        final_model = tfmot.sparsity.keras.strip_pruning(self.pruned_model)
        
        # Recompilar
        final_model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print("✅ Wrappers de pruning removidos")
        
        return final_model
    
    def compare_models(self, test_dataset: tf.data.Dataset) -> Dict[str, Any]:
        """
        Compara el modelo base vs el modelo con pruning
        
        Args:
            test_dataset: Dataset de prueba
            
        Returns:
            Diccionario con comparación de modelos
        """
        print("📊 Comparando modelos base vs pruning...")
        
        # Evaluar modelo base
        print("🔍 Evaluando modelo base...")
        base_metrics = self.base_model.evaluate(test_dataset, verbose=1)
        
        # Evaluar modelo con pruning (si existe)
        if self.pruned_model is not None:
            print("🔍 Evaluando modelo con pruning...")
            pruning_metrics = self.pruned_model.evaluate(test_dataset, verbose=1)
        else:
            pruning_metrics = None
        
        # Evaluar modelo final (stripped)
        final_model = self.strip_pruning()
        if final_model is not None:
            print("🔍 Evaluando modelo final (stripped)...")
            final_metrics = final_model.evaluate(test_dataset, verbose=1)
        else:
            final_metrics = None
        
        # Calcular reducción de parámetros
        base_params = self.base_model.count_params()
        if final_model is not None:
            final_params = final_model.count_params()
            param_reduction = (base_params - final_params) / base_params * 100
        else:
            final_params = 0
            param_reduction = 0
        
        # Calcular tamaño de modelos
        def get_model_size(model):
            if model is None:
                return 0
            with tempfile.NamedTemporaryFile() as tmp:
                model.save(tmp.name, save_format='h5')
                size = tmp.tell()
            return size / (1024 * 1024)  # MB
        
        base_size = get_model_size(self.base_model)
        final_size = get_model_size(final_model)
        size_reduction = (base_size - final_size) / base_size * 100 if base_size > 0 else 0
        
        comparison = {
            'base_model': {
                'accuracy': base_metrics[1],
                'loss': base_metrics[0],
                'parameters': base_params,
                'size_mb': base_size
            },
            'pruned_model': {
                'accuracy': pruning_metrics[1] if pruning_metrics else None,
                'loss': pruning_metrics[0] if pruning_metrics else None,
                'parameters': final_params,
                'size_mb': final_size
            },
            'improvements': {
                'parameter_reduction_percent': param_reduction,
                'size_reduction_percent': size_reduction,
                'accuracy_change': (final_metrics[1] - base_metrics[1]) if final_metrics else 0
            }
        }
        
        print(f"✅ Comparación completada:")
        print(f"   - Reducción de parámetros: {param_reduction:.2f}%")
        print(f"   - Reducción de tamaño: {size_reduction:.2f}%")
        print(f"   - Cambio en accuracy: {comparison['improvements']['accuracy_change']:.4f}")
        
        return comparison
    
    def analyze_sparsity_by_layer(self) -> Dict[str, Any]:
        """
        Analiza la esparsidad por capa del modelo
        
        Returns:
            Diccionario con análisis de esparsidad
        """
        print("🔍 Analizando esparsidad por capa...")
        
        if self.pruned_model is None:
            print("⚠️ No hay modelo con pruning para analizar")
            return {}
        
        layer_sparsity = {}
        total_weights = 0
        zero_weights = 0
        
        for layer in self.pruned_model.layers:
            if hasattr(layer, 'kernel') and layer.kernel is not None:
                weights = layer.kernel.numpy()
                layer_name = layer.name
                
                # Calcular esparsidad de la capa
                layer_total = np.prod(weights.shape)
                layer_zeros = np.sum(weights == 0)
                layer_sparsity_ratio = layer_zeros / layer_total
                
                layer_sparsity[layer_name] = {
                    'total_weights': layer_total,
                    'zero_weights': layer_zeros,
                    'sparsity_ratio': layer_sparsity_ratio,
                    'layer_type': type(layer).__name__
                }
                
                total_weights += layer_total
                zero_weights += layer_zeros
        
        overall_sparsity = zero_weights / total_weights if total_weights > 0 else 0
        
        analysis = {
            'layer_sparsity': layer_sparsity,
            'overall_sparsity': overall_sparsity,
            'total_weights': total_weights,
            'zero_weights': zero_weights
        }
        
        print(f"✅ Análisis de esparsidad:")
        print(f"   - Esparsidad total: {overall_sparsity:.2%}")
        print(f"   - Pesos totales: {total_weights:,}")
        print(f"   - Pesos cero: {zero_weights:,}")
        
        return analysis
    
    def plot_sparsity_analysis(self, sparsity_analysis: Dict[str, Any]):
        """
        Visualiza el análisis de esparsidad
        
        Args:
            sparsity_analysis: Resultados del análisis de esparsidad
        """
        if not sparsity_analysis:
            print("⚠️ No hay datos de esparsidad para visualizar")
            return
        
        layer_names = list(sparsity_analysis['layer_sparsity'].keys())
        sparsity_ratios = [info['sparsity_ratio'] for info in sparsity_analysis['layer_sparsity'].values()]
        layer_types = [info['layer_type'] for info in sparsity_analysis['layer_sparsity'].values()]
        
        # Colores por tipo de capa
        colors = ['red' if 'Conv' in t else 'blue' if 'Dense' in t else 'green' for t in layer_types]
        
        plt.figure(figsize=(12, 6))
        bars = plt.bar(range(len(layer_names)), sparsity_ratios, color=colors, alpha=0.7)
        
        plt.title('Esparsidad por Capa')
        plt.xlabel('Capa')
        plt.ylabel('Ratio de Esparsidad')
        plt.xticks(range(len(layer_names)), layer_names, rotation=45, ha='right')
        plt.ylim(0, 1)
        
        # Línea de esparsidad objetivo
        plt.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='Esparsidad Objetivo (50%)')
        
        # Leyenda
        red_patch = plt.Rectangle((0, 0), 1, 1, fc="red", alpha=0.7, label='Capas Conv')
        blue_patch = plt.Rectangle((0, 0), 1, 1, fc="blue", alpha=0.7, label='Capas Dense')
        plt.legend(handles=[red_patch, blue_patch, plt.Line2D([0], [0], color='red', linestyle='--', label='Objetivo')])
        
        plt.tight_layout()
        plt.show()
    
    def plot_training_comparison(self):
        """
        Visualiza comparación de entrenamiento
        """
        if self.pruning_history is None:
            print("⚠️ No hay historial de pruning para visualizar")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Accuracy
        axes[0].plot(self.pruning_history['accuracy'], label='Training')
        axes[0].plot(self.pruning_history['val_accuracy'], label='Validation')
        axes[0].set_title('Accuracy durante Pruning Fine-tuning')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Accuracy')
        axes[0].legend()
        axes[0].grid(True)
        
        # Loss
        axes[1].plot(self.pruning_history['loss'], label='Training')
        axes[1].plot(self.pruning_history['val_loss'], label='Validation')
        axes[1].set_title('Loss durante Pruning Fine-tuning')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def save_pruned_model(self, filepath: str = 'pruned_model.h5'):
        """
        Guarda el modelo optimizado con pruning
        
        Args:
            filepath: Ruta donde guardar el modelo
        """
        final_model = self.strip_pruning()
        if final_model is not None:
            print(f"💾 Guardando modelo con pruning en {filepath}...")
            final_model.save(filepath)
            print("✅ Modelo guardado exitosamente")
        else:
            print("⚠️ No hay modelo para guardar")


def main():
    """
    Función principal para probar el pruning
    """
    print("🚀 Iniciando prueba de pruning optimization")
    
    # Cargar modelo base (simulado)
    from teacher_model import TeacherModel
    from load_cifar10 import CIFAR10DataLoader
    
    # Crear datos
    loader = CIFAR10DataLoader(batch_size=32)
    datasets = loader.create_datasets()
    
    # Crear modelo teacher simplificado para prueba
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
    
    # Aplicar pruning
    pruner = PruningOptimizer(teacher.model)
    pruned_model = pruner.apply_structured_pruning(target_sparsity=0.3)
    
    # Fine-tuning
    pruner.fine_tune_pruned_model(
        datasets['train_dataset'],
        datasets['val_dataset'],
        epochs=3
    )
    
    # Comparar modelos
    comparison = pruner.compare_models(datasets['test_dataset'])
    
    # Analizar esparsidad
    sparsity_analysis = pruner.analyze_sparsity_by_layer()
    pruner.plot_sparsity_analysis(sparsity_analysis)
    
    # Visualizar entrenamiento
    pruner.plot_training_comparison()
    
    # Guardar modelo
    pruner.save_pruned_model()
    
    print("✅ Prueba de pruning completada exitosamente")


if __name__ == "__main__":
    main()
