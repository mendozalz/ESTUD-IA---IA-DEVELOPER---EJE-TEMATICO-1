"""
Laboratorio 5.3 - Benchmark de Optimizaciones
==============================================

Este script realiza un benchmark completo de todas las técnicas
de optimización: pruning, quantization y distillation.
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict, Any, List, Tuple
import time
import json

from load_cifar10 import CIFAR10DataLoader
from teacher_model import TeacherModel
from pruning_optimization import PruningOptimizer
from quantization_optimization import QuantizationOptimizer
from distillation_optimization import DistillationOptimizer

class OptimizationBenchmark:
    """
    Clase para realizar benchmark completo de optimizaciones
    """
    
    def __init__(self, input_shape: Tuple[int, int, int] = (32, 32, 3),
                 num_classes: int = 10):
        """
        Inicializa el benchmark
        
        Args:
            input_shape: Forma de entrada
            num_classes: Número de clases
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.results = {}
        self.teacher_model = None
        self.optimized_models = {}
        
    def setup_benchmark(self, train_dataset: tf.data.Dataset,
                       val_dataset: tf.data.Dataset,
                       test_dataset: tf.data.Dataset):
        """
        Configura el entorno de benchmark
        
        Args:
            train_dataset: Dataset de entrenamiento
            val_dataset: Dataset de validación
            test_dataset: Dataset de prueba
        """
        print("🔧 Configurando entorno de benchmark...")
        
        # Crear y entrenar modelo teacher
        print("🎓 Creando y entrenando modelo teacher...")
        self.teacher_model = TeacherModel(self.input_shape, self.num_classes)
        self.teacher_model.build_model()
        self.teacher_model.compile_model()
        
        # Entrenar con épocas reducidas para benchmark
        training_results = self.teacher_model.train_model(
            train_dataset, val_dataset,
            epochs=3,  # Reducido para benchmark
            fine_tune_epochs=2
        )
        
        # Evaluar modelo base
        print("📊 Evaluando modelo base...")
        base_metrics = self.teacher_model.evaluate_model(test_dataset)
        
        # Guardar resultados base
        self.results['base_model'] = {
            'accuracy': base_metrics['test_accuracy'],
            'loss': base_metrics['test_loss'],
            'parameters': self.teacher_model.model.count_params(),
            'training_time': training_results['training_time'],
            'model_info': self.teacher_model.get_model_info()
        }
        
        print("✅ Entorno de benchmark configurado")
        
    def run_pruning_benchmark(self, train_dataset: tf.data.Dataset,
                             val_dataset: tf.data.Dataset,
                             test_dataset: tf.data.Dataset) -> Dict[str, Any]:
        """
        Ejecuta benchmark de pruning
        
        Args:
            train_dataset: Dataset de entrenamiento
            val_dataset: Dataset de validación
            test_dataset: Dataset de prueba
            
        Returns:
            Resultados del benchmark de pruning
        """
        print("✂️ Ejecutando benchmark de pruning...")
        
        # Diferentes niveles de pruning
        pruning_levels = [0.3, 0.5, 0.7]
        pruning_results = {}
        
        for level in pruning_levels:
            print(f"🔍 Probando pruning level: {level}")
            
            # Crear optimizador
            pruner = PruningOptimizer(self.teacher_model.model)
            
            # Aplicar pruning
            pruned_model = pruner.apply_structured_pruning(
                target_sparsity=level,
                begin_step=0,
                end_step=1000
            )
            
            # Fine-tuning
            ft_results = pruner.fine_tune_pruned_model(
                train_dataset, val_dataset, epochs=3
            )
            
            # Comparar modelos
            comparison = pruner.compare_models(test_dataset)
            
            # Analizar esparsidad
            sparsity_analysis = pruner.analyze_sparsity_by_layer()
            
            # Guardar resultados
            pruning_results[f'pruning_{level}'] = {
                'pruning_level': level,
                'accuracy': comparison['pruned_model']['accuracy'],
                'parameter_reduction': comparison['improvements']['parameter_reduction_percent'],
                'size_reduction': comparison['improvements']['size_reduction_percent'],
                'fine_tune_time': ft_results['fine_tune_time'],
                'sparsity_analysis': sparsity_analysis
            }
            
            # Guardar modelo optimizado
            pruner.save_pruned_model(f'pruned_model_level_{level}.h5')
        
        self.results['pruning'] = pruning_results
        print("✅ Benchmark de pruning completado")
        
        return pruning_results
    
    def run_quantization_benchmark(self, train_dataset: tf.data.Dataset,
                                  val_dataset: tf.data.Dataset,
                                  test_dataset: tf.data.Dataset) -> Dict[str, Any]:
        """
        Ejecuta benchmark de quantization
        
        Args:
            train_dataset: Dataset de entrenamiento
            val_dataset: Dataset de validación
            test_dataset: Dataset de prueba
            
        Returns:
            Resultados del benchmark de quantization
        """
        print("🎯 Ejecutando benchmark de quantization...")
        
        # Crear optimizador
        quantizer = QuantizationOptimizer(self.teacher_model.model)
        
        # Aplicar QAT
        qat_model = quantizer.apply_quantization_aware_training(
            train_dataset, val_dataset, epochs=3
        )
        
        # Comparar métodos de quantization
        comparison = quantizer.compare_quantization_methods(
            test_dataset, train_dataset
        )
        
        # Benchmark de velocidad
        speed_benchmark = quantizer.benchmark_inference_speed(test_dataset, 50)
        
        # Guardar modelos
        for q_type in ['dynamic', 'float16', 'int8']:
            quantizer.save_tflite_model('quantized_model', q_type)
        
        # Guardar resultados
        quantization_results = {
            'qat_accuracy': comparison['qat_model']['accuracy'],
            'tflite_dynamic': comparison['tflite_models']['dynamic'],
            'tflite_float16': comparison['tflite_models']['float16'],
            'tflite_int8': comparison['tflite_models']['int8'],
            'size_reductions': comparison['size_reductions'],
            'speed_benchmark': speed_benchmark
        }
        
        self.results['quantization'] = quantization_results
        print("✅ Benchmark de quantization completado")
        
        return quantization_results
    
    def run_distillation_benchmark(self, train_dataset: tf.data.Dataset,
                                  val_dataset: tf.data.Dataset,
                                  test_dataset: tf.data.Dataset) -> Dict[str, Any]:
        """
        Ejecuta benchmark de knowledge distillation
        
        Args:
            train_dataset: Dataset de entrenamiento
            val_dataset: Dataset de validación
            test_dataset: Dataset de prueba
            
        Returns:
            Resultados del benchmark de distillation
        """
        print("🎓 Ejecutando benchmark de knowledge distillation...")
        
        # Diferentes configuraciones de distillation
        configs = [
            {'temperature': 3.0, 'alpha': 0.5},
            {'temperature': 5.0, 'alpha': 0.7},
            {'temperature': 2.0, 'alpha': 0.3}
        ]
        
        distillation_results = {}
        
        for i, config in enumerate(configs):
            print(f"🔍 Probando configuración {i+1}: T={config['temperature']}, α={config['alpha']}")
            
            # Crear optimizador
            distiller = DistillationOptimizer(
                self.teacher_model.model,
                temperature=config['temperature'],
                alpha=config['alpha']
            )
            
            # Crear modelo estudiante
            student_model = distiller.create_student_model()
            
            # Entrenar con distillation
            dist_results = distiller.train_student_with_distillation(
                train_dataset, val_dataset, epochs=5
            )
            
            # Comparar modelos
            comparison = distiller.compare_models(test_dataset)
            
            # Analizar transferencia
            transfer_analysis = distiller.analyze_knowledge_transfer(test_dataset, 50)
            
            # Guardar resultados
            config_name = f"distillation_T{config['temperature']}_A{config['alpha']}"
            distillation_results[config_name] = {
                'config': config,
                'accuracy': comparison['student_distilled']['accuracy'],
                'parameter_reduction': comparison['improvements']['parameter_reduction_vs_teacher'],
                'size_reduction': comparison['improvements']['size_reduction_vs_teacher'],
                'training_time': dist_results['training_time'] if 'training_time' in dist_results else 0,
                'transfer_analysis': transfer_analysis,
                'improvement_vs_baseline': comparison['improvements']['distillation_vs_baseline_accuracy']
            }
            
            # Guardar modelo
            distiller.save_student_model(f'student_model_{config_name}.h5')
        
        self.results['distillation'] = distillation_results
        print("✅ Benchmark de distillation completado")
        
        return distillation_results
    
    def run_combined_optimization(self, train_dataset: tf.data.Dataset,
                                 val_dataset: tf.data.Dataset,
                                 test_dataset: tf.data.Dataset) -> Dict[str, Any]:
        """
        Ejecuta benchmark de optimización combinada
        
        Args:
            train_dataset: Dataset de entrenamiento
            val_dataset: Dataset de validación
            test_dataset: Dataset de prueba
            
        Returns:
            Resultados de optimización combinada
        """
        print("🔄 Ejecutando benchmark de optimización combinada...")
        
        combined_results = {}
        
        # Estrategia 1: Pruning + Quantization
        print("🔍 Estrategia 1: Pruning + Quantization")
        
        # Aplicar pruning primero
        pruner = PruningOptimizer(self.teacher_model.model)
        pruned_model = pruner.apply_structured_pruning(target_sparsity=0.5)
        pruner.fine_tune_pruned_model(train_dataset, val_dataset, epochs=3)
        
        # Luego quantization
        final_model = pruner.strip_pruning()
        quantizer = QuantizationOptimizer(final_model)
        quantizer.apply_quantization_aware_training(train_dataset, val_dataset, epochs=2)
        
        # Evaluar
        comparison = quantizer.compare_quantization_methods(test_dataset, train_dataset)
        
        combined_results['pruning_quantization'] = {
            'accuracy': comparison['qat_model']['accuracy'],
            'size_reduction': comparison['size_reductions'].get('int8', 0),
            'strategy': 'Pruning -> Quantization'
        }
        
        # Estrategia 2: Distillation + Quantization
        print("🔍 Estrategia 2: Distillation + Quantization")
        
        # Aplicar distillation primero
        distiller = DistillationOptimizer(self.teacher_model.model)
        student_model = distiller.create_student_model()
        distiller.train_student_with_distillation(train_dataset, val_dataset, epochs=3)
        
        # Luego quantization
        quantizer = QuantizationOptimizer(distiller.student_model)
        quantizer.apply_quantization_aware_training(train_dataset, val_dataset, epochs=2)
        
        # Evaluar
        comparison = quantizer.compare_quantization_methods(test_dataset, train_dataset)
        
        combined_results['distillation_quantization'] = {
            'accuracy': comparison['qat_model']['accuracy'],
            'size_reduction': comparison['size_reductions'].get('int8', 0),
            'strategy': 'Distillation -> Quantization'
        }
        
        self.results['combined'] = combined_results
        print("✅ Benchmark de optimización combinada completado")
        
        return combined_results
    
    def generate_comprehensive_report(self) -> Dict[str, Any]:
        """
        Genera un reporte completo del benchmark
        
        Returns:
            Reporte completo en formato diccionario
        """
        print("📊 Generando reporte completo...")
        
        report = {
            'benchmark_summary': {
                'base_model': self.results['base_model'],
                'optimization_results': self.results
            },
            'recommendations': self._generate_recommendations(),
            'best_configurations': self._find_best_configurations(),
            'trade_off_analysis': self._analyze_trade_offs()
        }
        
        # Guardar reporte en JSON
        with open('optimization_benchmark_report.json', 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print("✅ Reporte generado y guardado")
        
        return report
    
    def _generate_recommendations(self) -> List[str]:
        """
        Genera recomendaciones basadas en resultados
        
        Returns:
            Lista de recomendaciones
        """
        recommendations = []
        
        if 'pruning' in self.results:
            best_pruning = max(self.results['pruning'].items(), 
                             key=lambda x: x[1]['accuracy'])
            recommendations.append(
                f"Mejor pruning: {best_pruning[0]} con "
                f"accuracy {best_pruning[1]['accuracy']:.3f} y "
                f"reducción {best_pruning[1]['parameter_reduction']:.1f}%"
            )
        
        if 'quantization' in self.results:
            best_quant = max(self.results['quantization']['tflite_models'].items(),
                           key=lambda x: x[1]['accuracy'])
            recommendations.append(
                f"Mejor quantization: {best_quant[0]} con "
                f"accuracy {best_quant[1]['accuracy']:.3f} y "
                f"reducción {self.results['quantization']['size_reductions'][best_quant[0]]:.1f}%"
            )
        
        if 'distillation' in self.results:
            best_dist = max(self.results['distillation'].items(),
                           key=lambda x: x[1]['accuracy'])
            recommendations.append(
                f"Mejor distillation: {best_dist[0]} con "
                f"accuracy {best_dist[1]['accuracy']:.3f} y "
                f"reducción {best_dist[1]['parameter_reduction']:.1f}%"
            )
        
        return recommendations
    
    def _find_best_configurations(self) -> Dict[str, Any]:
        """
        Encuentra las mejores configuraciones para diferentes objetivos
        
        Returns:
            Diccionario con mejores configuraciones
        """
        best_configs = {
            'best_accuracy': {'model': 'base', 'accuracy': self.results['base_model']['accuracy']},
            'best_size_reduction': {'model': 'none', 'reduction': 0},
            'best_balance': {'model': 'none', 'score': 0}
        }
        
        # Evaluar todas las configuraciones
        all_configs = []
        
        # Base model
        all_configs.append({
            'name': 'base_model',
            'accuracy': self.results['base_model']['accuracy'],
            'size_reduction': 0,
            'score': self.results['base_model']['accuracy']
        })
        
        # Pruning configs
        if 'pruning' in self.results:
            for name, config in self.results['pruning'].items():
                score = config['accuracy'] + (config['parameter_reduction'] / 100) * 0.1
                all_configs.append({
                    'name': name,
                    'accuracy': config['accuracy'],
                    'size_reduction': config['parameter_reduction'],
                    'score': score
                })
        
        # Quantization configs
        if 'quantization' in self.results:
            for q_type, config in self.results['quantization']['tflite_models'].items():
                reduction = self.results['quantization']['size_reductions'].get(q_type, 0)
                score = config['accuracy'] + (reduction / 100) * 0.1
                all_configs.append({
                    'name': f'quantization_{q_type}',
                    'accuracy': config['accuracy'],
                    'size_reduction': reduction,
                    'score': score
                })
        
        # Distillation configs
        if 'distillation' in self.results:
            for name, config in self.results['distillation'].items():
                score = config['accuracy'] + (config['parameter_reduction'] / 100) * 0.1
                all_configs.append({
                    'name': name,
                    'accuracy': config['accuracy'],
                    'size_reduction': config['parameter_reduction'],
                    'score': score
                })
        
        # Encontrar mejores
        best_configs['best_accuracy'] = max(all_configs, key=lambda x: x['accuracy'])
        best_configs['best_size_reduction'] = max(all_configs, key=lambda x: x['size_reduction'])
        best_configs['best_balance'] = max(all_configs, key=lambda x: x['score'])
        
        return best_configs
    
    def _analyze_trade_offs(self) -> Dict[str, Any]:
        """
        Analiza los trade-offs entre accuracy y tamaño
        
        Returns:
            Análisis de trade-offs
        """
        trade_offs = {
            'accuracy_vs_size': [],
            'efficiency_analysis': {}
        }
        
        # Recolectar datos
        all_configs = []
        
        # Base model
        all_configs.append({
            'name': 'base_model',
            'accuracy': self.results['base_model']['accuracy'],
            'size_mb': self.results['base_model']['model_info']['model_size_mb']
        })
        
        # Agregar otras configuraciones
        for category, results in self.results.items():
            if category == 'base_model':
                continue
            
            if isinstance(results, dict):
                for name, config in results.items():
                    if 'accuracy' in config:
                        size_mb = config.get('size_reduction', 0)
                        if size_mb > 0:
                            size_mb = self.results['base_model']['model_info']['model_size_mb'] * (1 - size_mb/100)
                        else:
                            size_mb = self.results['base_model']['model_info']['model_size_mb']
                        
                        all_configs.append({
                            'name': f"{category}_{name}",
                            'accuracy': config['accuracy'],
                            'size_mb': size_mb
                        })
        
        trade_offs['accuracy_vs_size'] = all_configs
        
        # Análisis de eficiencia
        for config in all_configs:
            efficiency = config['accuracy'] / config['size_mb']
            trade_offs['efficiency_analysis'][config['name']] = efficiency
        
        return trade_offs
    
    def plot_comprehensive_results(self):
        """
        Visualiza resultados completos del benchmark
        """
        print("📊 Visualizando resultados completos...")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Accuracy comparison
        if 'pruning' in self.results:
            pruning_names = list(self.results['pruning'].keys())
            pruning_accs = [self.results['pruning'][name]['accuracy'] for name in pruning_names]
            
            axes[0, 0].bar(pruning_names, pruning_accs, color='orange', alpha=0.7)
            axes[0, 0].set_title('Pruning - Accuracy')
            axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Size reduction comparison
        if 'pruning' in self.results:
            pruning_reductions = [self.results['pruning'][name]['parameter_reduction'] for name in pruning_names]
            
            axes[0, 1].bar(pruning_names, pruning_reductions, color='red', alpha=0.7)
            axes[0, 1].set_title('Pruning - Parameter Reduction')
            axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Quantization comparison
        if 'quantization' in self.results:
            quant_types = list(self.results['quantization']['tflite_models'].keys())
            quant_accs = [self.results['quantization']['tflite_models'][q_type]['accuracy'] for q_type in quant_types]
            
            axes[0, 2].bar(quant_types, quant_accs, color='green', alpha=0.7)
            axes[0, 2].set_title('Quantization - Accuracy')
            axes[0, 2].tick_params(axis='x', rotation=45)
        
        # Distillation comparison
        if 'distillation' in self.results:
            dist_names = list(self.results['distillation'].keys())
            dist_accs = [self.results['distillation'][name]['accuracy'] for name in dist_names]
            
            axes[1, 0].bar(dist_names, dist_accs, color='purple', alpha=0.7)
            axes[1, 0].set_title('Distillation - Accuracy')
            axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Combined optimization
        if 'combined' in self.results:
            combined_names = list(self.results['combined'].keys())
            combined_accs = [self.results['combined'][name]['accuracy'] for name in combined_names]
            
            axes[1, 1].bar(combined_names, combined_accs, color='brown', alpha=0.7)
            axes[1, 1].set_title('Combined - Accuracy')
            axes[1, 1].tick_params(axis='x', rotation=45)
        
        # Overall comparison
        all_methods = ['Base']
        all_accs = [self.results['base_model']['accuracy']]
        
        if 'pruning' in self.results:
            best_pruning = max(self.results['pruning'].values(), key=lambda x: x['accuracy'])
            all_methods.append('Best Pruning')
            all_accs.append(best_pruning['accuracy'])
        
        if 'quantization' in self.results:
            best_quant = max(self.results['quantization']['tflite_models'].values(), key=lambda x: x['accuracy'])
            all_methods.append('Best Quantization')
            all_accs.append(best_quant['accuracy'])
        
        if 'distillation' in self.results:
            best_dist = max(self.results['distillation'].values(), key=lambda x: x['accuracy'])
            all_methods.append('Best Distillation')
            all_accs.append(best_dist['accuracy'])
        
        axes[1, 2].bar(all_methods, all_accs, color=['blue', 'orange', 'green', 'purple'])
        axes[1, 2].set_title('Best Methods Comparison')
        axes[1, 2].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()
    
    def save_results_to_csv(self):
        """
        Guarda resultados en formato CSV para análisis
        """
        print("💾 Guardando resultados en CSV...")
        
        # Crear DataFrame con resultados
        results_data = []
        
        # Base model
        results_data.append({
            'method': 'base_model',
            'accuracy': self.results['base_model']['accuracy'],
            'parameter_reduction': 0,
            'size_reduction': 0,
            'training_time': self.results['base_model']['training_time']
        })
        
        # Agregar otros resultados
        for category, results in self.results.items():
            if category == 'base_model':
                continue
            
            if isinstance(results, dict):
                for name, config in results.items():
                    results_data.append({
                        'method': f"{category}_{name}",
                        'accuracy': config.get('accuracy', 0),
                        'parameter_reduction': config.get('parameter_reduction', 0),
                        'size_reduction': config.get('size_reduction', 0),
                        'training_time': config.get('training_time', 0)
                    })
        
        # Guardar CSV
        df = pd.DataFrame(results_data)
        df.to_csv('optimization_benchmark_results.csv', index=False)
        
        print("✅ Resultados guardados en CSV")


def main():
    """
    Función principal para ejecutar el benchmark completo
    """
    print("🚀 Iniciando benchmark completo de optimizaciones")
    
    # Crear datasets
    loader = CIFAR10DataLoader(batch_size=32)
    datasets = loader.create_datasets()
    
    # Crear benchmark
    benchmark = OptimizationBenchmark()
    
    # Configurar entorno
    benchmark.setup_benchmark(
        datasets['train_dataset'],
        datasets['val_dataset'],
        datasets['test_dataset']
    )
    
    # Ejecutar benchmarks individuales
    benchmark.run_pruning_benchmark(
        datasets['train_dataset'],
        datasets['val_dataset'],
        datasets['test_dataset']
    )
    
    benchmark.run_quantization_benchmark(
        datasets['train_dataset'],
        datasets['val_dataset'],
        datasets['test_dataset']
    )
    
    benchmark.run_distillation_benchmark(
        datasets['train_dataset'],
        datasets['val_dataset'],
        datasets['test_dataset']
    )
    
    # Ejecutar optimización combinada
    benchmark.run_combined_optimization(
        datasets['train_dataset'],
        datasets['val_dataset'],
        datasets['test_dataset']
    )
    
    # Generar reporte
    report = benchmark.generate_comprehensive_report()
    
    # Visualizar resultados
    benchmark.plot_comprehensive_results()
    
    # Guardar en CSV
    benchmark.save_results_to_csv()
    
    print("✅ Benchmark completado exitosamente")
    print(f"📊 Reporte guardado en: optimization_benchmark_report.json")
    print(f"📈 Resultados CSV guardados en: optimization_benchmark_results.csv")


if __name__ == "__main__":
    main()
