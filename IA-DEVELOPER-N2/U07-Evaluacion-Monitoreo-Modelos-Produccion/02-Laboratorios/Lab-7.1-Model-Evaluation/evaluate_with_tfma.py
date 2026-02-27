"""
Caso de Uso 1 - Evaluación de Modelos con TFMA
Fase 2: Evaluación Avanzada con TensorFlow Model Analysis
"""

import tensorflow as tf
import tensorflow_model_analysis as tfma
from tensorflow_model_analysis.view.render_slicing_metrics import render_slicing_metrics
from tensorflow_model_analysis.view import render_plot
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
from datetime import datetime

class FraudModelEvaluator:
    """
    Clase para evaluar modelos de detección de fraude usando TFMA
    """
    
    def __init__(self, model_path=None, data_path='data/eval.tfrecord'):
        self.model_path = model_path
        self.data_path = data_path
        self.eval_result = None
        self.eval_config = None
        
    def create_simple_model(self):
        """Crear un modelo simple de clasificación de fraudes"""
        print("Creando modelo de clasificación de fraudes...")
        
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(20,), name='features'),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid', name='output')
        ])
        
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
        )
        
        # Entrenar con datos dummy (en producción, usar datos reales)
        print("Entrenando modelo con datos de ejemplo...")
        dummy_X = np.random.random((1000, 20))
        dummy_y = np.random.randint(0, 2, (1000, 1))
        
        model.fit(
            dummy_X, dummy_y,
            epochs=5,
            batch_size=32,
            verbose=0
        )
        
        # Guardar modelo
        os.makedirs('models', exist_ok=True)
        model.save('models/fraud_classifier.h5')
        print("Modelo guardado en models/fraud_classifier.h5")
        
        return model
    
    def setup_eval_config(self):
        """Configurar la evaluación con TFMA"""
        print("Configurando evaluación con TFMA...")
        
        self.eval_config = tfma.EvalConfig(
            model_specs=[tfma.ModelSpec(
                label_key='label',
                preprocessing_fn=None
            )],
            metrics_specs=[
                # Métricas de clasificación binaria
                tfma.MetricConfig(class_name='BinaryAccuracy'),
                tfma.MetricConfig(class_name='Precision'),
                tfma.MetricConfig(class_name='Recall'),
                tfma.MetricConfig(class_name='F1Score'),
                
                # Métricas de curva ROC
                tfma.MetricConfig(class_name='AUC'),
                tfma.MetricConfig(class_name='AUC', name='pr_auc', options={'curve': 'PR'}),
                
                # Métricas de conteo
                tfma.MetricConfig(class_name='ExampleCount'),
                tfma.MetricConfig(class_name='TruePositives'),
                tfma.MetricConfig(class_name='FalsePositives'),
                tfma.MetricConfig(class_name='TrueNegatives'),
                tfma.MetricConfig(class_name='FalseNegatives'),
                
                # Métricas personalizadas
                tfma.MetricConfig(
                    class_name='MeanLabel',
                    options={'value_threshold': 0.5}
                ),
                tfma.MetricConfig(
                    class_name='MeanPrediction',
                    options={'value_threshold': 0.5}
                )
            ],
            slicing_specs=[
                # Evaluación general
                tfma.SlicingSpec(),
                
                # Slicing por región
                tfma.SlicingSpec(feature_keys=['region']),
                
                # Slicing por umbrales de monto (si estuviera disponible)
                # tfma.SlicingSpec(feature_keys=['amount_bucket']),
            ]
        )
        
        print("Configuración de evaluación completada")
        return self.eval_config
    
    def run_evaluation(self):
        """Ejecutar la evaluación con TFMA"""
        print("Iniciando evaluación con TFMA...")
        
        # Crear modelo si no existe
        if not os.path.exists('models/fraud_classifier.h5'):
            model = self.create_simple_model()
        else:
            model = tf.keras.models.load_model('models/fraud_classifier.h5')
            print("Modelo cargado desde models/fraud_classifier.h5")
        
        # Configurar evaluación
        eval_config = self.setup_eval_config()
        
        # Crear directorio de salida
        output_path = 'outputs/tfma_output'
        os.makedirs(output_path, exist_ok=True)
        
        try:
            # Ejecutar evaluación
            self.eval_result = tfma.run_model_analysis(
                tfma.load_model('models/fraud_classifier.h5'),
                data_location=self.data_path,
                eval_config=eval_config,
                output_path=output_path
            )
            
            print("Evaluación completada exitosamente")
            print(f"Resultados guardados en: {output_path}")
            
            return self.eval_result
            
        except Exception as e:
            print(f"Error durante la evaluación: {e}")
            print("Verificando disponibilidad de datos...")
            
            # Verificar si el archivo de datos existe
            if not os.path.exists(self.data_path):
                print(f"ERROR: Archivo de datos no encontrado: {self.data_path}")
                print("Ejecuta primero load_model_data.py para generar los datos")
                return None
            
            raise
    
    def generate_visualizations(self):
        """Generar visualizaciones de los resultados"""
        if self.eval_result is None:
            print("ERROR: No hay resultados de evaluación disponibles")
            return
        
        print("Generando visualizaciones...")
        
        # Crear directorio de salida
        os.makedirs('outputs', exist_ok=True)
        
        # 1. Reporte principal de TFMA
        print("Generando reporte principal de TFMA...")
        render_slicing_metrics(
            self.eval_result, 
            output_file='outputs/tfma_report.html'
        )
        
        # 2. Análisis de métricas por slicing
        self.analyze_slicing_metrics()
        
        # 3. Gráficos personalizados
        self.create_custom_plots()
        
        # 4. Reporte JSON para análisis programático
        self.generate_json_report()
        
        print("Visualizaciones generadas en el directorio outputs/")
    
    def analyze_slicing_metrics(self):
        """Analizar métricas por slicing (regiones)"""
        print("Analizando métricas por slicing...")
        
        # Extraer métricas por slice
        slicing_metrics = {}
        
        for slice_key, slice_metrics in self.eval_result.slicing_metrics().items():
            if slice_key:
                # Extraer información del slice
                slice_info = {}
                for feature, value in slice_key:
                    if feature == 'region':
                        slice_info['region'] = value.decode('utf-8')
                    else:
                        slice_info[feature] = value
                
                # Extraer métricas
                metrics = {}
                for metric_name, metric_value in slice_metrics.items():
                    if 'doubleValue' in metric_value:
                        metrics[metric_name] = metric_value['doubleValue']
                    elif 'confusionMatrix' in metric_value:
                        metrics[metric_name] = metric_value['confusionMatrix']
                
                slice_info['metrics'] = metrics
                slicing_metrics[f"slice_{len(slicing_metrics)}"] = slice_info
            else:
                # Slice general (overall)
                slicing_metrics['overall'] = {
                    'metrics': {
                        name: value.get('doubleValue', 0)
                        for name, value in slice_metrics.items()
                        if 'doubleValue' in value
                    }
                }
        
        # Guardar análisis
        with open('outputs/slicing_analysis.json', 'w') as f:
            json.dump(slicing_metrics, f, indent=2)
        
        print("Análisis de slicing guardado en outputs/slicing_analysis.json")
        
        return slicing_metrics
    
    def create_custom_plots(self):
        """Crear gráficos personalizados de las métricas"""
        print("Creando gráficos personalizados...")
        
        # Extraer datos para graficar
        metrics_data = []
        
        for slice_key, slice_metrics in self.eval_result.slicing_metrics().items():
            if slice_key:
                region = 'overall'
                for feature, value in slice_key:
                    if feature == 'region':
                        region = value.decode('utf-8')
                
                # Extraer métricas principales
                metrics = {
                    'region': region,
                    'accuracy': slice_metrics.get('binary_accuracy', {}).get('doubleValue', 0),
                    'precision': slice_metrics.get('precision', {}).get('doubleValue', 0),
                    'recall': slice_metrics.get('recall', {}).get('doubleValue', 0),
                    'auc': slice_metrics.get('auc', {}).get('doubleValue', 0),
                    'pr_auc': slice_metrics.get('pr_auc', {}).get('doubleValue', 0),
                    'example_count': slice_metrics.get('example_count', {}).get('doubleValue', 0)
                }
                
                metrics_data.append(metrics)
        
        # Convertir a DataFrame
        df_metrics = pd.DataFrame(metrics_data)
        
        # Crear gráficos
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Métricas de Evaluación por Región', fontsize=16, fontweight='bold')
        
        # Accuracy por región
        sns.barplot(data=df_metrics, x='region', y='accuracy', ax=axes[0, 0])
        axes[0, 0].set_title('Accuracy por Región')
        axes[0, 0].set_ylim(0, 1)
        
        # Precision por región
        sns.barplot(data=df_metrics, x='region', y='precision', ax=axes[0, 1])
        axes[0, 1].set_title('Precision por Región')
        axes[0, 1].set_ylim(0, 1)
        
        # Recall por región
        sns.barplot(data=df_metrics, x='region', y='recall', ax=axes[0, 2])
        axes[0, 2].set_title('Recall por Región')
        axes[0, 2].set_ylim(0, 1)
        
        # ROC-AUC por región
        sns.barplot(data=df_metrics, x='region', y='auc', ax=axes[1, 0])
        axes[1, 0].set_title('ROC-AUC por Región')
        axes[1, 0].set_ylim(0, 1)
        
        # PR-AUC por región
        sns.barplot(data=df_metrics, x='region', y='pr_auc', ax=axes[1, 1])
        axes[1, 1].set_title('PR-AUC por Región')
        axes[1, 1].set_ylim(0, 1)
        
        # Número de ejemplos por región
        sns.barplot(data=df_metrics, x='region', y='example_count', ax=axes[1, 2])
        axes[1, 2].set_title('Número de Ejemplos por Región')
        
        plt.tight_layout()
        plt.savefig('outputs/metrics_by_region.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Crear gráfico de comparación de métricas
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Normalizar métricas para comparación
        metrics_to_compare = ['accuracy', 'precision', 'recall', 'auc', 'pr_auc']
        df_melted = df_metrics.melt(
            id_vars=['region'], 
            value_vars=metrics_to_compare,
            var_name='metric', 
            value_name='value'
        )
        
        sns.barplot(data=df_melted, x='metric', y='value', hue='region', ax=ax)
        ax.set_title('Comparación de Métricas por Región')
        ax.set_xlabel('Métrica')
        ax.set_ylabel('Valor')
        ax.set_ylim(0, 1)
        ax.legend(title='Región')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig('outputs/metrics_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Gráficos personalizados guardados:")
        print("  - outputs/metrics_by_region.png")
        print("  - outputs/metrics_comparison.png")
    
    def generate_json_report(self):
        """Generar reporte en formato JSON"""
        print("Generando reporte JSON...")
        
        # Extraer métricas generales
        overall_metrics = {}
        for slice_key, slice_metrics in self.eval_result.slicing_metrics().items():
            if not slice_key:  # Overall slice
                overall_metrics = {
                    name: value.get('doubleValue', 0)
                    for name, value in slice_metrics.items()
                    if 'doubleValue' in value
                }
                break
        
        # Crear reporte completo
        report = {
            'evaluation_timestamp': datetime.now().isoformat(),
            'model_path': self.model_path or 'models/fraud_classifier.h5',
            'data_path': self.data_path,
            'overall_metrics': overall_metrics,
            'evaluation_summary': {
                'total_examples': overall_metrics.get('example_count', 0),
                'accuracy': overall_metrics.get('binary_accuracy', 0),
                'precision': overall_metrics.get('precision', 0),
                'recall': overall_metrics.get('recall', 0),
                'auc': overall_metrics.get('auc', 0),
                'pr_auc': overall_metrics.get('pr_auc', 0),
                'false_positive_rate': self.calculate_fpr(overall_metrics),
                'false_negative_rate': self.calculate_fnr(overall_metrics)
            },
            'recommendations': self.generate_recommendations(overall_metrics)
        }
        
        # Guardar reporte
        with open('outputs/evaluation_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        print("Reporte JSON guardado en outputs/evaluation_report.json")
        
        return report
    
    def calculate_fpr(self, metrics):
        """Calcular False Positive Rate"""
        fp = metrics.get('falsePositives', 0)
        tn = metrics.get('trueNegatives', 0)
        return fp / (fp + tn) if (fp + tn) > 0 else 0
    
    def calculate_fnr(self, metrics):
        """Calcular False Negative Rate"""
        fn = metrics.get('falseNegatives', 0)
        tp = metrics.get('truePositives', 0)
        return fn / (fn + tp) if (fn + tp) > 0 else 0
    
    def generate_recommendations(self, metrics):
        """Generar recomendaciones basadas en las métricas"""
        recommendations = []
        
        accuracy = metrics.get('binary_accuracy', 0)
        precision = metrics.get('precision', 0)
        recall = metrics.get('recall', 0)
        auc = metrics.get('auc', 0)
        pr_auc = metrics.get('pr_auc', 0)
        
        # Recomendaciones basadas en métricas
        if accuracy < 0.85:
            recommendations.append({
                'priority': 'high',
                'metric': 'accuracy',
                'value': accuracy,
                'recommendation': 'Considerar mejorar el modelo o añadir más características'
            })
        
        if precision < 0.80:
            recommendations.append({
                'priority': 'medium',
                'metric': 'precision',
                'value': precision,
                'recommendation': 'Ajustar umbral de decisión para reducir falsos positivos'
            })
        
        if recall < 0.75:
            recommendations.append({
                'priority': 'high',
                'metric': 'recall',
                'value': recall,
                'recommendation': 'Mejorar detección de fraudes para reducir falsos negativos'
            })
        
        if auc < 0.85:
            recommendations.append({
                'priority': 'high',
                'metric': 'auc',
                'value': auc,
                'recommendation': 'El modelo tiene capacidad de discriminación limitada'
            })
        
        if pr_auc < 0.80:
            recommendations.append({
                'priority': 'medium',
                'metric': 'pr_auc',
                'value': pr_auc,
                'recommendation': 'Considerar técnicas para manejo de clases desbalanceadas'
            })
        
        return recommendations
    
    def print_summary(self):
        """Imprimir resumen de la evaluación"""
        if self.eval_result is None:
            print("ERROR: No hay resultados de evaluación disponibles")
            return
        
        print("\n" + "=" * 60)
        print("RESUMEN DE EVALUACIÓN CON TFMA")
        print("=" * 60)
        
        # Extraer métricas generales
        for slice_key, slice_metrics in self.eval_result.slicing_metrics().items():
            if not slice_key:  # Overall slice
                print("\nMétricas Generales:")
                for metric_name, metric_value in slice_metrics.items():
                    if 'doubleValue' in metric_value:
                        value = metric_value['doubleValue']
                        print(f"  {metric_name}: {value:.4f}")
                break
        
        # Mostrar slicing por región
        print("\nMétricas por Región:")
        for slice_key, slice_metrics in self.eval_result.slicing_metrics().items():
            if slice_key:
                region = 'unknown'
                for feature, value in slice_key:
                    if feature == 'region':
                        region = value.decode('utf-8')
                        break
                
                print(f"\n  Región: {region}")
                for metric_name, metric_value in slice_metrics.items():
                    if 'doubleValue' in metric_value and metric_name in ['binary_accuracy', 'auc', 'pr_auc']:
                        value = metric_value['doubleValue']
                        print(f"    {metric_name}: {value:.4f}")
        
        print("\n" + "=" * 60)

def main():
    """Función principal para ejecutar la evaluación"""
    print("=" * 60)
    print("EVALUACIÓN DE MODELO DE FRAUDE CON TENSORFLOW MODEL ANALYSIS")
    print("=" * 60)
    
    # Crear evaluador
    evaluator = FraudModelEvaluator()
    
    # Ejecutar evaluación
    eval_result = evaluator.run_evaluation()
    
    if eval_result:
        # Generar visualizaciones
        evaluator.generate_visualizations()
        
        # Imprimir resumen
        evaluator.print_summary()
        
        print("\n" + "=" * 60)
        print("EVALUACIÓN COMPLETADA EXITOSAMENTE")
        print("Revisa los archivos generados en el directorio outputs/")
        print("=" * 60)
    else:
        print("ERROR: La evaluación falló")
        print("Verifica que los datos existan ejecutando load_model_data.py")

if __name__ == "__main__":
    main()
