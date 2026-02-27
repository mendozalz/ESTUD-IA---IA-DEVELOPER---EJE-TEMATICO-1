"""
Caso de Uso 1 - Evaluación de Modelos con TFMA
Fase 3: Análisis Detallado por Subgrupos
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow_model_analysis as tfma
from sklearn.metrics import (
    roc_auc_score, average_precision_score, 
    precision_recall_curve, roc_curve,
    confusion_matrix, classification_report
)
import json
import os
from datetime import datetime

class SubgroupAnalyzer:
    """
    Clase para analizar métricas de evaluación por subgrupos
    """
    
    def __init__(self, tfma_output_path='outputs/tfma_output'):
        self.tfma_output_path = tfma_output_path
        self.eval_result = None
        self.subgroup_metrics = None
        
    def load_tfma_results(self):
        """Cargar resultados de TFMA"""
        print("Cargando resultados de TFMA...")
        
        try:
            self.eval_result = tfma.load_eval_result(self.tfma_output_path)
            print("Resultados de TFMA cargados exitosamente")
            return True
        except Exception as e:
            print(f"Error cargando resultados de TFMA: {e}")
            print("Asegúrate de haber ejecutado evaluate_with_tfma.py primero")
            return False
    
    def extract_subgroup_metrics(self):
        """Extraer métricas por subgrupos"""
        if self.eval_result is None:
            print("ERROR: No hay resultados de TFMA disponibles")
            return None
        
        print("Extrayendo métricas por subgrupos...")
        
        # Crear DataFrame para análisis
        metrics_data = []
        
        for slice_key, slice_metrics in self.eval_result.slicing_metrics().items():
            row = {}
            
            # Extraer información del slice
            if slice_key:
                for feature, value in slice_key:
                    if feature == 'region':
                        row['region'] = value.decode('utf-8')
                    else:
                        row[feature] = value
            else:
                row['region'] = 'overall'
            
            # Extraer métricas
            for metric_name, metric_value in slice_metrics.items():
                if 'doubleValue' in metric_value:
                    row[metric_name] = metric_value['doubleValue']
                elif 'confusionMatrix' in metric_value:
                    # Procesar matriz de confusión
                    cm = metric_value['confusionMatrix']
                    row['true_negatives'] = cm['falseNegatives']  # TFMA usa nombres diferentes
                    row['false_positives'] = cm['falsePositives']
                    row['false_negatives'] = cm['falseNegatives']
                    row['true_positives'] = cm['truePositives']
            
            metrics_data.append(row)
        
        # Convertir a DataFrame
        self.subgroup_metrics = pd.DataFrame(metrics_data)
        
        print(f"Métricas extraídas para {len(metrics_data)} subgrupos")
        print(f"Columnas disponibles: {list(self.subgroup_metrics.columns)}")
        
        return self.subgroup_metrics
    
    def calculate_derived_metrics(self):
        """Calcular métricas derivadas"""
        if self.subgroup_metrics is None:
            return None
        
        print("Calculando métricas derivadas...")
        
        # Calcular métricas adicionales
        df = self.subgroup_metrics.copy()
        
        # False Positive Rate y False Negative Rate
        df['false_positive_rate'] = df['false_positives'] / (
            df['false_positives'] + df['true_negatives']
        ).replace(0, np.nan)
        
        df['false_negative_rate'] = df['false_negatives'] / (
            df['false_negatives'] + df['true_positives']
        ).replace(0, np.nan)
        
        # Specificity (True Negative Rate)
        df['specificity'] = df['true_negatives'] / (
            df['true_negatives'] + df['false_positives']
        ).replace(0, np.nan)
        
        # Balance accuracy
        df['balanced_accuracy'] = (df['binary_accuracy'] + df['specificity']) / 2
        
        # Matthews correlation coefficient
        tp, tn, fp, fn = df['true_positives'], df['true_negatives'], df['false_positives'], df['false_negatives']
        df['matthews_corrcoef'] = (tp * tn - fp * fn) / np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        df['matthews_corrcoef'] = df['matthews_corrcoef'].replace(0, np.nan)
        
        # F-beta scores
        beta = 2  # Dar más peso a recall
        precision = df['precision']
        recall = df['recall']
        df['f2_score'] = (1 + beta**2) * (precision * recall) / ((beta**2 * precision) + recall)
        df['f2_score'] = df['f2_score'].replace(0, np.nan)
        
        self.subgroup_metrics = df
        
        print("Métricas derivadas calculadas")
        return df
    
    def generate_comprehensive_analysis(self):
        """Generar análisis comprehensivo de subgrupos"""
        if self.subgroup_metrics is None:
            return None
        
        print("Generando análisis comprehensivo...")
        
        # Estadísticas descriptivas por región
        region_stats = self.subgroup_metrics.groupby('region').agg({
            'binary_accuracy': ['mean', 'std'],
            'precision': ['mean', 'std'],
            'recall': ['mean', 'std'],
            'auc': ['mean', 'std'],
            'pr_auc': ['mean', 'std'],
            'example_count': ['sum', 'mean']
        }).round(4)
        
        # Identificar mejores y peores regiones
        best_accuracy = self.subgroup_metrics.loc[self.subgroup_metrics['binary_accuracy'].idxmax()]
        worst_accuracy = self.subgroup_metrics.loc[self.subgroup_metrics['binary_accuracy'].idxmin()]
        
        best_auc = self.subgroup_metrics.loc[self.subgroup_metrics['auc'].idxmax()]
        worst_auc = self.subgroup_metrics.loc[self.subgroup_metrics['auc'].idxmin()]
        
        # Análisis de equidad
        fairness_metrics = self.calculate_fairness_metrics()
        
        # Crear reporte
        analysis_report = {
            'timestamp': datetime.now().isoformat(),
            'total_subgroups': len(self.subgroup_metrics),
            'regions': self.subgroup_metrics['region'].tolist(),
            'region_statistics': region_stats.to_dict(),
            'best_performing_regions': {
                'accuracy': {
                    'region': best_accuracy['region'],
                    'value': best_accuracy['binary_accuracy']
                },
                'auc': {
                    'region': best_auc['region'],
                    'value': best_auc['auc']
                }
            },
            'worst_performing_regions': {
                'accuracy': {
                    'region': worst_accuracy['region'],
                    'value': worst_accuracy['binary_accuracy']
                },
                'auc': {
                    'region': worst_auc['region'],
                    'value': worst_auc['auc']
                }
            },
            'fairness_metrics': fairness_metrics,
            'recommendations': self.generate_fairness_recommendations(fairness_metrics)
        }
        
        # Guardar reporte
        with open('outputs/subgroup_analysis_report.json', 'w') as f:
            json.dump(analysis_report, f, indent=2, default=str)
        
        print("Reporte de análisis guardado en outputs/subgroup_analysis_report.json")
        
        return analysis_report
    
    def calculate_fairness_metrics(self):
        """Calcular métricas de equidad"""
        print("Calculando métricas de equidad...")
        
        df = self.subgroup_metrics.copy()
        
        # Excluir el slice 'overall' para análisis de equidad
        df_regions = df[df['region'] != 'overall'].copy()
        
        if len(df_regions) == 0:
            return {}
        
        fairness_metrics = {}
        
        # Disparidad en accuracy
        accuracy_max = df_regions['binary_accuracy'].max()
        accuracy_min = df_regions['binary_accuracy'].min()
        fairness_metrics['accuracy_disparity'] = accuracy_max - accuracy_min
        
        # Disparidad en AUC
        auc_max = df_regions['auc'].max()
        auc_min = df_regions['auc'].min()
        fairness_metrics['auc_disparity'] = auc_max - auc_min
        
        # Coeficiente de variación (CV)
        fairness_metrics['accuracy_cv'] = df_regions['binary_accuracy'].std() / df_regions['binary_accuracy'].mean()
        fairness_metrics['auc_cv'] = df_regions['auc'].std() / df_regions['auc'].mean()
        
        # Índice de equidad (1 - disparidad_normalizada)
        max_possible_disparity = 1.0
        fairness_metrics['fairness_index'] = 1 - (fairness_metrics['accuracy_disparity'] / max_possible_disparity)
        
        # Análisis por par de regiones
        region_pairs = []
        regions = df_regions['region'].unique()
        for i, region1 in enumerate(regions):
            for region2 in regions[i+1:]:
                metrics1 = df_regions[df_regions['region'] == region1].iloc[0]
                metrics2 = df_regions[df_regions['region'] == region2].iloc[0]
                
                region_pairs.append({
                    'region1': region1,
                    'region2': region2,
                    'accuracy_diff': abs(metrics1['binary_accuracy'] - metrics2['binary_accuracy']),
                    'auc_diff': abs(metrics1['auc'] - metrics2['auc']),
                    'precision_diff': abs(metrics1['precision'] - metrics2['precision']),
                    'recall_diff': abs(metrics1['recall'] - metrics2['recall'])
                })
        
        fairness_metrics['region_pairwise_comparisons'] = region_pairs
        
        return fairness_metrics
    
    def generate_fairness_recommendations(self, fairness_metrics):
        """Generar recomendaciones basadas en métricas de equidad"""
        recommendations = []
        
        # Umbral para disparidad aceptable
        disparity_threshold = 0.10  # 10%
        cv_threshold = 0.15  # 15%
        
        # Recomendaciones por disparidad en accuracy
        if fairness_metrics.get('accuracy_disparity', 0) > disparity_threshold:
            recommendations.append({
                'type': 'fairness',
                'metric': 'accuracy_disparity',
                'value': fairness_metrics['accuracy_disparity'],
                'threshold': disparity_threshold,
                'priority': 'high',
                'recommendation': 'La disparidad en accuracy entre regiones es alta. Considerar recolectar más datos balanceados o ajustar el modelo por región.'
            })
        
        # Recomendaciones por disparidad en AUC
        if fairness_metrics.get('auc_disparity', 0) > disparity_threshold:
            recommendations.append({
                'type': 'fairness',
                'metric': 'auc_disparity',
                'value': fairness_metrics['auc_disparity'],
                'threshold': disparity_threshold,
                'priority': 'high',
                'recommendation': 'La disparidad en AUC entre regiones es alta. Considerar técnicas de equidad como reponderación o modelos específicos por región.'
            })
        
        # Recomendaciones por coeficiente de variación
        if fairness_metrics.get('accuracy_cv', 0) > cv_threshold:
            recommendations.append({
                'type': 'fairness',
                'metric': 'accuracy_cv',
                'value': fairness_metrics['accuracy_cv'],
                'threshold': cv_threshold,
                'priority': 'medium',
                'recommendation': 'Alta variabilidad en el rendimiento entre regiones. Implementar monitoreo continuo por subgrupo.'
            })
        
        # Índice de equidad bajo
        if fairness_metrics.get('fairness_index', 1) < 0.8:
            recommendations.append({
                'type': 'fairness',
                'metric': 'fairness_index',
                'value': fairness_metrics['fairness_index'],
                'threshold': 0.8,
                'priority': 'high',
                'recommendation': 'El índice de equidad es bajo. Implementar auditorías regulares de sesgo algorítmico.'
            })
        
        return recommendations
    
    def create_visualizations(self):
        """Crear visualizaciones del análisis de subgrupos"""
        if self.subgroup_metrics is None:
            return
        
        print("Creando visualizaciones de subgrupos...")
        
        # Configurar estilo
        plt.style.use('seaborn-v0_8')
        
        # 1. Gráfico de barras comparativo
        self.create_comparison_plots()
        
        # 2. Gráfico de dispersión (accuracy vs AUC)
        self.create_scatter_plots()
        
        # 3. Heatmap de correlaciones
        self.create_correlation_heatmap()
        
        # 4. Gráfico de disparidad
        self.create_disparity_plots()
        
        print("Visualizaciones guardadas en outputs/")
    
    def create_comparison_plots(self):
        """Crear gráficos comparativos de métricas"""
        df = self.subgroup_metrics.copy()
        
        # Excluir 'overall' para mejor visualización
        df_regions = df[df['region'] != 'overall'].copy()
        
        if len(df_regions) == 0:
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Métricas por Región - Comparación Detallada', fontsize=16, fontweight='bold')
        
        # Configurar paleta de colores
        colors = sns.color_palette("husl", len(df_regions))
        
        # Accuracy
        ax1 = axes[0, 0]
        bars1 = ax1.bar(df_regions['region'], df_regions['binary_accuracy'], color=colors)
        ax1.set_title('Accuracy por Región')
        ax1.set_ylabel('Accuracy')
        ax1.set_ylim(0, 1)
        ax1.set_xticklabels(df_regions['region'], rotation=45)
        # Añadir valores sobre las barras
        for bar, value in zip(bars1, df_regions['binary_accuracy']):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom')
        
        # Precision
        ax2 = axes[0, 1]
        bars2 = ax2.bar(df_regions['region'], df_regions['precision'], color=colors)
        ax2.set_title('Precision por Región')
        ax2.set_ylabel('Precision')
        ax2.set_ylim(0, 1)
        ax2.set_xticklabels(df_regions['region'], rotation=45)
        for bar, value in zip(bars2, df_regions['precision']):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom')
        
        # Recall
        ax3 = axes[0, 2]
        bars3 = ax3.bar(df_regions['region'], df_regions['recall'], color=colors)
        ax3.set_title('Recall por Región')
        ax3.set_ylabel('Recall')
        ax3.set_ylim(0, 1)
        ax3.set_xticklabels(df_regions['region'], rotation=45)
        for bar, value in zip(bars3, df_regions['recall']):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom')
        
        # ROC-AUC
        ax4 = axes[1, 0]
        bars4 = ax4.bar(df_regions['region'], df_regions['auc'], color=colors)
        ax4.set_title('ROC-AUC por Región')
        ax4.set_ylabel('ROC-AUC')
        ax4.set_ylim(0, 1)
        ax4.set_xticklabels(df_regions['region'], rotation=45)
        for bar, value in zip(bars4, df_regions['auc']):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom')
        
        # PR-AUC
        ax5 = axes[1, 1]
        bars5 = ax5.bar(df_regions['region'], df_regions['pr_auc'], color=colors)
        ax5.set_title('PR-AUC por Región')
        ax5.set_ylabel('PR-AUC')
        ax5.set_ylim(0, 1)
        ax5.set_xticklabels(df_regions['region'], rotation=45)
        for bar, value in zip(bars5, df_regions['pr_auc']):
            ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom')
        
        # Número de ejemplos
        ax6 = axes[1, 2]
        bars6 = ax6.bar(df_regions['region'], df_regions['example_count'], color=colors)
        ax6.set_title('Número de Ejemplos por Región')
        ax6.set_ylabel('Cantidad')
        ax6.set_xticklabels(df_regions['region'], rotation=45)
        for bar, value in zip(bars6, df_regions['example_count']):
            ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(df_regions['example_count'])*0.01,
                    f'{int(value)}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('outputs/subgroup_detailed_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_scatter_plots(self):
        """Crear gráficos de dispersión"""
        df = self.subgroup_metrics.copy()
        df_regions = df[df['region'] != 'overall'].copy()
        
        if len(df_regions) < 2:
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Accuracy vs AUC
        scatter1 = ax1.scatter(df_regions['binary_accuracy'], df_regions['auc'], 
                            s=100, alpha=0.7, c=range(len(df_regions)), cmap='viridis')
        ax1.set_xlabel('Accuracy')
        ax1.set_ylabel('ROC-AUC')
        ax1.set_title('Accuracy vs ROC-AUC por Región')
        ax1.grid(True, alpha=0.3)
        
        # Añadir etiquetas
        for i, region in enumerate(df_regions['region']):
            ax1.annotate(region, (df_regions.iloc[i]['binary_accuracy'], df_regions.iloc[i]['auc']),
                        xytext=(5, 5), textcoords='offset points')
        
        # Precision vs Recall
        scatter2 = ax2.scatter(df_regions['precision'], df_regions['recall'], 
                            s=100, alpha=0.7, c=range(len(df_regions)), cmap='viridis')
        ax2.set_xlabel('Precision')
        ax2.set_ylabel('Recall')
        ax2.set_title('Precision vs Recall por Región')
        ax2.grid(True, alpha=0.3)
        
        # Añadir etiquetas
        for i, region in enumerate(df_regions['region']):
            ax2.annotate(region, (df_regions.iloc[i]['precision'], df_regions.iloc[i]['recall']),
                        xytext=(5, 5), textcoords='offset points')
        
        plt.tight_layout()
        plt.savefig('outputs/subgroup_scatter_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_correlation_heatmap(self):
        """Crear heatmap de correlaciones entre métricas"""
        df = self.subgroup_metrics.copy()
        
        # Seleccionar métricas numéricas
        numeric_metrics = [
            'binary_accuracy', 'precision', 'recall', 'auc', 'pr_auc',
            'false_positive_rate', 'false_negative_rate', 'specificity',
            'balanced_accuracy', 'matthews_corrcoef', 'f2_score'
        ]
        
        # Filtrar columnas existentes
        available_metrics = [m for m in numeric_metrics if m in df.columns]
        
        if len(available_metrics) < 2:
            return
        
        correlation_matrix = df[available_metrics].corr()
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                   square=True, fmt='.3f', cbar_kws={'label': 'Correlación'})
        plt.title('Matriz de Correlación entre Métricas por Subgrupo')
        plt.tight_layout()
        plt.savefig('outputs/subgroup_correlation_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_disparity_plots(self):
        """Crear gráficos de disparidad"""
        fairness_metrics = self.calculate_fairness_metrics()
        
        if not fairness_metrics:
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Disparidad en métricas principales
        metrics = ['accuracy_disparity', 'auc_disparity']
        values = [fairness_metrics.get(m, 0) for m in metrics]
        
        bars1 = ax1.bar(metrics, values, color=['#ff7f0e', '#2ca02c'])
        ax1.set_title('Disparidad entre Regiones')
        ax1.set_ylabel('Disparidad (diferencia máxima)')
        ax1.set_ylim(0, max(values) * 1.2)
        
        # Añadir línea de umbral
        threshold = 0.10
        ax1.axhline(y=threshold, color='red', linestyle='--', alpha=0.7, label=f'Umbral aceptable ({threshold})')
        ax1.legend()
        
        # Coeficientes de variación
        cv_metrics = ['accuracy_cv', 'auc_cv']
        cv_values = [fairness_metrics.get(m, 0) for m in cv_metrics]
        
        bars2 = ax2.bar(cv_metrics, cv_values, color=['#1f77b4', '#d62728'])
        ax2.set_title('Coeficiente de Variación por Métrica')
        ax2.set_ylabel('CV (desviación estándar / media)')
        ax2.set_ylim(0, max(cv_values) * 1.2)
        
        # Añadir línea de umbral
        cv_threshold = 0.15
        ax2.axhline(y=cv_threshold, color='red', linestyle='--', alpha=0.7, label=f'Umbral aceptable ({cv_threshold})')
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig('outputs/subgroup_disparity_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def print_detailed_summary(self):
        """Imprimir resumen detallado del análisis"""
        if self.subgroup_metrics is None:
            return
        
        print("\n" + "=" * 80)
        print("ANÁLISIS DETALLADO POR SUBGRUPOS")
        print("=" * 80)
        
        # Tabla de métricas principales
        print("\nMÉTRICAS PRINCIPALES POR REGIÓN:")
        print("-" * 80)
        
        df_display = self.subgroup_metrics[['region', 'binary_accuracy', 'precision', 'recall', 
                                     'auc', 'pr_auc', 'example_count']].copy()
        df_display = df_display.round(4)
        print(df_display.to_string(index=False))
        
        # Análisis de equidad
        fairness_metrics = self.calculate_fairness_metrics()
        if fairness_metrics:
            print("\nMÉTRICAS DE EQUIDAD:")
            print("-" * 80)
            print(f"Disparidad en Accuracy: {fairness_metrics.get('accuracy_disparity', 0):.4f}")
            print(f"Disparidad en AUC: {fairness_metrics.get('auc_disparity', 0):.4f}")
            print(f"CV en Accuracy: {fairness_metrics.get('accuracy_cv', 0):.4f}")
            print(f"CV en AUC: {fairness_metrics.get('auc_cv', 0):.4f}")
            print(f"Índice de Equidad: {fairness_metrics.get('fairness_index', 0):.4f}")
        
        # Recomendaciones
        recommendations = self.generate_fairness_recommendations(fairness_metrics)
        if recommendations:
            print("\nRECOMENDACIONES:")
            print("-" * 80)
            for i, rec in enumerate(recommendations, 1):
                print(f"{i}. [{rec['priority'].upper()}] {rec['recommendation']}")
                print(f"   Métrica: {rec['metric']} = {rec['value']:.4f} (umbral: {rec['threshold']})")
        
        print("\n" + "=" * 80)

def main():
    """Función principal para el análisis de subgrupos"""
    print("=" * 80)
    print("ANÁLISIS DETALLADO POR SUBGRUPOS - EVALUACIÓN DE MODELOS")
    print("=" * 80)
    
    # Crear analizador
    analyzer = SubgroupAnalyzer()
    
    # Cargar resultados de TFMA
    if not analyzer.load_tfma_results():
        return
    
    # Extraer métricas
    analyzer.extract_subgroup_metrics()
    
    # Calcular métricas derivadas
    analyzer.calculate_derived_metrics()
    
    # Generar análisis comprehensivo
    analyzer.generate_comprehensive_analysis()
    
    # Crear visualizaciones
    analyzer.create_visualizations()
    
    # Imprimir resumen
    analyzer.print_detailed_summary()
    
    print("\n" + "=" * 80)
    print("ANÁLISIS DE SUBGRUPOS COMPLETADO")
    print("Revisa los archivos generados en el directorio outputs/")
    print("=" * 80)

if __name__ == "__main__":
    main()
