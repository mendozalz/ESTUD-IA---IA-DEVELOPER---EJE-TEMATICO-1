"""
Laboratorio 5.4 - Análisis de Resultados NAS
===============================================

Este script analiza en profundidad los resultados de Neural
Architecture Search y genera insights sobre las arquitecturas encontradas.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Tuple
import json
import networkx as nx
from collections import defaultdict
import itertools

class NASAnalyzer:
    """
    Clase para analizar resultados de Neural Architecture Search
    """
    
    def __init__(self):
        """
        Inicializa el analizador NAS
        """
        self.search_results = {}
        self.architecture_patterns = {}
        self.performance_analysis = {}
        self.insights = {}
        
    def load_search_results(self, results_file: str = 'nas_results.json'):
        """
        Carga resultados de NAS desde archivo JSON
        
        Args:
            results_file: Ruta del archivo de resultados
        """
        print(f"📂 Cargando resultados desde {results_file}...")
        
        try:
            with open(results_file, 'r') as f:
                self.search_results = json.load(f)
            print("✅ Resultados cargados exitosamente")
        except FileNotFoundError:
            print(f"⚠️ Archivo {results_file} no encontrado")
            self.search_results = {}
    
    def analyze_architecture_patterns(self) -> Dict[str, Any]:
        """
        Analiza patrones en las arquitecturas encontradas
        
        Returns:
            Análisis de patrones arquitectónicos
        """
        print("🔍 Analizando patrones arquitectónicos...")
        
        if not self.search_results or 'nas_results' not in self.search_results:
            print("⚠️ No hay resultados de NAS para analizar")
            return {}
        
        nas_results = self.search_results['nas_results']
        
        # Extraer información de trials
        trials_info = []
        
        if 'search_summary' in nas_results and 'top_5_trials' in nas_results['search_summary']:
            for trial in nas_results['search_summary']['top_5_trials']:
                trials_info.append({
                    'trial_id': trial['trial_id'],
                    'score': trial['score'],
                    'hyperparameters': trial['hyperparameters']
                })
        
        # Analizar patrones de capas
        layer_patterns = self._analyze_layer_patterns(trials_info)
        
        # Analizar patrones de hiperparámetros
        hyperparameter_patterns = self._analyze_hyperparameter_patterns(trials_info)
        
        # Analizar correlaciones
        correlations = self._analyze_parameter_correlations(trials_info)
        
        # Identificar arquitecturas exitosas
        successful_architectures = self._identify_successful_architectures(trials_info)
        
        patterns_analysis = {
            'layer_patterns': layer_patterns,
            'hyperparameter_patterns': hyperparameter_patterns,
            'correlations': correlations,
            'successful_architectures': successful_architectures,
            'total_trials_analyzed': len(trials_info)
        }
        
        self.architecture_patterns = patterns_analysis
        
        print("✅ Análisis de patrones completado")
        
        return patterns_analysis
    
    def _analyze_layer_patterns(self, trials_info: List[Dict]) -> Dict[str, Any]:
        """
        Analiza patrones de capas en los trials
        
        Args:
            trials_info: Información de trials
            
        Returns:
            Patrones de capas identificados
        """
        layer_stats = defaultdict(lambda: defaultdict(int))
        layer_sequences = []
        
        for trial in trials_info:
            hyperparams = trial['hyperparameters']
            
            # Extraer información de capas (depende del formato de AutoKeras)
            layers = []
            for key, value in hyperparams.items():
                if 'layer' in key.lower() or 'block' in key.lower():
                    layer_type = key.split('_')[0]
                    layers.append(layer_type)
                    layer_stats[layer_type]['count'] += 1
                    layer_stats[layer_type]['total_score'] += trial['score']
            
            layer_sequences.append(layers)
        
        # Calcular estadísticas
        for layer_type in layer_stats:
            layer_stats[layer_type]['avg_score'] = (
                layer_stats[layer_type]['total_score'] / layer_stats[layer_type]['count']
            )
            layer_stats[layer_type]['frequency'] = (
                layer_stats[layer_type]['count'] / len(trials_info)
            )
        
        # Analizar secuencias comunes
        sequence_patterns = self._find_common_sequences(layer_sequences)
        
        return {
            'layer_statistics': dict(layer_stats),
            'common_sequences': sequence_patterns,
            'layer_diversity': len(layer_stats)
        }
    
    def _analyze_hyperparameter_patterns(self, trials_info: List[Dict]) -> Dict[str, Any]:
        """
        Analiza patrones de hiperparámetros
        
        Args:
            trials_info: Información de trials
            
        Returns:
            Patrones de hiperparámetros
        """
        param_stats = defaultdict(list)
        
        for trial in trials_info:
            hyperparams = trial['hyperparameters']
            
            for param, value in hyperparams.items():
                if isinstance(value, (int, float)):
                    param_stats[param].append((value, trial['score']))
        
        # Analizar distribución y correlación con performance
        param_analysis = {}
        
        for param, values_scores in param_stats.items():
            values = [vs[0] for vs in values_scores]
            scores = [vs[1] for vs in values_scores]
            
            param_analysis[param] = {
                'values': values,
                'scores': scores,
                'mean_value': np.mean(values),
                'std_value': np.std(values),
                'correlation_with_score': np.corrcoef(values, scores)[0, 1] if len(values) > 1 else 0,
                'optimal_range': self._find_optimal_range(values, scores)
            }
        
        return param_analysis
    
    def _analyze_parameter_correlations(self, trials_info: List[Dict]) -> Dict[str, float]:
        """
        Analiza correlaciones entre parámetros
        
        Args:
            trials_info: Información de trials
            
        Returns:
            Matriz de correlaciones
        """
        # Extraer parámetros numéricos
        param_matrix = []
        param_names = []
        
        for trial in trials_info:
            hyperparams = trial['hyperparameters']
            numeric_params = []
            
            for param, value in sorted(hyperparams.items()):
                if isinstance(value, (int, float)):
                    if not param_names:
                        param_names.append(param)
                    elif param != param_names[-1]:
                        param_names.append(param)
                    numeric_params.append(value)
            
            if numeric_params:
                param_matrix.append(numeric_params)
        
        if not param_matrix:
            return {}
        
        # Calcular matriz de correlación
        param_array = np.array(param_matrix)
        correlation_matrix = np.corrcoef(param_array.T)
        
        # Convertir a diccionario
        correlations = {}
        for i, param1 in enumerate(param_names):
            for j, param2 in enumerate(param_names):
                if i < j:  # Solo guardar pares únicos
                    correlations[f"{param1}_vs_{param2}"] = correlation_matrix[i, j]
        
        return correlations
    
    def _identify_successful_architectures(self, trials_info: List[Dict]) -> Dict[str, Any]:
        """
        Identifica características de arquitecturas exitosas
        
        Args:
            trials_info: Información de trials
            
        Returns:
            Características de arquitecturas exitosas
        """
        if not trials_info:
            return {}
        
        # Ordenar por score (mejor primero)
        sorted_trials = sorted(trials_info, key=lambda x: x['score'])
        
        # Top 10% y bottom 10%
        top_10_percent = int(len(sorted_trials) * 0.1)
        bottom_10_percent = int(len(sorted_trials) * 0.1)
        
        top_trials = sorted_trials[:top_10_percent]
        bottom_trials = sorted_trials[-bottom_10_percent:]
        
        # Analizar diferencias
        successful_features = self._compare_trial_groups(top_trials, bottom_trials)
        
        return {
            'top_trials': top_trials,
            'bottom_trials': bottom_trials,
            'successful_features': successful_features,
            'best_trial': sorted_trials[0],
            'worst_trial': sorted_trials[-1]
        }
    
    def _compare_trial_groups(self, top_trials: List[Dict], bottom_trials: List[Dict]) -> Dict[str, Any]:
        """
        Compara características entre grupos de trials
        
        Args:
            top_trials: Trials con mejor performance
            bottom_trials: Trials con peor performance
            
        Returns:
            Diferencias significativas
        """
        differences = {}
        
        # Extraer hiperparámetros comunes
        if top_trials and bottom_trials:
            top_params = top_trials[0]['hyperparameters'].keys()
            bottom_params = bottom_trials[0]['hyperparameters'].keys()
            common_params = set(top_params) & set(bottom_params)
            
            for param in common_params:
                top_values = [trial['hyperparameters'][param] for trial in top_trials 
                            if isinstance(trial['hyperparameters'][param], (int, float))]
                bottom_values = [trial['hyperparameters'][param] for trial in bottom_trials 
                                if isinstance(trial['hyperparameters'][param], (int, float))]
                
                if top_values and bottom_values:
                    differences[param] = {
                        'top_mean': np.mean(top_values),
                        'bottom_mean': np.mean(bottom_values),
                        'difference': np.mean(top_values) - np.mean(bottom_values),
                        'significance': abs(np.mean(top_values) - np.mean(bottom_values)) > 0.1
                    }
        
        return differences
    
    def _find_common_sequences(self, layer_sequences: List[List[str]]) -> List[Dict]:
        """
        Encuentra secuencias comunes de capas
        
        Args:
            layer_sequences: Secuencias de capas por trial
            
        Returns:
            Secuencias comunes con frecuencia
        """
        sequence_counts = defaultdict(int)
        
        for sequence in layer_sequences:
            # Generar todas las subsecuencias de longitud 2-4
            for length in range(2, min(5, len(sequence) + 1)):
                for i in range(len(sequence) - length + 1):
                    subsequence = tuple(sequence[i:i+length])
                    sequence_counts[subsequence] += 1
        
        # Ordenar por frecuencia
        common_sequences = [
            {
                'sequence': seq,
                'frequency': count,
                'relative_frequency': count / len(layer_sequences)
            }
            for seq, count in sorted(sequence_counts.items(), key=lambda x: x[1], reverse=True)
        ]
        
        return common_sequences[:10]  # Top 10 secuencias
    
    def _find_optimal_range(self, values: List[float], scores: List[float]) -> Dict[str, float]:
        """
        Encuentra rango óptimo para un parámetro
        
        Args:
            values: Valores del parámetro
            scores: Scores correspondientes
            
        Returns:
            Rango óptimo
        """
        if len(values) < 3:
            return {'min': min(values), 'max': max(values), 'mean': np.mean(values)}
        
        # Ordenar por valor
        sorted_pairs = sorted(zip(values, scores))
        values_sorted, scores_sorted = zip(*sorted_pairs)
        
        # Encontrar rango con mejor score promedio
        best_score = -np.inf
        best_range = {'min': min(values), 'max': max(values)}
        
        window_size = max(3, len(values) // 4)
        
        for i in range(len(values) - window_size + 1):
            window_values = values_sorted[i:i+window_size]
            window_scores = scores_sorted[i:i+window_size]
            avg_score = np.mean(window_scores)
            
            if avg_score > best_score:
                best_score = avg_score
                best_range = {
                    'min': min(window_values),
                    'max': max(window_values),
                    'mean': np.mean(window_values),
                    'score': avg_score
                }
        
        return best_range
    
    def analyze_search_efficiency(self) -> Dict[str, Any]:
        """
        Analiza la eficiencia de la búsqueda NAS
        
        Returns:
            Análisis de eficiencia
        """
        print("⚡ Analizando eficiencia de búsqueda...")
        
        if not self.search_results:
            return {}
        
        nas_results = self.search_results['nas_results']
        
        # Métricas de eficiencia
        efficiency_metrics = {
            'search_time': nas_results.get('search_time', 0),
            'total_trials': nas_results.get('search_summary', {}).get('total_trials', 0),
            'trials_per_second': 0,
            'convergence_rate': 0,
            'search_space_coverage': 0
        }
        
        if efficiency_metrics['search_time'] > 0:
            efficiency_metrics['trials_per_second'] = (
                efficiency_metrics['total_trials'] / efficiency_metrics['search_time']
            )
        
        # Análisis de convergencia
        if 'top_5_trials' in nas_results.get('search_summary', {}):
            trials = nas_results['search_summary']['top_5_trials']
            scores = [trial['score'] for trial in trials]
            
            if len(scores) > 1:
                # Calcular tasa de mejora
                score_improvement = (scores[0] - scores[-1]) / abs(scores[-1])
                efficiency_metrics['convergence_rate'] = score_improvement
        
        return efficiency_metrics
    
    def generate_architecture_recommendations(self) -> Dict[str, Any]:
        """
        Genera recomendaciones basadas en el análisis
        
        Returns:
            Recomendaciones de arquitectura
        """
        print("💡 Generando recomendaciones de arquitectura...")
        
        recommendations = {
            'layer_recommendations': {},
            'parameter_recommendations': {},
            'architecture_patterns': {},
            'optimization_suggestions': []
        }
        
        if not self.architecture_patterns:
            return recommendations
        
        # Recomendaciones de capas
        layer_stats = self.architecture_patterns.get('layer_patterns', {}).get('layer_statistics', {})
        
        for layer_type, stats in layer_stats.items():
            if stats['frequency'] > 0.5 and stats['avg_score'] < 0.1:  # Frecuente y buen score
                recommendations['layer_recommendations'][layer_type] = {
                    'recommended': True,
                    'reason': f"Alta frecuencia ({stats['frequency']:.2f}) y buen performance ({stats['avg_score']:.4f})"
                }
        
        # Recomendaciones de parámetros
        param_patterns = self.architecture_patterns.get('hyperparameter_patterns', {})
        
        for param, analysis in param_patterns.items():
            if abs(analysis['correlation_with_score']) > 0.3:
                recommendations['parameter_recommendations'][param] = {
                    'optimal_range': analysis['optimal_range'],
                    'correlation': analysis['correlation_with_score'],
                    'recommendation': 'Ajustar a este rango para mejor performance'
                }
        
        # Patrones de arquitectura
        common_sequences = self.architecture_patterns.get('layer_patterns', {}).get('common_sequences', [])
        
        if common_sequences:
            recommendations['architecture_patterns'] = {
                'most_common': common_sequences[0],
                'recommendation': 'Considerar usar esta secuencia como base'
            }
        
        # Sugerencias de optimización
        efficiency = self.analyze_search_efficiency()
        
        if efficiency.get('trials_per_second', 0) < 1:
            recommendations['optimization_suggestions'].append(
                "Considerar optimizar la velocidad de búsqueda para reducir tiempo total"
            )
        
        if efficiency.get('convergence_rate', 0) < 0.1:
            recommendations['optimization_suggestions'].append(
                "La búsqueda convergió lentamente, considerar ajustar estrategia de búsqueda"
            )
        
        return recommendations
    
    def visualize_nas_results(self, save_plots: bool = False):
        """
        Visualiza resultados de NAS
        
        Args:
            save_plots: Si guardar los gráficos
        """
        print("📊 Visualizando resultados de NAS...")
        
        if not self.search_results:
            print("⚠️ No hay resultados para visualizar")
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. Scores de trials
        if 'search_summary' in self.search_results['nas_results'] and 'top_5_trials' in self.search_results['nas_results']['search_summary']:
            trials = self.search_results['nas_results']['search_summary']['top_5_trials']
            trial_ids = [trial['trial_id'] for trial in trials]
            scores = [trial['score'] for trial in trials]
            
            axes[0, 0].plot(trial_ids, scores, 'o-', color='blue')
            axes[0, 0].set_title('Scores de Trials')
            axes[0, 0].set_xlabel('Trial ID')
            axes[0, 0].set_ylabel('Score')
            axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Distribución de scores
        if 'top_5_trials' in self.search_results['nas_results'].get('search_summary', {}):
            scores = [trial['score'] for trial in self.search_results['nas_results']['search_summary']['top_5_trials']]
            axes[0, 1].hist(scores, bins=10, alpha=0.7, color='skyblue')
            axes[0, 1].set_title('Distribución de Scores')
            axes[0, 1].set_xlabel('Score')
            axes[0, 1].set_ylabel('Frecuencia')
        
        # 3. Eficiencia de búsqueda
        efficiency = self.analyze_search_efficiency()
        if efficiency:
            metrics = ['search_time', 'total_trials', 'trials_per_second']
            values = [efficiency.get(m, 0) for m in metrics]
            
            axes[0, 2].bar(metrics, values, color=['orange', 'green', 'purple'])
            axes[0, 2].set_title('Métricas de Eficiencia')
            axes[0, 2].tick_params(axis='x', rotation=45)
        
        # 4. Patrones de capas (si disponibles)
        if self.architecture_patterns and 'layer_patterns' in self.architecture_patterns:
            layer_stats = self.architecture_patterns['layer_patterns'].get('layer_statistics', {})
            
            if layer_stats:
                layer_names = list(layer_stats.keys())
                frequencies = [layer_stats[name]['frequency'] for name in layer_names]
                
                axes[1, 0].bar(layer_names, frequencies, color='lightcoral')
                axes[1, 0].set_title('Frecuencia de Tipos de Capa')
                axes[1, 0].tick_params(axis='x', rotation=45)
        
        # 5. Correlaciones de parámetros
        if self.architecture_patterns and 'correlations' in self.architecture_patterns:
            correlations = self.architecture_patterns['correlations']
            
            if correlations:
                param_pairs = list(correlations.keys())[:10]  # Top 10
                corr_values = list(correlations.values())[:10]
                
                colors = ['red' if abs(c) > 0.5 else 'blue' for c in corr_values]
                axes[1, 1].bar(range(len(param_pairs)), corr_values, color=colors, alpha=0.7)
                axes[1, 1].set_title('Correlaciones de Parámetros')
                axes[1, 1].set_xticks(range(len(param_pairs)))
                axes[1, 1].set_xticklabels([p[:15] + '...' if len(p) > 15 else p for p in param_pairs], rotation=45, ha='right')
                axes[1, 1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # 6. Comparación NAS vs Modelos Manuales
        if 'comparison' in self.search_results:
            comparison = self.search_results['comparison']
            
            models = ['NAS'] + list(comparison.get('manual_models', {}).keys())
            metrics = [comparison.get('nas_model', {}).get('test_metric', 0)]
            
            for manual_model in comparison.get('manual_models', {}):
                metrics.append(comparison['manual_models'][manual_model].get('test_metric', 0))
            
            colors = ['green' if i == 0 else 'blue' for i in range(len(models))]
            axes[1, 2].bar(models, metrics, color=colors, alpha=0.7)
            axes[1, 2].set_title('NAS vs Modelos Manuales')
            axes[1, 2].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_plots:
            plt.savefig('nas_analysis.png', dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def generate_insights_report(self, filepath: str = 'nas_insights.json'):
        """
        Genera reporte de insights del NAS
        
        Args:
            filepath: Ruta del archivo JSON
        """
        print("📄 Generando reporte de insights...")
        
        # Analizar patrones
        patterns = self.analyze_architecture_patterns()
        
        # Analizar eficiencia
        efficiency = self.analyze_search_efficiency()
        
        # Generar recomendaciones
        recommendations = self.generate_architecture_recommendations()
        
        # Insights clave
        key_insights = []
        
        if patterns:
            diversity = patterns.get('layer_diversity', 0)
            key_insights.append(f"Diversidad de arquitecturas exploradas: {diversity} tipos de capas")
        
        if efficiency:
            if efficiency.get('trials_per_second', 0) > 1:
                key_insights.append("Búsqueda eficiente: más de 1 trial por segundo")
            else:
                key_insights.append("Búsqueda podría optimizarse para mayor velocidad")
        
        if self.search_results and 'comparison' in self.search_results:
            comparison = self.search_results['comparison']
            improvement = comparison.get('improvement', {})
            
            for model_type, improvement_pct in improvement.items():
                if improvement_pct > 0:
                    key_insights.append(f"NAS supera a {model_type} en {improvement_pct:.1f}%")
        
        insights_report = {
            'analysis_summary': {
                'total_trials_analyzed': patterns.get('total_trials_analyzed', 0),
                'search_efficiency': efficiency,
                'architecture_diversity': patterns.get('layer_diversity', 0)
            },
            'architecture_patterns': patterns,
            'efficiency_analysis': efficiency,
            'recommendations': recommendations,
            'key_insights': key_insights,
            'next_steps': [
                "Considerar ejecutar más trials con patrones identificados",
                "Optimizar hiperparámetros según rangos recomendados",
                "Explorar arquitecturas híbridas basadas en patrones comunes"
            ]
        }
        
        # Guardar reporte
        with open(filepath, 'w') as f:
            json.dump(insights_report, f, indent=2, default=str)
        
        print(f"✅ Reporte de insights guardado en {filepath}")
        
        return insights_report


def main():
    """
    Función principal para demostrar análisis NAS
    """
    print("🚀 Iniciando análisis de resultados NAS")
    
    # Crear analizador
    analyzer = NASAnalyzer()
    
    # Cargar resultados (simulado)
    # analyzer.load_search_results('nas_results.json')
    
    # Analizar patrones
    patterns = analyzer.analyze_architecture_patterns()
    
    # Analizar eficiencia
    efficiency = analyzer.analyze_search_efficiency()
    
    # Generar recomendaciones
    recommendations = analyzer.generate_architecture_recommendations()
    
    # Visualizar resultados
    analyzer.visualize_nas_results()
    
    # Generar reporte
    insights = analyzer.generate_insights_report()
    
    print("✅ Análisis NAS completado")
    print("💡 En uso real, primero ejecute AutoKeras NAS y luego analice los resultados")


if __name__ == "__main__":
    main()
