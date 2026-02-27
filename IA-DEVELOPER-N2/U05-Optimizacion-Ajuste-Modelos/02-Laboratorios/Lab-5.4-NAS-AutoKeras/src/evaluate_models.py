"""
Laboratorio 5.4 - Evaluación Comparativa de Modelos
==================================================

Este script evalúa y compara los modelos NAS vs manuales
con métricas completas y análisis estadístico.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Tuple
import time
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy import stats
import json

class ModelEvaluator:
    """
    Clase para evaluación comparativa de modelos
    """
    
    def __init__(self):
        """
        Inicializa el evaluador
        """
        self.evaluation_results = {}
        self.statistical_tests = {}
        self.comparison_metrics = {}
        
    def evaluate_model_comprehensive(self, model_name: str,
                                   model, X_test: np.ndarray, y_test: np.ndarray,
                                   prediction_horizon: int = 6) -> Dict[str, Any]:
        """
        Evaluación comprehensiva de un modelo
        
        Args:
            model_name: Nombre del modelo
            model: Modelo a evaluar
            X_test: Datos de prueba
            y_test: Labels verdaderos
            prediction_horizon: Horizonte de predicción
            
        Returns:
            Resultados completos de evaluación
        """
        print(f"🧪 Evaluación comprehensiva de {model_name}...")
        
        start_time = time.time()
        
        # Predicciones
        predictions = model.predict(X_test, verbose=0)
        
        # Asegurar shapes correctos
        if len(predictions.shape) > 1 and predictions.shape[1] == 1:
            predictions = predictions.flatten()
        
        if len(y_test.shape) > 1 and y_test.shape[1] == 1:
            y_test = y_test.flatten()
        
        evaluation_time = time.time() - start_time
        
        # Métricas básicas
        mae = mean_absolute_error(y_test, predictions)
        mse = mean_squared_error(y_test, predictions)
        rmse = np.sqrt(mse)
        
        # MAPE (Mean Absolute Percentage Error)
        mape = np.mean(np.abs((y_test - predictions) / np.maximum(y_test, 1e-8))) * 100
        
        # SMAPE (Symmetric MAPE)
        smape = np.mean(2 * np.abs(predictions - y_test) / 
                       (np.abs(predictions) + np.abs(y_test) + 1e-8)) * 100
        
        # R² Score
        ss_res = np.sum((y_test - predictions) ** 2)
        ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        
        # Error relativo
        relative_error = np.abs((y_test - predictions) / np.maximum(y_test, 1e-8))
        
        # Análisis de residuos
        residuals = y_test - predictions
        
        # Métricas de residuos
        residuals_mean = np.mean(residuals)
        residuals_std = np.std(residuals)
        residuals_skew = stats.skew(residuals)
        residuals_kurtosis = stats.kurtosis(residuals)
        
        # Test de normalidad de residuos
        shapiro_stat, shapiro_p = stats.shapiro(residuals[:5000])  # Limitar a 5000 muestras
        
        # Durbin-Watson test (autocorrelación de residuos)
        durbin_watson = self._durbin_watson(residuals)
        
        # Análisis de errores por cuartil
        error_by_quartile = self._analyze_errors_by_quartile(y_test, predictions)
        
        # Análisis de errores por magnitud
        error_by_magnitude = self._analyze_errors_by_magnitude(y_test, predictions)
        
        # Análisis de predicciones extremas
        extreme_predictions = self._analyze_extreme_predictions(y_test, predictions)
        
        # Consistencia de predicciones
        prediction_consistency = self._analyze_prediction_consistency(predictions)
        
        results = {
            'model_name': model_name,
            'basic_metrics': {
                'mae': mae,
                'mse': mse,
                'rmse': rmse,
                'mape': mape,
                'smape': smape,
                'r2': r2
            },
            'error_analysis': {
                'relative_error_mean': np.mean(relative_error),
                'relative_error_std': np.std(relative_error),
                'relative_error_median': np.median(relative_error),
                'relative_error_95th': np.percentile(relative_error, 95)
            },
            'residuals_analysis': {
                'mean': residuals_mean,
                'std': residuals_std,
                'skewness': residuals_skew,
                'kurtosis': residuals_kurtosis,
                'shapiro_stat': shapiro_stat,
                'shapiro_p_value': shapiro_p,
                'durbin_watson': durbin_watson
            },
            'error_by_quartile': error_by_quartile,
            'error_by_magnitude': error_by_magnitude,
            'extreme_predictions': extreme_predictions,
            'prediction_consistency': prediction_consistency,
            'performance_metrics': {
                'evaluation_time': evaluation_time,
                'prediction_accuracy': np.mean(relative_error < 0.1),  # 10% de error
                'prediction_quality': np.mean(relative_error < 0.05)  # 5% de error
            },
            'predictions': predictions,
            'residuals': residuals
        }
        
        self.evaluation_results[model_name] = results
        
        print(f"✅ Evaluación completada para {model_name}")
        print(f"   - MAE: {mae:.4f}")
        print(f"   - RMSE: {rmse:.4f}")
        print(f"   - MAPE: {mape:.2f}%")
        print(f"   - R²: {r2:.4f}")
        
        return results
    
    def _durbin_watson(self, residuals: np.ndarray) -> float:
        """
        Calcula estadístico de Durbin-Watson
        
        Args:
            residuals: Residuos del modelo
            
        Returns:
            Estadístico Durbin-Watson
        """
        diff = np.diff(residuals)
        return np.sum(diff ** 2) / np.sum(residuals ** 2)
    
    def _analyze_errors_by_quartile(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Analiza errores por cuartil de valores verdaderos
        
        Args:
            y_true: Valores verdaderos
            y_pred: Predicciones
            
        Returns:
            Errores por cuartil
        """
        quartiles = {}
        for i in range(1, 5):
            mask = (y_true >= np.percentile(y_true, (i-1)*25)) & (y_true < np.percentile(y_true, i*25))
            if np.sum(mask) > 0:
                error = np.mean(np.abs(y_true[mask] - y_pred[mask]))
                quartiles[f'q{i}'] = error
        
        return quartiles
    
    def _analyze_errors_by_magnitude(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Analiza errores por magnitud de valores verdaderos
        
        Args:
            y_true: Valores verdaderos
            y_pred: Predicciones
            
        Returns:
            Errores por magnitud
        """
        # Definir rangos de magnitud
        ranges = {
            'low': (0, np.percentile(y_true, 33)),
            'medium': (np.percentile(y_true, 33), np.percentile(y_true, 67)),
            'high': (np.percentile(y_true, 67), np.max(y_true))
        }
        
        errors = {}
        for range_name, (low, high) in ranges.items():
            mask = (y_true >= low) & (y_true < high)
            if np.sum(mask) > 0:
                error = np.mean(np.abs(y_true[mask] - y_pred[mask]))
                errors[range_name] = error
        
        return errors
    
    def _analyze_extreme_predictions(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
        """
        Analiza predicciones extremas
        
        Args:
            y_true: Valores verdaderos
            y_pred: Predicciones
            
        Returns:
            Análisis de predicciones extremas
        """
        # Identificar valores extremos (top 5% y bottom 5%)
        high_threshold = np.percentile(y_true, 95)
        low_threshold = np.percentile(y_true, 5)
        
        high_mask = y_true >= high_threshold
        low_mask = y_true <= low_threshold
        
        extreme_analysis = {
            'high_values': {
                'count': np.sum(high_mask),
                'mean_error': np.mean(np.abs(y_true[high_mask] - y_pred[high_mask])) if np.sum(high_mask) > 0 else 0,
                'relative_error': np.mean(np.abs((y_true[high_mask] - y_pred[high_mask]) / np.maximum(y_true[high_mask], 1e-8))) if np.sum(high_mask) > 0 else 0
            },
            'low_values': {
                'count': np.sum(low_mask),
                'mean_error': np.mean(np.abs(y_true[low_mask] - y_pred[low_mask])) if np.sum(low_mask) > 0 else 0,
                'relative_error': np.mean(np.abs((y_true[low_mask] - y_pred[low_mask]) / np.maximum(y_true[low_mask], 1e-8))) if np.sum(low_mask) > 0 else 0
            }
        }
        
        return extreme_analysis
    
    def _analyze_prediction_consistency(self, predictions: np.ndarray) -> Dict[str, float]:
        """
        Analiza consistencia de predicciones
        
        Args:
            predictions: Predicciones del modelo
            
        Returns:
            Métricas de consistencia
        """
        # Cambios entre predicciones consecutivas
        changes = np.abs(np.diff(predictions))
        
        consistency_metrics = {
            'mean_change': np.mean(changes),
            'std_change': np.std(changes),
            'max_change': np.max(changes),
            'smoothness_score': 1 - (np.std(changes) / np.mean(np.abs(predictions))) if np.mean(np.abs(predictions)) > 0 else 0
        }
        
        return consistency_metrics
    
    def compare_models_statistically(self, model_names: List[str]) -> Dict[str, Any]:
        """
        Comparación estadística entre modelos
        
        Args:
            model_names: Nombres de modelos a comparar
            
        Returns:
            Resultados de tests estadísticos
        """
        print("🔬 Realizando comparación estadística...")
        
        if len(model_names) < 2:
            print("⚠️ Se necesitan al menos 2 modelos para comparar")
            return {}
        
        # Extraer errores absolutos de cada modelo
        model_errors = {}
        for model_name in model_names:
            if model_name in self.evaluation_results:
                y_true = self.evaluation_results[model_name]['predictions'] - self.evaluation_results[model_name]['residuals']
                y_pred = self.evaluation_results[model_name]['predictions']
                errors = np.abs(y_true - y_pred)
                model_errors[model_name] = errors
        
        # Tests estadísticos pairwise
        statistical_results = {}
        
        for i, model1 in enumerate(model_names):
            for j, model2 in enumerate(model_names[i+1:], i+1):
                if model1 in model_errors and model2 in model_errors:
                    errors1 = model_errors[model1]
                    errors2 = model_errors[model2]
                    
                    # Test t de Student para diferencias de medias
                    t_stat, t_p = stats.ttest_ind(errors1, errors2)
                    
                    # Test U de Mann-Whitney (no paramétrico)
                    u_stat, u_p = stats.mannwhitneyu(errors1, errors2, alternative='two-sided')
                    
                    # Test de Wilcoxon para muestras pareadas (si mismo tamaño)
                    if len(errors1) == len(errors2):
                        w_stat, w_p = stats.wilcoxon(errors1, errors2)
                    else:
                        w_stat, w_p = None, None
                    
                    # Efecto del tamaño (Cohen's d)
                    pooled_std = np.sqrt(((len(errors1) - 1) * np.var(errors1) + 
                                        (len(errors2) - 1) * np.var(errors2)) / 
                                       (len(errors1) + len(errors2) - 2))
                    cohens_d = (np.mean(errors1) - np.mean(errors2)) / pooled_std if pooled_std > 0 else 0
                    
                    pair_key = f"{model1}_vs_{model2}"
                    statistical_results[pair_key] = {
                        't_test': {'statistic': t_stat, 'p_value': t_p},
                        'mann_whitney': {'statistic': u_stat, 'p_value': u_p},
                        'wilcoxon': {'statistic': w_stat, 'p_value': w_p} if w_stat is not None else None,
                        'cohens_d': cohens_d,
                        'mean_diff': np.mean(errors1) - np.mean(errors2),
                        'significance': t_p < 0.05  # Significancia al 5%
                    }
        
        self.statistical_tests = statistical_results
        
        print("✅ Comparación estadística completada")
        
        return statistical_results
    
    def create_comprehensive_comparison_table(self) -> pd.DataFrame:
        """
        Crea tabla comparativa comprehensiva
        
        Returns:
            DataFrame con comparación completa
        """
        print("📊 Creando tabla comparativa...")
        
        comparison_data = []
        
        for model_name, results in self.evaluation_results.items():
            row = {
                'Model': model_name,
                'MAE': results['basic_metrics']['mae'],
                'RMSE': results['basic_metrics']['rmse'],
                'MAPE (%)': results['basic_metrics']['mape'],
                'SMAPE (%)': results['basic_metrics']['smape'],
                'R²': results['basic_metrics']['r2'],
                'Mean Relative Error': results['error_analysis']['relative_error_mean'],
                'Median Relative Error': results['error_analysis']['relative_error_median'],
                '95th Percentile Error': results['error_analysis']['relative_error_95th'],
                'Residuals Mean': results['residuals_analysis']['mean'],
                'Residuals Std': results['residuals_analysis']['std'],
                'Durbin-Watson': results['residuals_analysis']['durbin_watson'],
                'Shapiro p-value': results['residuals_analysis']['shapiro_p_value'],
                'Evaluation Time (s)': results['performance_metrics']['evaluation_time'],
                'Accuracy (<10% error)': results['performance_metrics']['prediction_accuracy'],
                'Quality (<5% error)': results['performance_metrics']['prediction_quality']
            }
            
            # Agregar errores por cuartil
            for quartile, error in results['error_by_quartile'].items():
                row[f'Error {quartile}'] = error
            
            # Agregar errores por magnitud
            for magnitude, error in results['error_by_magnitude'].items():
                row[f'Error {magnitude}'] = error
            
            comparison_data.append(row)
        
        df = pd.DataFrame(comparison_data)
        
        # Ordenar por MAE (menor es mejor)
        df = df.sort_values('MAE')
        
        print("✅ Tabla comparativa creada")
        
        return df
    
    def plot_comprehensive_comparison(self, save_plots: bool = False):
        """
        Visualización comprehensiva de comparación
        
        Args:
            save_plots: Si guardar los gráficos
        """
        print("📊 Visualizando comparación comprehensiva...")
        
        if not self.evaluation_results:
            print("⚠️ No hay resultados para visualizar")
            return
        
        model_names = list(self.evaluation_results.keys())
        
        # Crear figura con múltiples subplots
        fig, axes = plt.subplots(3, 3, figsize=(20, 15))
        
        # 1. Métricas básicas
        metrics = ['mae', 'rmse', 'mape']
        metric_labels = ['MAE', 'RMSE', 'MAPE (%)']
        
        for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
            values = [self.evaluation_results[model]['basic_metrics'][metric] for model in model_names]
            axes[0, i].bar(model_names, values, color='skyblue', alpha=0.7)
            axes[0, i].set_title(label)
            axes[0, i].tick_params(axis='x', rotation=45)
        
        # 2. Error relativo
        rel_errors = [self.evaluation_results[model]['error_analysis']['relative_error_mean'] 
                     for model in model_names]
        axes[1, 0].bar(model_names, rel_errors, color='lightcoral', alpha=0.7)
        axes[1, 0].set_title('Mean Relative Error')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # 3. R² Score
        r2_scores = [self.evaluation_results[model]['basic_metrics']['r2'] for model in model_names]
        axes[1, 1].bar(model_names, r2_scores, color='lightgreen', alpha=0.7)
        axes[1, 1].set_title('R² Score')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        # 4. Tiempo de evaluación
        eval_times = [self.evaluation_results[model]['performance_metrics']['evaluation_time'] 
                     for model in model_names]
        axes[1, 2].bar(model_names, eval_times, color='gold', alpha=0.7)
        axes[1, 2].set_title('Evaluation Time (s)')
        axes[1, 2].tick_params(axis='x', rotation=45)
        
        # 5. Errores por cuartil
        quartiles = ['q1', 'q2', 'q3', 'q4']
        for i, quartile in enumerate(quartiles):
            values = []
            for model in model_names:
                if quartile in self.evaluation_results[model]['error_by_quartile']:
                    values.append(self.evaluation_results[model]['error_by_quartile'][quartile])
                else:
                    values.append(0)
            
            axes[2, i].bar(model_names, values, color='orange', alpha=0.7)
            axes[2, i].set_title(f'Error - {quartile.upper()}')
            axes[2, i].tick_params(axis='x', rotation=45)
        
        # 6. Residuos (scatter plot del mejor modelo)
        best_model = min(self.evaluation_results.keys(), 
                        key=lambda x: self.evaluation_results[x]['basic_metrics']['mae'])
        
        residuals = self.evaluation_results[best_model]['residuals']
        predictions = self.evaluation_results[best_model]['predictions']
        
        axes[2, 2].scatter(predictions, residuals, alpha=0.5, s=1)
        axes[2, 2].axhline(y=0, color='r', linestyle='--')
        axes[2, 2].set_title(f'Residuals - {best_model}')
        axes[2, 2].set_xlabel('Predictions')
        axes[2, 2].set_ylabel('Residuals')
        
        plt.tight_layout()
        
        if save_plots:
            plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_residuals_analysis(self, model_name: str, save_plots: bool = False):
        """
        Análisis detallado de residuos para un modelo
        
        Args:
            model_name: Nombre del modelo a analizar
            save_plots: Si guardar los gráficos
        """
        if model_name not in self.evaluation_results:
            print(f"⚠️ Modelo {model_name} no encontrado")
            return
        
        results = self.evaluation_results[model_name]
        residuals = results['residuals']
        predictions = results['predictions']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Histograma de residuos
        axes[0, 0].hist(residuals, bins=50, alpha=0.7, color='skyblue', density=True)
        axes[0, 0].set_title('Histograma de Residuos')
        axes[0, 0].set_xlabel('Residuals')
        axes[0, 0].set_ylabel('Density')
        
        # Superponer curva normal
        x_range = np.linspace(residuals.min(), residuals.max(), 100)
        normal_curve = stats.norm.pdf(x_range, 0, np.std(residuals))
        axes[0, 0].plot(x_range, normal_curve, 'r-', linewidth=2, label='Normal')
        axes[0, 0].legend()
        
        # 2. Q-Q plot
        stats.probplot(residuals, dist="norm", plot=axes[0, 1])
        axes[0, 1].set_title('Q-Q Plot')
        
        # 3. Residuos vs Predicciones
        axes[1, 0].scatter(predictions, residuals, alpha=0.5, s=1)
        axes[1, 0].axhline(y=0, color='r', linestyle='--')
        axes[1, 0].set_title('Residuos vs Predicciones')
        axes[1, 0].set_xlabel('Predicciones')
        axes[1, 0].set_ylabel('Residuos')
        
        # 4. Autocorrelación de residuos
        from statsmodels.tsa.stattools import acf
        autocorr = acf(residuals, nlags=40, fft=True)
        
        axes[1, 1].plot(range(len(autocorr)), autocorr, 'bo-')
        axes[1, 1].axhline(y=0, color='r', linestyle='--')
        axes[1, 1].set_title('Autocorrelación de Residuos')
        axes[1, 1].set_xlabel('Lag')
        axes[1, 1].set_ylabel('Autocorrelación')
        
        plt.suptitle(f'Análisis de Residuos - {model_name}')
        plt.tight_layout()
        
        if save_plots:
            plt.savefig(f'residuals_analysis_{model_name}.png', dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def generate_evaluation_report(self, filepath: str = 'evaluation_report.json'):
        """
        Genera reporte completo de evaluación
        
        Args:
            filepath: Ruta del archivo JSON
        """
        print(f"📄 Generando reporte de evaluación...")
        
        # Tabla comparativa
        comparison_df = self.create_comprehensive_comparison_table()
        
        # Tests estadísticos
        statistical_summary = {}
        if self.statistical_tests:
            for pair, tests in self.statistical_tests.items():
                statistical_summary[pair] = {
                    'significant_difference': tests['significance'],
                    'effect_size': tests['cohens_d'],
                    'p_value_t_test': tests['t_test']['p_value']
                }
        
        # Ranking de modelos
        model_ranking = {}
        for model_name in self.evaluation_results.keys():
            # Score compuesto (ponderación de métricas)
            mae_norm = self.evaluation_results[model_name]['basic_metrics']['mae'] / comparison_df['MAE'].max()
            rmse_norm = self.evaluation_results[model_name]['basic_metrics']['rmse'] / comparison_df['RMSE'].max()
            mape_norm = self.evaluation_results[model_name]['basic_metrics']['mape'] / comparison_df['MAPE (%)'].max()
            r2_norm = 1 - (self.evaluation_results[model_name]['basic_metrics']['r2'] / comparison_df['R²'].max())
            
            composite_score = (mae_norm + rmse_norm + mape_norm + r2_norm) / 4
            model_ranking[model_name] = {
                'composite_score': composite_score,
                'rank': 0  # Se llenará después
            }
        
        # Ordenar por score
        sorted_models = sorted(model_ranking.items(), key=lambda x: x[1]['composite_score'])
        for i, (model_name, _) in enumerate(sorted_models):
            model_ranking[model_name]['rank'] = i + 1
        
        # Recomendaciones
        best_model = sorted_models[0][0]
        recommendations = {
            'best_overall': best_model,
            'most_accurate': comparison_df.iloc[0]['Model'],
            'fastest': comparison_df.iloc[comparison_df['Evaluation Time (s)'].idxmin()]['Model'],
            'most_consistent': min(self.evaluation_results.keys(), 
                                 key=lambda x: self.evaluation_results[x]['prediction_consistency']['smoothness_score'])
        }
        
        report = {
            'evaluation_summary': {
                'models_evaluated': list(self.evaluation_results.keys()),
                'total_samples': len(list(self.evaluation_results.values())[0]['predictions']),
                'evaluation_timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            },
            'comparison_table': comparison_df.to_dict('records'),
            'model_ranking': model_ranking,
            'statistical_tests': statistical_summary,
            'recommendations': recommendations,
            'detailed_results': self.evaluation_results
        }
        
        # Guardar reporte
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"✅ Reporte guardado en {filepath}")
        print(f"🏆 Mejor modelo: {best_model}")
        
        return report


def main():
    """
    Función principal para demostrar evaluación
    """
    print("🚀 Iniciando evaluación comprehensiva de modelos")
    
    # Simular resultados de evaluación (en uso real, vendrían de modelos entrenados)
    evaluator = ModelEvaluator()
    
    # Ejemplo de cómo evaluar un modelo (simulado)
    # En uso real: evaluator.evaluate_model_comprehensive(model_name, model, X_test, y_test)
    
    print("✅ Evaluación comprehensiva completada")
    print("💡 En uso real, primero ejecute los modelos y luego evalúe con esta clase")


if __name__ == "__main__":
    main()
