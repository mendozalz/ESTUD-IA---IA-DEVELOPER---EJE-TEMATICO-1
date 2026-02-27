"""
Caso de Uso 4 - Reentrenamiento Automático con MLflow y TFX
Fase 3: Configuración de MLflow para Tracking de Experimentos
"""

import mlflow
import mlflow.tensorflow
import mlflow.sklearn
import pandas as pd
import numpy as np
import tensorflow as tf
import json
import logging
import os
from datetime import datetime
from typing import Dict, Any, List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MLflowTracker:
    """
    Clase para tracking de experimentos y modelos con MLflow
    """
    
    def __init__(self, experiment_name="recommendation_retraining"):
        self.experiment_name = experiment_name
        self.experiment_id = None
        self.tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
        self.artifact_path = "mlruns"
        
    def setup_mlflow(self):
        """Configurar MLflow"""
        try:
            # Configurar tracking URI
            mlflow.set_tracking_uri(self.tracking_uri)
            logger.info(f"MLflow tracking URI: {self.tracking_uri}")
            
            # Crear o obtener experimento
            try:
                self.experiment_id = mlflow.create_experiment(self.experiment_name)
                logger.info(f"Experimento creado: {self.experiment_name} (ID: {self.experiment_id})")
            except mlflow.exceptions.MlflowException:
                # El experimento ya existe
                experiment = mlflow.get_experiment_by_name(self.experiment_name)
                self.experiment_id = experiment.experiment_id if experiment else None
                logger.info(f"Usando experimento existente: {self.experiment_name}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error configurando MLflow: {e}")
            return False
    
    def log_training_run(self, model_path: str, metrics: Dict[str, Any], 
                        params: Dict[str, Any], artifacts: List[str] = None):
        """
        Registrar un run de entrenamiento
        """
        try:
            with mlflow.start_run(experiment_id=self.experiment_id) as run:
                # Loguear parámetros
                for param_name, param_value in params.items():
                    mlflow.log_param(param_name, param_value)
                
                # Loguear métricas
                for metric_name, metric_value in metrics.items():
                    mlflow.log_metric(metric_name, metric_value)
                
                # Loguear modelo TensorFlow
                if os.path.exists(model_path):
                    mlflow.tensorflow.log_model(
                        model_path,
                        "model",
                        registered_model_name="recommendation_model"
                    )
                    logger.info(f"Modelo registrado: {model_path}")
                
                # Loguear artefactos adicionales
                if artifacts:
                    for artifact_path in artifacts:
                        if os.path.exists(artifact_path):
                            mlflow.log_artifact(artifact_path)
                            logger.info(f"Artefacto registrado: {artifact_path}")
                
                # Loguear información adicional
                mlflow.set_tag("model_type", "recommendation")
                mlflow.set_tag("framework", "tensorflow")
                mlflow.set_tag("timestamp", datetime.now().isoformat())
                
                logger.info(f"Run registrado: {run.info.run_id}")
                return run.info.run_id
                
        except Exception as e:
            logger.error(f"Error registrando run: {e}")
            raise
    
    def register_model_production(self, run_id: str, model_name: str = "recommendation_model"):
        """
        Transicionar modelo a producción
        """
        try:
            client = mlflow.tracking.MlflowClient()
            
            # Obtener información del modelo
            model_version = client.get_latest_versions(model_name, stages=["None"])[0]
            
            # Transicionar a Staging primero
            client.transition_model_version_stage(
                name=model_name,
                version=model_version.version,
                stage="Staging"
            )
            
            # Validar métricas antes de producción
            run = client.get_run(run_id)
            metrics = run.data.metrics
            
            # Criterios para producción
            production_criteria = {
                'val_accuracy': 0.8,
                'val_auc': 0.85,
                'val_loss': 0.5
            }
            
            can_promote = True
            for metric, threshold in production_criteria.items():
                if metric not in metrics or metrics[metric] < threshold:
                    logger.warning(f"Métrica {metric} no cumple criterio: {metrics.get(metric, 0)} < {threshold}")
                    can_promote = False
            
            if can_promote:
                # Transicionar a Producción
                client.transition_model_version_stage(
                    name=model_name,
                    version=model_version.version,
                    stage="Production"
                )
                logger.info(f"Modelo {model_name} versión {model_version.version} promovido a Producción")
                return True
            else:
                logger.warning("Modelo no cumple criterios para producción")
                return False
                
        except Exception as e:
            logger.error(f"Error registrando modelo en producción: {e}")
            return False
    
    def compare_models(self, model_name: str = "recommendation_model") -> Dict[str, Any]:
        """
        Comparar modelos registrados
        """
        try:
            client = mlflow.tracking.MlflowClient()
            
            # Obtener todas las versiones del modelo
            model_versions = client.search_model_versions(f"name='{model_name}'")
            
            comparison = {
                'model_name': model_name,
                'total_versions': len(model_versions),
                'versions': []
            }
            
            for mv in model_versions:
                run = client.get_run(mv.run_id)
                metrics = run.data.metrics
                params = run.data.params
                
                version_info = {
                    'version': mv.version,
                    'stage': mv.current_stage,
                    'run_id': mv.run_id,
                    'created_at': mv.creation_timestamp,
                    'metrics': metrics,
                    'params': params
                }
                comparison['versions'].append(version_info)
            
            # Ordenar por versión
            comparison['versions'].sort(key=lambda x: int(x['version']), reverse=True)
            
            return comparison
            
        except Exception as e:
            logger.error(f"Error comparando modelos: {e}")
            return {}
    
    def get_production_model_info(self, model_name: str = "recommendation_model") -> Dict[str, Any]:
        """
        Obtener información del modelo en producción
        """
        try:
            client = mlflow.tracking.MlflowClient()
            
            # Obtener versión en producción
            production_versions = client.get_latest_versions(model_name, stages=["Production"])
            
            if not production_versions:
                return {"status": "no_production_model"}
            
            prod_version = production_versions[0]
            run = client.get_run(prod_version.run_id)
            
            model_info = {
                "model_name": model_name,
                "version": prod_version.version,
                "stage": prod_version.current_stage,
                "run_id": prod_version.run_id,
                "created_at": prod_version.creation_timestamp,
                "metrics": run.data.metrics,
                "params": run.data.params,
                "tags": run.data.tags
            }
            
            return model_info
            
        except Exception as e:
            logger.error(f"Error obteniendo información del modelo de producción: {e}")
            return {"status": "error", "error": str(e)}
    
    def log_model_performance(self, model_name: str, performance_data: Dict[str, Any]):
        """
        Registrar datos de rendimiento del modelo en producción
        """
        try:
            with mlflow.start_run(experiment_id=self.experiment_id) as run:
                # Loguear métricas de rendimiento
                for metric_name, metric_value in performance_data.items():
                    mlflow.log_metric(f"prod_{metric_name}", metric_value)
                
                # Loguear tags
                mlflow.set_tag("run_type", "production_monitoring")
                mlflow.set_tag("model_name", model_name)
                mlflow.set_tag("timestamp", datetime.now().isoformat())
                
                logger.info(f"Rendimiento del modelo {model_name} registrado")
                return run.info.run_id
                
        except Exception as e:
            logger.error(f"Error registrando rendimiento del modelo: {e}")
            raise

def main():
    """
    Función principal para demostrar el uso de MLflow
    """
    logger.info("=" * 80)
    logger.info("CONFIGURANDO MLFLOW PARA TRACKING DE MODELOS")
    logger.info("=" * 80)
    
    # Crear tracker
    tracker = MLflowTracker()
    
    # Configurar MLflow
    if not tracker.setup_mlflow():
        return
    
    # Datos de ejemplo (simulados)
    metrics = {
        'train_loss': 0.345,
        'val_loss': 0.389,
        'train_accuracy': 0.876,
        'val_accuracy': 0.842,
        'train_auc': 0.923,
        'val_auc': 0.887,
        'dataset_size': 10000,
        'num_users': 1000,
        'num_items': 500
    }
    
    params = {
        'batch_size': 256,
        'learning_rate': 0.001,
        'epochs': 10,
        'embedding_dim': 32,
        'optimizer': 'adam',
        'loss_function': 'binary_crossentropy'
    }
    
    # Simular modelo
    model_path = "models/dummy_model.h5"
    os.makedirs("models", exist_ok=True)
    
    # Crear modelo dummy
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(32,)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.save(model_path)
    
    # Registrar run
    run_id = tracker.log_training_run(
        model_path=model_path,
        metrics=metrics,
        params=params
    )
    
    # Registrar en producción
    success = tracker.register_model_production(run_id)
    
    if success:
        logger.info("Modelo promovido a producción exitosamente")
    
    # Comparar modelos
    comparison = tracker.compare_models()
    logger.info(f"Comparación de modelos: {len(comparison.get('versions', []))} versiones")
    
    # Obtener información del modelo de producción
    prod_info = tracker.get_production_model_info()
    logger.info(f"Modelo en producción: versión {prod_info.get('version', 'N/A')}")
    
    # Registrar rendimiento en producción
    performance_data = {
        'latency_p95': 125.5,
        'throughput': 1000.0,
        'error_rate': 0.02,
        'memory_usage': 512.0
    }
    
    tracker.log_model_performance("recommendation_model", performance_data)
    
    logger.info("=" * 80)
    logger.info("MLFLOW CONFIGURADO Y DEMOSTRACIÓN COMPLETADA")
    logger.info("=" * 80)

if __name__ == "__main__":
    main()
