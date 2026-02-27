"""
Caso de Uso 4 - Reentrenamiento Automático con MLflow y TFX
Fase 1: Pipeline Completo de Reentrenamiento Automático
"""

from tfx.orchestration import pipeline
from tfx.components import (
    CsvExampleGen, StatisticsGen, SchemaGen, Transform,
    Trainer, Tuner, Evaluator, Pusher
)
from tfx.proto import pusher_pb2, trainer_pb2
from tfx.dsl.components.common import resolver
from tfx.dsl.input_resolution.strategies.latest_blessed_model import LatestBlessedModelStrategy
from tfx.types import standard_artifacts
from tfx.orchestration.kubeflow import kubeflow_dag_runner
from tfx.orchestration.data_types import RuntimeParameter
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RecommendationRetrainingPipeline:
    """
    Pipeline completo para reentrenamiento automático de modelo de recomendación
    """
    
    def __init__(self, pipeline_name="recommendation_retraining"):
        self.pipeline_name = pipeline_name
        self.data_base_path = "data/interactions"
        self.pipeline_root = "pipeline_root"
        self.serving_model_dir = "serving_model"
        
    def create_pipeline(self):
        """Crear pipeline completo de TFX"""
        logger.info(f"Creando pipeline: {self.pipeline_name}")
        
        # 1. Ingestión de datos
        example_gen = CsvExampleGen(
            input_base=self.data_base_path,
            runtime_parameter=RuntimeParameter(
                name="input_data",
                default=self.data_base_path,
                ptype=RuntimeParameter.PropertyType.STRING
            )
        )
        
        # 2. Estadísticas y validación de esquema
        statistics_gen = StatisticsGen(
            examples=example_gen.outputs['examples']
        )
        
        schema_gen = SchemaGen(
            statistics=statistics_gen.outputs['statistics'],
            infer_feature_shape=True
        )
        
        # 3. Transformación de datos
        transform = Transform(
            examples=example_gen.outputs['examples'],
            schema=schema_gen.outputs['schema'],
            runtime_parameter=RuntimeParameter(
                name="transform_module",
                default="transform_module",
                ptype=RuntimeParameter.PropertyType.STRING
            )
        )
        
        # 4. Resolver modelo actual (baseline)
        model_resolver = resolver.Resolver(
            strategy_class=LatestBlessedModelStrategy,
            model=standard_artifacts.Model,
            model_blessing=standard_artifacts.ModelBlessing
        ).with_id('latest_blessed_model_resolver')
        
        # 5. Tuning de hiperparámetros (opcional)
        tuner = None
        if os.getenv("ENABLE_TUNING", "false").lower() == "true":
            tuner = Tuner(
                module_file="tuner_module",
                examples=transform.outputs['transformed_examples'],
                schema=schema_gen.outputs['schema'],
                transform_graph=transform.outputs['transform_graph'],
                train_args=trainer_pb2.TrainArgs(num_steps=5000),
                eval_args=trainer_pb2.EvalArgs(num_steps=1000)
            )
        
        # 6. Entrenamiento
        trainer = Trainer(
            module_file="trainer_module",
            examples=transform.outputs['transformed_examples'],
            schema=schema_gen.outputs['schema'],
            transform_graph=transform.outputs['transform_graph'],
            train_args=trainer_pb2.TrainArgs(
                num_steps=RuntimeParameter(
                    name="train_steps",
                    default=10000,
                    ptype=RuntimeParameter.PropertyType.INT
                )
            ),
            eval_args=trainer_pb2.EvalArgs(
                num_steps=RuntimeParameter(
                    name="eval_steps",
                    default=5000,
                    ptype=RuntimeParameter.PropertyType.INT
                )
            ),
            custom_config={
                "learning_rate": RuntimeParameter(
                    name="learning_rate",
                    default=0.001,
                    ptype=RuntimeParameter.PropertyType.DOUBLE
                ),
                "batch_size": RuntimeParameter(
                    name="batch_size",
                    default=32,
                    ptype=RuntimeParameter.PropertyType.INT
                )
            }
        )
        
        # 7. Evaluación comparativa
        evaluator = Evaluator(
            examples=example_gen.outputs['examples'],
            model=trainer.outputs['model'],
            baseline_model=model_resolver.outputs['model'],
            eval_config="eval_config.json",
            runtime_parameter=RuntimeParameter(
                name="eval_config",
                default="eval_config.json",
                ptype=RuntimeParameter.PropertyType.STRING
            )
        )
        
        # 8. Despliegue condicional
        pusher = Pusher(
            model=trainer.outputs['model'],
            model_blessing=evaluator.outputs['blessing'],
            push_destination=pusher_pb2.PushDestination(
                filesystem=pusher_pb2.PushDestination.Filesystem(
                    base_directory=self.serving_model_dir
                )
            ),
            custom_config={
                "push_destination": RuntimeParameter(
                    name="push_destination",
                    default=self.serving_model_dir,
                    ptype=RuntimeParameter.PropertyType.STRING
                )
            }
        )
        
        # Componentes del pipeline
        components = [
            example_gen,
            statistics_gen,
            schema_gen,
            transform,
            model_resolver,
            trainer,
            evaluator,
            pusher
        ]
        
        # Añadir tuner si está habilitado
        if tuner:
            components.insert(-2, tuner)
            trainer.custom_config["tuner"] = tuner.outputs["best_hyperparameters"]
        
        # Crear pipeline
        pipeline_obj = pipeline.Pipeline(
            pipeline_name=self.pipeline_name,
            pipeline_root=self.pipeline_root,
            components=components,
            enable_cache=True,
            metadata={
                "project_name": "recommendation_system",
                "description": "Pipeline automático de reentrenamiento de modelo de recomendación",
                "version": "1.0.0"
            }
        )
        
        logger.info(f"Pipeline creado con {len(components)} componentes")
        return pipeline_obj
    
    def create_kubeflow_dag(self):
        """Crear DAG para Kubeflow"""
        logger.info("Creando DAG para Kubeflow")
        
        pipeline_obj = self.create_pipeline()
        
        # Configuración para Kubeflow
        kubeflow_dag_runner = kubeflow_dag_runner.KubeflowDagRunner(
            config=RuntimeParameter(
                name="pipeline_config",
                default="kubeflow_config.json",
                ptype=RuntimeParameter.PropertyType.STRING
            ),
            tfx_image="tensorflow/tfx:2.0.0",
            kubeflow_pipeline_operator_version="1.8.1"
        )
        
        return kubeflow_dag_runner.run(pipeline_obj)
    
    def create_airflow_dag(self):
        """Crear DAG para Airflow"""
        logger.info("Creando DAG para Airflow")
        
        # Importar aquí para evitar dependencias circulares
        try:
            from airflow import DAG
            from airflow.operators.python import PythonOperator
            from datetime import datetime, timedelta
            
            default_args = {
                'owner': 'ml-team',
                'depends_on_past': False,
                'start_date': datetime(2026, 1, 1),
                'email_on_failure': True,
                'email_on_retry': False,
                'retries': 1,
                'retry_delay': timedelta(minutes=5),
            }
            
            dag = DAG(
                dag_id='recommendation_retraining',
                default_args=default_args,
                description='Pipeline de reentrenamiento automático de recomendación',
                schedule_interval='@daily',
                catchup=False,
                max_active_runs=1
            )
            
            def run_tfx_pipeline():
                """Ejecutar pipeline TFX desde Airflow"""
                import subprocess
                import sys
                
                try:
                    # Ejecutar pipeline TFX
                    result = subprocess.run([
                        sys.executable, "-m", "tfx",
                        "run", "pipeline",
                        "--pipeline-path", "retraining_pipeline.py",
                        "--engine", "kubeflow"
                    ], capture_output=True, text=True, check=True)
                    
                    logger.info(f"Pipeline ejecutado exitosamente: {result.stdout}")
                    return True
                    
                except subprocess.CalledProcessError as e:
                    logger.error(f"Error ejecutando pipeline: {e.stderr}")
                    raise Exception(f"Pipeline failed: {e.stderr}")
            
            # Definir tareas
            run_pipeline_task = PythonOperator(
                task_id='run_retraining_pipeline',
                python_callable=run_tfx_pipeline,
                dag=dag
            )
            
            return dag
            
        except ImportError as e:
            logger.warning(f"No se pudo crear DAG de Airflow: {e}")
            return None
    
    def validate_pipeline(self):
        """Validar configuración del pipeline"""
        logger.info("Validando configuración del pipeline...")
        
        validation_errors = []
        
        # Validar directorios
        if not os.path.exists(self.data_base_path):
            validation_errors.append(f"Directorio de datos no encontrado: {self.data_base_path}")
        
        # Validar archivos de configuración
        required_files = [
            "trainer_module.py",
            "transform_module.py",
            "eval_config.json"
        ]
        
        for file in required_files:
            if not os.path.exists(file):
                validation_errors.append(f"Archivo requerido no encontrado: {file}")
        
        # Validar variables de entorno
        env_vars = [
            "KUBEFLOW_ENDPOINT",
            "MLFLOW_TRACKING_URI"
        ]
        
        for var in env_vars:
            if not os.getenv(var):
                logger.warning(f"Variable de entorno no configurada: {var}")
        
        if validation_errors:
            error_msg = "Errores de validación:\n" + "\n".join(validation_errors)
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        logger.info("Validación completada exitosamente")
        return True
    
    def setup_monitoring(self):
        """Configurar monitoreo del pipeline"""
        logger.info("Configurando monitoreo del pipeline...")
        
        # Crear directorios para logs
        os.makedirs("logs", exist_ok=True)
        os.makedirs("metrics", exist_ok=True)
        
        # Configurar logging estructurado
        logging.config.dictConfig({
            'version': 1,
            'disable_existing_loggers': False,
            'formatters': {
                'detailed': {
                    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                }
            },
            'handlers': {
                'file': {
                    'class': 'logging.FileHandler',
                    'formatter': 'detailed',
                    'filename': 'logs/pipeline.log',
                    'mode': 'a'
                },
                'console': {
                    'class': 'logging.StreamHandler',
                    'formatter': 'detailed'
                }
            },
            'loggers': {
                'tfx': {
                    'handlers': ['file', 'console'],
                    'level': 'INFO',
                    'propagate': False
                }
            },
            'root': {
                'handlers': ['console'],
                'level': 'INFO'
            }
        })
        
        logger.info("Monitoreo configurado exitosamente")

def main():
    """Función principal para ejecutar el pipeline"""
    logger.info("=" * 80)
    logger.info("INICIANDO PIPELINE DE REENTRENAMIENTO AUTOMÁTICO")
    logger.info("=" * 80)
    
    # Crear pipeline
    pipeline_creator = RecommendationRetrainingPipeline()
    
    try:
        # Validar configuración
        pipeline_creator.validate_pipeline()
        
        # Configurar monitoreo
        pipeline_creator.setup_monitoring()
        
        # Crear y ejecutar pipeline
        pipeline_obj = pipeline_creator.create_pipeline()
        
        # Ejecutar en Kubeflow
        logger.info("Ejecutando pipeline en Kubeflow...")
        kubeflow_dag = pipeline_creator.create_kubeflow_dag()
        
        logger.info("Pipeline iniciado exitosamente")
        logger.info(f"Nombre del pipeline: {pipeline_creator.pipeline_name}")
        logger.info(f"Directorio raíz: {pipeline_creator.pipeline_root}")
        logger.info(f"Modelo serving: {pipeline_creator.serving_model_dir}")
        
        # Intentar crear DAG de Airflow también
        airflow_dag = pipeline_creator.create_airflow_dag()
        if airflow_dag:
            logger.info("DAG de Airflow creado también")
        
    except Exception as e:
        logger.error(f"Error ejecutando pipeline: {e}")
        raise
    
    logger.info("=" * 80)
    logger.info("PIPELINE DE REENTRENAMIENTO INICIADO")
    logger.info("Monitorea el progreso en Kubeflow o Airflow")
    logger.info("=" * 80)

if __name__ == "__main__":
    main()
