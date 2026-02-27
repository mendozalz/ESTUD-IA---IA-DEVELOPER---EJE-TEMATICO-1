"""
Laboratorio 1 - Pipeline de Datos Automatizado
Pipeline completo de ventas retail con TFX y Airflow
"""

import tensorflow as tf
from tfx import v1 as tfx
from tfx.components import (
    CsvExampleGen,
    StatisticsGen,
    SchemaGen,
    ExampleValidator,
    Transform,
    Trainer,
    Evaluator,
    Pusher
)
from tfx.proto import trainer_pb2, pusher_pb2
from tfx.orchestration import data_types
from tfx.components.base import executor_spec

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yaml
import logging

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RetailSalesDataGenerator:
    """
    Generador de datos sintéticos de ventas retail para testing
    """
    
    def __init__(self, num_days=365, num_stores=10, num_products=100):
        self.num_days = num_days
        self.num_stores = num_stores
        self.num_products = num_products
        self.start_date = datetime.now() - timedelta(days=num_days)
    
    def generate_sales_data(self):
        """
        Genera datos de ventas con patrones realistas
        """
        logger.info(f"Generando datos para {self.num_days} días, {self.num_stores} tiendas, {self.num_products} productos")
        
        data = []
        
        for day in range(self.num_days):
            current_date = self.start_date + timedelta(days=day)
            
            # Factores estacionales y semanales
            day_of_week = current_date.weekday()
            is_weekend = day_of_week >= 5
            month = current_date.month
            
            # Factor estacional (más ventas en diciembre, menos en enero)
            seasonal_factor = 1.0 + 0.3 * np.sin(2 * np.pi * month / 12)
            
            # Factor de fin de semana
            weekend_factor = 1.2 if is_weekend else 1.0
            
            for store_id in range(self.num_stores):
                for product_id in range(self.num_products):
                    # Ventas base con variación aleatoria
                    base_sales = np.random.lognormal(mean=3.0, sigma=0.5)
                    
                    # Aplicar factores
                    sales = max(0, base_sales * seasonal_factor * weekend_factor)
                    
                    # Otros atributos
                    inventory = np.random.lognormal(mean=5.0, sigma=0.3)
                    price = np.random.uniform(10, 100)
                    promotion = np.random.choice([0, 1], p=[0.8, 0.2])
                    
                    # Temperatura (factor externo)
                    temperature = 20 + 10 * np.sin(2 * np.pi * day / 365) + np.random.normal(0, 5)
                    
                    data.append({
                        'date': current_date.strftime('%Y-%m-%d'),
                        'store_id': store_id,
                        'product_id': product_id,
                        'sales': round(sales, 2),
                        'inventory': round(inventory, 0),
                        'price': round(price, 2),
                        'promotion': promotion,
                        'temperature': round(temperature, 1),
                        'day_of_week': day_of_week,
                        'is_weekend': int(is_weekend),
                        'month': month
                    })
        
        df = pd.DataFrame(data)
        logger.info(f"Generados {len(df)} registros de ventas")
        return df
    
    def save_to_csv(self, df, filepath):
        """
        Guarda datos en formato CSV para TFX
        """
        df.to_csv(filepath, index=False)
        logger.info(f"Datos guardados en {filepath}")

class RetailSalesPipeline:
    """
    Pipeline TFX completo para análisis de ventas retail
    """
    
    def __init__(self, pipeline_name, pipeline_root, data_path, module_file, serving_model_dir):
        self.pipeline_name = pipeline_name
        self.pipeline_root = pipeline_root
        self.data_path = data_path
        self.module_file = module_file
        self.serving_model_dir = serving_model_dir
        
    def create_pipeline(self):
        """
        Crea el pipeline completo de TFX
        """
        logger.info(f"Creando pipeline: {self.pipeline_name}")
        
        # 1. Generación de ejemplos desde CSV
        example_gen = CsvExampleGen(
            input_base=self.data_path,
            name='example_gen'
        )
        
        # 2. Estadísticas de datos
        statistics_gen = StatisticsGen(
            examples=example_gen.outputs['examples'],
            name='statistics_gen'
        )
        
        # 3. Generación de schema
        schema_gen = SchemaGen(
            statistics=statistics_gen.outputs['statistics'],
            infer_feature_shape=True,
            name='schema_gen'
        )
        
        # 4. Validación de ejemplos
        example_validator = ExampleValidator(
            statistics=statistics_gen.outputs['statistics'],
            schema=schema_gen.outputs['schema'],
            name='example_validator'
        )
        
        # 5. Transformación de features
        transform = Transform(
            examples=example_gen.outputs['examples'],
            schema=schema_gen.outputs['schema'],
            module_file=self.module_file,
            name='transform'
        )
        
        # 6. Entrenamiento del modelo
        trainer = Trainer(
            module_file=self.module_file,
            custom_executor_spec=executor_spec.ExecutorClassSpec(trainer_pb2.Trainer),
            examples=transform.outputs['transformed_examples'],
            schema=schema_gen.outputs['schema'],
            transform_graph=transform.outputs['transform_graph'],
            train_args=tfx.proto.TrainArgs(num_steps=10000),
            eval_args=tfx.proto.EvalArgs(num_steps=5000),
            name='trainer'
        )
        
        # 7. Evaluación del modelo
        evaluator = Evaluator(
            examples=transform.outputs['transformed_examples'],
            model=trainer.outputs['model'],
            schema=schema_gen.outputs['schema'],
            name='evaluator'
        )
        
        # 8. Despliegue del modelo
        pusher = Pusher(
            model=trainer.outputs['model'],
            model_blessing=evaluator.outputs['blessing'],
            push_destination=pusher_pb2.PushDestination(
                filesystem=pusher_pb2.PushDestination.Filesystem(
                    base_directory=self.serving_model_dir
                )
            ),
            name='pusher'
        )
        
        # Construir el pipeline
        pipeline = tfx.dsl.Pipeline(
            pipeline_name=self.pipeline_name,
            pipeline_root=self.pipeline_root,
            components=[
                example_gen,
                statistics_gen,
                schema_gen,
                example_validator,
                transform,
                trainer,
                evaluator,
                pusher
            ],
            enable_cache=True,
            metadata_connection_config=tfx.orchestration.metadata.sqlite_metadata_connection_config(
                f'{self.pipeline_root}/metadata.db'
            )
        )
        
        logger.info("Pipeline creado exitosamente")
        return pipeline

class PipelineConfig:
    """
    Gestor de configuración del pipeline
    """
    
    def __init__(self, config_file='pipeline_config.yaml'):
        self.config_file = config_file
        self.config = self.load_config()
    
    def load_config(self):
        """
        Carga configuración desde archivo YAML
        """
        try:
            with open(self.config_file, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Configuración cargada desde {self.config_file}")
            return config
        except FileNotFoundError:
            logger.warning(f"Archivo de configuración {self.config_file} no encontrado, usando defaults")
            return self.get_default_config()
    
    def get_default_config(self):
        """
        Configuración por defecto
        """
        return {
            'pipeline': {
                'name': 'retail_sales_pipeline',
                'root': '/tmp/tfx/retail_sales',
                'data_path': '/tmp/data/retail_sales',
                'module_file': 'retail_transform.py',
                'serving_model_dir': '/tmp/serving_model/retail_sales'
            },
            'training': {
                'train_steps': 10000,
                'eval_steps': 5000,
                'learning_rate': 0.001,
                'batch_size': 32
            },
            'data_generation': {
                'num_days': 365,
                'num_stores': 10,
                'num_products': 100
            }
        }
    
    def save_config(self):
        """
        Guarda configuración actual
        """
        with open(self.config_file, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)
        logger.info(f"Configuración guardada en {self.config_file}")

class PipelineMonitor:
    """
    Monitor de estado y métricas del pipeline
    """
    
    def __init__(self):
        self.metrics = {}
    
    def log_pipeline_start(self, pipeline_name):
        """
        Registra inicio de pipeline
        """
        start_time = datetime.now()
        self.metrics[pipeline_name] = {
            'start_time': start_time,
            'status': 'running',
            'components_completed': [],
            'errors': []
        }
        logger.info(f"Pipeline {pipeline_name} iniciado a las {start_time}")
    
    def log_component_completion(self, pipeline_name, component_name, duration):
        """
        Registra completación de componente
        """
        if pipeline_name in self.metrics:
            self.metrics[pipeline_name]['components_completed'].append({
                'component': component_name,
                'duration': duration,
                'completed_at': datetime.now()
            })
        logger.info(f"Componente {component_name} completado en {duration:.2f} segundos")
    
    def log_pipeline_completion(self, pipeline_name, success=True):
        """
        Registra finalización de pipeline
        """
        if pipeline_name in self.metrics:
            end_time = datetime.now()
            start_time = self.metrics[pipeline_name]['start_time']
            total_duration = (end_time - start_time).total_seconds()
            
            self.metrics[pipeline_name].update({
                'end_time': end_time,
                'status': 'completed' if success else 'failed',
                'total_duration': total_duration
            })
            
            status_msg = "exitosamente" if success else "con errores"
            logger.info(f"Pipeline {pipeline_name} finalizado {status_msg} en {total_duration:.2f} segundos")
    
    def get_pipeline_summary(self, pipeline_name):
        """
        Obtiene resumen del pipeline
        """
        if pipeline_name not in self.metrics:
            return None
        
        metrics = self.metrics[pipeline_name]
        return {
            'pipeline_name': pipeline_name,
            'status': metrics['status'],
            'total_duration': metrics.get('total_duration', 0),
            'components_completed': len(metrics['components_completed']),
            'errors': len(metrics['errors'])
        }

def main():
    """
    Función principal para ejecutar el pipeline
    """
    logger.info("Iniciando Pipeline de Ventas Retail")
    
    # 1. Cargar configuración
    config_manager = PipelineConfig()
    config = config_manager.config
    
    # 2. Generar datos de ejemplo
    data_generator = RetailSalesDataGenerator(
        num_days=config['data_generation']['num_days'],
        num_stores=config['data_generation']['num_stores'],
        num_products=config['data_generation']['num_products']
    )
    
    sales_data = data_generator.generate_sales_data()
    
    # Crear directorios necesarios
    import os
    os.makedirs(config['pipeline']['data_path'], exist_ok=True)
    os.makedirs(config['pipeline']['serving_model_dir'], exist_ok=True)
    
    # Guardar datos
    data_file = os.path.join(config['pipeline']['data_path'], 'retail_sales.csv')
    data_generator.save_to_csv(sales_data, data_file)
    
    # 3. Crear y ejecutar pipeline
    pipeline_manager = RetailSalesPipeline(
        pipeline_name=config['pipeline']['name'],
        pipeline_root=config['pipeline']['root'],
        data_path=config['pipeline']['data_path'],
        module_file=config['pipeline']['module_file'],
        serving_model_dir=config['pipeline']['serving_model_dir']
    )
    
    # 4. Iniciar monitoreo
    monitor = PipelineMonitor()
    monitor.log_pipeline_start(config['pipeline']['name'])
    
    try:
        # Crear pipeline
        pipeline = pipeline_manager.create_pipeline()
        
        # Ejecutar pipeline (en producción, esto se ejecutaría con TFX CLI)
        logger.info("Pipeline creado. Para ejecutar, use:")
        logger.info(f"tfx pipeline create --pipeline-path retail_pipeline.py --engine local")
        
        # Simulación de ejecución para demostración
        import time
        components = ['example_gen', 'statistics_gen', 'schema_gen', 'transform', 'trainer', 'evaluator', 'pusher']
        
        for component in components:
            start_time = time.time()
            time.sleep(1)  # Simular procesamiento
            duration = time.time() - start_time
            monitor.log_component_completion(config['pipeline']['name'], component, duration)
        
        monitor.log_pipeline_completion(config['pipeline']['name'], success=True)
        
        # 5. Mostrar resumen
        summary = monitor.get_pipeline_summary(config['pipeline']['name'])
        print("\n=== RESUMEN DEL PIPELINE ===")
        print(f"Pipeline: {summary['pipeline_name']}")
        print(f"Estado: {summary['status']}")
        print(f"Duración total: {summary['total_duration']:.2f} segundos")
        print(f"Componentes completados: {summary['components_completed']}")
        print(f"Errores: {summary['errors']}")
        
    except Exception as e:
        logger.error(f"Error en el pipeline: {str(e)}")
        monitor.log_pipeline_completion(config['pipeline']['name'], success=False)
        raise

if __name__ == "__main__":
    main()
