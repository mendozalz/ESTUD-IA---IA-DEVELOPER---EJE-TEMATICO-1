# Laboratorio 1: Pipeline de Datos Automatizado con TFX y Airflow

## 🎯 Objetivo Principal

Construir un pipeline completo de datos desde la recolección hasta el modelo entrenado, utilizando TensorFlow Extended (TFX) para el pipeline de ML y Apache Airflow para la orquestación de tareas.

## 📋 Contexto del Proyecto

### **Caso de Uso: Sistema de Predicción de Ventas Retail**
Desarrollaremos un pipeline automatizado para una cadena de retail que necesita:
- Predecir ventas diarias por tienda y producto
- Procesar datos de múltiples fuentes (POS, inventario, clima, eventos)
- Actualizar modelos automáticamente cada 24 horas
- Monitorear la calidad de datos y rendimiento del modelo

## 🏗️ Arquitectura del Pipeline

```mermaid
graph TD
    A[Data Sources] --> B[Data Ingestion]
    B --> C[Data Validation]
    C --> D[Feature Engineering]
    D --> E[Model Training]
    E --> F[Model Evaluation]
    F --> G[Model Deployment]
    G --> H[Monitoring]
    H --> B
    
    subgraph "Airflow DAG"
        B
        C
        D
        E
        F
        G
        H
    end
    
    subgraph "TFX Components"
        I[ExampleGen]
        J[StatisticsGen]
        K[SchemaGen]
        L[ExampleValidator]
        M[Transform]
        N[Trainer]
        O[Evaluator]
        P[Pusher]
    end
```

## 🔧 Tecnologías Utilizadas

### **Core Pipeline**
- **TensorFlow Extended (TFX)**: Framework de pipelines de ML
- **Apache Airflow**: Orquestación de workflows
- **Great Expectations**: Validación de datos
- **MLflow**: Tracking de experimentos

### **Data Processing**
- **Apache Beam**: Procesamiento distribuido de datos
- **Pandas**: Manipulación de datos
- **NumPy**: Computación numérica
- **SQL**: Consultas a bases de datos

### **Infrastructure**
- **Docker**: Contenerización
- **PostgreSQL**: Base de datos
- **Redis**: Caché y cola de mensajes
- **MinIO**: Storage compatible con S3

## 📁 Estructura del Proyecto

```
Laboratorio 1 - Pipeline de Datos Automatizado/
├── README.md
├── requirements.txt
├── docker-compose.yml
├── airflow/
│   ├── dags/
│   │   ├── retail_sales_pipeline.py
│   │   └── data_quality_monitoring.py
│   ├── plugins/
│   └── config/
├── tfx/
│   ├── pipelines/
│   │   ├── __init__.py
│   │   ├── retail_pipeline.py
│   │   └── pipeline_config.py
│   ├── components/
│   │   ├── __init__.py
│   │   ├── custom_components.py
│   │   └── data_validation.py
│   └── examples/
│       ├── __init__.py
│       └── retail_example_gen.py
├── data/
│   ├── raw/
│   ├── processed/
│   └── schema/
├── models/
│   ├── trained/
│   └── serving/
├── monitoring/
│   ├── data_quality.py
│   ├── model_performance.py
│   └── alerts.py
├── tests/
│   ├── unit/
│   ├── integration/
│   └── fixtures/
├── docs/
│   ├── architecture.md
│   ├── api_documentation.md
│   └── deployment_guide.md
└── scripts/
    ├── setup_environment.py
    ├── generate_sample_data.py
    └── deploy_pipeline.py
```

## 🚀 Fases de Implementación

### **Fase 1: Configuración del Entorno**
```bash
# 1. Clonar repositorio y configurar entorno virtual
git clone <repository_url>
cd retail-sales-pipeline
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Iniciar servicios con Docker Compose
docker-compose up -d

# 4. Inicializar base de datos
python scripts/setup_environment.py
```

### **Fase 2: Generación de Datos de Ejemplo**
```python
# scripts/generate_sample_data.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_retail_data(days=365, stores=10, products=100):
    """Genera datos sintéticos de ventas retail"""
    
    # Generar fechas
    start_date = datetime.now() - timedelta(days=days)
    dates = [start_date + timedelta(days=i) for i in range(days)]
    
    data = []
    for date in dates:
        for store_id in range(stores):
            for product_id in range(products):
                # Simular ventas con patrones estacionales
                base_sales = np.random.normal(50, 15)
                seasonal_factor = 1 + 0.3 * np.sin(2 * np.pi * date.timetuple().tm_yday / 365)
                weekend_factor = 1.2 if date.weekday() >= 5 else 1.0
                
                sales = max(0, base_sales * seasonal_factor * weekend_factor)
                
                data.append({
                    'date': date,
                    'store_id': store_id,
                    'product_id': product_id,
                    'sales': sales,
                    'inventory': np.random.normal(200, 50),
                    'price': np.random.uniform(10, 100),
                    'promotion': np.random.choice([0, 1], p=[0.8, 0.2])
                })
    
    return pd.DataFrame(data)

# Generar y guardar datos
df = generate_retail_data()
df.to_csv('data/raw/retail_sales.csv', index=False)
```

### **Fase 3: Pipeline TFX Principal**
```python
# tfx/pipelines/retail_pipeline.py
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

from components.custom_components import (
    DataQualityValidator,
    FeatureEngineering
)

def create_retail_pipeline(
    pipeline_name: str,
    pipeline_root: str,
    data_path: str,
    module_file: str,
    serving_model_dir: str,
    metadata_path: str
) -> tfx.dsl.Pipeline:
    """
    Crea el pipeline completo de ventas retail
    """
    
    # 1. Generación de ejemplos desde CSV
    example_gen = CsvExampleGen(
        input_base=data_path
    )
    
    # 2. Estadísticas de datos
    statistics_gen = StatisticsGen(
        examples=example_gen.outputs['examples']
    )
    
    # 3. Generación de schema
    schema_gen = SchemaGen(
        statistics=statistics_gen.outputs['statistics'],
        infer_feature_shape=True
    )
    
    # 4. Validación de ejemplos
    example_validator = ExampleValidator(
        statistics=statistics_gen.outputs['statistics'],
        schema=schema_gen.outputs['schema']
    )
    
    # 5. Componente personalizado de calidad de datos
    data_quality_validator = DataQualityValidator(
        examples=example_gen.outputs['examples'],
        statistics=statistics_gen.outputs['statistics'],
        schema=schema_gen.outputs['schema']
    )
    
    # 6. Transformación de features
    transform = Transform(
        examples=example_gen.outputs['examples'],
        schema=schema_gen.outputs['schema'],
        module_file=module_file
    )
    
    # 7. Entrenamiento del modelo
    trainer = Trainer(
        module_file=module_file,
        custom_executor_spec=executor_spec.ExecutorClassSpec(
            trainer_pb2.Trainer
        ),
        examples=transform.outputs['transformed_examples'],
        schema=schema_gen.outputs['schema'],
        transform_graph=transform.outputs['transform_graph'],
        train_args=tfx.proto.TrainArgs(num_steps=10000),
        eval_args=tfx.proto.EvalArgs(num_steps=5000)
    )
    
    # 8. Evaluación del modelo
    evaluator = Evaluator(
        examples=transform.outputs['transformed_examples'],
        model=trainer.outputs['model'],
        schema=schema_gen.outputs['schema']
    )
    
    # 9. Despliegue del modelo
    pusher = Pusher(
        model=trainer.outputs['model'],
        model_blessing=evaluator.outputs['blessing'],
        push_destination=pusher_pb2.PushDestination(
            filesystem=pusher_pb2.PushDestination.Filesystem(
                base_directory=serving_model_dir
            )
        )
    )
    
    # Construir el pipeline
    pipeline = tfx.dsl.Pipeline(
        pipeline_name=pipeline_name,
        pipeline_root=pipeline_root,
        components=[
            example_gen,
            statistics_gen,
            schema_gen,
            example_validator,
            data_quality_validator,
            transform,
            trainer,
            evaluator,
            pusher
        ],
        enable_cache=True,
        metadata_connection_config=tfx.orchestration.metadata
            .sqlite_metadata_connection_config(metadata_path)
    )
    
    return pipeline
```

### **Fase 4: DAG de Airflow**
```python
# airflow/dags/retail_sales_pipeline.py
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.sensors.filesystem import FileSensor

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

def run_tfx_pipeline():
    """Ejecuta el pipeline TFX"""
    import subprocess
    result = subprocess.run([
        'tfx', 'pipeline', 'create',
        '--pipeline-path', 'tfx/pipelines/retail_pipeline.py',
        '--engine', 'airflow'
    ], capture_output=True, text=True)
    
    if result.returncode != 0:
        raise Exception(f"Pipeline execution failed: {result.stderr}")
    
    return result.stdout

def generate_data_quality_report():
    """Genera reporte de calidad de datos"""
    from monitoring.data_quality import DataQualityMonitor
    
    monitor = DataQualityMonitor()
    report = monitor.generate_daily_report()
    
    # Enviar alerta si la calidad es baja
    if report['overall_score'] < 0.8:
        from monitoring.alerts import send_alert
        send_alert(
            subject="Data Quality Alert",
            message=f"Data quality score: {report['overall_score']:.2f}"
        )

with DAG(
    'retail_sales_pipeline',
    default_args=default_args,
    description='Pipeline automatizado de ventas retail',
    schedule_interval='@daily',
    catchup=False,
    max_active_runs=1
) as dag:
    
    # Sensor para detectar nuevos datos
    data_sensor = FileSensor(
        task_id='wait_for_new_data',
        filepath='/opt/airflow/data/raw/',
        poke_interval=300,
        timeout=3600
    )
    
    # Generar datos de ejemplo (para desarrollo)
    generate_data = PythonOperator(
        task_id='generate_sample_data',
        python_callable=lambda: subprocess.run([
            'python', 'scripts/generate_sample_data.py'
        ])
    )
    
    # Ejecutar pipeline TFX
    run_pipeline = PythonOperator(
        task_id='run_tfx_pipeline',
        python_callable=run_tfx_pipeline
    )
    
    # Validar calidad de datos
    validate_quality = PythonOperator(
        task_id='validate_data_quality',
        python_callable=generate_data_quality_report
    )
    
    # Monitorear modelo en producción
    monitor_model = DockerOperator(
        task_id='monitor_model_performance',
        image='python:3.9',
        command='python monitoring/model_performance.py',
        volumes=['./monitoring:/app/monitoring']
    )
    
    # Definir dependencias
    data_sensor >> generate_data >> run_pipeline >> validate_quality >> monitor_model
```

## 📊 Componentes Personalizados

### **Validador de Calidad de Datos**
```python
# tfx/components/data_validation.py
import tensorflow as tf
from tfx.components.base import base_component
from tfx.types import standard_component_specs
from tfx.types.component_spec import ChannelParameter

class DataQualityValidator(base_component.BaseComponent):
    """
    Componente personalizado para validación avanzada de calidad de datos
    """
    
    SPEC_CLASS = standard_component_specs.StandardComponentSpec
    
    def __init__(
        self,
        examples: ChannelParameter,
        statistics: ChannelParameter,
        schema: ChannelParameter,
        quality_threshold: float = 0.8
    ):
        spec = self.SPEC_CLASS(
            examples=examples,
            statistics=statistics,
            schema=schema,
            quality_threshold=quality_threshold
        )
        super().__init__(spec=spec)
    
    def _run(self):
        """Ejecuta la validación de calidad de datos"""
        
        # Cargar estadísticas y schema
        stats = self._get_statistics()
        schema = self._get_schema()
        
        # Calcular métricas de calidad
        quality_metrics = self._calculate_quality_metrics(stats, schema)
        
        # Generar reporte
        report = self._generate_quality_report(quality_metrics)
        
        # Validar contra umbral
        if quality_metrics['overall_score'] < self._spec.quality_threshold:
            raise ValueError(
                f"Data quality score {quality_metrics['overall_score']:.2f} "
                f"below threshold {self._spec.quality_threshold}"
            )
        
        return {'quality_report': report}
```

## 📈 Monitoreo y Alertas

### **Dashboard de Monitoreo**
```python
# monitoring/dashboard.py
import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta

def create_monitoring_dashboard():
    """Crea dashboard de monitoreo del pipeline"""
    
    st.title("Pipeline de Ventas Retail - Monitoreo")
    
    # Métricas principales
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Data Quality Score", "0.92", "↑ 0.05")
    
    with col2:
        st.metric("Model Accuracy", "0.87", "↑ 0.02")
    
    with col3:
        st.metric("Pipeline Latency", "4.2 min", "↓ 0.3 min")
    
    with col4:
        st.metric("Daily Predictions", "1.2M", "↑ 100K")
    
    # Gráficos
    tab1, tab2, tab3 = st.tabs(["Data Quality", "Model Performance", "Pipeline Health"])
    
    with tab1:
        # Gráfico de calidad de datos
        quality_data = get_quality_metrics()
        fig = px.line(
            quality_data, 
            x='date', 
            y='quality_score',
            title='Data Quality Over Time'
        )
        st.plotly_chart(fig)
    
    with tab2:
        # Gráfico de rendimiento del modelo
        model_data = get_model_metrics()
        fig = px.scatter(
            model_data,
            x='prediction_date',
            y='mae',
            color='model_version',
            title='Model Performance Over Time'
        )
        st.plotly_chart(fig)
    
    with tab3:
        # Estado del pipeline
        pipeline_status = get_pipeline_status()
        st.dataframe(pipeline_status)

if __name__ == "__main__":
    create_monitoring_dashboard()
```

## 🧪 Testing y Validación

### **Tests Unitarios**
```python
# tests/unit/test_pipeline_components.py
import unittest
import pandas as pd
from tfx.components import StatisticsGen, SchemaGen
from components.custom_components import DataQualityValidator

class TestPipelineComponents(unittest.TestCase):
    
    def setUp(self):
        """Configuración inicial de tests"""
        self.sample_data = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=100),
            'store_id': range(100),
            'product_id': range(100),
            'sales': [50] * 100,
            'inventory': [200] * 100,
            'price': [25.0] * 100
        })
    
    def test_data_quality_validator(self):
        """Test del validador de calidad de datos"""
        validator = DataQualityValidator(quality_threshold=0.8)
        
        # Test con datos buenos
        good_quality = validator.validate_data(self.sample_data)
        self.assertGreater(good_quality['overall_score'], 0.8)
        
        # Test con datos malos
        bad_data = self.sample_data.copy()
        bad_data.loc[:10, 'sales'] = -1  # Valores inválidos
        bad_quality = validator.validate_data(bad_data)
        self.assertLess(bad_quality['overall_score'], 0.8)
    
    def test_feature_engineering(self):
        """Test del componente de feature engineering"""
        from components.custom_components import FeatureEngineering
        
        fe = FeatureEngineering()
        features = fe.transform(self.sample_data)
        
        # Verificar que se crearon las features esperadas
        expected_features = [
            'day_of_week', 'month', 'is_weekend', 
            'sales_lag_1', 'sales_ma_7'
        ]
        
        for feature in expected_features:
            self.assertIn(feature, features.columns)

if __name__ == '__main__':
    unittest.main()
```

## 🚀 Despliegue en Producción

### **Docker Compose para Producción**
```yaml
# docker-compose.prod.yml
version: '3.8'

services:
  airflow:
    image: apache/airflow:2.5.0
    environment:
      - AIRFLOW__CORE__EXECUTOR=LocalExecutor
      - AIRFLOW__CORE__SQL_ALCHEMY_CONN=postgresql://airflow:airflow@postgres:5432/airflow
    volumes:
      - ./airflow:/opt/airflow
      - ./tfx:/opt/tfx
    depends_on:
      - postgres
      - redis
  
  postgres:
    image: postgres:13
    environment:
      - POSTGRES_USER=airflow
      - POSTGRES_PASSWORD=airflow
      - POSTGRES_DB=airflow
    volumes:
      - postgres_data:/var/lib/postgresql/data
  
  redis:
    image: redis:6-alpine
    ports:
      - "6379:6379"
  
  minio:
    image: minio/minio
    command: server /data --console-address ":9001"
    environment:
      - MINIO_ROOT_USER=minioadmin
      - MINIO_ROOT_PASSWORD=minioadmin
    ports:
      - "9000:9000"
      - "9001:9001"
    volumes:
      - minio_data:/data
  
  mlflow:
    image: python:3.9
    command: >
      bash -c "pip install mlflow && 
               mlflow server --host 0.0.0.0 --port 5000 
               --default-artifact-root /mlflow/artifacts"
    ports:
      - "5000:5000"
    volumes:
      - ./mlflow:/mlflow

volumes:
  postgres_data:
  minio_data:
```

## 📋 Entregables del Laboratorio

### **📁 Código Fuente**
- Pipeline TFX completo y funcional
- DAGs de Airflow orquestados
- Componentes personalizados de validación
- Sistema de monitoreo y alertas

### **📊 Documentación**
- Diagrama de arquitectura completo
- API documentation
- Guía de despliegue
- Manual de operación

### **🧪 Testing**
- Suite de tests unitarios
- Tests de integración
- Tests de performance
- Fixtures de datos

### **🚀 Infraestructura**
- Configuración Docker completa
- Scripts de setup y deploy
- Configuración de monitoreo
- Backup y recovery

## 🎯 Criterios de Evaluación

### **Funcionalidad (40%)**
- Pipeline ejecuta correctamente de principio a fin
- Manejo apropiado de errores y excepciones
- Validación de datos funciona correctamente
- Modelo se entrena y despliega automáticamente

### **Calidad del Código (30%)**
- Código limpio y bien documentado
- Buenas prácticas de desarrollo
- Testing adecuado
- Modularidad y reusabilidad

### **Infraestructura (20%)**
- Configuración Docker correcta
- Despliegue automatizado funciona
- Monitoreo y alertas operativas
- Documentación completa

### **Innovación (10%)**
- Componentes personalizados útiles
- Optimizaciones de performance
- Mejoras en usabilidad
- Extensiones creativas

---

**Duración Estimada**: 2 semanas  
**Dificultad**: Intermedia-Avanzada  
**Prerrequisitos**: Python, Docker, SQL, Machine Learning básico
