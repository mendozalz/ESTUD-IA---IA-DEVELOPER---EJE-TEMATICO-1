"""
Caso de Uso 2 - Detección de Drift con Evidently AI y Alertas en Grafana
Fase 2: Pipeline de Reentrenamiento Automático con Airflow
"""

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.providers.http.operators.http import HttpOperator
from airflow.providers.http.sensors.http import HttpSensor
from airflow.providers.slack.operators.slack import SlackAPIPostOperator
from airflow.operators.email import EmailOperator
from datetime import datetime, timedelta
import requests
import pandas as pd
import numpy as np
import json
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuración por defecto
default_args = {
    'owner': 'ml-team',
    'depends_on_past': False,
    'start_date': datetime(2026, 1, 1),
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
    'sla': timedelta(hours=2),
}

def check_drift_status(**context):
    """
    Verificar el estado de drift del modelo
    """
    logger.info("Verificando estado de drift del modelo...")
    
    try:
        # Obtener reporte de drift de la API
        response = requests.get("http://localhost:8000/drift/report", timeout=30)
        response.raise_for_status()
        
        drift_data = response.json()
        
        logger.info(f"Drift score: {drift_data.get('dataset_drift_score', 0):.4f}")
        logger.info(f"Drift detected: {drift_data.get('drift_detected', False)}")
        
        # Almacenar en XCom para uso posterior
        context['ti'].xcom_push(key='drift_data', value=drift_data)
        
        # Determinar si se necesita reentrenamiento
        drift_threshold = 0.2
        drift_score = drift_data.get('dataset_drift_score', 0)
        drift_detected = drift_score > drift_threshold
        
        logger.info(f"Decisión de reentrenamiento: {'SÍ' if drift_detected else 'NO'}")
        
        return drift_detected
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Error verificando drift: {e}")
        raise Exception(f"No se pudo conectar a la API del modelo: {e}")
    except Exception as e:
        logger.error(f"Error inesperado: {e}")
        raise

def collect_new_data(**context):
    """
    Recolectar nuevos datos para reentrenamiento
    """
    logger.info("Recolectando nuevos datos para reentrenamiento...")
    
    try:
        # En producción, esto conectaría a bases de datos reales
        # Aquí simulamos la recolección de datos
        
        # Cargar datos actuales
        current_data_path = 'data/current_data.csv'
        if os.path.exists(current_data_path):
            current_data = pd.read_csv(current_data_path)
            logger.info(f"Datos actuales cargados: {len(current_data)} muestras")
        else:
            logger.warning("No se encontraron datos actuales")
            return False
        
        # Validar calidad de datos
        if len(current_data) < 100:
            logger.warning("Insuficientes datos para reentrenamiento")
            return False
        
        # Verificar columnas requeridas
        required_columns = ['product_id', 'store_id', 'price', 'demand', 'promotion']
        missing_columns = [col for col in required_columns if col not in current_data.columns]
        
        if missing_columns:
            logger.error(f"Columnas faltantes: {missing_columns}")
            return False
        
        # Guardar datos para reentrenamiento
        training_data_path = 'data/training_data.csv'
        current_data.to_csv(training_data_path, index=False)
        
        logger.info(f"Datos guardados para reentrenamiento: {len(current_data)} muestras")
        
        # Almacenar estadísticas
        stats = {
            'total_samples': len(current_data),
            'columns': list(current_data.columns),
            'date_range': {
                'start': current_data.get('timestamp', pd.Series([None])).min(),
                'end': current_data.get('timestamp', pd.Series([None])).max()
            }
        }
        
        context['ti'].xcom_push(key='data_stats', value=stats)
        
        return True
        
    except Exception as e:
        logger.error(f"Error recolectando datos: {e}")
        raise

def train_model(**context):
    """
    Entrenar nuevo modelo con datos actualizados
    """
    logger.info("Iniciando entrenamiento del modelo...")
    
    try:
        # Cargar datos de entrenamiento
        training_data_path = 'data/training_data.csv'
        if not os.path.exists(training_data_path):
            raise FileNotFoundError("No se encontraron datos de entrenamiento")
        
        data = pd.read_csv(training_data_path)
        logger.info(f"Datos de entrenamiento cargados: {len(data)} muestras")
        
        # Preparar características y objetivo
        feature_columns = ['price', 'promotion', 'day_of_week', 'month', 'competitor_price']
        target_column = 'demand'
        
        # Verificar columnas disponibles
        available_features = [col for col in feature_columns if col in data.columns]
        if len(available_features) < 3:
            raise ValueError("Insuficientes características para entrenamiento")
        
        X = data[available_features].fillna(0)
        y = data[target_column].fillna(data[target_column].median())
        
        # Simular entrenamiento (en producción, usar modelo real)
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import mean_absolute_error, r2_score
        
        # Dividir datos
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Entrenar modelo
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        
        model.fit(X_train, y_train)
        
        # Evaluar modelo
        y_pred = model.predict(X_val)
        mae = mean_absolute_error(y_val, y_pred)
        r2 = r2_score(y_val, y_pred)
        
        logger.info(f"Métricas de evaluación - MAE: {mae:.2f}, R2: {r2:.3f}")
        
        # Guardar modelo
        import joblib
        model_path = 'models/demand_model_v2.pkl'
        joblib.dump(model, model_path)
        
        logger.info(f"Modelo guardado en {model_path}")
        
        # Almacenar métricas
        model_metrics = {
            'mae': float(mae),
            'r2': float(r2),
            'features': available_features,
            'model_path': model_path,
            'training_samples': len(X_train),
            'validation_samples': len(X_val)
        }
        
        context['ti'].xcom_push(key='model_metrics', value=model_metrics)
        
        # Validar calidad del modelo
        if mae > 20 or r2 < 0.7:
            logger.warning("Modelo no cumple criterios de calidad")
            return False
        
        logger.info("Modelo entrenado exitosamente")
        return True
        
    except Exception as e:
        logger.error(f"Error entrenando modelo: {e}")
        raise

def validate_model(**context):
    """
    Validar nuevo modelo antes de despliegue
    """
    logger.info("Validando nuevo modelo...")
    
    try:
        # Obtener métricas del modelo
        model_metrics = context['ti'].xcom_pull(key='model_metrics', task_ids='train_model')
        
        if not model_metrics:
            raise ValueError("No se encontraron métricas del modelo")
        
        # Criterios de validación
        validation_criteria = {
            'max_mae': 20.0,
            'min_r2': 0.7,
            'min_samples': 100
        }
        
        validation_results = {
            'mae_valid': model_metrics['mae'] <= validation_criteria['max_mae'],
            'r2_valid': model_metrics['r2'] >= validation_criteria['min_r2'],
            'samples_valid': model_metrics['training_samples'] >= validation_criteria['min_samples']
        }
        
        # Validación general
        all_valid = all(validation_results.values())
        
        logger.info(f"Resultados de validación: {validation_results}")
        logger.info(f"Validación general: {'APROBADA' if all_valid else 'RECHAZADA'}")
        
        if all_valid:
            # Almacenar resultado de validación
            context['ti'].xcom_push(key='validation_passed', value=True)
            return True
        else:
            context['ti'].xcom_push(key='validation_passed', value=False)
            logger.warning("Modelo no pasó validación")
            return False
            
    except Exception as e:
        logger.error(f"Error en validación: {e}")
        raise

def deploy_model(**context):
    """
    Desplegar nuevo modelo en producción
    """
    logger.info("Desplegando nuevo modelo en producción...")
    
    try:
        # Verificar que la validación pasó
        validation_passed = context['ti'].xcom_pull(key='validation_passed', task_ids='validate_model')
        
        if not validation_passed:
            raise ValueError("Modelo no validado, no se puede desplegar")
        
        # Obtener ruta del modelo
        model_metrics = context['ti'].xcom_pull(key='model_metrics', task_ids='train_model')
        model_path = model_metrics['model_path']
        
        # Simular despliegue (en producción, esto actualizaría el servicio)
        # Aquí simplemente movemos el modelo a la ubicación de producción
        
        production_model_path = 'models/production/demand_model.pkl'
        os.makedirs(os.path.dirname(production_model_path), exist_ok=True)
        
        import shutil
        shutil.copy2(model_path, production_model_path)
        
        logger.info(f"Modelo desplegado en {production_model_path}")
        
        # Actualizar metadatos del despliegue
        deployment_info = {
            'deployment_time': datetime.now().isoformat(),
            'model_path': production_model_path,
            'model_version': 'v2',
            'previous_version': 'v1',
            'validation_metrics': model_metrics,
            'deployment_status': 'success'
        }
        
        # Guardar información de despliegue
        with open('outputs/deployment_logs/latest_deployment.json', 'w') as f:
            json.dump(deployment_info, f, indent=2)
        
        context['ti'].xcom_push(key='deployment_info', value=deployment_info)
        
        logger.info("Modelo desplegado exitosamente")
        return True
        
    except Exception as e:
        logger.error(f"Error en despliegue: {e}")
        raise

def update_reference_data(**context):
    """
    Actualizar datos de referencia con los datos más recientes
    """
    logger.info("Actualizando datos de referencia...")
    
    try:
        # Cargar datos actuales
        current_data_path = 'data/current_data.csv'
        if not os.path.exists(current_data_path):
            raise FileNotFoundError("No se encontraron datos actuales")
        
        current_data = pd.read_csv(current_data_path)
        
        # Filtrar datos recientes (últimos 30 días)
        if 'timestamp' in current_data.columns:
            current_data['timestamp'] = pd.to_datetime(current_data['timestamp'])
            thirty_days_ago = datetime.now() - timedelta(days=30)
            recent_data = current_data[current_data['timestamp'] >= thirty_days_ago]
        else:
            recent_data = current_data.tail(1000)  # Últimos 1000 registros
        
        # Actualizar datos de referencia
        reference_data_path = 'data/reference_data.csv'
        recent_data.to_csv(reference_data_path, index=False)
        
        logger.info(f"Datos de referencia actualizados: {len(recent_data)} muestras")
        
        # Almacenar información de actualización
        update_info = {
            'update_time': datetime.now().isoformat(),
            'reference_samples': len(recent_data),
            'date_range': {
                'start': recent_data['timestamp'].min() if 'timestamp' in recent_data.columns else None,
                'end': recent_data['timestamp'].max() if 'timestamp' in recent_data.columns else None
            }
        }
        
        context['ti'].xcom_push(key='reference_update_info', value=update_info)
        
        return True
        
    except Exception as e:
        logger.error(f"Error actualizando datos de referencia: {e}")
        raise

def send_success_notification(**context):
    """
    Enviar notificación de éxito del reentrenamiento
    """
    logger.info("Enviando notificación de éxito...")
    
    try:
        # Obtener información del despliegue
        deployment_info = context['ti'].xcom_pull(key='deployment_info', task_ids='deploy_model')
        
        if not deployment_info:
            raise ValueError("No se encontró información de despliegue")
        
        # Crear mensaje de notificación
        message = f"""
        ✅ **Reentrenamiento de Modelo Exitoso**
        
        **Modelo:** Predicción de Demanda
        **Versión:** {deployment_info['model_version']}
        **Tiempo de despliegue:** {deployment_info['deployment_time']}
        
        **Métricas de Validación:**
        - MAE: {deployment_info['validation_metrics']['mae']:.2f}
        - R²: {deployment_info['validation_metrics']['r2']:.3f}
        
        **Estado:** Desplegado y funcionando
        """
        
        logger.info("Notificación de éxito enviada")
        return message
        
    except Exception as e:
        logger.error(f"Error enviando notificación: {e}")
        raise

def send_failure_notification(**context):
    """
    Enviar notificación de fallo del reentrenamiento
    """
    logger.info("Enviando notificación de fallo...")
    
    try:
        # Obtener información del error
        task_instance = context['task_instance']
        error_message = str(task_instance.error) if task_instance.error else "Error desconocido"
        
        # Crear mensaje de notificación
        message = f"""
        ❌ **Fallo en Reentrenamiento de Modelo**
        
        **Modelo:** Predicción de Demanda
        **Tiempo del fallo:** {datetime.now().isoformat()}
        **Tarea fallida:** {task_instance.task_id}
        **Error:** {error_message}
        
        **Acción requerida:** Investigar y resolver el problema
        """
        
        logger.info("Notificación de fallo enviada")
        return message
        
    except Exception as e:
        logger.error(f"Error enviando notificación de fallo: {e}")
        raise

# Crear DAG
with DAG(
    dag_id='demand_model_retraining',
    default_args=default_args,
    description='Pipeline automático de reentrenamiento de modelo de predicción de demanda',
    schedule_interval='@daily',
    catchup=False,
    max_active_runs=1,
    tags=['ml', 'retraining', 'demand-prediction'],
) as dag:
    
    # Tarea para verificar drift
    check_drift_task = PythonOperator(
        task_id='check_drift',
        python_callable=check_drift_status,
        doc_md="""Verificar si el modelo presenta drift significativo que requiera reentrenamiento."""
    )
    
    # Tarea para recolectar datos
    collect_data_task = PythonOperator(
        task_id='collect_data',
        python_callable=collect_new_data,
        doc_md="""Recolectar nuevos datos de producción para el reentrenamiento."""
    )
    
    # Tarea para entrenar modelo
    train_model_task = PythonOperator(
        task_id='train_model',
        python_callable=train_model,
        doc_md="""Entrenar nuevo modelo con los datos actualizados."""
    )
    
    # Tarea para validar modelo
    validate_model_task = PythonOperator(
        task_id='validate_model',
        python_callable=validate_model,
        doc_md="""Validar calidad del nuevo modelo antes del despliegue."""
    )
    
    # Tarea para desplegar modelo
    deploy_model_task = PythonOperator(
        task_id='deploy_model',
        python_callable=deploy_model,
        doc_md="""Desplegar nuevo modelo en producción."""
    )
    
    # Tarea para actualizar datos de referencia
    update_reference_task = PythonOperator(
        task_id='update_reference_data',
        python_callable=update_reference_data,
        doc_md="""Actualizar datos de referencia para futuras detecciones de drift."""
    )
    
    # Notificación de éxito
    success_notification = PythonOperator(
        task_id='send_success_notification',
        python_callable=send_success_notification,
        trigger_rule='all_success',
        doc_md="""Enviar notificación cuando el reentrenamiento es exitoso."""
    )
    
    # Notificación de fallo
    failure_notification = PythonOperator(
        task_id='send_failure_notification',
        python_callable=send_failure_notification,
        trigger_rule='one_failed',
        doc_md="""Enviar notificación cuando el reentrenamiento falla."""
    )
    
    # Definir dependencias
    check_drift_task >> collect_data_task >> train_model_task >> validate_model_task >> deploy_model_task >> update_reference_task >> success_notification
    
    # Notificación de fallo para cualquier tarea
    [check_drift_task, collect_data_task, train_model_task, validate_model_task, deploy_model_task, update_reference_task] >> failure_notification
