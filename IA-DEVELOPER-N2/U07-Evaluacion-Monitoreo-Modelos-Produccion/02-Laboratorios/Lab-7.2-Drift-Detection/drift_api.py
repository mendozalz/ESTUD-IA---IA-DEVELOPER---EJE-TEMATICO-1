"""
Caso de Uso 2 - Detección de Drift con Evidently AI y Alertas en Grafana
Fase 1: API con Métricas de Drift para Predicción de Demanda
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
from prometheus_client import make_asgi_app, Gauge, Counter, Histogram
from evidently.report import Report
from evidently.metrics import DatasetDriftMetric, ColumnDriftMetric
from evidently.test_suite import TestSuite
from evidently.tests import TestNumberOfRowsWithMissingValues
import json
import time
import logging
from datetime import datetime, timedelta
import os

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DemandPredictionModel:
    """
    Modelo de predicción de demanda con monitoreo de drift
    """
    
    def __init__(self):
        self.reference_data = None
        self.current_data = None
        self.model_version = "1.0.0"
        self.last_retrain_date = datetime.now()
        
    def load_reference_data(self):
        """Cargar datos de referencia históricos"""
        try:
            self.reference_data = pd.read_csv('data/reference_data.csv')
            logger.info(f"Datos de referencia cargados: {len(self.reference_data)} muestras")
            return True
        except FileNotFoundError:
            logger.warning("No se encontraron datos de referencia, generando datos de ejemplo")
            self.reference_data = self._generate_reference_data()
            self.reference_data.to_csv('data/reference_data.csv', index=False)
            return True
        except Exception as e:
            logger.error(f"Error cargando datos de referencia: {e}")
            return False
    
    def _generate_reference_data(self):
        """Generar datos de referencia de ejemplo"""
        np.random.seed(42)
        n_samples = 1000
        
        data = {
            'product_id': np.random.choice(['P001', 'P002', 'P003', 'P004', 'P005'], n_samples),
            'store_id': np.random.choice(['S001', 'S002', 'S003'], n_samples),
            'day_of_week': np.random.randint(0, 7, n_samples),
            'month': np.random.randint(1, 13, n_samples),
            'price': np.random.uniform(10, 100, n_samples),
            'promotion': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
            'competitor_price': np.random.uniform(8, 120, n_samples),
            'demand': np.random.poisson(50, n_samples) + np.random.normal(0, 10, n_samples)
        }
        
        return pd.DataFrame(data)
    
    def predict_demand(self, features: Dict[str, Any]) -> float:
        """Predecir demanda (modelo simplificado)"""
        # En producción, usar modelo real entrenado
        base_demand = 50
        
        # Factores de influencia
        price_factor = max(0.5, 1 - (features.get('price', 50) - 50) / 100)
        promotion_factor = 1.3 if features.get('promotion', 0) == 1 else 1.0
        competitor_factor = min(1.2, 1 + (features.get('competitor_price', 60) - features.get('price', 50)) / 100)
        
        # Efecto estacional
        month = features.get('month', 6)
        seasonal_factor = 1.0 + 0.3 * np.sin(2 * np.pi * month / 12)
        
        # Predicción final
        prediction = base_demand * price_factor * promotion_factor * competitor_factor * seasonal_factor
        prediction += np.random.normal(0, 5)  # Ruido
        
        return max(0, int(prediction))
    
    def add_current_data(self, features: Dict[str, Any], prediction: float):
        """Agregar datos actuales para monitoreo"""
        new_row = features.copy()
        new_row['demand'] = prediction
        new_row['timestamp'] = datetime.now().isoformat()
        
        if self.current_data is None:
            self.current_data = pd.DataFrame([new_row])
        else:
            self.current_data = pd.concat([self.current_data, pd.DataFrame([new_row])], ignore_index=True)
        
        # Mantener solo últimos 1000 registros
        if len(self.current_data) > 1000:
            self.current_data = self.current_data.tail(1000)
        
        # Guardar periódicamente
        if len(self.current_data) % 100 == 0:
            self.current_data.to_csv('data/current_data.csv', index=False)

class DriftDetectionSystem:
    """
    Sistema de detección de drift con Evidently AI
    """
    
    def __init__(self, reference_data: pd.DataFrame):
        self.reference_data = reference_data
        self.drift_history = []
        self.drift_threshold = 0.2
        
    def detect_drift(self, current_data: pd.DataFrame) -> Dict[str, Any]:
        """Detectar drift en los datos actuales"""
        try:
            # Crear reporte de drift
            drift_report = Report(metrics=[
                DatasetDriftMetric(),
                ColumnDriftMetric(column_name='price'),
                ColumnDriftMetric(column_name='demand'),
                ColumnDriftMetric(column_name='competitor_price')
            ])
            
            drift_report.run(
                reference_data=self.reference_data,
                current_data=current_data
            )
            
            # Extraer resultados
            dataset_drift = drift_report.metrics[0].result['drift_score']
            column_drift = {}
            
            for metric in drift_report.metrics[1:]:
                column_name = metric.column_name
                drift_score = metric.result['drift_score']
                column_drift[column_name] = drift_score
            
            # Determinar si hay drift
            drift_detected = dataset_drift > self.drift_threshold
            
            # Registrar en historial
            drift_record = {
                'timestamp': datetime.now().isoformat(),
                'dataset_drift_score': dataset_drift,
                'column_drift_scores': column_drift,
                'drift_detected': drift_detected,
                'current_data_size': len(current_data)
            }
            
            self.drift_history.append(drift_record)
            
            logger.info(f"Drift detectado: {drift_detected}, Score: {dataset_drift:.4f}")
            
            return {
                'drift_detected': drift_detected,
                'dataset_drift_score': dataset_drift,
                'column_drift_scores': column_drift,
                'drift_threshold': self.drift_threshold,
                'timestamp': drift_record['timestamp']
            }
            
        except Exception as e:
            logger.error(f"Error detectando drift: {e}")
            return {
                'drift_detected': False,
                'error': str(e)
            }
    
    def get_drift_trend(self, window_size: int = 10) -> Dict[str, Any]:
        """Analizar tendencia de drift"""
        if len(self.drift_history) < 2:
            return {'trend': 'insufficient_data'}
        
        recent_scores = [record['dataset_drift_score'] for record in self.drift_history[-window_size:]]
        
        if len(recent_scores) < 2:
            return {'trend': 'insufficient_data'}
        
        # Calcular tendencia
        slope = (recent_scores[-1] - recent_scores[0]) / len(recent_scores)
        
        if slope > 0.01:
            trend = 'increasing'
        elif slope < -0.01:
            trend = 'decreasing'
        else:
            trend = 'stable'
        
        return {
            'trend': trend,
            'slope': slope,
            'recent_average': np.mean(recent_scores),
            'current': recent_scores[-1],
            'max': max(recent_scores),
            'min': min(recent_scores)
        }

class MonitoringMetrics:
    """
    Sistema de métricas para Prometheus
    """
    
    def __init__(self):
        # Métricas de Prometheus
        self.request_count = Counter(
            'demand_prediction_requests_total',
            'Total number of demand prediction requests',
            ['endpoint', 'status']
        )
        
        self.prediction_latency = Histogram(
            'demand_prediction_latency_seconds',
            'Time spent processing demand prediction requests'
        )
        
        self.drift_score = Gauge(
            'dataset_drift_score',
            'Current dataset drift score'
        )
        
        self.column_drift_scores = Gauge(
            'column_drift_score',
            'Drift score for specific columns',
            ['column']
        )
        
        self.model_accuracy = Gauge(
            'demand_model_accuracy',
            'Current model accuracy'
        )
        
        self.active_connections = Gauge(
            'active_connections',
            'Number of active connections'
        )
    
    def record_request(self, endpoint: str, status: str, duration: float):
        """Registrar métricas de solicitud"""
        self.request_count.labels(endpoint=endpoint, status=status).inc()
        self.prediction_latency.observe(duration)
    
    def update_drift_metrics(self, drift_result: Dict[str, Any]):
        """Actualizar métricas de drift"""
        if 'dataset_drift_score' in drift_result:
            self.drift_score.set(drift_result['dataset_drift_score'])
        
        if 'column_drift_scores' in drift_result:
            for column, score in drift_result['column_drift_scores'].items():
                self.column_drift_scores.labels(column=column).set(score)
    
    def update_model_metrics(self, accuracy: float):
        """Actualizar métricas del modelo"""
        self.model_accuracy.set(accuracy)

# Modelos Pydantic
class PredictionRequest(BaseModel):
    product_id: str
    store_id: str
    day_of_week: int
    month: int
    price: float
    promotion: int
    competitor_price: float

class PredictionResponse(BaseModel):
    predicted_demand: int
    confidence: float
    model_version: str
    timestamp: str
    processing_time_ms: float

class DriftReportResponse(BaseModel):
    drift_detected: bool
    dataset_drift_score: float
    column_drift_scores: Dict[str, float]
    drift_threshold: float
    timestamp: str

# Inicializar componentes
model = DemandPredictionModel()
drift_detector = None
metrics = MonitoringMetrics()

# Crear aplicación FastAPI
app = FastAPI(
    title="Demand Prediction API",
    description="API para predicción de demanda con detección de drift",
    version="1.0.0"
)

# Crear aplicación de métricas de Prometheus
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

@app.on_event("startup")
async def startup_event():
    """Evento de inicio de la aplicación"""
    logger.info("Iniciando API de predicción de demanda...")
    
    # Cargar datos de referencia
    if not model.load_reference_data():
        raise RuntimeError("No se pudieron cargar los datos de referencia")
    
    # Inicializar detector de drift
    global drift_detector
    drift_detector = DriftDetectionSystem(model.reference_data)
    
    logger.info("API iniciada exitosamente")

@app.post("/predict", response_model=PredictionResponse)
async def predict_demand(request: PredictionRequest):
    """Endpoint principal de predicción"""
    start_time = time.time()
    
    try:
        # Convertir a diccionario
        features = request.dict()
        
        # Realizar predicción
        predicted_demand = model.predict_demand(features)
        
        # Calcular confianza (simulada)
        confidence = min(0.95, max(0.5, 1.0 - abs(predicted_demand - 50) / 100))
        
        # Agregar a datos actuales
        model.add_current_data(features, predicted_demand)
        
        # Detectar drift si hay suficientes datos
        drift_result = {}
        if len(model.current_data) >= 50:
            drift_result = drift_detector.detect_drift(model.current_data)
            metrics.update_drift_metrics(drift_result)
        
        # Calcular tiempo de procesamiento
        processing_time = (time.time() - start_time) * 1000
        
        # Registrar métricas
        metrics.record_request("/predict", "success", processing_time / 1000)
        
        response = PredictionResponse(
            predicted_demand=predicted_demand,
            confidence=confidence,
            model_version=model.model_version,
            timestamp=datetime.now().isoformat(),
            processing_time_ms=processing_time
        )
        
        logger.info(f"Predicción exitosa: {predicted_demand} unidades")
        return response
        
    except Exception as e:
        processing_time = (time.time() - start_time) * 1000
        metrics.record_request("/predict", "error", processing_time / 1000)
        logger.error(f"Error en predicción: {e}")
        raise HTTPException(status_code=500, detail="Error interno del servidor")

@app.get("/drift/report", response_model=DriftReportResponse)
async def get_drift_report():
    """Obtener reporte de drift actual"""
    if drift_detector is None:
        raise HTTPException(status_code=503, detail="Sistema de drift no inicializado")
    
    if model.current_data is None or len(model.current_data) < 50:
        return DriftReportResponse(
            drift_detected=False,
            dataset_drift_score=0.0,
            column_drift_scores={},
            drift_threshold=drift_detector.drift_threshold,
            timestamp=datetime.now().isoformat()
        )
    
    drift_result = drift_detector.detect_drift(model.current_data)
    
    return DriftReportResponse(
        drift_detected=drift_result.get('drift_detected', False),
        dataset_drift_score=drift_result.get('dataset_drift_score', 0.0),
        column_drift_scores=drift_result.get('column_drift_scores', {}),
        drift_threshold=drift_detector.drift_threshold,
        timestamp=drift_result.get('timestamp', datetime.now().isoformat())
    )

@app.get("/drift/trend")
async def get_drift_trend():
    """Obtener tendencia de drift"""
    if drift_detector is None:
        raise HTTPException(status_code=503, detail="Sistema de drift no inicializado")
    
    trend = drift_detector.get_drift_trend()
    return trend

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model.reference_data is not None,
        "drift_detector_active": drift_detector is not None,
        "current_data_size": len(model.current_data) if model.current_data is not None else 0,
        "reference_data_size": len(model.reference_data) if model.reference_data is not None else 0,
        "uptime_seconds": time.time()
    }

@app.get("/model/info")
async def get_model_info():
    """Obtener información del modelo"""
    return {
        "model_name": "Demand Prediction Model",
        "version": model.model_version,
        "last_retrain_date": model.last_retrain_date.isoformat(),
        "features": [
            "product_id", "store_id", "day_of_week", "month",
            "price", "promotion", "competitor_price"
        ],
        "target": "demand",
        "reference_data_size": len(model.reference_data) if model.reference_data is not None else 0,
        "current_data_size": len(model.current_data) if model.current_data is not None else 0
    }

@app.get("/predictions/stats")
async def get_prediction_stats():
    """Obtener estadísticas de predicciones"""
    if model.current_data is None:
        return {
            "total_predictions": 0,
            "average_demand": 0,
            "demand_range": {"min": 0, "max": 0},
            "last_24h_predictions": 0
        }
    
    # Estadísticas básicas
    stats = {
        "total_predictions": len(model.current_data),
        "average_demand": float(model.current_data['demand'].mean()),
        "demand_range": {
            "min": int(model.current_data['demand'].min()),
            "max": int(model.current_data['demand'].max())
        }
    }
    
    # Predicciones últimas 24 horas
    if 'timestamp' in model.current_data.columns:
        last_24h = datetime.now() - timedelta(hours=24)
        recent_predictions = model.current_data[
            pd.to_datetime(model.current_data['timestamp']) >= last_24h
        ]
        stats["last_24h_predictions"] = len(recent_predictions)
    else:
        stats["last_24h_predictions"] = 0
    
    return stats

if __name__ == "__main__":
    import uvicorn
    
    logger.info("Iniciando servidor de API de predicción de demanda...")
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
