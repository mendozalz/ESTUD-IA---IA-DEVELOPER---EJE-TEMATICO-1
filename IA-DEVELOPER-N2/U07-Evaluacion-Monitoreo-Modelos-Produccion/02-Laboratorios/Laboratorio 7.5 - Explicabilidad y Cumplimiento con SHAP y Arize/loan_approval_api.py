"""
Caso de Uso 5 - Explicabilidad y Cumplimiento con SHAP y Arize
Fase 4: API de Aprobación de Préstamos con Explicaciones Integradas
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
import tensorflow as tf
import json
import logging
from datetime import datetime
import os
from shap_explanations import LoanExplainer
from arize_integration import ArizeMonitor

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Modelos Pydantic
class LoanApplication(BaseModel):
    application_id: str = Field(..., description="ID único de la solicitud")
    income: float = Field(..., ge=0, description="Ingreso anual del solicitante")
    credit_score: int = Field(..., ge=300, le=850, description="Puntaje de crédito")
    loan_amount: float = Field(..., ge=0, description="Monto del préstamo solicitado")
    employment_years: int = Field(..., ge=0, description="Años en el empleo actual")
    age: int = Field(..., ge=18, le=100, description="Edad del solicitante")
    debt_to_income: float = Field(..., ge=0, le=1, description="Ratio deuda-ingreso")
    has_collateral: bool = Field(..., description="Tiene garantía")
    loan_purpose: int = Field(..., ge=0, le=5, description="Propósito del préstamo (0-5)")

class PredictionResponse(BaseModel):
    application_id: str
    approved: bool
    confidence: float
    probability: float
    explanation: Optional[Dict[str, Any]] = None
    timestamp: str
    processing_time_ms: float

class ComplianceInfo(BaseModel):
    explanation_available: bool
    human_understandable: bool
    gdpr_compliant: bool
    audit_trail: bool
    regulatory_frameworks: List[str]

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    explainer_ready: bool
    arize_connected: bool
    timestamp: str

class LoanApprovalAPI:
    """
    API para aprobación de préstamos con explicaciones integradas
    """
    
    def __init__(self):
        self.explainer = None
        self.arize_monitor = None
        self.model_loaded = False
        self.explainer_ready = False
        self.arize_connected = False
        
    def initialize(self):
        """Inicializar componentes de la API"""
        logger.info("Inicializando API de aprobación de préstamos...")
        
        # Inicializar explainer SHAP
        try:
            self.explainer = LoanExplainer()
            if self.explainer.load_model():
                # Generar datos de fondo para SHAP
                background_data = self._generate_background_data()
                if self.explainer.setup_explainer(background_data):
                    self.explainer_ready = True
                    self.model_loaded = True
                    logger.info("Explainer SHAP inicializado")
        except Exception as e:
            logger.error(f"Error inicializando explainer: {e}")
        
        # Inicializar monitor Arize
        try:
            space_key = os.getenv("ARIZE_SPACE_KEY", "YOUR_SPACE_KEY")
            api_key = os.getenv("ARIZE_API_KEY", "YOUR_API_KEY")
            
            self.arize_monitor = ArizeMonitor(space_key, api_key)
            if self.arize_monitor.setup_arize():
                self.arize_connected = True
                logger.info("Monitor Arize inicializado")
        except Exception as e:
            logger.error(f"Error inicializando Arize: {e}")
        
        logger.info(f"Estado inicialización - Modelo: {self.model_loaded}, Explainer: {self.explainer_ready}, Arize: {self.arize_connected}")
    
    def _generate_background_data(self):
        """Generar datos de fondo para SHAP"""
        np.random.seed(42)
        n_samples = 100
        
        background_data = np.array([
            np.random.randint(20000, 150000, n_samples),  # income
            np.random.randint(300, 850, n_samples),        # credit_score
            np.random.randint(5000, 100000, n_samples),    # loan_amount
            np.random.randint(0, 30, n_samples),            # employment_years
            np.random.randint(18, 80, n_samples),           # age
            np.random.uniform(0, 0.5, n_samples),         # debt_to_income
            np.random.choice([0, 1], n_samples),           # has_collateral
            np.random.randint(0, 6, n_samples)             # loan_purpose
        ]).T
        
        return background_data
    
    def process_application(self, application: LoanApplication) -> Dict[str, Any]:
        """Procesar solicitud de préstamo"""
        start_time = datetime.now()
        
        try:
            # Convertir a diccionario
            app_data = application.dict()
            
            # Generar explicación
            explanation = self.explainer.explain_application(
                app_data, 
                application.application_id
            )
            
            if not explanation:
                raise HTTPException(status_code=500, detail="Error generando explicación")
            
            # Preparar respuesta
            response = {
                "application_id": application.application_id,
                "approved": explanation["prediction"]["approved"],
                "confidence": explanation["prediction"]["confidence"],
                "probability": explanation["prediction"]["probability"],
                "explanation": {
                    "top_features": explanation["top_features"],
                    "explanation_type": explanation["explanation_type"],
                    "compliance_info": explanation["compliance_info"]
                },
                "timestamp": datetime.now().isoformat(),
                "processing_time_ms": (datetime.now() - start_time).total_seconds() * 1000
            }
            
            # Enviar a Arize (asíncrono)
            if self.arize_connected:
                try:
                    self.arize_monitor.log_explanations([explanation])
                except Exception as e:
                    logger.warning(f"Error enviando a Arize: {e}")
            
            return response
            
        except Exception as e:
            logger.error(f"Error procesando aplicación {application.application_id}: {e}")
            raise HTTPException(status_code=500, detail=f"Error interno: {str(e)}")
    
    def get_compliance_info(self) -> ComplianceInfo:
        """Obtener información de cumplimiento"""
        return ComplianceInfo(
            explanation_available=self.explainer_ready,
            human_understandable=True,
            gdpr_compliant=True,
            audit_trail=True,
            regulatory_frameworks=["GDPR", "ECOA", "FCRA", "CCPA"]
        )
    
    def get_health_status(self) -> HealthResponse:
        """Obtener estado de salud de la API"""
        return HealthResponse(
            status="healthy" if self.model_loaded and self.explainer_ready else "degraded",
            model_loaded=self.model_loaded,
            explainer_ready=self.explainer_ready,
            arize_connected=self.arize_connected,
            timestamp=datetime.now().isoformat()
        )

# Crear instancia de la API
api_instance = LoanApprovalAPI()

# Crear aplicación FastAPI
app = FastAPI(
    title="Loan Approval API",
    description="API para aprobación de préstamos con explicaciones SHAP y cumplimiento regulatorio",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

@app.on_event("startup")
async def startup_event():
    """Evento de inicio de la aplicación"""
    logger.info("Iniciando API de aprobación de préstamos...")
    api_instance.initialize()

@app.post("/predict", response_model=PredictionResponse)
async def predict_loan_approval(application: LoanApplication):
    """
    Evaluar solicitud de préstamo y proporcionar explicación
    
    - **application_id**: ID único de la solicitud
    - **income**: Ingreso anual del solicitante
    - **credit_score**: Puntaje de crédito (300-850)
    - **loan_amount**: Monto del préstamo solicitado
    - **employment_years**: Años en el empleo actual
    - **age**: Edad del solicitante (18-100)
    - **debt_to_income**: Ratio deuda-ingreso (0-1)
    - **has_collateral**: Tiene garantía
    - **loan_purpose**: Propósito del préstamo (0-5)
    """
    logger.info(f"Procesando solicitud: {application.application_id}")
    
    if not api_instance.explainer_ready:
        raise HTTPException(
            status_code=503, 
            detail="Servicio no disponible: Explainer no inicializado"
        )
    
    result = api_instance.process_application(application)
    logger.info(f"Solicitud {application.application_id} procesada: {'APROBADA' if result['approved'] else 'RECHAZADA'}")
    
    return result

@app.get("/compliance", response_model=ComplianceInfo)
async def get_compliance_info():
    """
    Obtener información de cumplimiento regulatorio
    """
    return api_instance.get_compliance_info()

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Verificar estado de salud de la API
    """
    return api_instance.get_health_status()

@app.get("/model/info")
async def get_model_info():
    """
    Obtener información del modelo
    """
    return {
        "model_name": "Loan Approval Model",
        "version": "1.0.0",
        "type": "Neural Network",
        "features": [
            "income", "credit_score", "loan_amount", "employment_years",
            "age", "debt_to_income", "has_collateral", "loan_purpose"
        ],
        "explanation_method": "SHAP DeepExplainer",
        "compliance_standards": ["GDPR", "ECOA", "FCRA", "CCPA"],
        "last_updated": datetime.now().isoformat()
    }

@app.get("/applications/stats")
async def get_application_stats():
    """
    Obtener estadísticas de solicitudes procesadas
    """
    # En producción, esto consultaría una base de datos
    # Aquí simulamos estadísticas
    return {
        "total_applications": 1000,
        "approved_applications": 650,
        "rejected_applications": 350,
        "approval_rate": 0.65,
        "average_confidence": 0.82,
        "top_approval_factors": [
            {"factor": "credit_score", "importance": 0.35},
            {"factor": "income", "importance": 0.25},
            {"factor": "debt_to_income", "importance": 0.20}
        ],
        "compliance_rate": 0.98,
        "last_updated": datetime.now().isoformat()
    }

@app.get("/audit/logs")
async def get_audit_logs(limit: int = 50):
    """
    Obtener logs de auditoría recientes
    """
    # En producción, esto consultaría logs reales
    # Aquí simulamos logs de auditoría
    logs = []
    for i in range(min(limit, 10)):
        log = {
            "timestamp": datetime.now().isoformat(),
            "application_id": f"APP_{i:04d}",
            "decision": i % 3 != 0,  # 2/3 aprobadas
            "confidence": np.random.uniform(0.7, 0.95),
            "explanation_method": "SHAP DeepExplainer",
            "compliance_check": {
                "explanation_available": True,
                "human_understandable": True,
                "audit_trail": True
            }
        }
        logs.append(log)
    
    return {
        "logs": logs,
        "total_count": len(logs),
        "limit": limit,
        "generated_at": datetime.now().isoformat()
    }

@app.get("/explanations/sample")
async def get_sample_explanations():
    """
    Obtener ejemplos de explicaciones generadas
    """
    try:
        with open('outputs/explanations/loan_explanations.json', 'r') as f:
            explanations = json.load(f)
        
        # Devolver primeros 5 ejemplos
        sample_explanations = explanations[:5]
        
        return {
            "sample_explanations": sample_explanations,
            "total_available": len(explanations),
            "sample_size": len(sample_explanations),
            "generated_at": datetime.now().isoformat()
        }
        
    except FileNotFoundError:
        return {
            "sample_explanations": [],
            "message": "No hay explicaciones disponibles. Ejecute shap_explanations.py primero.",
            "generated_at": datetime.now().isoformat()
        }

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Manejador global de excepciones"""
    logger.error(f"Error no manejado: {exc}")
    return {
        "error": "Error interno del servidor",
        "detail": str(exc),
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    import uvicorn
    
    logger.info("Iniciando servidor de API de aprobación de préstamos...")
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
        reload=False
    )
