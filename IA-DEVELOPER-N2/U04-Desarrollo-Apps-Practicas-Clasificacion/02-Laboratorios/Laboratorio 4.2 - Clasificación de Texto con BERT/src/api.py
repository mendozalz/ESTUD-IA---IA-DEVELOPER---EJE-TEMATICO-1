"""
API REST para Clasificación de Texto con BERT y Adaptadores
Laboratorio 4.2 - Clasificación de Texto con BERT y Adaptadores
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import torch
import numpy as np
from transformers import BertTokenizer
from bert_adapters import BERTWithAdapters
from explain_model import BERTExplainer
import logging
import json
from datetime import datetime
import os

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Modelos Pydantic
class TextInput(BaseModel):
    text: str
    threshold: Optional[float] = 0.5

class BatchTextInput(BaseModel):
    texts: List[str]
    threshold: Optional[float] = 0.5

class PredictionResponse(BaseModel):
    text: str
    prediction: str
    confidence: float
    probabilities: Dict[str, float]
    timestamp: str

class BatchPredictionResponse(BaseModel):
    total_texts: int
    successful_predictions: int
    results: List[Dict[str, Any]]
    timestamp: str

class ExplanationResponse(BaseModel):
    text: str
    prediction: Dict[str, Any]
    explanation: Dict[str, Any]
    timestamp: str

# Inicializar FastAPI
app = FastAPI(
    title="BERT Text Classification API",
    description="API para clasificación de texto usando BERT con adaptadores y explicabilidad SHAP",
    version="1.0.0"
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Variables globales
model = None
tokenizer = None
explainer = None
class_names = ['Negative', 'Positive']
model_path = 'models/best_bert_adapters.pth'

class TextClassificationService:
    """
    Servicio para clasificación de texto
    """
    
    def __init__(self, model_path: str, class_names: List[str]):
        self.model_path = model_path
        self.class_names = class_names
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Cargar modelo y tokenizer
        self.load_model()
        self.setup_explainer()
    
    def load_model(self):
        """
        Carga el modelo BERT con adaptadores
        """
        global model, tokenizer
        
        try:
            if os.path.exists(self.model_path):
                model = BERTWithAdapters.load_model(self.model_path)
                tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
                model.to(self.device)
                model.eval()
                logger.info(f"✅ Modelo cargado desde {self.model_path}")
            else:
                logger.error(f"❌ Modelo no encontrado en {self.model_path}")
                raise FileNotFoundError(f"Modelo no encontrado: {self.model_path}")
        except Exception as e:
            logger.error(f"❌ Error cargando modelo: {str(e)}")
            raise
    
    def setup_explainer(self):
        """
        Configura el explicador SHAP
        """
        global explainer
        
        try:
            # Textos de fondo para SHAP
            background_texts = [
                "This product is amazing and works perfectly",
                "Terrible experience, would not recommend",
                "Average quality, nothing special",
                "Good value for money",
                "Poor customer service"
            ]
            
            explainer = BERTExplainer(model, tokenizer, self.device)
            explainer.setup_explainer(background_texts)
            logger.info("✅ Explainer SHAP configurado")
        except Exception as e:
            logger.error(f"❌ Error configurando explainer: {str(e)}")
            explainer = None
    
    def predict_text(self, text: str, threshold: float = 0.5) -> Dict[str, Any]:
        """
        Predice la clase de un texto
        """
        try:
            # Tokenizar
            inputs = tokenizer(
                text,
                max_length=128,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Predicción
            with torch.no_grad():
                outputs = model(**inputs)
                probabilities = torch.softmax(outputs, dim=-1)
                confidence, predicted_class = torch.max(probabilities, dim=-1)
            
            # Aplicar threshold
            if confidence.item() < threshold:
                predicted_class = 0  # Default a clase negativa
            
            result = {
                'prediction': self.class_names[predicted_class.item()],
                'confidence': confidence.item(),
                'probabilities': {
                    self.class_names[i]: prob.item() 
                    for i, prob in enumerate(probabilities[0])
                },
                'predicted_class_id': predicted_class.item(),
                'threshold_applied': threshold
            }
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Error en predicción: {str(e)}")
            raise
    
    def explain_text(self, text: str, threshold: float = 0.5) -> Dict[str, Any]:
        """
        Explica una predicción
        """
        if explainer is None:
            raise HTTPException(status_code=503, detail="Explainer no disponible")
        
        try:
            # Predicción
            prediction = self.predict_text(text, threshold)
            
            # Explicación SHAP
            explanation = explainer.explain_text(text, self.class_names)
            
            # Procesar explicación para respuesta
            token_importance = explanation['token_importance'][:20]  # Top 20 tokens
            
            result = {
                'prediction': prediction,
                'explanation': {
                    'tokens': [token for token, _ in token_importance],
                    'shap_values': [value for _, value in token_importance],
                    'top_positive_tokens': [
                        {'token': token, 'impact': value}
                        for token, value in token_importance if value > 0
                    ][:5],
                    'top_negative_tokens': [
                        {'token': token, 'impact': value}
                        for token, value in token_importance if value < 0
                    ][:5]
                }
            }
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Error en explicación: {str(e)}")
            raise

# Inicializar servicio
try:
    service = TextClassificationService(model_path, class_names)
except Exception as e:
    logger.error(f"❌ Error inicializando servicio: {str(e)}")
    service = None

@app.get("/")
async def root():
    """
    Endpoint principal
    """
    return {
        "message": "BERT Text Classification API",
        "version": "1.0.0",
        "status": "active" if service is not None else "error",
        "classes": class_names,
        "device": service.device if service else None
    }

@app.get("/health")
async def health_check():
    """
    Endpoint de health check
    """
    return {
        "status": "healthy" if service is not None else "unhealthy",
        "model_loaded": service is not None,
        "explainer_available": explainer is not None,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict_text(input_data: TextInput):
    """
    Predice la clase de un texto
    """
    if service is None:
        raise HTTPException(status_code=503, detail="Servicio no disponible")
    
    try:
        result = service.predict_text(input_data.text, input_data.threshold)
        
        response = PredictionResponse(
            text=input_data.text,
            prediction=result['prediction'],
            confidence=result['confidence'],
            probabilities=result['probabilities'],
            timestamp=datetime.now().isoformat()
        )
        
        logger.info(f"✅ Predicción exitosa: {result['prediction']} (confianza: {result['confidence']:.3f})")
        
        return response
        
    except Exception as e:
        logger.error(f"❌ Error en predicción: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error en predicción: {str(e)}")

@app.post("/explain", response_model=ExplanationResponse)
async def explain_text(input_data: TextInput):
    """
    Predice y explica la clasificación de un texto
    """
    if service is None:
        raise HTTPException(status_code=503, detail="Servicio no disponible")
    
    try:
        result = service.explain_text(input_data.text, input_data.threshold)
        
        response = ExplanationResponse(
            text=input_data.text,
            prediction=result['prediction'],
            explanation=result['explanation'],
            timestamp=datetime.now().isoformat()
        )
        
        logger.info(f"✅ Explicación generada para: {input_data.text[:50]}...")
        
        return response
        
    except Exception as e:
        logger.error(f"❌ Error en explicación: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error en explicación: {str(e)}")

@app.post("/batch_predict", response_model=BatchPredictionResponse)
async def batch_predict(input_data: BatchTextInput):
    """
    Predice múltiples textos en batch
    """
    if service is None:
        raise HTTPException(status_code=503, detail="Servicio no disponible")
    
    results = []
    successful = 0
    
    for text in input_data.texts:
        try:
            result = service.predict_text(text, input_data.threshold)
            result['text'] = text
            result['success'] = True
            results.append(result)
            successful += 1
        except Exception as e:
            results.append({
                'text': text,
                'success': False,
                'error': str(e)
            })
    
    response = BatchPredictionResponse(
        total_texts=len(input_data.texts),
        successful_predictions=successful,
        results=results,
        timestamp=datetime.now().isoformat()
    )
    
    logger.info(f"✅ Batch prediction completado: {successful}/{len(input_data.texts)} exitosas")
    
    return response

@app.get("/model/info")
async def model_info():
    """
    Retorna información del modelo
    """
    if service is None:
        raise HTTPException(status_code=503, detail="Servicio no disponible")
    
    try:
        # Obtener parámetros del modelo
        trainable_params, total_params = model.get_trainable_parameters()
        
        model_info = {
            'model_path': model_path,
            'model_name': model.model_name,
            'num_classes': model.num_classes,
            'class_names': class_names,
            'trainable_parameters': trainable_params,
            'total_parameters': total_params,
            'parameter_efficiency': (trainable_params / total_params) * 100,
            'device': service.device,
            'explainer_available': explainer is not None
        }
        
        return model_info
        
    except Exception as e:
        logger.error(f"❌ Error obteniendo info del modelo: {str(e)}")
        raise HTTPException(status_code=500, detail="Error obteniendo información del modelo")

@app.get("/classes")
async def get_classes():
    """
    Retorna las clases disponibles
    """
    return {
        "classes": class_names,
        "num_classes": len(class_names)
    }

if __name__ == "__main__":
    import uvicorn
    
    # Configuración del servidor
    host = "0.0.0.0"
    port = 8000
    
    logger.info(f"🚀 Iniciando servidor BERT Text Classification API")
    logger.info(f"📍 http://{host}:{port}")
    logger.info(f"📖 Docs: http://{host}:{port}/docs")
    
    uvicorn.run(app, host=host, port=port)
