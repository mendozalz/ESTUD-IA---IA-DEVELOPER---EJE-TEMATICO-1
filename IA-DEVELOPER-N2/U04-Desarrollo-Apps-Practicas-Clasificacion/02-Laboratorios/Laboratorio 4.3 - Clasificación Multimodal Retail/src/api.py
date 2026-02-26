"""
API REST para Clasificación Multimodal Retail
Laboratorio 4.3 - Clasificación Multimodal Retail
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import torch
import numpy as np
import cv2
import json
import os
import logging
from datetime import datetime
import redis
import hashlib

from multimodal_model import MultimodalRetailClassifier
from data_preprocessing import MultimodalDataPreprocessor

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Modelos Pydantic
class TextInput(BaseModel):
    text: str

class MultimodalInput(BaseModel):
    text: str
    image_path: Optional[str] = None

class BatchMultimodalInput(BaseModel):
    items: List[MultimodalInput]

class PredictionResponse(BaseModel):
    text: str
    prediction: str
    confidence: float
    probabilities: Dict[str, float]
    timestamp: str

class BatchPredictionResponse(BaseModel):
    total_items: int
    successful_predictions: int
    results: List[Dict[str, Any]]
    timestamp: str

# Inicializar FastAPI
app = FastAPI(
    title="Multimodal Retail Classification API",
    description="API para clasificación de productos usando imágenes y texto",
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
preprocessor = None
class_names = None
redis_client = None
model_path = 'models/best_multimodal_model.pth'
preprocessor_path = 'models/preprocessor_state.json'
class_names_path = 'models/class_names.json'

class MultimodalClassificationService:
    """
    Servicio para clasificación multimodal
    """
    
    def __init__(self, model_path: str, preprocessor_path: str, class_names_path: str):
        self.model_path = model_path
        self.preprocessor_path = preprocessor_path
        self.class_names_path = class_names_path
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Cargar componentes
        self.load_model()
        self.load_preprocessor()
        self.load_class_names()
        
        # Inicializar Redis para caché
        self.init_redis()
        
        logger.info("Servicio de clasificación multimodal inicializado")
    
    def load_model(self):
        """
        Carga el modelo multimodal
        """
        global model
        
        try:
            if os.path.exists(self.model_path):
                model = MultimodalRetailClassifier.load_model(self.model_path)
                model.to(self.device)
                model.eval()
                logger.info(f"✅ Modelo cargado desde {self.model_path}")
            else:
                raise FileNotFoundError(f"Modelo no encontrado: {self.model_path}")
        except Exception as e:
            logger.error(f"❌ Error cargando modelo: {str(e)}")
            raise
    
    def load_preprocessor(self):
        """
        Carga el preprocesador
        """
        global preprocessor
        
        try:
            preprocessor = MultimodalDataPreprocessor()
            if os.path.exists(self.preprocessor_path):
                preprocessor.load_preprocessor(self.preprocessor_path)
                logger.info(f"✅ Preprocesador cargado desde {self.preprocessor_path}")
            else:
                logger.warning("⚠️ Preprocesador no encontrado, usando configuración por defecto")
        except Exception as e:
            logger.error(f"❌ Error cargando preprocesador: {str(e)}")
            raise
    
    def load_class_names(self):
        """
        Carga los nombres de las clases
        """
        global class_names
        
        try:
            if os.path.exists(self.class_names_path):
                with open(self.class_names_path, 'r') as f:
                    class_names = json.load(f)
                logger.info(f"✅ Nombres de clases cargados: {class_names}")
            else:
                class_names = [f"Class_{i}" for i in range(10)]
                logger.warning("⚠️ Usando nombres de clases por defecto")
        except Exception as e:
            logger.error(f"❌ Error cargando nombres de clases: {str(e)}")
            raise
    
    def init_redis(self):
        """
        Inicializa cliente Redis para caché
        """
        global redis_client
        
        try:
            redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
            redis_client.ping()
            logger.info("✅ Redis conectado para caché")
        except Exception as e:
            logger.warning(f"⚠️ No se pudo conectar a Redis: {e}")
            redis_client = None
    
    def get_cache_key(self, text: str, image_hash: str) -> str:
        """
        Genera clave de caché
        """
        content = f"{text}_{image_hash}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def get_image_hash(self, image_path: str) -> str:
        """
        Calcula hash de una imagen
        """
        try:
            with open(image_path, 'rb') as f:
                return hashlib.md5(f.read()).hexdigest()
        except:
            return "unknown"
    
    def predict_multimodal(self, text: str, image_path: str = None, 
                          use_cache: bool = True) -> Dict[str, Any]:
        """
        Realiza predicción multimodal
        """
        try:
            # Verificar caché
            if use_cache and redis_client and image_path:
                image_hash = self.get_image_hash(image_path)
                cache_key = self.get_cache_key(text, image_hash)
                cached_result = redis_client.get(cache_key)
                
                if cached_result:
                    logger.info(f"Resultado obtenido desde caché: {cache_key}")
                    return json.loads(cached_result)
            
            # Preprocesar datos
            if image_path and os.path.exists(image_path):
                # Predicción multimodal completa
                processed = preprocessor.preprocess_single_sample(image_path, text)
                
                with torch.no_grad():
                    image_tensor = processed['image'].unsqueeze(0).to(self.device)
                    input_ids = processed['input_ids'].unsqueeze(0).to(self.device)
                    attention_mask = processed['attention_mask'].unsqueeze(0).to(self.device)
                    
                    outputs = model(image_tensor, input_ids, attention_mask)
                    probabilities = torch.softmax(outputs, dim=-1)
                    confidence, predicted_class = torch.max(probabilities, dim=-1)
            else:
                # Solo texto (imagen por defecto)
                dummy_image = torch.zeros(1, 3, 300, 300).to(self.device)
                processed = preprocessor.preprocess_single_sample("", text)
                
                with torch.no_grad():
                    input_ids = processed['input_ids'].unsqueeze(0).to(self.device)
                    attention_mask = processed['attention_mask'].unsqueeze(0).to(self.device)
                    
                    outputs = model(dummy_image, input_ids, attention_mask)
                    probabilities = torch.softmax(outputs, dim=-1)
                    confidence, predicted_class = torch.max(probabilities, dim=-1)
            
            # Construir resultado
            result = {
                'prediction': class_names[predicted_class.item()],
                'confidence': confidence.item(),
                'probabilities': {
                    class_names[i]: prob.item() 
                    for i, prob in enumerate(probabilities[0])
                },
                'predicted_class_id': predicted_class.item(),
                'input_type': 'multimodal' if image_path else 'text_only'
            }
            
            # Guardar en caché
            if use_cache and redis_client and image_path:
                image_hash = self.get_image_hash(image_path)
                cache_key = self.get_cache_key(text, image_hash)
                redis_client.setex(cache_key, 3600, json.dumps(result))  # 1 hora TTL
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Error en predicción multimodal: {str(e)}")
            raise
    
    def predict_batch(self, items: List[MultimodalInput]) -> List[Dict[str, Any]]:
        """
        Realiza predicciones en batch
        """
        results = []
        
        for item in items:
            try:
                result = self.predict_multimodal(item.text, item.image_path)
                result['text'] = item.text
                result['image_path'] = item.image_path
                result['success'] = True
                results.append(result)
            except Exception as e:
                results.append({
                    'text': item.text,
                    'image_path': item.image_path,
                    'success': False,
                    'error': str(e)
                })
        
        return results

# Inicializar servicio
try:
    service = MultimodalClassificationService(model_path, preprocessor_path, class_names_path)
except Exception as e:
    logger.error(f"❌ Error inicializando servicio: {str(e)}")
    service = None

@app.get("/")
async def root():
    """
    Endpoint principal
    """
    return {
        "message": "Multimodal Retail Classification API",
        "version": "1.0.0",
        "status": "active" if service is not None else "error",
        "classes": class_names,
        "device": service.device if service else None,
        "cache_enabled": redis_client is not None
    }

@app.get("/health")
async def health_check():
    """
    Endpoint de health check
    """
    return {
        "status": "healthy" if service is not None else "unhealthy",
        "model_loaded": model is not None,
        "preprocessor_loaded": preprocessor is not None,
        "redis_connected": redis_client is not None,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/predict/text", response_model=PredictionResponse)
async def predict_text_only(input_data: TextInput):
    """
    Predice usando solo texto
    """
    if service is None:
        raise HTTPException(status_code=503, detail="Servicio no disponible")
    
    try:
        result = service.predict_multimodal(input_data.text, image_path=None)
        
        response = PredictionResponse(
            text=input_data.text,
            prediction=result['prediction'],
            confidence=result['confidence'],
            probabilities=result['probabilities'],
            timestamp=datetime.now().isoformat()
        )
        
        logger.info(f"✅ Predicción texto-only: {result['prediction']} (conf: {result['confidence']:.3f})")
        
        return response
        
    except Exception as e:
        logger.error(f"❌ Error en predicción texto: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error en predicción: {str(e)}")

@app.post("/predict/multimodal", response_model=PredictionResponse)
async def predict_multimodal(image: UploadFile = File(...), text: str = ""):
    """
    Predice usando imagen y texto
    """
    if service is None:
        raise HTTPException(status_code=503, detail="Servicio no disponible")
    
    try:
        # Guardar imagen temporalmente
        temp_image_path = f"temp_{datetime.now().timestamp()}.jpg"
        
        with open(temp_image_path, "wb") as buffer:
            content = await image.read()
            buffer.write(content)
        
        # Realizar predicción
        result = service.predict_multimodal(text, temp_image_path)
        
        # Limpiar archivo temporal
        os.remove(temp_image_path)
        
        response = PredictionResponse(
            text=text,
            prediction=result['prediction'],
            confidence=result['confidence'],
            probabilities=result['probabilities'],
            timestamp=datetime.now().isoformat()
        )
        
        logger.info(f"✅ Predicción multimodal: {result['prediction']} (conf: {result['confidence']:.3f})")
        
        return response
        
    except Exception as e:
        # Limpiar archivo temporal en caso de error
        if 'temp_image_path' in locals():
            try:
                os.remove(temp_image_path)
            except:
                pass
        
        logger.error(f"❌ Error en predicción multimodal: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error en predicción: {str(e)}")

@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(input_data: BatchMultimodalInput):
    """
    Realiza predicciones en batch
    """
    if service is None:
        raise HTTPException(status_code=503, detail="Servicio no disponible")
    
    try:
        results = service.predict_batch(input_data.items)
        successful = sum(1 for r in results if r['success'])
        
        response = BatchPredictionResponse(
            total_items=len(input_data.items),
            successful_predictions=successful,
            results=results,
            timestamp=datetime.now().isoformat()
        )
        
        logger.info(f"✅ Batch prediction: {successful}/{len(input_data.items)} exitosas")
        
        return response
        
    except Exception as e:
        logger.error(f"❌ Error en batch prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error en predicción batch: {str(e)}")

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
            'num_classes': len(class_names),
            'class_names': class_names,
            'trainable_parameters': trainable_params,
            'total_parameters': total_params,
            'parameter_efficiency': (trainable_params / total_params) * 100,
            'device': service.device,
            'image_model': model.image_encoder.__class__.__name__,
            'text_model': 'DistilBERT',
            'cache_enabled': redis_client is not None
        }
        
        return model_info
        
    except Exception as e:
        logger.error(f"❌ Error obteniendo info del modelo: {str(e)}")
        raise HTTPException(status_code=500, detail="Error obteniendo información del modelo")

@app.get("/cache/stats")
async def cache_stats():
    """
    Retorna estadísticas de caché
    """
    if redis_client is None:
        return {"cache_enabled": False}
    
    try:
        info = redis_client.info()
        return {
            "cache_enabled": True,
            "connected_clients": info.get('connected_clients', 0),
            "used_memory": info.get('used_memory_human', 'N/A'),
            "keyspace_hits": info.get('keyspace_hits', 0),
            "keyspace_misses": info.get('keyspace_misses', 0),
            "hit_rate": info.get('keyspace_hits', 0) / max(1, info.get('keyspace_hits', 0) + info.get('keyspace_misses', 0))
        }
    except Exception as e:
        logger.error(f"❌ Error obteniendo stats de caché: {str(e)}")
        return {"cache_enabled": True, "error": str(e)}

@app.delete("/cache/clear")
async def clear_cache():
    """
    Limpia la caché
    """
    if redis_client is None:
        return {"cache_enabled": False, "message": "Caché no disponible"}
    
    try:
        redis_client.flushdb()
        logger.info("🧹 Caché limpiada")
        return {"cache_enabled": True, "message": "Caché limpiada exitosamente"}
    except Exception as e:
        logger.error(f"❌ Error limpiando caché: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error limpiando caché: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    
    # Configuración del servidor
    host = "0.0.0.0"
    port = 8000
    
    logger.info(f"🚀 Iniciando servidor Multimodal Retail Classification API")
    logger.info(f"📍 http://{host}:{port}")
    logger.info(f"📖 Docs: http://{host}:{port}/docs")
    
    uvicorn.run(app, host=host, port=port)
