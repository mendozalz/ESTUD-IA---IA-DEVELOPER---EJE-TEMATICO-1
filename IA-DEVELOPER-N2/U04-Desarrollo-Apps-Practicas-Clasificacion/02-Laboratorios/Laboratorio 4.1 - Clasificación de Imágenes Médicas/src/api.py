"""
API REST para Clasificación de Imágenes Médicas
Laboratorio 4.1 - Clasificación de Imágenes Médicas con EfficientNetV3 y ViT
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import os
import numpy as np
import cv2
import tensorflow as tf
from PIL import Image
import io
import base64
import logging
from typing import Optional
import json
from datetime import datetime

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Inicializar FastAPI
app = FastAPI(
    title="Medical Image Classification API",
    description="API para clasificación de imágenes médicas usando EfficientNetV3 + ViT",
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
class_names = ['Normal', 'Pneumonia']
model_path = 'models/best_model.h5'

class MedicalImageClassifier:
    """
    Clase para manejar la clasificación de imágenes médicas
    """
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None
        self.load_model()
    
    def load_model(self):
        """
        Carga el modelo pre-entrenado
        """
        try:
            if os.path.exists(self.model_path):
                self.model = tf.keras.models.load_model(self.model_path)
                logger.info(f"✅ Modelo cargado desde {self.model_path}")
            else:
                logger.error(f"❌ Modelo no encontrado en {self.model_path}")
                raise FileNotFoundError(f"Modelo no encontrado: {self.model_path}")
        except Exception as e:
            logger.error(f"❌ Error cargando modelo: {str(e)}")
            raise
    
    def preprocess_image(self, image_bytes: bytes) -> np.ndarray:
        """
        Preprocesa la imagen para predicción
        """
        try:
            # Convertir bytes a imagen PIL
            image = Image.open(io.BytesIO(image_bytes))
            image = image.convert('RGB')
            
            # Redimensionar a 224x224
            image = image.resize((224, 224))
            
            # Convertir a array numpy y normalizar
            image_array = np.array(image) / 255.0
            
            # Añadir dimensión batch
            image_array = np.expand_dims(image_array, axis=0)
            
            return image_array
            
        except Exception as e:
            logger.error(f"❌ Error preprocesando imagen: {str(e)}")
            raise
    
    def predict(self, image_array: np.ndarray, threshold: float = 0.5) -> dict:
        """
        Realiza la predicción de la imagen médica
        """
        try:
            # Realizar predicción
            prediction = self.model.predict(image_array)[0][0]
            confidence = float(prediction)
            
            # Determinar clase basado en threshold
            class_idx = 1 if confidence > threshold else 0
            class_name = class_names[class_idx]
            
            # Calcular confianza adicional
            if confidence > threshold:
                adjusted_confidence = confidence
            else:
                adjusted_confidence = 1 - confidence
            
            return {
                'prediction': class_name,
                'confidence': round(adjusted_confidence, 4),
                'raw_score': round(confidence, 4),
                'threshold': threshold,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Error en predicción: {str(e)}")
            raise

# Inicializar clasificador
try:
    classifier = MedicalImageClassifier(model_path)
except Exception as e:
    logger.error(f"❌ Error inicializando clasificador: {str(e)}")
    classifier = None

@app.get("/")
async def root():
    """
    Endpoint principal
    """
    return {
        "message": "Medical Image Classification API",
        "version": "1.0.0",
        "status": "active" if classifier is not None else "error",
        "classes": class_names
    }

@app.get("/health")
async def health_check():
    """
    Endpoint de health check
    """
    return {
        "status": "healthy" if classifier is not None else "unhealthy",
        "model_loaded": classifier is not None,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/predict")
async def predict_image(
    file: UploadFile = File(...),
    threshold: Optional[float] = 0.5
):
    """
    Predice la clase de una imagen médica
    """
    if classifier is None:
        raise HTTPException(status_code=503, detail="Modelo no disponible")
    
    # Validar tipo de archivo
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="El archivo debe ser una imagen")
    
    try:
        # Leer archivo
        image_bytes = await file.read()
        
        # Preprocesar imagen
        image_array = classifier.preprocess_image(image_bytes)
        
        # Realizar predicción
        result = classifier.predict(image_array, threshold)
        
        # Añadir metadatos del archivo
        result['file_info'] = {
            'filename': file.filename,
            'content_type': file.content_type,
            'size': len(image_bytes)
        }
        
        logger.info(f"✅ Predicción exitosa: {result['prediction']} (confianza: {result['confidence']})")
        
        return result
        
    except Exception as e:
        logger.error(f"❌ Error en predicción: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error procesando imagen: {str(e)}")

@app.post("/predict_base64")
async def predict_image_base64(
    image_data: dict,
    threshold: Optional[float] = 0.5
):
    """
    Predice la clase de una imagen médica desde base64
    """
    if classifier is None:
        raise HTTPException(status_code=503, detail="Modelo no disponible")
    
    try:
        # Decodificar imagen base64
        image_bytes = base64.b64decode(image_data['image'])
        
        # Preprocesar imagen
        image_array = classifier.preprocess_image(image_bytes)
        
        # Realizar predicción
        result = classifier.predict(image_array, threshold)
        
        logger.info(f"✅ Predicción exitosa desde base64: {result['prediction']}")
        
        return result
        
    except Exception as e:
        logger.error(f"❌ Error en predicción base64: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error procesando imagen: {str(e)}")

@app.get("/model/info")
async def model_info():
    """
    Retorna información del modelo
    """
    if classifier is None:
        raise HTTPException(status_code=503, detail="Modelo no disponible")
    
    try:
        model_info = {
            'model_path': model_path,
            'input_shape': classifier.model.input_shape,
            'output_shape': classifier.model.output_shape,
            'classes': class_names,
            'parameters': classifier.model.count_params(),
            'model_loaded': True
        }
        
        return model_info
        
    except Exception as e:
        logger.error(f"❌ Error obteniendo info del modelo: {str(e)}")
        raise HTTPException(status_code=500, detail="Error obteniendo información del modelo")

@app.post("/batch_predict")
async def batch_predict(
    background_tasks: BackgroundTasks,
    files: list[UploadFile] = File(...),
    threshold: Optional[float] = 0.5
):
    """
    Predice múltiples imágenes en batch
    """
    if classifier is None:
        raise HTTPException(status_code=503, detail="Modelo no disponible")
    
    results = []
    
    for file in files:
        if not file.content_type.startswith('image/'):
            results.append({
                'filename': file.filename,
                'error': 'El archivo debe ser una imagen'
            })
            continue
        
        try:
            # Leer archivo
            image_bytes = await file.read()
            
            # Preprocesar imagen
            image_array = classifier.preprocess_image(image_bytes)
            
            # Realizar predicción
            result = classifier.predict(image_array, threshold)
            result['filename'] = file.filename
            
            results.append(result)
            
        except Exception as e:
            results.append({
                'filename': file.filename,
                'error': str(e)
            })
    
    # Log de resultados
    successful_predictions = sum(1 for r in results if 'error' not in r)
    logger.info(f"✅ Batch prediction completado: {successful_predictions}/{len(files)} exitosas")
    
    return {
        'total_files': len(files),
        'successful_predictions': successful_predictions,
        'results': results,
        'timestamp': datetime.now().isoformat()
    }

if __name__ == "__main__":
    # Configuración del servidor
    host = "0.0.0.0"
    port = 8000
    
    logger.info(f"🚀 Iniciando servidor Medical Image Classification API")
    logger.info(f"📍 http://{host}:{port}")
    logger.info(f"📖 Docs: http://{host}:{port}/docs")
    
    uvicorn.run(app, host=host, port=port)
