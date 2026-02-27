from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
import uvicorn
import logging
import time
import os
import io
import numpy as np
from PIL import Image
import tensorflow as tf
from typing import List, Optional, Dict, Any
from datetime import datetime
import redis
import json

from models.image_classifier import ImageClassifier
from utils.image_preprocessing import ImagePreprocessor
from utils.metrics import (
    REQUEST_COUNT, REQUEST_LATENCY, 
    PREDICTION_COUNT, MODEL_ACCURACY
)

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Inicialización de FastAPI
app = FastAPI(
    title="Image Classification API",
    description="API for real-time image classification using TensorFlow",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://your-react-app.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Servir archivos estáticos
app.mount("/static", StaticFiles(directory="static"), name="static")

# Inicializar componentes
image_classifier = ImageClassifier()
image_preprocessor = ImagePreprocessor()

# Conexión a Redis para caché
try:
    redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
    redis_client.ping()
    logger.info("Redis connection established")
except:
    redis_client = None
    logger.warning("Redis connection failed, running without cache")

# Dependencias
async def get_current_user():
    """Obtener usuario actual (simplificado para demo)"""
    return {"user_id": "demo_user", "role": "user"}

@app.on_event("startup")
async def startup_event():
    """Inicializar componentes al iniciar la aplicación"""
    logger.info("Starting Image Classification API...")
    
    # Cargar modelo
    await image_classifier.load_model()
    
    # Crear directorios necesarios
    os.makedirs("static/uploads", exist_ok=True)
    os.makedirs("static/results", exist_ok=True)
    
    logger.info("Image Classification API started successfully")

@app.on_event("shutdown")
async def shutdown_event():
    """Limpiar recursos al cerrar la aplicación"""
    logger.info("Shutting down Image Classification API...")

@app.get("/")
async def root():
    """Endpoint raíz"""
    return {
        "message": "Image Classification API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "model_loaded": image_classifier.is_loaded(),
        "redis_connected": redis_client is not None
    }

@app.post("/predict")
async def predict_image(
    file: UploadFile = File(...),
    confidence_threshold: float = 0.5,
    top_k: int = 5,
    current_user: Dict = Depends(get_current_user)
):
    """Predecir clase de una imagen"""
    start_time = time.time()
    
    try:
        # Validar archivo
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Leer imagen
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))
        
        # Preprocesar imagen
        processed_image = image_preprocessor.preprocess(image)
        
        # Realizar predicción
        prediction = await image_classifier.predict(
            processed_image,
            confidence_threshold=confidence_threshold,
            top_k=top_k
        )
        
        # Agregar metadata
        prediction.update({
            "filename": file.filename,
            "file_size": len(image_data),
            "processing_time_ms": (time.time() - start_time) * 1000,
            "timestamp": datetime.now().isoformat(),
            "user_id": current_user["user_id"]
        })
        
        # Guardar en caché si Redis está disponible
        if redis_client:
            cache_key = f"prediction:{hash(image_data)}"
            redis_client.setex(
                cache_key, 
                3600,  # 1 hora TTL
                json.dumps(prediction)
            )
        
        # Actualizar métricas
        PREDICTION_COUNT.labels(
            model="image_classifier",
            status="success"
        ).inc()
        
        return prediction
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        PREDICTION_COUNT.labels(
            model="image_classifier",
            status="error"
        ).inc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict-batch")
async def predict_batch(
    files: List[UploadFile] = File(...),
    confidence_threshold: float = 0.5,
    top_k: int = 5,
    current_user: Dict = Depends(get_current_user)
):
    """Predecir clases de múltiples imágenes"""
    start_time = time.time()
    
    try:
        if len(files) > 10:
            raise HTTPException(status_code=400, detail="Too many files (max 10)")
        
        results = []
        
        for i, file in enumerate(files):
            try:
                # Validar archivo
                if not file.content_type.startswith('image/'):
                    results.append({
                        "id": i,
                        "filename": file.filename,
                        "error": "File must be an image"
                    })
                    continue
                
                # Leer imagen
                image_data = await file.read()
                image = Image.open(io.BytesIO(image_data))
                
                # Preprocesar imagen
                processed_image = image_preprocessor.preprocess(image)
                
                # Realizar predicción
                prediction = await image_classifier.predict(
                    processed_image,
                    confidence_threshold=confidence_threshold,
                    top_k=top_k
                )
                
                # Agregar metadata
                prediction.update({
                    "id": i,
                    "filename": file.filename,
                    "file_size": len(image_data)
                })
                
                results.append(prediction)
                
            except Exception as e:
                logger.error(f"Error processing file {file.filename}: {e}")
                results.append({
                    "id": i,
                    "filename": file.filename,
                    "error": str(e)
                })
        
        response = {
            "results": results,
            "total_files": len(files),
            "successful_predictions": len([r for r in results if 'error' not in r]),
            "processing_time_ms": (time.time() - start_time) * 1000,
            "timestamp": datetime.now().isoformat(),
            "user_id": current_user["user_id"]
        }
        
        return response
        
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/model/info")
async def get_model_info():
    """Obtener información del modelo"""
    try:
        info = await image_classifier.get_model_info()
        return info
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/model/classes")
async def get_model_classes():
    """Obtener lista de clases del modelo"""
    try:
        classes = await image_classifier.get_classes()
        return {"classes": classes}
    except Exception as e:
        logger.error(f"Error getting model classes: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/model/evaluate")
async def evaluate_model(
    test_data_path: str,
    current_user: Dict = Depends(get_current_user)
):
    """Evaluar modelo con datos de prueba"""
    try:
        # Validar que el usuario tenga permisos
        if current_user.get("role") != "admin":
            raise HTTPException(status_code=403, detail="Admin access required")
        
        # Evaluar modelo
        results = await image_classifier.evaluate_model(test_data_path)
        
        # Actualizar métricas
        if "accuracy" in results:
            MODEL_ACCURACY.set(results["accuracy"])
        
        return results
        
    except Exception as e:
        logger.error(f"Model evaluation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/predictions/history")
async def get_prediction_history(
    limit: int = 50,
    offset: int = 0,
    current_user: Dict = Depends(get_current_user)
):
    """Obtener historial de predicciones del usuario"""
    try:
        # En una implementación real, esto vendría de una base de datos
        # Por ahora, simulamos algunos datos
        history = [
            {
                "id": i,
                "filename": f"image_{i}.jpg",
                "prediction": f"class_{i % 10}",
                "confidence": 0.8 + (i % 20) * 0.01,
                "timestamp": datetime.now().isoformat()
            }
            for i in range(offset, min(offset + limit, 100))
        ]
        
        return {
            "history": history,
            "total": 100,
            "limit": limit,
            "offset": offset
        }
        
    except Exception as e:
        logger.error(f"Error getting prediction history: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/predictions/{prediction_id}")
async def delete_prediction(
    prediction_id: int,
    current_user: Dict = Depends(get_current_user)
):
    """Eliminar una predicción del historial"""
    try:
        # En una implementación real, esto eliminaría de la base de datos
        # Por ahora, simulamos la eliminación
        logger.info(f"User {current_user['user_id']} deleted prediction {prediction_id}")
        
        return {"message": "Prediction deleted successfully"}
        
    except Exception as e:
        logger.error(f"Error deleting prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics")
async def get_metrics():
    """Endpoint para métricas de Prometheus"""
    from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
    return generate_latest(), 200, {'Content-Type': CONTENT_TYPE_LATEST}

@app.get("/stats")
async def get_stats():
    """Obtener estadísticas del sistema"""
    try:
        stats = {
            "model_loaded": image_classifier.is_loaded(),
            "redis_connected": redis_client is not None,
            "total_predictions": PREDICTION_COUNT._value._value.get("image_classifier", {}).get("success", 0),
            "error_predictions": PREDICTION_COUNT._value._value.get("image_classifier", {}).get("error", 0),
            "average_latency": REQUEST_LATENCY._sum._value.get() / max(REQUEST_LATENCY._count._value.get(), 1),
            "uptime": datetime.now().isoformat()
        }
        
        return stats
        
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Middleware para métricas
@app.middleware("http")
async def add_metrics_middleware(request, call_next):
    """Middleware para recolectar métricas"""
    start_time = time.time()
    
    response = await call_next(request)
    
    # Actualizar métricas
    REQUEST_LATENCY.observe(time.time() - start_time)
    REQUEST_COUNT.labels(
        method=request.method,
        endpoint=request.url.path,
        status_code=response.status_code
    ).inc()
    
    return response

# Manejador de errores
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Manejar excepciones HTTP"""
    logger.error(f"HTTP exception: {exc.status_code} - {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail, "status_code": exc.status_code}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Manejar excepciones generales"""
    logger.error(f"General exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "status_code": 500}
    )

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
