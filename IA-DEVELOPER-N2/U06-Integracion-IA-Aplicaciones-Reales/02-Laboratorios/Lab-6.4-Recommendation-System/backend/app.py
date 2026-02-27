from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer
from contextlib import asynccontextmanager
import uvicorn
import logging
from datetime import datetime
from typing import List, Optional, Dict, Any

from models.model_manager import ModelManager
from services.recommendation_service import RecommendationService
from services.user_service import UserService
from services.product_service import ProductService
from services.ab_testing_service import ABTestingService
from streaming.kafka_consumer import KafkaConsumerManager
from utils.evaluation_metrics import EvaluationMetrics

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Security
security = HTTPBearer()

# Gestión del ciclo de vida
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gestionar el ciclo de vida de la aplicación"""
    # Startup
    logger.info("Starting Recommendation System...")
    
    # Inicializar componentes
    app.state.model_manager = ModelManager()
    app.state.recommendation_service = RecommendationService()
    app.state.user_service = UserService()
    app.state.product_service = ProductService()
    app.state.ab_testing_service = ABTestingService()
    app.state.kafka_consumer = KafkaConsumerManager()
    
    # Cargar modelos
    await app.state.model_manager.load_all_models()
    
    # Iniciar consumidor Kafka
    await app.state.kafka_consumer.start()
    
    logger.info("Recommendation System started successfully")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Recommendation System...")
    await app.state.kafka_consumer.stop()
    logger.info("Recommendation System shutdown complete")

# Inicialización de FastAPI
app = FastAPI(
    title="E-commerce Recommendation System",
    description="Advanced recommendation system with real-time processing and A/B testing",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://admin.yourdomain.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Dependencias
async def get_current_user(token: str = Depends(security)):
    """Obtener usuario actual desde token"""
    # Implementar validación JWT
    return {"user_id": "demo_user", "role": "admin"}

# Endpoints de recomendaciones
@app.get("/recommendations/{user_id}")
async def get_recommendations(
    user_id: str,
    limit: int = 10,
    category: Optional[str] = None,
    context: Optional[Dict[str, Any]] = None,
    current_user: Dict = Depends(get_current_user)
):
    """Obtener recomendaciones para un usuario"""
    try:
        # Determinar si el usuario está en un test A/B
        ab_test = await app.state.ab_testing_service.get_user_test(user_id)
        
        # Obtener recomendaciones según el test
        if ab_test:
            recommendations = await app.state.recommendation_service.get_recommendations(
                user_id=user_id,
                limit=limit,
                category=category,
                model_version=ab_test['model_version'],
                context=context
            )
            
            # Registrar evento de A/B testing
            await app.state.ab_testing_service.record_impression(
                test_id=ab_test['test_id'],
                user_id=user_id,
                recommendations=recommendations
            )
        else:
            recommendations = await app.state.recommendation_service.get_recommendations(
                user_id=user_id,
                limit=limit,
                category=category,
                context=context
            )
        
        return {
            "user_id": user_id,
            "recommendations": recommendations,
            "ab_test": ab_test,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting recommendations for user {user_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/recommendations/batch")
async def get_batch_recommendations(
    request: Dict[str, Any],
    current_user: Dict = Depends(get_current_user)
):
    """Obtener recomendaciones para múltiples usuarios"""
    try:
        user_ids = request.get('user_ids', [])
        limit = request.get('limit', 10)
        category = request.get('category')
        
        if len(user_ids) > 100:
            raise HTTPException(status_code=400, detail="Too many users (max 100)")
        
        results = {}
        
        for user_id in user_ids:
            try:
                recommendations = await app.state.recommendation_service.get_recommendations(
                    user_id=user_id,
                    limit=limit,
                    category=category
                )
                results[user_id] = recommendations
            except Exception as e:
                logger.error(f"Error for user {user_id}: {e}")
                results[user_id] = {"error": str(e)}
        
        return {
            "results": results,
            "total_users": len(user_ids),
            "successful": len([r for r in results.values() if "error" not in r])
        }
        
    except Exception as e:
        logger.error(f"Batch recommendation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/recommendations/feedback")
async def record_feedback(
    feedback: Dict[str, Any],
    current_user: Dict = Depends(get_current_user)
):
    """Registrar feedback de recomendaciones"""
    try:
        user_id = feedback.get('user_id')
        item_id = feedback.get('item_id')
        feedback_type = feedback.get('type')  # click, purchase, rating, etc.
        value = feedback.get('value')
        
        # Validar datos
        if not all([user_id, item_id, feedback_type]):
            raise HTTPException(status_code=400, detail="Missing required fields")
        
        # Registrar feedback
        await app.state.recommendation_service.record_feedback(
            user_id=user_id,
            item_id=item_id,
            feedback_type=feedback_type,
            value=value
        )
        
        # Verificar si es parte de un test A/B
        ab_test = await app.state.ab_testing_service.get_user_test(user_id)
        if ab_test:
            await app.state.ab_testing_service.record_conversion(
                test_id=ab_test['test_id'],
                user_id=user_id,
                item_id=item_id,
                conversion_type=feedback_type,
                value=value
            )
        
        return {"status": "success", "message": "Feedback recorded"}
        
    except Exception as e:
        logger.error(f"Error recording feedback: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Endpoints de productos
@app.get("/products")
async def get_products(
    limit: int = 50,
    offset: int = 0,
    category: Optional[str] = None,
    search: Optional[str] = None,
    current_user: Dict = Depends(get_current_user)
):
    """Obtener lista de productos"""
    try:
        products = await app.state.product_service.get_products(
            limit=limit,
            offset=offset,
            category=category,
            search=search
        )
        
        return {
            "products": products,
            "total": len(products),
            "limit": limit,
            "offset": offset
        }
        
    except Exception as e:
        logger.error(f"Error getting products: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/products/{product_id}")
async def get_product(
    product_id: str,
    current_user: Dict = Depends(get_current_user)
):
    """Obtener detalles de un producto"""
    try:
        product = await app.state.product_service.get_product(product_id)
        
        if not product:
            raise HTTPException(status_code=404, detail="Product not found")
        
        return product
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting product {product_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Endpoints de A/B Testing
@app.get("/ab-tests")
async def get_ab_tests(
    current_user: Dict = Depends(get_current_user)
):
    """Obtener lista de tests A/B activos"""
    try:
        tests = await app.state.ab_testing_service.get_active_tests()
        return {"tests": tests}
        
    except Exception as e:
        logger.error(f"Error getting A/B tests: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ab-tests")
async def create_ab_test(
    test_config: Dict[str, Any],
    current_user: Dict = Depends(get_current_user)
):
    """Crear nuevo test A/B"""
    try:
        # Validar configuración
        required_fields = ['name', 'description', 'model_a', 'model_b', 'traffic_split']
        for field in required_fields:
            if field not in test_config:
                raise HTTPException(status_code=400, detail=f"Missing field: {field}")
        
        test_id = await app.state.ab_testing_service.create_test(test_config)
        
        return {
            "test_id": test_id,
            "status": "created",
            "message": "A/B test created successfully"
        }
        
    except Exception as e:
        logger.error(f"Error creating A/B test: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/ab-tests/{test_id}/results")
async def get_ab_test_results(
    test_id: str,
    current_user: Dict = Depends(get_current_user)
):
    """Obtener resultados de un test A/B"""
    try:
        results = await app.state.ab_testing_service.get_test_results(test_id)
        
        if not results:
            raise HTTPException(status_code=404, detail="Test not found")
        
        return results
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting A/B test results: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Endpoints de analytics
@app.get("/analytics/users/{user_id}")
async def get_user_analytics(
    user_id: str,
    days: int = 30,
    current_user: Dict = Depends(get_current_user)
):
    """Obtener analytics de un usuario"""
    try:
        analytics = await app.state.user_service.get_user_analytics(
            user_id=user_id,
            days=days
        )
        
        return analytics
        
    except Exception as e:
        logger.error(f"Error getting user analytics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/analytics/dashboard")
async def get_dashboard_analytics(
    days: int = 7,
    current_user: Dict = Depends(get_current_user)
):
    """Obtener analytics del dashboard"""
    try:
        analytics = await app.state.recommendation_service.get_dashboard_analytics(
            days=days
        )
        
        return analytics
        
    except Exception as e:
        logger.error(f"Error getting dashboard analytics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Health check
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "models_loaded": len(app.state.model_manager.loaded_models),
        "kafka_connected": app.state.kafka_consumer.is_connected()
    }

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
