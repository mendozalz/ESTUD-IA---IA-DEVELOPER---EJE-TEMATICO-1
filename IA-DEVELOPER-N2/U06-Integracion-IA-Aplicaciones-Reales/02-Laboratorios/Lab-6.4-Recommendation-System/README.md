# Laboratorio 6.4: Sistema de Recomendación E-commerce

## 🎯 Objetivos del Laboratorio

### Objetivo General
Desarrollar un sistema completo de recomendación para e-commerce con procesamiento en tiempo real, aprendizaje continuo y A/B testing integrado.

### Objetivos Específicos
- Implementar algoritmos de recomendación (collaborative filtering, content-based, hybrid)
- Crear pipeline de streaming con Apache Kafka
- Desarrollar sistema de A/B testing para modelos
- Construir dashboard de administración con Vue.js
- Aplicar metodología Windsor para gestión del proyecto

## 📋 Marco Lógico del Proyecto

| Componente | Objetivo | Indicadores | Medios de Verificación |
|------------|----------|-------------|------------------------|
| **Motor de Recomendación** | Generar recomendaciones personalizadas | >85% precisión en tests | Modelos implementados y evaluados |
| **Pipeline Streaming** | Procesar eventos en tiempo real | <100ms latencia de procesamiento | Kafka pipeline funcional |
| **A/B Testing** | Evaluar rendimiento de modelos | Tests estadísticos significativos | Sistema de experimentación completo |
| **Dashboard Admin** | Gestión de productos y análisis | Interfaz responsiva y funcional | Frontend Vue.js completo |

## 🛠️ Tecnologías y Herramientas

### Principales Tecnologías
- **Apache Kafka 3.5**: Streaming de eventos
- **Apache Spark 3.4**: Procesamiento distribuido
- **MLflow 2.7**: Tracking de modelos
- **Vue.js 3.3**: Frontend de administración
- **FastAPI 2026.3**: API de recomendaciones

### Dependencias Adicionales
- **Redis 7.0**: Caching de recomendaciones
- **Elasticsearch 8.8**: Búsqueda de productos
- **PostgreSQL 15**: Perfiles de usuarios
- **Grafana 10.0**: Dashboards de monitoreo

## 📁 Estructura del Proyecto

```
Laboratorio 6.4 - Sistema de Recomendación/
├── README.md                           # Guía del laboratorio
├── windsor_plan.md                    # Planificación Windsor
├── requirements.txt                    # Dependencias Python
├── docker-compose.yml                 # Configuración local
├── Dockerfile                          # Imagen Docker
├── backend/                           # API y servicios
│   ├── app.py                         # Aplicación FastAPI principal
│   ├── models/                        # Modelos de recomendación
│   │   ├── __init__.py
│   │   ├── collaborative_filtering.py # Filtrado colaborativo
│   │   ├── content_based.py           # Basado en contenido
│   │   ├── hybrid_model.py            # Modelo híbrido
│   │   └── model_manager.py           # Gestión de modelos
│   ├── services/                       # Servicios de negocio
│   │   ├── __init__.py
│   │   ├── recommendation_service.py  # Servicio de recomendaciones
│   │   ├── user_service.py             # Gestión de usuarios
│   │   ├── product_service.py          # Gestión de productos
│   │   └── ab_testing_service.py      # A/B testing
│   ├── streaming/                      # Pipeline de streaming
│   │   ├── __init__.py
│   │   ├── kafka_consumer.py           # Consumidor Kafka
│   │   ├── spark_processor.py         # Procesamiento Spark
│   │   └── event_handlers.py          # Manejadores de eventos
│   ├── utils/                          # Utilidades
│   │   ├── __init__.py
│   │   ├── data_preprocessing.py       # Preprocesamiento de datos
│   │   ├── feature_engineering.py      # Ingeniería de características
│   │   └── evaluation_metrics.py       # Métricas de evaluación
│   └── tests/                         # Tests
│       ├── test_models.py
│       ├── test_services.py
│       └── test_streaming.py
├── frontend/                          # Dashboard Vue.js
│   ├── package.json                   # Dependencias Node.js
│   ├── vite.config.js                 # Configuración Vite
│   ├── public/                        # Archivos estáticos
│   ├── src/                           # Código fuente
│   │   ├── components/                # Componentes Vue
│   │   │   ├── ProductManager.vue     # Gestión de productos
│   │   │   ├── UserAnalytics.vue      # Análisis de usuarios
│   │   │   ├── ABTesting.vue          # Panel A/B testing
│   │   │   └── RecommendationPreview.vue # Vista previa
│   │   ├── views/                     # Vistas principales
│   │   │   ├── Dashboard.vue          # Dashboard principal
│   │   │   ├── Products.vue           # Gestión productos
│   │   │   ├── Analytics.vue          # Análisis
│   │   │   └── Settings.vue           # Configuración
│   │   ├── services/                  # Servicios API
│   │   │   ├── api.js                 # Cliente HTTP
│   │   │   ├── websocket.js           # Cliente WebSocket
│   │   │   └── auth.js                # Autenticación
│   │   ├── stores/                    # Pinia stores
│   │   │   ├── products.js            # Store productos
│   │   │   ├── users.js               # Store usuarios
│   │   │   └── recommendations.js      # Store recomendaciones
│   │   ├── router/                    # Vue Router
│   │   │   └── index.js               # Configuración rutas
│   │   ├── App.vue                    # Componente principal
│   │   └── main.js                    # Punto de entrada
│   └── tests/                         # Tests Vue
├── data/                              # Datos y modelos
│   ├── models/                        # Modelos entrenados
│   ├── datasets/                      # Datasets de entrenamiento
│   └── cache/                         # Caché Redis
├── mlflow/                            # Configuración MLflow
│   ├── tracking/                      # Tracking de experimentos
│   └── models/                        # Modelos registrados
├── kafka/                             # Configuración Kafka
│   ├── topics/                        # Definición de topics
│   └── schemas/                       # Esquemas de eventos
├── monitoring/                        # Monitoreo
│   ├── prometheus/
│   └── grafana/
└── docs/                              # Documentación
    ├── api_documentation.md
    ├── deployment_guide.md
    └── user_manual.md
```

## 🔧 Implementación Detallada

### Fase 1: API FastAPI Principal

#### backend/app.py:
```python
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
```

### Fase 2: Modelos de Recomendación

#### backend/models/collaborative_filtering.py:
```python
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import NMF
from scipy.sparse import csr_matrix
import pickle
import logging
from typing import List, Dict, Tuple, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

class CollaborativeFilteringModel:
    def __init__(self, n_factors: int = 50, regularization: float = 0.01):
        self.n_factors = n_factors
        self.regularization = regularization
        self.model = None
        self.user_item_matrix = None
        self.user_mapping = {}
        self.item_mapping = {}
        self.reverse_user_mapping = {}
        self.reverse_item_mapping = {}
        self.is_trained = False
        
    def prepare_data(self, interactions_df: pd.DataFrame):
        """Preparar datos para el modelo"""
        logger.info("Preparing data for collaborative filtering...")
        
        # Crear mapeos de usuarios y items
        unique_users = interactions_df['user_id'].unique()
        unique_items = interactions_df['item_id'].unique()
        
        self.user_mapping = {user: idx for idx, user in enumerate(unique_users)}
        self.item_mapping = {item: idx for idx, item in enumerate(unique_items)}
        
        self.reverse_user_mapping = {idx: user for user, idx in self.user_mapping.items()}
        self.reverse_item_mapping = {idx: item for item, idx in self.item_mapping.items()}
        
        # Crear matriz usuario-item
        n_users = len(unique_users)
        n_items = len(unique_items)
        
        self.user_item_matrix = np.zeros((n_users, n_items))
        
        for _, row in interactions_df.iterrows():
            user_idx = self.user_mapping[row['user_id']]
            item_idx = self.item_mapping[row['item_id']]
            rating = row.get('rating', row.get('interaction', 1))
            self.user_item_matrix[user_idx, item_idx] = rating
        
        logger.info(f"Created user-item matrix: {n_users} users, {n_items} items")
        
    def train(self, interactions_df: pd.DataFrame):
        """Entrenar el modelo de collaborative filtering"""
        logger.info("Training collaborative filtering model...")
        
        self.prepare_data(interactions_df)
        
        # Usar NMF (Non-negative Matrix Factorization)
        self.model = NMF(
            n_components=self.n_factors,
            init='random',
            random_state=42,
            alpha=self.regularization,
            max_iter=500
        )
        
        # Entrenar el modelo
        W = self.model.fit_transform(self.user_item_matrix)
        H = self.model.components_
        
        # Reconstruir la matriz
        self.reconstructed_matrix = np.dot(W, H)
        
        self.is_trained = True
        logger.info("Collaborative filtering model trained successfully")
        
    def predict(self, user_id: str, item_id: str) -> float:
        """Predecir rating para un usuario-item"""
        if not self.is_trained:
            raise RuntimeError("Model not trained")
        
        if user_id not in self.user_mapping or item_id not in self.item_mapping:
            return 0.0
        
        user_idx = self.user_mapping[user_id]
        item_idx = self.item_mapping[item_id]
        
        return self.reconstructed_matrix[user_idx, item_idx]
    
    def recommend(self, user_id: str, n_recommendations: int = 10, 
                  exclude_seen: bool = True) -> List[Tuple[str, float]]:
        """Generar recomendaciones para un usuario"""
        if not self.is_trained:
            raise RuntimeError("Model not trained")
        
        if user_id not in self.user_mapping:
            logger.warning(f"User {user_id} not found in training data")
            return []
        
        user_idx = self.user_mapping[user_id]
        user_ratings = self.reconstructed_matrix[user_idx]
        
        # Excluir items ya vistos
        if exclude_seen:
            seen_items = np.where(self.user_item_matrix[user_idx] > 0)[0]
            user_ratings[seen_items] = -np.inf
        
        # Obtener top N recomendaciones
        top_indices = np.argsort(user_ratings)[::-1][:n_recommendations]
        
        recommendations = []
        for idx in top_indices:
            item_id = self.reverse_item_mapping[idx]
            score = user_ratings[idx]
            recommendations.append((item_id, float(score)))
        
        return recommendations
    
    def get_similar_users(self, user_id: str, n_similar: int = 10) -> List[Tuple[str, float]]:
        """Encontrar usuarios similares"""
        if not self.is_trained:
            raise RuntimeError("Model not trained")
        
        if user_id not in self.user_mapping:
            return []
        
        user_idx = self.user_mapping[user_id]
        user_vector = self.user_item_matrix[user_idx].reshape(1, -1)
        
        # Calcular similitud con todos los usuarios
        similarities = cosine_similarity(user_vector, self.user_item_matrix)[0]
        
        # Excluir al mismo usuario
        similarities[user_idx] = -1
        
        # Obtener top N usuarios similares
        top_indices = np.argsort(similarities)[::-1][:n_similar]
        
        similar_users = []
        for idx in top_indices:
            similar_user_id = self.reverse_user_mapping[idx]
            similarity_score = similarities[idx]
            similar_users.append((similar_user_id, float(similarity_score)))
        
        return similar_users
    
    def get_similar_items(self, item_id: str, n_similar: int = 10) -> List[Tuple[str, float]]:
        """Encontrar items similares basado en interacciones de usuarios"""
        if not self.is_trained:
            raise RuntimeError("Model not trained")
        
        if item_id not in self.item_mapping:
            return []
        
        item_idx = self.item_mapping[item_id]
        item_vector = self.user_item_matrix[:, item_idx].reshape(1, -1)
        
        # Calcular similitud con todos los items
        similarities = cosine_similarity(item_vector, self.user_item_matrix.T)[0]
        
        # Excluir al mismo item
        similarities[item_idx] = -1
        
        # Obtener top N items similares
        top_indices = np.argsort(similarities)[::-1][:n_similar]
        
        similar_items = []
        for idx in top_indices:
            similar_item_id = self.reverse_item_mapping[idx]
            similarity_score = similarities[idx]
            similar_items.append((similar_item_id, float(similarity_score)))
        
        return similar_items
    
    def evaluate(self, test_df: pd.DataFrame) -> Dict[str, float]:
        """Evaluar el modelo con métricas estándar"""
        if not self.is_trained:
            raise RuntimeError("Model not trained")
        
        predictions = []
        actuals = []
        
        for _, row in test_df.iterrows():
            user_id = row['user_id']
            item_id = row['item_id']
            actual_rating = row.get('rating', row.get('interaction', 1))
            
            if user_id in self.user_mapping and item_id in self.item_mapping:
                predicted_rating = self.predict(user_id, item_id)
                predictions.append(predicted_rating)
                actuals.append(actual_rating)
        
        if not predictions:
            return {"error": "No valid predictions"}
        
        predictions = np.array(predictions)
        actuals = np.array(actuals)
        
        # Calcular métricas
        mse = np.mean((predictions - actuals) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(predictions - actuals))
        
        # Para precisión@k y recall@k
        precision_at_5 = self._calculate_precision_at_k(test_df, k=5)
        recall_at_5 = self._calculate_recall_at_k(test_df, k=5)
        
        return {
            "mse": float(mse),
            "rmse": float(rmse),
            "mae": float(mae),
            "precision_at_5": precision_at_5,
            "recall_at_5": recall_at_5
        }
    
    def _calculate_precision_at_k(self, test_df: pd.DataFrame, k: int = 5) -> float:
        """Calcular precisión@k"""
        precisions = []
        
        for user_id in test_df['user_id'].unique():
            if user_id in self.user_mapping:
                # Obtener recomendaciones
                recommendations = self.recommend(user_id, n_recommendations=k)
                recommended_items = [item_id for item_id, _ in recommendations]
                
                # Obtener items relevantes del usuario
                user_test_data = test_df[test_df['user_id'] == user_id]
                relevant_items = set(user_test_data['item_id'].tolist())
                
                if recommended_items:
                    # Calcular precisión
                    relevant_recommended = len(set(recommended_items) & relevant_items)
                    precision = relevant_recommended / len(recommended_items)
                    precisions.append(precision)
        
        return np.mean(precisions) if precisions else 0.0
    
    def _calculate_recall_at_k(self, test_df: pd.DataFrame, k: int = 5) -> float:
        """Calcular recall@k"""
        recalls = []
        
        for user_id in test_df['user_id'].unique():
            if user_id in self.user_mapping:
                # Obtener recomendaciones
                recommendations = self.recommend(user_id, n_recommendations=k)
                recommended_items = [item_id for item_id, _ in recommendations]
                
                # Obtener items relevantes del usuario
                user_test_data = test_df[test_df['user_id'] == user_id]
                relevant_items = set(user_test_data['item_id'].tolist())
                
                if relevant_items:
                    # Calcular recall
                    relevant_recommended = len(set(recommended_items) & relevant_items)
                    recall = relevant_recommended / len(relevant_items)
                    recalls.append(recall)
        
        return np.mean(recalls) if recalls else 0.0
    
    def save_model(self, filepath: str):
        """Guardar el modelo"""
        model_data = {
            'model': self.model,
            'user_item_matrix': self.user_item_matrix,
            'reconstructed_matrix': self.reconstructed_matrix,
            'user_mapping': self.user_mapping,
            'item_mapping': self.item_mapping,
            'reverse_user_mapping': self.reverse_user_mapping,
            'reverse_item_mapping': self.reverse_item_mapping,
            'n_factors': self.n_factors,
            'regularization': self.regularization,
            'is_trained': self.is_trained
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Cargar el modelo"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.user_item_matrix = model_data['user_item_matrix']
        self.reconstructed_matrix = model_data['reconstructed_matrix']
        self.user_mapping = model_data['user_mapping']
        self.item_mapping = model_data['item_mapping']
        self.reverse_user_mapping = model_data['reverse_user_mapping']
        self.reverse_item_mapping = model_data['reverse_item_mapping']
        self.n_factors = model_data['n_factors']
        self.regularization = model_data['regularization']
        self.is_trained = model_data['is_trained']
        
        logger.info(f"Model loaded from {filepath}")
```

## 📊 Entregables del Laboratorio

### 1. Código Fuente Completo
- **backend/app.py**: API FastAPI con todos los endpoints
- **backend/models/**: Modelos de recomendación completos
- **backend/services/**: Servicios de negocio
- **frontend/src/**: Dashboard Vue.js completo

### 2. Configuración de Streaming
- **kafka/**: Configuración completa de topics y schemas
- **backend/streaming/**: Pipeline de streaming con Spark

### 3. Sistema de A/B Testing
- **backend/services/ab_testing_service.py**: Sistema completo de experimentación
- **MLflow tracking**: Integración para tracking de modelos

### 4. Tests y Documentación
- **tests/**: Suite completa de tests
- **docs/**: Documentación de API y despliegue

---

**Duración Estimada**: 8-10 horas  
**Dificultad**: Avanzada  
**Prerrequisitos**: Conocimientos de sistemas de recomendación, Kafka y Spark
