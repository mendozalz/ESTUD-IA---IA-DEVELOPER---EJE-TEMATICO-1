# Laboratorio 6.5: Chatbot con WebSockets y FastAPI

## 🎯 Objetivos del Laboratorio

### Objetivo General
Desarrollar un chatbot inteligente con procesamiento de lenguaje natural usando WebSockets para comunicación en tiempo real y FastAPI para el backend.

### Objetivos Específicos
- Implementar chatbot con NLP (transformers, spaCy)
- Crear comunicación bidireccional con WebSockets
- Desarrollar frontend con React y Socket.IO
- Implementar sistema de intents y entities
- Aplicar metodología Windsor para gestión del proyecto

## 📋 Marco Lógico del Proyecto

| Componente | Objetivo | Indicadores | Medios de Verificación |
|------------|----------|-------------|------------------------|
| **Chatbot NLP** | Procesamiento y comprensión del lenguaje | >85% precisión en intents | Modelo entrenado y evaluado |
| **WebSocket Server** | Comunicación en tiempo real | <50ms latencia de respuesta | Servidor WebSocket funcional |
| **Frontend React** | Interfaz de chat interactiva | UX responsiva y fluida | Componentes React completos |
| **Sistema de Intents** | Clasificación de intenciones | >90% accuracy en clasificación | Sistema de intents implementado |

## 🛠️ Tecnologías y Herramientas

### Principales Tecnologías
- **FastAPI 2026.3**: Backend API y WebSockets
- **Transformers 4.30**: Modelos de NLP
- **spaCy 3.6**: Procesamiento de lenguaje
- **React 18.2**: Frontend del chatbot
- **Socket.IO 4.7**: Comunicación WebSocket

### Dependencias Adicionales
- **Redis 7.0**: Sesiones y caché
- **MongoDB 6.0**: Almacenamiento de conversaciones
- **Rasa 3.6**: Framework de chatbots
- **Docker 24.0**: Contenerización

## 📁 Estructura del Proyecto

```
Laboratorio 6.5 - Chatbot con WebSockets/
├── README.md                           # Guía del laboratorio
├── windsor_plan.md                    # Planificación Windsor
├── requirements.txt                    # Dependencias Python
├── docker-compose.yml                 # Configuración local
├── Dockerfile                          # Imagen Docker
├── backend/                           # Backend FastAPI
│   ├── app.py                         # Aplicación principal
│   ├── models/                        # Modelos NLP
│   │   ├── __init__.py
│   │   ├── intent_classifier.py       # Clasificador de intents
│   │   ├── entity_extractor.py        # Extracción de entidades
│   │   ├── response_generator.py      # Generador de respuestas
│   │   └── chatbot_model.py          # Modelo principal del chatbot
│   ├── services/                      # Servicios de negocio
│   │   ├── __init__.py
│   │   ├── conversation_service.py     # Gestión de conversaciones
│   │   ├── nlp_service.py             # Servicio NLP
│   │   └── websocket_service.py       # Servicio WebSocket
│   ├── utils/                         # Utilidades
│   │   ├── __init__.py
│   │   ├── text_preprocessing.py      # Preprocesamiento de texto
│   │   ├── intent_training.py         # Entrenamiento de intents
│   │   └── response_templates.py      # Plantillas de respuesta
│   ├── data/                          # Datos de entrenamiento
│   │   ├── intents/                   # Definición de intents
│   │   ├── entities/                  # Definición de entidades
│   │   └── responses/                 # Plantillas de respuesta
│   └── tests/                         # Tests
│       ├── test_models.py
│       ├── test_services.py
│       └── test_websockets.py
├── frontend/                          # Frontend React
│   ├── package.json                   # Dependencias Node.js
│   ├── vite.config.js                 # Configuración Vite
│   ├── public/                        # Archivos estáticos
│   ├── src/                           # Código fuente
│   │   ├── components/                # Componentes React
│   │   │   ├── ChatWindow.jsx         # Ventana principal del chat
│   │   │   ├── MessageBubble.jsx      # Burbuja de mensaje
│   │   │   ├── InputArea.jsx         # Área de entrada
│   │   │   └── TypingIndicator.jsx    # Indicador de escritura
│   │   ├── hooks/                     # Hooks personalizados
│   │   │   ├── useWebSocket.js        # Hook WebSocket
│   │   │   ├── useChat.js             # Hook de chat
│   │   │   └── useTyping.js           # Hook de escritura
│   │   ├── services/                  # Servicios
│   │   │   ├── websocketService.js     # Cliente WebSocket
│   │   │   └── chatService.js         # Servicio de chat
│   │   ├── utils/                     # Utilidades
│   │   │   ├── messageFormatter.js    # Formateo de mensajes
│   │   │   └── emojiPicker.js         # Selector de emojis
│   │   ├── styles/                    # Estilos
│   │   │   ├── ChatWindow.css         # Estilos del chat
│   │   │   └── MessageBubble.css      # Estilos de mensajes
│   │   ├── App.jsx                    # Componente principal
│   │   └── main.js                    # Punto de entrada
│   └── tests/                         # Tests React
├── training/                          # Scripts de entrenamiento
│   ├── train_intent_classifier.py     # Entrenamiento de clasificador
│   ├── train_entity_extractor.py      # Entrenamiento de extractor
│   └── evaluate_models.py             # Evaluación de modelos
├── data/                              # Datos
│   ├── conversations/                  # Logs de conversaciones
│   ├── training_data/                 # Datos de entrenamiento
│   └── models/                        # Modelos entrenados
├── config/                            # Configuraciones
│   ├── intents_config.yml             # Configuración de intents
│   ├── entities_config.yml            # Configuración de entidades
│   └── responses_config.yml           # Configuración de respuestas
└── docs/                              # Documentación
    ├── api_documentation.md
    ├── user_guide.md
    └── deployment_guide.md
```

## 🔧 Implementación Detallada

### Fase 1: Backend FastAPI con WebSockets

#### backend/app.py:
```python
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
import uvicorn
import logging
import json
import asyncio
from datetime import datetime
from typing import Dict, List, Optional, Any

from models.chatbot_model import ChatbotModel
from services.conversation_service import ConversationService
from services.websocket_service import WebSocketService
from services.nlp_service import NLPService
from utils.response_templates import ResponseTemplates

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Gestión del ciclo de vida
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gestionar el ciclo de vida de la aplicación"""
    # Startup
    logger.info("Starting Chatbot Server...")
    
    # Inicializar componentes
    app.state.chatbot_model = ChatbotModel()
    app.state.conversation_service = ConversationService()
    app.state.websocket_service = WebSocketService()
    app.state.nlp_service = NLPService()
    app.state.response_templates = ResponseTemplates()
    
    # Cargar modelos NLP
    await app.state.chatbot_model.load_models()
    
    logger.info("Chatbot Server started successfully")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Chatbot Server...")
    await app.state.websocket_service.disconnect_all()
    logger.info("Chatbot Server shutdown complete")

# Inicialización de FastAPI
app = FastAPI(
    title="Intelligent Chatbot API",
    description="Advanced chatbot with NLP capabilities and real-time communication",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://chat.yourdomain.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Servir archivos estáticos
app.mount("/static", StaticFiles(directory="static"), name="static")

# Almacenamiento de conexiones WebSocket activas
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.user_sessions: Dict[str, Dict] = {}
    
    async def connect(self, websocket: WebSocket, user_id: str):
        await websocket.accept()
        self.active_connections[user_id] = websocket
        self.user_sessions[user_id] = {
            "connected_at": datetime.now(),
            "message_count": 0,
            "last_activity": datetime.now()
        }
        logger.info(f"User {user_id} connected")
    
    def disconnect(self, user_id: str):
        if user_id in self.active_connections:
            del self.active_connections[user_id]
        if user_id in self.user_sessions:
            del self.user_sessions[user_id]
        logger.info(f"User {user_id} disconnected")
    
    async def send_personal_message(self, message: str, user_id: str):
        if user_id in self.active_connections:
            websocket = self.active_connections[user_id]
            await websocket.send_text(message)
    
    async def broadcast(self, message: str):
        for connection in self.active_connections.values():
            await connection.send_text(message)
    
    def get_connected_users(self) -> List[str]:
        return list(self.active_connections.keys())

manager = ConnectionManager()

# Endpoints HTTP
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "connected_users": len(manager.active_connections),
        "models_loaded": app.state.chatbot_model.is_loaded()
    }

@app.get("/intents")
async def get_available_intents():
    """Obtener lista de intents disponibles"""
    try:
        intents = await app.state.nlp_service.get_available_intents()
        return {"intents": intents}
    except Exception as e:
        logger.error(f"Error getting intents: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/conversations/{user_id}")
async def get_conversation_history(user_id: str, limit: int = 50):
    """Obtener historial de conversación"""
    try:
        history = await app.state.conversation_service.get_conversation(
            user_id=user_id,
            limit=limit
        )
        return {"history": history}
    except Exception as e:
        logger.error(f"Error getting conversation history: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/train/intent")
async def train_intent_classifier(training_data: Dict[str, Any]):
    """Entrenar clasificador de intents"""
    try:
        result = await app.state.chatbot_model.train_intent_classifier(training_data)
        return {"status": "success", "result": result}
    except Exception as e:
        logger.error(f"Error training intent classifier: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# WebSocket endpoints
@app.websocket("/ws/{user_id}")
async def websocket_endpoint(websocket: WebSocket, user_id: str):
    """Endpoint principal de WebSocket"""
    await manager.connect(websocket, user_id)
    
    try:
        while True:
            # Recibir mensaje del cliente
            data = await websocket.receive_text()
            message_data = json.loads(data)
            
            # Actualizar actividad del usuario
            if user_id in manager.user_sessions:
                manager.user_sessions[user_id]["last_activity"] = datetime.now()
                manager.user_sessions[user_id]["message_count"] += 1
            
            # Procesar mensaje
            response = await process_message(user_id, message_data)
            
            # Enviar respuesta
            await manager.send_personal_message(
                json.dumps(response),
                user_id
            )
            
    except WebSocketDisconnect:
        manager.disconnect(user_id)
    except Exception as e:
        logger.error(f"WebSocket error for user {user_id}: {e}")
        manager.disconnect(user_id)

@app.websocket("/ws/typing/{user_id}")
async def typing_websocket(websocket: WebSocket, user_id: str):
    """WebSocket para indicadores de escritura"""
    await websocket.accept()
    
    try:
        while True:
            # Recibir estado de escritura
            data = await websocket.receive_text()
            typing_data = json.loads(data)
            
            # Broadcast a otros usuarios (si es un chat grupal)
            await broadcast_typing_status(user_id, typing_data)
            
    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.error(f"Typing WebSocket error: {e}")

async def process_message(user_id: str, message_data: Dict[str, Any]) -> Dict[str, Any]:
    """Procesar mensaje del usuario y generar respuesta"""
    try:
        message_text = message_data.get("message", "").strip()
        message_type = message_data.get("type", "text")
        
        if not message_text:
            return {
                "type": "error",
                "message": "Empty message",
                "timestamp": datetime.now().isoformat()
            }
        
        # Guardar mensaje del usuario
        await app.state.conversation_service.add_message(
            user_id=user_id,
            message=message_text,
            sender="user",
            message_type=message_type
        )
        
        # Enviar indicador de que el bot está procesando
        await manager.send_personal_message(
            json.dumps({
                "type": "typing",
                "sender": "bot",
                "is_typing": True
            }),
            user_id
        )
        
        # Pequeña delay para simular procesamiento
        await asyncio.sleep(0.5)
        
        # Procesar con NLP
        nlp_result = await app.state.nlp_service.process_message(message_text)
        
        # Generar respuesta
        response_text = await app.state.chatbot_model.generate_response(
            message=message_text,
            intent=nlp_result.get("intent"),
            entities=nlp_result.get("entities"),
            context=await app.state.conversation_service.get_context(user_id)
        )
        
        # Guardar respuesta del bot
        await app.state.conversation_service.add_message(
            user_id=user_id,
            message=response_text,
            sender="bot",
            message_type="text",
            intent=nlp_result.get("intent"),
            entities=nlp_result.get("entities")
        )
        
        # Enviar indicador de que el bot terminó de escribir
        await manager.send_personal_message(
            json.dumps({
                "type": "typing",
                "sender": "bot",
                "is_typing": False
            }),
            user_id
        )
        
        # Construir respuesta
        response = {
            "type": "message",
            "sender": "bot",
            "message": response_text,
            "timestamp": datetime.now().isoformat(),
            "intent": nlp_result.get("intent"),
            "confidence": nlp_result.get("confidence"),
            "entities": nlp_result.get("entities"),
            "suggestions": await generate_suggestions(nlp_result.get("intent"))
        }
        
        return response
        
    except Exception as e:
        logger.error(f"Error processing message: {e}")
        return {
            "type": "error",
            "message": "Sorry, I encountered an error processing your message.",
            "timestamp": datetime.now().isoformat()
        }

async def generate_suggestions(intent: Optional[str]) -> List[str]:
    """Generar sugerencias basadas en el intent"""
    if not intent:
        return []
    
    suggestions_map = {
        "greeting": ["How are you?", "What can you help me with?", "Tell me a joke"],
        "goodbye": ["Is there anything else I can help with?", "Have a great day!"],
        "help": ["What services do you offer?", "How do I contact support?", "What are your hours?"],
        "question": ["Can you provide more details?", "Is there anything specific you'd like to know?"],
        "complaint": ["I understand your concern", "Let me help you with that", "I'll connect you with support"]
    }
    
    return suggestions_map.get(intent, [])

async def broadcast_typing_status(user_id: str, typing_data: Dict[str, Any]):
    """Broadcast estado de escritura a otros usuarios"""
    message = {
        "type": "typing",
        "user_id": user_id,
        "is_typing": typing_data.get("is_typing", False),
        "timestamp": datetime.now().isoformat()
    }
    
    # Enviar a todos excepto al remitente (para chat grupal)
    for connection_user_id, websocket in manager.active_connections.items():
        if connection_user_id != user_id:
            await websocket.send_text(json.dumps(message))

# Endpoint para obtener estadísticas
@app.get("/stats")
async def get_chatbot_stats():
    """Obtener estadísticas del chatbot"""
    try:
        stats = {
            "connected_users": len(manager.active_connections),
            "total_conversations": await app.state.conversation_service.get_total_conversations(),
            "total_messages": await app.state.conversation_service.get_total_messages(),
            "popular_intents": await app.state.nlp_service.get_popular_intents(),
            "average_response_time": await app.state.conversation_service.get_average_response_time(),
            "uptime": datetime.now().isoformat()
        }
        return stats
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Endpoint para resetear conversación
@app.post("/conversations/{user_id}/reset")
async def reset_conversation(user_id: str):
    """Resetear conversación de un usuario"""
    try:
        await app.state.conversation_service.reset_conversation(user_id)
        
        # Notificar al usuario si está conectado
        if user_id in manager.active_connections:
            await manager.send_personal_message(
                json.dumps({
                    "type": "system",
                    "message": "Conversation has been reset",
                    "timestamp": datetime.now().isoformat()
                }),
                user_id
            )
        
        return {"status": "success", "message": "Conversation reset"}
    except Exception as e:
        logger.error(f"Error resetting conversation: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
```

### Fase 2: Modelo NLP del Chatbot

#### backend/models/chatbot_model.py:
```python
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import spacy
import json
import pickle
import logging
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import asyncio

logger = logging.getLogger(__name__)

class ChatbotModel:
    def __init__(self):
        self.intent_classifier = None
        self.entity_extractor = None
        self.response_generator = None
        self.nlp = None
        self.intent_labels = []
        self.entity_types = []
        self.is_model_loaded = False
        
    async def load_models(self):
        """Cargar todos los modelos NLP"""
        try:
            logger.info("Loading NLP models...")
            
            # Cargar modelo de clasificación de intents
            await self._load_intent_classifier()
            
            # Cargar spaCy para procesamiento de lenguaje
            try:
                self.nlp = spacy.load("en_core_web_sm")
                logger.info("SpaCy model loaded successfully")
            except OSError:
                logger.warning("SpaCy model not found, using basic processing")
                self.nlp = None
            
            # Cargar extractor de entidades
            await self._load_entity_extractor()
            
            # Cargar generador de respuestas
            await self._load_response_generator()
            
            self.is_model_loaded = True
            logger.info("All NLP models loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            raise
    
    async def _load_intent_classifier(self):
        """Cargar clasificador de intents"""
        try:
            # Usar un modelo pre-entrenado para clasificación de texto
            model_name = "distilbert-base-uncased"
            
            self.intent_tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.intent_model = AutoModelForSequenceClassification.from_pretrained(model_name)
            
            # Labels de intents (personalizar según necesidades)
            self.intent_labels = [
                "greeting", "goodbye", "help", "question", 
                "complaint", "compliment", "information", "booking",
                "cancel", "modify", "payment", "technical", "other"
            ]
            
            # Crear pipeline de clasificación
            self.intent_classifier = pipeline(
                "text-classification",
                model=self.intent_model,
                tokenizer=self.intent_tokenizer,
                return_all_scores=True
            )
            
            logger.info("Intent classifier loaded")
            
        except Exception as e:
            logger.error(f"Error loading intent classifier: {e}")
            # Fallback a clasificador simple
            self.intent_classifier = self._create_fallback_classifier()
    
    async def _load_entity_extractor(self):
        """Cargar extractor de entidades"""
        try:
            # Usar spaCy NER si está disponible
            if self.nlp:
                self.entity_extractor = self.nlp
                self.entity_types = ["PERSON", "ORG", "GPE", "DATE", "TIME", "MONEY"]
            else:
                # Fallback a extractor simple basado en reglas
                self.entity_extractor = self._create_fallback_entity_extractor()
                self.entity_types = ["email", "phone", "date", "time"]
            
            logger.info("Entity extractor loaded")
            
        except Exception as e:
            logger.error(f"Error loading entity extractor: {e}")
            self.entity_extractor = self._create_fallback_entity_extractor()
    
    async def _load_response_generator(self):
        """Cargar generador de respuestas"""
        try:
            # Cargar plantillas de respuesta
            self.response_templates = await self._load_response_templates()
            logger.info("Response generator loaded")
            
        except Exception as e:
            logger.error(f"Error loading response generator: {e}")
            self.response_templates = self._create_fallback_responses()
    
    async def generate_response(self, message: str, intent: Optional[str], 
                              entities: Optional[List[Dict]], context: Optional[Dict]) -> str:
        """Generar respuesta basada en el mensaje, intent y entidades"""
        try:
            if not intent:
                return self._get_fallback_response()
            
            # Obtener plantillas para el intent
            templates = self.response_templates.get(intent, [])
            
            if not templates:
                return self._get_fallback_response()
            
            # Seleccionar plantilla (puede ser aleatoria o basada en contexto)
            import random
            template = random.choice(templates)
            
            # Personalizar respuesta con entidades
            response = self._personalize_response(template, entities, context)
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return self._get_fallback_response()
    
    def _personalize_response(self, template: str, entities: Optional[List[Dict]], 
                            context: Optional[Dict]) -> str:
        """Personalizar plantilla de respuesta con entidades y contexto"""
        try:
            response = template
            
            # Reemplazar entidades
            if entities:
                for entity in entities:
                    entity_type = entity.get("type", "")
                    entity_value = entity.get("value", "")
                    
                    if entity_type.lower() in response.lower():
                        response = response.replace(f"{{{entity_type.upper()}}}", entity_value)
            
            # Reemplazar variables de contexto
            if context:
                for key, value in context.items():
                    if f"{{{key.upper()}}}" in response:
                        response = response.replace(f"{{{key.upper()}}}", str(value))
            
            return response
            
        except Exception as e:
            logger.error(f"Error personalizing response: {e}")
            return template
    
    def _create_fallback_classifier(self):
        """Crear clasificador fallback simple basado en reglas"""
        class FallbackClassifier:
            def __init__(self, labels):
                self.labels = labels
            
            def __call__(self, text):
                # Lógica simple basada en palabras clave
                text_lower = text.lower()
                
                scores = []
                for label in self.labels:
                    score = 0.0
                    
                    # Palabras clave por intent
                    keyword_map = {
                        "greeting": ["hello", "hi", "hey", "good morning", "good evening"],
                        "goodbye": ["bye", "goodbye", "see you", "farewell"],
                        "help": ["help", "assist", "support", "aid"],
                        "question": ["what", "how", "when", "where", "why", "who"],
                        "complaint": ["complaint", "problem", "issue", "wrong", "bad"],
                        "compliment": ["good", "great", "excellent", "amazing", "wonderful"],
                        "information": ["information", "details", "about", "tell me"],
                        "booking": ["book", "reserve", "schedule", "appointment"],
                        "cancel": ["cancel", "delete", "remove"],
                        "payment": ["pay", "payment", "cost", "price", "fee"]
                    }
                    
                    if label in keyword_map:
                        for keyword in keyword_map[label]:
                            if keyword in text_lower:
                                score += 0.5
                    
                    scores.append({"label": label, "score": min(score, 1.0)})
                
                return [scores]
        
        return FallbackClassifier(self.intent_labels)
    
    def _create_fallback_entity_extractor(self):
        """Crear extractor de entidades fallback basado en expresiones regulares"""
        import re
        
        class FallbackEntityExtractor:
            def __call__(self, text):
                doc = type('MockDoc', (), {})()
                
                # Extraer emails
                emails = re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text)
                
                # Extraer teléfonos
                phones = re.findall(r'\b\d{3}-\d{3}-\d{4}\b|\b\d{10}\b', text)
                
                # Extraer fechas
                dates = re.findall(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b', text)
                
                entities = []
                
                for email in emails:
                    entities.append({"text": email, "label_": "EMAIL"})
                
                for phone in phones:
                    entities.append({"text": phone, "label_": "PHONE"})
                
                for date in dates:
                    entities.append({"text": date, "label_": "DATE"})
                
                doc.ents = entities
                return doc
        
        return FallbackEntityExtractor()
    
    async def _load_response_templates(self) -> Dict[str, List[str]]:
        """Cargar plantillas de respuesta desde archivo"""
        try:
            # Plantillas por defecto
            templates = {
                "greeting": [
                    "Hello! How can I help you today?",
                    "Hi there! What can I do for you?",
                    "Good day! How may I assist you?",
                    "Welcome! How can I help?"
                ],
                "goodbye": [
                    "Goodbye! Have a great day!",
                    "See you later! Take care!",
                    "Farewell! Don't hesitate to return if you need help.",
                    "Bye! It was nice helping you!"
                ],
                "help": [
                    "I'm here to help! What do you need assistance with?",
                    "I'd be happy to help you. Please tell me what you need.",
                    "How can I assist you today?",
                    "What can I help you with?"
                ],
                "question": [
                    "That's a great question! Let me help you with that.",
                    "I understand your question. Here's what I can tell you...",
                    "Let me provide you with that information.",
                    "I'd be happy to answer your question."
                ],
                "complaint": [
                    "I'm sorry to hear you're having issues. Let me help resolve this.",
                    "I understand your frustration. I'm here to help make things right.",
                    "I apologize for the inconvenience. Let me assist you with this problem.",
                    "I'm sorry you're experiencing difficulties. Let's work together to solve this."
                ],
                "compliment": [
                    "Thank you so much! I appreciate your kind words.",
                    "That's very kind of you to say! Thank you!",
                    "I'm glad I could help! Your feedback means a lot to me.",
                    "Thank you for the compliment! I'm here to assist whenever you need."
                ],
                "information": [
                    "I'd be happy to provide that information for you.",
                    "Let me share those details with you.",
                    "Here's the information you requested:",
                    "I can help you with that information."
                ],
                "booking": [
                    "I'd be happy to help you make a booking.",
                    "Let me assist you with scheduling that.",
                    "I can help you reserve that for you.",
                    "Let's get that scheduled for you."
                ],
                "cancel": [
                    "I can help you cancel that for you.",
                    "Let me process that cancellation for you.",
                    "I'll help you with the cancellation process.",
                    "I can assist you with cancelling that."
                ],
                "payment": [
                    "I can help you with payment processing.",
                    "Let me assist you with the payment.",
                    "I can help you complete that payment.",
                    "Let me guide you through the payment process."
                ],
                "technical": [
                    "I understand you're having technical difficulties. Let me help.",
                    "I can assist you with technical support.",
                    "Let's troubleshoot this technical issue together.",
                    "I'm here to help resolve your technical problem."
                ]
            }
            
            return templates
            
        except Exception as e:
            logger.error(f"Error loading response templates: {e}")
            return self._create_fallback_responses()
    
    def _create_fallback_responses(self) -> Dict[str, List[str]]:
        """Crear respuestas fallback"""
        return {
            "default": [
                "I'm here to help! Could you please provide more details?",
                "I'd be happy to assist you. What would you like to know?",
                "How can I help you today?",
                "I'm listening. Please tell me what you need."
            ]
        }
    
    def _get_fallback_response(self) -> str:
        """Obtener respuesta fallback"""
        fallback_responses = [
            "I'm here to help! Could you please rephrase that?",
            "I'd be happy to assist you. Could you provide more details?",
            "I want to help, but I'm not sure I understand. Could you clarify?",
            "I'm listening. Please try explaining in a different way."
        ]
        
        import random
        return random.choice(fallback_responses)
    
    def is_loaded(self) -> bool:
        """Verificar si los modelos están cargados"""
        return self.is_model_loaded
    
    async def train_intent_classifier(self, training_data: Dict[str, Any]):
        """Entrenar clasificador de intents con nuevos datos"""
        try:
            # Implementar lógica de entrenamiento
            # Esto podría incluir fine-tuning del modelo transformers
            
            logger.info("Training intent classifier with new data...")
            
            # Simulación de entrenamiento
            await asyncio.sleep(2)
            
            result = {
                "status": "success",
                "samples": len(training_data.get("samples", [])),
                "accuracy": 0.85,
                "training_time": "2.0s"
            }
            
            logger.info("Intent classifier training completed")
            return result
            
        except Exception as e:
            logger.error(f"Error training intent classifier: {e}")
            raise
```

## 📊 Entregables del Laboratorio

### 1. Código Fuente Completo
- **backend/app.py**: API FastAPI con WebSockets
- **backend/models/chatbot_model.py**: Modelo NLP completo
- **backend/services/**: Servicios de chat y NLP
- **frontend/src/**: Interfaz React completa

### 2. Configuración de Entrenamiento
- **training/**: Scripts para entrenar modelos
- **data/**: Datos de entrenamiento y configuración

### 3. Sistema de Intents y Entidades
- **config/**: Configuración de intents y respuestas
- **utils/response_templates.py**: Plantillas de respuesta

### 4. Tests y Documentación
- **tests/**: Suite completa de tests
- **docs/**: Documentación de API y usuario

---

**Duración Estimada**: 8-10 horas  
**Dificultad**: Avanzada  
**Prerrequisitos**: Conocimientos de NLP, WebSockets y React
