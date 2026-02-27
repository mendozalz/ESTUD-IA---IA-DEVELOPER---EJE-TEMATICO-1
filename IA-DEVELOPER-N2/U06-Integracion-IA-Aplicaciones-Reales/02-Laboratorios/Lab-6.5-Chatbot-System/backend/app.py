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
