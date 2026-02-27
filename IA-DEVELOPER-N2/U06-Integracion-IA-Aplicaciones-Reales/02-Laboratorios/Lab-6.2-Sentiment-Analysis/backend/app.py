from flask import Flask, request, jsonify
from flask_socketio import SocketIO, emit
from flask_cors import CORS
import time
import logging
from datetime import datetime
import json

from models.sentiment_analyzer import SentimentAnalyzer
from utils.text_processing import TextProcessor
from utils.metrics import (
    REQUEST_COUNT, REQUEST_LATENCY, 
    SENTIMENT_COUNT, REAL_TIME_CONNECTIONS
)

# Configuración
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key'
CORS(app)

# Configurar SocketIO
socketio = SocketIO(app, cors_allowed_origins="*", logger=True, engineio_logger=True)

# Inicializar componentes
sentiment_analyzer = SentimentAnalyzer()
text_processor = TextProcessor()

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "model_loaded": sentiment_analyzer.is_loaded()
    })

@app.route('/analyze', methods=['POST'])
def analyze_sentiment():
    """Endpoint para análisis de sentimiento"""
    start_time = time.time()
    
    try:
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({"error": "Text field is required"}), 400
        
        text = data['text']
        
        # Validar texto
        if len(text.strip()) == 0:
            return jsonify({"error": "Text cannot be empty"}), 400
        
        if len(text) > 1000:
            return jsonify({"error": "Text too long (max 1000 characters)"}), 400
        
        # Preprocesar texto
        processed_text = text_processor.preprocess(text)
        
        # Realizar análisis
        result = sentiment_analyzer.analyze(processed_text)
        
        # Agregar metadata
        result.update({
            "original_text": text,
            "processed_text": processed_text,
            "timestamp": datetime.now().isoformat(),
            "processing_time_ms": (time.time() - start_time) * 1000
        })
        
        # Emitir resultado a través de WebSocket
        socketio.emit('sentiment_result', result)
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Analysis error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/analyze-batch', methods=['POST'])
def analyze_batch():
    """Endpoint para análisis batch"""
    start_time = time.time()
    
    try:
        data = request.get_json()
        
        if not data or 'texts' not in data:
            return jsonify({"error": "Texts field is required"}), 400
        
        texts = data['texts']
        
        if len(texts) > 100:
            return jsonify({"error": "Too many texts (max 100)"}), 400
        
        results = []
        
        for i, text in enumerate(texts):
            try:
                processed_text = text_processor.preprocess(text)
                result = sentiment_analyzer.analyze(processed_text)
                result.update({
                    "id": i,
                    "original_text": text,
                    "processed_text": processed_text
                })
                results.append(result)
            except Exception as e:
                results.append({
                    "id": i,
                    "error": str(e),
                    "original_text": text
                })
        
        response = {
            "results": results,
            "total_texts": len(texts),
            "successful_analyses": len([r for r in results if 'error' not in r]),
            "processing_time_ms": (time.time() - start_time) * 1000
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Batch analysis error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/metrics')
def metrics():
    """Endpoint para métricas de Prometheus"""
    from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
    return generate_latest(), 200, {'Content-Type': CONTENT_TYPE_LATEST}

# Eventos WebSocket
@socketio.on('connect')
def handle_connect():
    """Manejar conexión de cliente"""
    logger.info(f"Client connected: {request.sid}")
    REAL_TIME_CONNECTIONS.inc()
    emit('connected', {'message': 'Connected to sentiment analysis server'})

@socketio.on('disconnect')
def handle_disconnect():
    """Manejar desconexión de cliente"""
    logger.info(f"Client disconnected: {request.sid}")
    REAL_TIME_CONNECTIONS.dec()

@socketio.on('analyze_realtime')
def handle_realtime_analysis(data):
    """Manejar análisis en tiempo real"""
    try:
        text = data.get('text', '')
        
        if len(text.strip()) == 0:
            emit('error', {'message': 'Text cannot be empty'})
            return
        
        # Preprocesar y analizar
        processed_text = text_processor.preprocess(text)
        result = sentiment_analyzer.analyze(processed_text)
        
        result.update({
            "original_text": text,
            "processed_text": processed_text,
            "timestamp": datetime.now().isoformat(),
            "client_id": request.sid
        })
        
        emit('realtime_result', result)
        
    except Exception as e:
        logger.error(f"Realtime analysis error: {e}")
        emit('error', {'message': str(e)})

# Middleware para métricas
@app.before_request
def before_request():
    request.start_time = time.time()

@app.after_request
def after_request(response):
    if hasattr(request, 'start_time'):
        REQUEST_LATENCY.observe(time.time() - request.start_time)
        REQUEST_COUNT.labels(
            method=request.method,
            endpoint=request.endpoint,
            status=response.status_code
        ).inc()
    return response

if __name__ == '__main__':
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)
