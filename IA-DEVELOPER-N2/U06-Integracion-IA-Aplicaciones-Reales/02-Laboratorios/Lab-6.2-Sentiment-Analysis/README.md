# Laboratorio 6.2: Análisis de Sentimiento con Vue.js y Flask

## 🎯 Objetivos del Laboratorio

### Objetivo General
Crear una aplicación de análisis de sentimiento en tiempo real usando Flask para el backend y Vue.js para el frontend, implementando WebSockets para comunicación bidireccional.

### Objetivos Específicos
- Desarrollar una API RESTful con Flask para análisis de sentimiento
- Crear una interfaz interactiva con Vue.js 3 Composition API
- Implementar WebSockets para análisis en tiempo real
- Aplicar metodología Windsor para gestión del proyecto
- Configurar monitoreo con Prometheus y Grafana

## 📋 Marco Lógico del Proyecto

| Componente | Objetivo | Indicadores | Medios de Verificación |
|------------|----------|-------------|------------------------|
| **API Flask** | Servir modelo de análisis de sentimiento | Endpoints funcionales con <100ms latencia | Código app.py con tests unitarios |
| **Frontend Vue.js** | Interfaz para análisis en tiempo real | Componentes reutilizables con reactividad | Código Vue.js con Composition API |
| **WebSockets** | Comunicación bidireccional en tiempo real | <50ms latencia en mensajes | Implementación socket.io |
| **Monitoreo** | Observabilidad de la aplicación | Métricas recolectadas y dashboards | Configuración Prometheus/Grafana |

## 🛠️ Tecnologías y Herramientas

### Principales Tecnologías
- **Flask 3.0**: Framework web ligero y flexible
- **Vue.js 3.3**: Framework progresivo de JavaScript
- **Flask-SocketIO**: WebSockets para comunicación en tiempo real
- **Transformers 4.30**: Modelo BERT para análisis de sentimiento
- **Redis 7.0**: Caching y manejo de sesiones

### Dependencias Adicionales
- **Prometheus 3.0**: Monitoreo de métricas
- **Grafana 10.0**: Visualización de dashboards
- **Docker 24.0**: Contenerización
- **Nginx**: Servidor web y proxy reverso

## 📁 Estructura del Proyecto

```
Laboratorio 6.2 - Análisis de Sentimiento/
├── README.md                           # Guía del laboratorio
├── windsor_plan.md                    # Planificación Windsor
├── requirements.txt                    # Dependencias Python
├── docker-compose.yml                 # Configuración local
├── Dockerfile                          # Imagen Docker
├── backend/                           # API Flask
│   ├── app.py                         # Aplicación principal
│   ├── models/                        # Modelo de análisis
│   │   ├── __init__.py
│   │   └── sentiment_analyzer.py      # Analizador de sentimiento
│   ├── utils/                         # Utilidades
│   │   ├── __init__.py
│   │   ├── text_processing.py         # Procesamiento de texto
│   │   └── metrics.py                 # Métricas Prometheus
│   └── tests/                         # Tests
│       ├── test_api.py
│       └── test_models.py
├── frontend/                          # Aplicación Vue.js
│   ├── package.json                   # Dependencias Node.js
│   ├── vite.config.js                 # Configuración Vite
│   ├── public/                        # Archivos estáticos
│   ├── src/                           # Código fuente
│   │   ├── components/                # Componentes Vue
│   │   │   ├── SentimentAnalyzer.vue   # Analizador principal
│   │   │   ├── RealTimeChart.vue      # Gráfico en tiempo real
│   │   │   └── MetricsDashboard.vue   # Dashboard de métricas
│   │   ├── composables/               # Composables
│   │   │   ├── useSentiment.js        # Lógica de análisis
│   │   │   └── useWebSocket.js        # Lógica WebSocket
│   │   ├── services/                  # Servicios API
│   │   │   └── api.js                 # Cliente HTTP
│   │   ├── App.vue                    # Componente principal
│   │   └── main.js                    # Punto de entrada
│   └── tests/                         # Tests Vue
└── monitoring/                        # Configuración monitoreo
    ├── prometheus/
    └── grafana/
```

## 🔧 Implementación Detallada

### Fase 1: API Flask Completa

#### backend/app.py:
```python
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
```

### Fase 2: Modelo de Análisis de Sentimiento

#### backend/models/sentiment_analyzer.py:
```python
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

class SentimentAnalyzer:
    def __init__(self, model_name: str = "cardiffnlp/twitter-roberta-base-sentiment-latest"):
        self.model_name = model_name
        self.pipeline = None
        self.tokenizer = None
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def load_model(self) -> bool:
        """Cargar el modelo de análisis de sentimiento"""
        try:
            logger.info(f"Loading sentiment model: {self.model_name}")
            
            # Cargar tokenizer y modelo
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            
            # Crear pipeline
            self.pipeline = pipeline(
                "sentiment-analysis",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if torch.cuda.is_available() else -1,
                return_all_scores=True
            )
            
            logger.info("Model loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False
    
    def is_loaded(self) -> bool:
        """Verificar si el modelo está cargado"""
        return self.pipeline is not None
    
    def analyze(self, text: str) -> Dict:
        """Analizar sentimiento del texto"""
        if not self.is_loaded():
            if not self.load_model():
                raise RuntimeError("Failed to load sentiment model")
        
        try:
            # Realizar análisis
            results = self.pipeline(text)
            
            # Procesar resultados
            scores = {result['label'].lower(): result['score'] for result in results[0]}
            
            # Determinar sentimiento principal
            sentiment = max(scores, key=scores.get)
            confidence = scores[sentiment]
            
            # Mapear etiquetas
            sentiment_mapping = {
                'positive': 'positivo',
                'negative': 'negativo', 
                'neutral': 'neutral'
            }
            
            mapped_sentiment = sentiment_mapping.get(sentiment, sentiment)
            
            # Calcular métricas adicionales
            polarity = self._calculate_polarity(scores)
            subjectivity = self._calculate_subjectivity(scores)
            
            return {
                'sentiment': mapped_sentiment,
                'confidence': confidence,
                'scores': scores,
                'polarity': polarity,
                'subjectivity': subjectivity,
                'model_name': self.model_name
            }
            
        except Exception as e:
            logger.error(f"Analysis error: {e}")
            raise
    
    def _calculate_polarity(self, scores: Dict) -> float:
        """Calcular polaridad (-1 a 1)"""
        positive = scores.get('positive', 0)
        negative = scores.get('negative', 0)
        
        if positive + negative == 0:
            return 0.0
        
        return (positive - negative) / (positive + negative)
    
    def _calculate_subjectivity(self, scores: Dict) -> float:
        """Calcular subjetividad (0 a 1)"""
        neutral = scores.get('neutral', 0)
        return 1.0 - neutral
    
    def get_model_info(self) -> Dict:
        """Obtener información del modelo"""
        return {
            'model_name': self.model_name,
            'device': str(self.device),
            'is_loaded': self.is_loaded(),
            'labels': ['positive', 'negative', 'neutral'] if self.is_loaded() else []
        }
```

### Fase 3: Frontend Vue.js Completo

#### frontend/src/components/SentimentAnalyzer.vue:
```vue
<template>
  <div class="sentiment-analyzer">
    <div class="analyzer-header">
      <h2>Análisis de Sentimiento en Tiempo Real</h2>
      <div class="connection-status">
        <span :class="['status-indicator', isConnected ? 'connected' : 'disconnected']"></span>
        {{ isConnected ? 'Conectado' : 'Desconectado' }}
      </div>
    </div>

    <div class="input-section">
      <div class="text-input">
        <textarea
          v-model="inputText"
          placeholder="Escribe el texto para analizar..."
          :disabled="isAnalyzing"
          @input="handleTextInput"
          rows="4"
          maxlength="1000"
        ></textarea>
        <div class="input-footer">
          <span class="char-count">{{ inputText.length }}/1000</span>
          <button 
            @click="analyzeText" 
            :disabled="!inputText.trim() || isAnalyzing"
            class="analyze-btn"
          >
            {{ isAnalyzing ? 'Analizando...' : 'Analizar' }}
          </button>
        </div>
      </div>
    </div>

    <div v-if="currentResult" class="result-section">
      <SentimentResult :result="currentResult" />
    </div>

    <div class="realtime-section">
      <h3>Análisis en Tiempo Real</h3>
      <div class="realtime-controls">
        <button 
          @click="toggleRealtime" 
          :class="['realtime-btn', realtimeEnabled ? 'active' : '']"
        >
          {{ realtimeEnabled ? 'Detener' : 'Iniciar' }} Análisis en Tiempo Real
        </button>
      </div>
      
      <div v-if="realtimeEnabled" class="realtime-input">
        <input
          v-model="realtimeText"
          placeholder="Escribe para análisis en tiempo real..."
          @input="handleRealtimeInput"
          class="realtime-text-input"
        />
      </div>
    </div>

    <div v-if="realtimeResults.length > 0" class="chart-section">
      <RealTimeChart :data="realtimeResults" />
    </div>

    <div class="history-section">
      <h3>Historial de Análisis</h3>
      <div class="history-list">
        <div 
          v-for="(result, index) in analysisHistory" 
          :key="index"
          class="history-item"
        >
          <div class="history-text">{{ result.original_text }}</div>
          <div class="history-result">
            <span :class="['sentiment-badge', result.sentiment]">
              {{ result.sentiment }}
            </span>
            <span class="confidence">{{ (result.confidence * 100).toFixed(1) }}%</span>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, reactive, onMounted, onUnmounted } from 'vue'
import { useSentiment } from '../composables/useSentiment'
import { useWebSocket } from '../composables/useWebSocket'
import SentimentResult from './SentimentResult.vue'
import RealTimeChart from './RealTimeChart.vue'

// Estado reactivo
const inputText = ref('')
const realtimeText = ref('')
const realtimeEnabled = ref(false)
const isAnalyzing = ref(false)
const currentResult = ref(null)
const realtimeResults = ref([])
const analysisHistory = ref([])

// Composables
const { analyzeSentiment, isAnalyzing: analyzing } = useSentiment()
const { 
  isConnected, 
  connect, 
  disconnect, 
  sendRealtimeAnalysis,
  onRealtimeResult 
} = useWebSocket()

// Manejar análisis de texto
const analyzeText = async () => {
  if (!inputText.value.trim()) return
  
  isAnalyzing.value = true
  
  try {
    const result = await analyzeSentiment(inputText.value)
    currentResult.value = result
    analysisHistory.value.unshift(result)
    
    // Limitar historial a 50 elementos
    if (analysisHistory.value.length > 50) {
      analysisHistory.value = analysisHistory.value.slice(0, 50)
    }
    
  } catch (error) {
    console.error('Analysis error:', error)
  } finally {
    isAnalyzing.value = false
  }
}

// Manejar input de texto
const handleTextInput = () => {
  if (realtimeEnabled.value) {
    realtimeText.value = inputText.value
    handleRealtimeInput()
  }
}

// Toggle análisis en tiempo real
const toggleRealtime = () => {
  realtimeEnabled.value = !realtimeEnabled.value
  
  if (realtimeEnabled.value) {
    realtimeText.value = inputText.value
  } else {
    realtimeText.value = ''
  }
}

// Manejar input en tiempo real
const handleRealtimeInput = () => {
  if (realtimeEnabled.value && realtimeText.value.trim()) {
    sendRealtimeAnalysis(realtimeText.value)
  }
}

// Escuchar resultados en tiempo real
onRealtimeResult((result) => {
  realtimeResults.value.push(result)
  
  // Limitar resultados a 100 elementos
  if (realtimeResults.value.length > 100) {
    realtimeResults.value = realtimeResults.value.slice(-100)
  }
})

// Ciclo de vida
onMounted(() => {
  connect()
})

onUnmounted(() => {
  disconnect()
})
</script>

<style scoped>
.sentiment-analyzer {
  max-width: 800px;
  margin: 0 auto;
  padding: 20px;
}

.analyzer-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 20px;
}

.connection-status {
  display: flex;
  align-items: center;
  gap: 8px;
}

.status-indicator {
  width: 10px;
  height: 10px;
  border-radius: 50%;
}

.status-indicator.connected {
  background-color: #52c41a;
}

.status-indicator.disconnected {
  background-color: #ff4d4f;
}

.text-input {
  margin-bottom: 20px;
}

.text-input textarea {
  width: 100%;
  padding: 12px;
  border: 1px solid #d9d9d9;
  border-radius: 6px;
  font-size: 14px;
  resize: vertical;
}

.input-footer {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-top: 8px;
}

.char-count {
  color: #666;
  font-size: 12px;
}

.analyze-btn {
  padding: 8px 16px;
  background-color: #1890ff;
  color: white;
  border: none;
  border-radius: 4px;
  cursor: pointer;
}

.analyze-btn:disabled {
  background-color: #d9d9d9;
  cursor: not-allowed;
}

.result-section {
  margin-bottom: 30px;
}

.realtime-section {
  margin-bottom: 30px;
  padding: 20px;
  background-color: #f5f5f5;
  border-radius: 6px;
}

.realtime-controls {
  margin-bottom: 16px;
}

.realtime-btn {
  padding: 8px 16px;
  background-color: #d9d9d9;
  border: none;
  border-radius: 4px;
  cursor: pointer;
}

.realtime-btn.active {
  background-color: #52c41a;
  color: white;
}

.realtime-text-input {
  width: 100%;
  padding: 8px;
  border: 1px solid #d9d9d9;
  border-radius: 4px;
}

.chart-section {
  margin-bottom: 30px;
}

.history-section {
  margin-top: 30px;
}

.history-list {
  max-height: 300px;
  overflow-y: auto;
}

.history-item {
  padding: 12px;
  border-bottom: 1px solid #f0f0f0;
}

.history-text {
  font-size: 14px;
  margin-bottom: 8px;
}

.history-result {
  display: flex;
  gap: 12px;
  align-items: center;
}

.sentiment-badge {
  padding: 4px 8px;
  border-radius: 4px;
  font-size: 12px;
  font-weight: bold;
}

.sentiment-badge.positivo {
  background-color: #f6ffed;
  color: #52c41a;
}

.sentiment-badge.negativo {
  background-color: #fff2f0;
  color: #ff4d4f;
}

.sentiment-badge.neutral {
  background-color: #f0f0f0;
  color: #666;
}

.confidence {
  font-size: 12px;
  color: #666;
}
</style>
```

## 📊 Entregables del Laboratorio

### 1. Código Fuente Completo
- **windsor_plan.md**: Planificación detallada
- **backend/app.py**: API Flask con WebSockets
- **backend/models/sentiment_analyzer.py**: Modelo BERT
- **frontend/src/components/**: Componentes Vue.js completos

### 2. Configuración de Despliegue
- **Dockerfile**: Imagen Docker optimizada
- **docker-compose.yml**: Configuración local
- **nginx.conf**: Configuración del proxy reverso

### 3. Monitoreo y Observabilidad
- **monitoring/prometheus/**: Configuración Prometheus
- **monitoring/grafana/**: Dashboards de métricas

### 4. Tests y Documentación
- **backend/tests/**: Tests unitarios y de integración
- **frontend/tests/**: Tests de componentes Vue
- **docs/**: Documentación completa

---

**Duración Estimada**: 6-8 horas  
**Dificultad**: Intermedia  
**Prerrequisitos**: Conocimientos de Flask, Vue.js y WebSockets
