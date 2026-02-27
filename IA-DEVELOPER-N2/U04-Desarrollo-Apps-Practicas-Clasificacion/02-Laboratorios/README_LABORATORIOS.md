# U04 - Laboratorios de Desarrollo de Apps de Clasificación

## 📋 Visión General

Los laboratorios de la Unidad 4 están diseñados para desarrollar aplicaciones prácticas de clasificación utilizando técnicas avanzadas de IA. Cada laboratorio se enfoca en un dominio específico para construir soluciones completas y funcionales del mundo real.

## 🎯 Metodología de Marco Lógico

Cada laboratorio sigue la metodología del marco lógico para garantizar:

- **Claridad de objetivos** y resultados esperados
- **Medición sistemática** del progreso
- **Verificación objetiva** de logros
- **Identificación de supuestos críticos**

## 🏗️ Estructura de Laboratorios

### **Lab-4.1-Medical-Images**
- **Objetivo**: Clasificación de imágenes médicas
- **Contenido**: Diagnóstico asistido por IA
- **Técnicas**: CNN, Transfer Learning, Grad-CAM
- **Duración estimada**: 8-10 horas

### **Lab-4.2-Fraud-Detection**
- **Objetivo**: Detección de fraudes financieros
- **Contenido**: Análisis de transacciones anómalas
- **Técnicas**: Grafos, LSTM, Ensemble
- **Duración estimada**: 6-8 horas

### **Lab-4.3-Ecommerce-Classification**
- **Objetivo**: Clasificación de productos e-commerce
- **Contenido**: Categorización automática de productos
- **Técnicas**: NLP, Computer Vision, Multi-modal
- **Duración estimada**: 6-8 horas

### **Lab-4.4-Sentiment-Analysis**
- **Objetivo**: Análisis de sentimientos en texto
- **Contenido**: Clasificación de opiniones y reseñas
- **Técnicas**: BERT, Transformers, Fine-tuning
- **Duración estimada**: 4-6 horas

### **Lab-4.5-Realtime-Classification**
- **Objetivo**: Clasificación en tiempo real
- **Contenido**: Sistema de clasificación streaming
- **Técnicas**: Kafka, FastAPI, Modelos ligeros
- **Duración estimada**: 6-8 horas

## 📊 Contenido Detallado

### **Lab-4.1-Medical-Images**

#### **Módulo 1: Preprocesamiento de Imágenes Médicas**
- **Objetivo**: Procesar imágenes médicas (X-rays, MRI, CT)
- **Tareas**: Normalización, augmentación, segmentación
- **Archivo**: `1-medical_preprocessing.py`

#### **Módulo 2: Modelo CNN para Diagnóstico**
- **Objetivo**: Construir modelo de clasificación médica
- **Tareas**: Arquitectura CNN, transfer learning
- **Archivo**: `2-medical_cnn.py`

#### **Módulo 3: Explicabilidad con Grad-CAM**
- **Objetivo**: Implementar explicabilidad visual
- **Tareas**: Grad-CAM, visualización de atención
- **Archivo**: `3-gradcam_explanation.py`

#### **Módulo 4: API FastAPI para Diagnóstico**
- **Objetivo**: Desarrollar servicio web
- **Tareas**: API REST, validación, seguridad
- **Archivo**: `4-medical_api.py`

### **Lab-4.2-Fraud-Detection**

#### **Módulo 1: Análisis de Transacciones**
- **Objetivo**: Procesar datos financieros
- **Tareas**: Feature engineering, análisis temporal
- **Archivo**: `1-transaction_analysis.py`

#### **Módulo 2: Grafos de Relaciones**
- **Objetivo**: Construir grafos de entidades
- **Tareas**: Grafos de transacciones, detección de patrones
- **Archivo**: `2-fraud_graphs.py`

#### **Módulo 3: Modelo LSTM para Series**
- **Objetivo**: Detectar patrones temporales
- **Tareas**: LSTM, secuencias, anomalías
- **Archivo**: `3-lstm_fraud.py`

#### **Módulo 4: Sistema Ensemble**
- **Objetivo**: Combinar múltiples modelos
- **Tareas**: Ensemble, voting, stacking
- **Archivo**: `4-ensemble_fraud.py`

### **Lab-4.3-Ecommerce-Classification**

#### **Módulo 1: Procesamiento Multi-modal**
- **Objetivo**: Procesar imágenes y texto
- **Tareas**: Feature extraction, fusión de datos
- **Archivo**: `1-multimodal_processing.py`

#### **Módulo 2: Clasificación de Imágenes**
- **Objetivo**: Categorizar productos por imagen
- **Tareas**: CNN, fine-tuning, embeddings
- **Archivo**: `2-image_classification.py`

#### **Módulo 3: Clasificación de Texto**
- **Objetivo**: Categorizar por descripción
- **Tareas**: NLP, embeddings, clasificación
- **Archivo**: `3-text_classification.py`

#### **Módulo 4: Fusión Multi-modal**
- **Objetivo**: Combinar imágenes y texto
- **Tareas**: Fusión tardía, atención, ensemble
- **Archivo**: `4-multimodal_fusion.py`

### **Lab-4.4-Sentiment-Analysis**

#### **Módulo 1: Preprocesamiento de Texto**
- **Objetivo**: Limpiar y preparar texto
- **Tareas**: Tokenización, limpieza, normalización
- **Archivo**: `1-text_preprocessing.py`

#### **Módulo 2: Modelo BERT Base**
- **Objetivo**: Implementar BERT para clasificación
- **Tareas**: Fine-tuning, embeddings, clasificación
- **Archivo**: `2-bert_model.py`

#### **Módulo 3: Transformers Avanzados**
- **Objetivo**: Explorar arquitecturas transformers
- **Tareas**: RoBERTa, DistilBERT, comparación
- **Archivo**: `3-advanced_transformers.py`

#### **Módulo 4: Análisis de Aspectos**
- **Objetivo**: Detectar sentimientos por aspectos
- **Tareas**: ABSA, multi-label classification
- **Archivo**: `4-aspect_analysis.py`

### **Lab-4.5-Realtime-Classification**

#### **Módulo 1: Streaming con Kafka**
- **Objetivo**: Implementar streaming de datos
- **Tareas**: Producer, consumer, topics
- **Archivo**: `1-kafka_streaming.py`

#### **Módulo 2: Modelos Ligeros**
- **Objetivo**: Optimizar para inferencia rápida
- **Tareas**: Modelos comprimidos, cuantización
- **Archivo**: `2-lightweight_models.py`

#### **Módulo 3: API en Tiempo Real**
- **Objetivo**: Desarrollar API streaming
- **Tareas**: FastAPI, WebSocket, async
- **Archivo**: `3-realtime_api.py`

#### **Módulo 4: Monitoreo y Escalabilidad**
- **Objetivo**: Implementar monitoreo activo
- **Tareas**: Métricas, alertas, auto-scaling
- **Archivo**: `4_monitoring_scaling.py`

## 🔧 Requisitos Técnicos

### **Software Requerido**
- Python 3.8+
- TensorFlow 2.12+
- FastAPI
- Kafka
- Docker

### **Hardware Recomendado**
- CPU: 8+ cores
- RAM: 16GB+ (32GB recomendado)
- GPU: NVIDIA con CUDA (recomendado)
- Almacenamiento: 20GB disponibles

### **Dependencias Principales**
```bash
pip install tensorflow==2.12.0
pip install fastapi==0.104.0
pip install uvicorn==0.24.0
pip install kafka-python==2.0.2
pip install transformers==4.35.0
pip install spektral==1.2.0
pip install opencv-python==4.8.0
pip install scikit-learn==1.3.0
```

## 📈 Secuencia de Aprendizaje

### **Fase 1: Visión por Computadora (Lab-4.1)**
- Procesamiento de imágenes médicas
- Modelos CNN y explicabilidad
- **Tiempo estimado**: 10 horas

### **Fase 2: Datos Estructurados (Lab-4.2)**
- Análisis de transacciones
- Grafos y series temporales
- **Tiempo estimado**: 8 horas

### **Fase 3: Multi-modal (Lab-4.3)**
- Fusión de imágenes y texto
- Clasificación multi-modal
- **Tiempo estimado**: 8 horas

### **Fase 4: NLP (Lab-4.4)**
- Procesamiento de lenguaje natural
- Transformers y BERT
- **Tiempo estimado**: 6 horas

### **Fase 5: Tiempo Real (Lab-4.5)**
- Streaming y APIs
- Optimización y monitoreo
- **Tiempo estimado**: 8 horas

## 🎯 Criterios de Evaluación

### **Comprensión Teórica (25%)**
- Explicación clara de técnicas de clasificación
- Justificación de arquitecturas
- Comprensión de dominios específicos

### **Implementación Práctica (50%)**
- Aplicaciones funcionales y completas
- Uso correcto de APIs y frameworks
- Integración de componentes

### **Resultados y Métricas (25%)**
- Modelos con buen rendimiento
- APIs funcionales y eficientes
- Sistemas escalables

## 📚 Recursos Adicionales

### **Documentación Oficial**
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Kafka Documentation](https://kafka.apache.org/documentation/)
- [Transformers Documentation](https://huggingface.co/docs/transformers/)

### **Recursos de Aprendizaje**
- Casos de uso médicos
- Ejemplos de fintech
- Tutoriales de e-commerce

### **Herramientas Útiles**
- Postman para testing de APIs
- Docker para contenerización
- Cloud services para despliegue

## 🚀 Tips para el Éxito

### **Antes de Empezar**
- Entender dominios específicos
- Configurar entorno de desarrollo
- Preparar datasets relevantes

### **Durante los Ejercicios**
- Validar resultados en cada etapa
- Documentar APIs completamente
- Considerar aspectos de seguridad

### **Para Profundizar**
- Explorar casos de uso reales
- Optimizar para producción
- Implementar pruebas automatizadas

## 📞 Soporte y Ayuda

### **Recursos Internos**
- Foros de discusión de aplicaciones
- Sesiones de code review
- Tutorías especializadas

### **Recursos Externos**
- Comunidades de FastAPI
- Stack Overflow ML
- Documentación médica y financiera

---

**¡Estos laboratorios te permitirán construir aplicaciones de clasificación completas y profesionales!**
