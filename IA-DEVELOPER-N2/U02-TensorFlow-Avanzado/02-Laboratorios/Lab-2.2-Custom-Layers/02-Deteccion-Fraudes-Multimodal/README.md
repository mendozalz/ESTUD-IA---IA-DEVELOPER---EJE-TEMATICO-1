# Detección de Fraudes Multimodal

## 🌐 **PROYECTO RENDERIZADO - VER EN LÍNEA**
### **[🔗 VER PROTOTIPO REALIZADO CON STUDIO IA](https://ai.studio/apps/b8ffd6bd-c45d-46d5-8b67-9f8f7858cd54)**
*Acceso interactivo para estudiantes - Prueba el sistema de detección de fraudes multimodal*

---

## 📋 Descripción

Este ejercicio implementa un sistema de detección de fraudes utilizando múltiples modalidades de datos:
- **Texto**: Descripciones de transacciones
- **Imágenes**: Cheques escaneados o documentos
- **Grafos**: Patrones de transacciones y relaciones

## 🎯 Objetivos

- Implementar capas personalizadas para fusión multimodal
- Desarrollar mecanismos de atención para diferentes modalidades
- Crear sistema de detección de fraudes robusto
- Evaluar performance con métricas apropiadas

## 📁 Archivos

### **Versión Simplificada (Recomendada)**
- `simple_multimodal_detector.py`: Implementación completa sin dependencias pesadas
- `requirements_simple.txt`: Dependencias mínimas (TensorFlow, NumPy, Matplotlib)

### **Versión Completa**
- `multimodal_fusion_layer.py`: Implementación con SHAP y análisis avanzado
- `requirements.txt`: Todas las dependencias incluyendo SHAP

## 🚀 Ejecución

### **Opción 1: Versión Simplificada (Recomendada)**
```bash
# Instalar dependencias mínimas
pip install -r requirements_simple.txt

# Ejecutar detector
python simple_multimodal_detector.py
```

### **Opción 2: Versión Completa**
```bash
# Instalar todas las dependencias
pip install -r requirements.txt

# Ejecutar versión completa
python multimodal_fusion_layer.py
```

## 📊 Características del Sistema

### **Arquitectura Multimodal**
- **Fusión con Atención**: Pesos aprendidos para cada modalidad
- **Procesamiento Paralelo**: Cada modalidad se procesa independientemente
- **Clasificación Binaria**: Fraude vs Normal

### **Mecanismos de Atención**
- **Atención Ponderada**: Da más importancia a modalidades relevantes
- **Aprendizaje Adaptativo**: Los pesos se ajustan durante entrenamiento
- **Interpretabilidad**: Se puede analizar qué modalidad contribuye más

## 📈 Métricas de Evaluación

### **Métricas de Clasificación**
- **Accuracy**: Proporción de predicciones correctas
- **Precision**: Verdaderos positivos / (Verdaderos + Falsos positivos)
- **Recall**: Verdaderos positivos / (Verdaderos + Falsos negativos)
- **F1-Score**: Media armónica de precision y recall

### **Métricas de Negocio**
- **False Positive Rate**: Costo de investigar transacciones legítimas
- **False Negative Rate**: Pérdidas por fraudes no detectados
- **ROI**: Ahorro generado por el sistema

## 🛠️ Implementación Técnica

### **Capa de Fusión Personalizada**
```python
class SimpleMultimodalFusion(layers.Layer):
    def call(self, inputs):
        text_features, image_features, graph_features = inputs
        
        # Procesar cada modalidad
        text_out = self.text_dense(text_features)
        image_out = self.image_dense(image_features)
        graph_out = self.graph_dense(graph_features)
        
        # Fusión con atención
        attention_weights = self.attention(tf.reduce_mean(combined, axis=-1))
        fused = (attention_weights[:, 0] * text_out + 
                 attention_weights[:, 1] * image_out + 
                 attention_weights[:, 2] * graph_out)
        
        return fused
```

### **Flujo de Datos**
1. **Entrada Multimodal**: 3 inputs separados
2. **Procesamiento Individual**: Dense/CNN para cada modalidad
3. **Fusión con Atención**: Combinación ponderada inteligente
4. **Clasificación Final**: Output binario con sigmoid

## 🎯 Casos de Uso

### **Sector Financiero**
- Detección de fraudes en tarjetas de crédito
- Análisis de transacciones sospechosas
- Verificación de identidad con documentos

### **Seguros**
- Detección de reclamaciones fraudulentas
- Análisis de documentos de seguro
- Identificación de patrones anómalos

### **E-commerce**
- Detección de cuentas falsas
- Análisis de comportamiento de compra
- Prevención de fraudes en pagos

## 📝 Resultados Esperados

### **Métricas de Performance**
- **Accuracy**: >90%
- **Precision**: >85%
- **Recall**: >80%
- **F1-Score**: >82%

### **Métricas de Sistema**
- **Tiempo de inferencia**: <100ms por transacción
- **Throughput**: >100 transacciones/segundo
- **Memoria**: <1GB RAM

## 🔧 Solución de Problemas Comunes

### **Error: Ruta demasiado larga**
**Solución**: Usar `requirements_simple.txt` y ejecutar desde directorio corto

### **Error: TensorFlow version**
**Solución**: Usar versión >=2.16.0 (incluida en requirements_simple.txt)

### **Error: SHAP installation**
**Solución**: Usar versión simplificada sin SHAP (`simple_multimodal_detector.py`)

---

**Este sistema proporciona una base sólida para detección de fraudes multimodal con TensorFlow.**
