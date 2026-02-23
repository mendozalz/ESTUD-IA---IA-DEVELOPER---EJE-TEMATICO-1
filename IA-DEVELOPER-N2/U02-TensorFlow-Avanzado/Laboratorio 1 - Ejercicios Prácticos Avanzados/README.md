# Laboratorio 1 - Ejercicios Prácticos Avanzados

## 🎯 Objetivo del Laboratorio

Este laboratorio está diseñado para estudiantes de maestría con pregrados en Ingeniería de Sistemas, IA, Desarrollo de Software, Big Data e Ingeniería de Datos. Los ejercicios cubren técnicas avanzadas de TensorFlow aplicadas a problemas empresariales reales.

## 📋 Estructura del Laboratorio

### 🏭 **Ejercicio 1: Carga de Imágenes con tf.data**
**Caso de Uso**: Detección de defectos en placas de circuito impreso (PCB)
**Sector**: Manufactura electrónica
**Tecnologías**: CNN + tf.data + TensorFlow Lite
**Archivos**: `01-Carga-Imagenes-tfdata/README.md` y `ejercicio_completo.py`

### ⚡ **Ejercicio 2: Datos de Series Temporales con Ventanas Deslizantes**
**Caso de Uso**: Predicción de consumo energético en planta industrial
**Sector**: Energía
**Tecnologías**: LSTM + Keras Tuner + MQTT
**Archivos**: `02-Series-Temporales-Ventanas/README.md` y `ejercicio_completo.py`

### 🕸️ **Ejercicio 3: Datos de Grafos con Spektral**
**Caso de Uso**: Detección de fraudes en transacciones bancarias
**Sector**: Banca
**Tecnologías**: GNN + Spektral + FastAPI
**Archivos**: `03-Datos-Grafos-Spektral/README.md` y `ejercicio_completo.py`

### 📝 **Ejercicio 4: Modelo con Capa de Atención Personalizada para Clasificación de Texto**
**Caso de Uso**: Análisis de sentimientos en reseñas de productos
**Sector**: Marketing Digital
**Tecnologías**: BERT + atención personalizada + TF Serving
**Archivos**: `04-Atencion-Personalizada-Texto/README.md` y `ejercicio_completo.py`

### 🗺️ **Ejercicio 5: Modelo con Capas Personalizadas para Datos de Grafos**
**Caso de Uso**: Optimización de rutas en logística
**Sector**: Logística
**Tecnologías**: GAT personalizado + OpenStreetMap + OR-Tools
**Archivos**: `05-Grafos-Personalizados-Logistica/README.md` y `ejercicio_completo.py`

## 🚀 Flujo de Trabajo

1. **Configuración del Entorno**: Instalar dependencias necesarias
2. **Ejecución Individual**: Cada ejercicio es independiente
3. **Evaluación**: Métricas específicas por ejercicio
4. **Despliegue**: Prototipos funcionales para producción
5. **Documentación**: Informe técnico completo

## 📊 Métricas de Éxito

| Ejercicio | Métrica Principal | Objetivo |
|-----------|-------------------|-----------|
| 1 - Imágenes | Precisión/Recall | >95% |
| 2 - Series Temporales | MAE | <2% |
| 3 - Grafos | AUC-ROC | >0.95 |
| 4 - Texto | F1-Score | >0.85 |
| 5 - Grafos Personalizados | Reducción de distancia | >10% |

## 🎓 Habilidades Desarrolladas

- **Procesamiento de datos**: tf.data, pipelines eficientes
- **Arquitecturas avanzadas**: Capas personalizadas, atención
- **Optimización**: Hiperparámetros, mixed precision
- **Despliegue**: APIs, Docker, TensorFlow Serving
- **Integración**: Sistemas empresariales, IoT, APIs

## 📋 Requisitos Previos

- Python 3.8+
- TensorFlow 2.x
- Conocimientos básicos de Deep Learning
- Experiencia con APIs y bases de datos

## 🛠️ Instalación

```bash
pip install tensorflow tensorflow-datasets spektral transformers
pip install fastapi uvicorn paho-mqtt networkx
pip install keras-tuner opencv-python matplotlib
pip install scikit-learn pandas numpy
```

## 📝 Entrega

Cada estudiante debe entregar:
1. **Código funcional** para cada ejercicio
2. **Informe técnico** con arquitectura y resultados
3. **Prototipo desplegado** (Docker + API)
4. **Análisis de ROI** para el caso de uso específico

## 🏆 Evaluación

| Criterio | Peso | Descripción |
|----------|------|-------------|
| Funcionalidad | 30% | Código que ejecuta correctamente |
| Optimización | 20% | Uso eficiente de recursos |
| Documentación | 20% | Código bien documentado |
| Despliegue | 20% | Prototipo funcional en producción |
| Creatividad | 10% | Soluciones innovadoras |

---

**¡Comienza con el Ejercicio 1 y avanza secuencialmente!**
