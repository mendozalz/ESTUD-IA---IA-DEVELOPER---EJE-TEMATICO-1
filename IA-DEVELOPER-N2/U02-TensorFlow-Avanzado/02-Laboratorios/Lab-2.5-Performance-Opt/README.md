# Laboratorio 5 - Redes Neuronales Detalladas

## Descripción
Exploración profunda de cada uno de los 12 tipos de redes neuronales con explicación individual, ejercicios prácticos y adaptación al proyecto integrador.

## Metodología

### 🎯 1er Momento: Explicación Individual por Red Neuronal
Para cada uno de los 12 tipos de redes neuronales:
- **Explicación teórica completa**
- **Características y componentes específicos**
- **Ejercicio práctico de caso de uso real**
- **Identificación del paradigma** (supervisado, no supervisado, reforzado)

### 🚀 2do Momento: Adaptación al Proyecto Integrador
- **Apropiación del ejercicio** por los estudiantes
- **Adaptación a sus proyectos específicos**
- **Generación de 2 ejercicios adicionales** por cada tipo de red
- **Asegurar cobertura de los 2 tipos de aprendizaje** que faltan por cada red

## Estructura del Laboratorio

### 📁 Organización por Red Neuronal
```
Laboratorio 5 - Redes Neuronales Detalladas/
├── 01-Red-Densa/
├── 02-Red-CNN/
├── 03-Red-RNN/
├── 04-Red-LSTM/
├── 05-Red-GRU/
├── 06-Red-Autoencoder/
├── 07-Red-GAN/
├── 08-Red-GNN/
├── 09-Red-Transformer/
├── 10-Red-ResNet/
├── 11-Red-UNet/
└── 12-Red-DQN/
```

### 📋 Contenido por cada Red Neuronal
Cada carpeta contiene:
- `README.md` - Explicación teórica completa
- `ejemplo_basico.py` - Ejercicio principal de caso de uso
- `ejemplo_adaptado_1.py` - Primera adaptación estudiantil
- `ejemplo_adaptado_2.py` - Segunda adaptación estudiantil
- `recursos/` - Datasets y materiales adicionales

## 🧠 Las 12 Redes Neuronales

### 1. 🔗 Red Neuronal Densa (Fully Connected)
- **Paradigma principal:** Supervisado
- **Paradigpos secundarios:** No supervisado (Autoencoder), Reforzado (DQN base)
- **Casos de uso:** Clasificación tabular, regresión

### 2. 🖼️ Red Neuronal Convolucional (CNN)
- **Paradigma principal:** Supervisado
- **Paradigmas secundarios:** No supervisado (Segmentación), Reforzado (Deep Q-CNN)
- **Casos de uso:** Imágenes, video, procesamiento espacial

### 3. 🔄 Red Neuronal Recurrente (RNN)
- **Paradigma principal:** Supervisado
- **Paradigmas secundarios:** No supervisado (Predicción), Reforzado (Policy RNN)
- **Casos de uso:** Series temporales, texto básico

### 4. 🧠 Long Short-Term Memory (LSTM)
- **Paradigma principal:** Supervisado
- **Paradigmas secundarios:** No supervisado (Anomalías), Reforzado (LSTM-DDPG)
- **Casos de uso:** Traducción, speech recognition, series largas

### 5. 🚪 Gated Recurrent Units (GRU)
- **Paradigma principal:** Supervisado
- **Paradigmas secundarios:** No supervisado (Clustering secuencial), Reforzado (GRU-PPO)
- **Casos de uso:** Procesamiento de lenguaje eficiente

### 6. 🗜️ Autoencoder
- **Paradigma principal:** No supervisado
- **Paradigmas secundarios:** Supervisado (Clasificación con encoder), Reforzado (World Models)
- **Casos de uso:** Reducción dimensionalidad, anomalías

### 7. ⚔️ Generative Adversarial Networks (GAN)
- **Paradigma principal:** No supervisado
- **Paradigmas secundarios:** Supervisado (Semi-supervisado), Reforzado (GAN-RL)
- **Casos de uso:** Generación de imágenes, data augmentation

### 8. 🕸️ Redes Neuronales de Grafos (GNN)
- **Paradigma principal:** Supervisado
- **Paradigmas secundarios:** No supervisado (Clustering de grafos), Reforzado (Graph RL)
- **Casos de uso:** Redes sociales, moléculas, sistemas de recomendación

### 9. 🤖 Transformer Networks
- **Paradigma principal:** Supervisado
- **Paradigmas secundarios:** No supervisado (BERT pre-entrenamiento), Reforzado (Decision Transformer)
- **Casos de uso:** NLP, visión por computadora, multimodal

### 10. 🔀 Redes Neuronales Residuales (ResNet)
- **Paradigma principal:** Supervisado
- **Paradigmas secundarios:** No supervisado (ResNet-Autoencoder), Reforzado (ResNet-DQN)
- **Casos de uso:** Deep learning profundo, imágenes

### 11. 🏥 U-Net
- **Paradigma principal:** Supervisado
- **Paradigmas secundarios:** No supervisado (U-Net-GAN), Reforzado (U-Net-RL)
- **Casos de uso:** Segmentación semántica, imágenes médicas

### 12. 🎮 Deep Q-Networks (DQN)
- **Paradigma principal:** Reforzado
- **Paradigmas secundarios:** Supervisado (DQN-Clasificación), No supervisado (DQN-Clustering)
- **Casos de uso:** Videojuegos, control, robótica

## 🎯 Entregables

### Fase 1: Exploración Teórica
- **12 READMEs** con explicación detallada de cada red
- **12 ejercicios básicos** funcionales
- **Identificación correcta** de paradigmas para cada red

### Fase 2: Adaptación Proyectual
- **24 ejercicios adaptados** (2 por cada red)
- **Cobertura completa** de los 3 paradigmas por red
- **Documentación** de adaptaciones a proyectos específicos
- **Recursos generados** para proyectos integradores

### Fase 3: Análisis y Conclusiones
- **Comparación cruzada** entre redes y paradigmas
- **Recomendaciones** de uso por tipo de problema
- **Reflexión personal** sobre aprendizaje obtenido

## 🔧 Herramientas y Librerías

### Librerías Principales
- **TensorFlow/Keras** - Para la mayoría de redes
- **PyTorch** - Alternativa para redes complejas
- **PyTorch Geometric** - Para Graph Neural Networks
- **Hugging Face** - Para Transformers
- **OpenCV** - Para procesamiento de imágenes

### Herramientas de Visualización
- **Matplotlib/Seaborn** - Gráficos básicos
- **Plotly** - Visualizaciones interactivas
- **TensorBoard** - Monitoreo de entrenamiento

## 📈 Evaluación

### Criterios de Evaluación
- **Comprensión teórica** (30%)
- **Implementación correcta** (40%)
- **Adaptación creativa** (20%)
- **Documentación** (10%)

### Proyecto Final
- **Portfolio de implementaciones** de las 12 redes
- **Proyecto integrador** que combine múltiples tipos
- **Presentación técnica** de aprendizajes

## 🚀 Flujo de Trabajo Sugerido

1. **Semana 1-2:** Redes 1-4 (Dense, CNN, RNN, LSTM)
2. **Semana 3-4:** Redes 5-8 (GRU, Autoencoder, GAN, GNN)
3. **Semana 5-6:** Redes 9-12 (Transformer, ResNet, U-Net, DQN)
4. **Semana 7-8:** Adaptación y proyectos integradores
5. **Semana 9-10:** Integración final y presentaciones

## 🎓 Objetivos de Aprendizaje

Al finalizar este laboratorio, los estudiantes serán capaces de:
- **Explicar** cualquier tipo de red neuronal
- **Implementar** redes en los 3 paradigmas
- **Adaptar** soluciones a problemas reales
- **Combinar** múltiples tipos de redes en proyectos complejos
- **Evaluar** y seleccionar la arquitectura adecuada para cada problema
