# Laboratorio 4 - Tipos de Redes Neuronales

## Descripción
Exploración teórica y práctica de los 12 tipos principales de redes neuronales, sus características, composiciones y aplicaciones en diferentes paradigmas de aprendizaje.

## Objetivos
- Comprender los 12 tipos principales de redes neuronales
- Identificar sus características y componentes
- Implementar ejemplos prácticos en supervisado, no supervisado y refuerzo
- Relacionar cada tipo con problemas del mundo real

## Estructura del Laboratorio

### 1. Redes Neuronales Densas (Fully Connected)
- **Características:** Capas totalmente conectadas
- **Composición:** Input → Hidden Layers → Output
- **Librerías:** TensorFlow/Keras, PyTorch
- **Aplicaciones:** Clasificación, regresión

### 2. Redes Neuronales Convolucionales (CNN)
- **Características:** Convoluciones, pooling, jerarquía espacial
- **Composición:** Conv → Pool → FC
- **Librerías:** TensorFlow/Keras, PyTorch, OpenCV
- **Aplicaciones:** Imágenes, video, procesamiento espacial

### 3. Redes Neuronales Recurrentes (RNN)
- **Características:** Memoria temporal, secuencias
- **Composición:** Input → RNN Layers → Output
- **Librerías:** TensorFlow/Keras, PyTorch
- **Aplicaciones:** Series temporales, texto

### 4. Long Short-Term Memory (LSTM)
- **Características:** Memoria a largo plazo, puertas
- **Composición:** Input → LSTM → Output
- **Librerías:** TensorFlow/Keras, PyTorch
- **Aplicaciones:** Traducción, speech recognition

### 5. Gated Recurrent Units (GRU)
- **Características:** Simplificación de LSTM
- **Composición:** Input → GRU → Output
- **Librerías:** TensorFlow/Keras, PyTorch
- **Aplicaciones:** Procesamiento de lenguaje

### 6. Redes Neuronales Autoencoder
- **Características:** Compresión-descompresión, aprendizaje no supervisado
- **Composición:** Encoder → Latent → Decoder
- **Librerías:** TensorFlow/Keras, PyTorch
- **Aplicaciones:** Reducción dimensionalidad, anomalías

### 7. Generative Adversarial Networks (GAN)
- **Características:** Generador vs Discriminador, competencia
- **Composición:** Generator ↔ Discriminator
- **Librerías:** TensorFlow/Keras, PyTorch
- **Aplicaciones:** Generación de imágenes, data augmentation

### 8. Redes Neuronales de Grafos (GNN)
- **Características:** Procesamiento de grafos, nodos y aristas
- **Composición:** Graph Conv → Pool → Readout
- **Librerías:** PyTorch Geometric, DGL
- **Aplicaciones:** Redes sociales, moléculas

### 9. Transformer Networks
- **Características:** Mecanismo de atención, paralelización
- **Composición:** Embedding → Multi-Head Attention → FFN
- **Librerías:** Hugging Face, TensorFlow/Keras
- **Aplicaciones:** NLP, visión por computadora

### 10. Redes Neuronales Residuales (ResNet)
- **Características:** Skip connections, gradient flow
- **Composición:** Conv Blocks + Skip Connections
- **Librerías:** TensorFlow/Keras, PyTorch
- **Aplicaciones:** Deep learning profundo, imágenes

### 11. U-Net
- **Características:** Encoder-decoder simétrico, skip connections
- **Composición:** Encoder → Bottleneck → Decoder
- **Librerías:** TensorFlow/Keras, PyTorch
- **Aplicaciones:** Segmentación semántica, imágenes médicas

### 12. Deep Q-Networks (DQN)
- **Características:** Q-learning + redes profundas
- **Composición:** CNN → FC → Q-Values
- **Librerías:** TensorFlow/Keras, PyTorch, Gym
- **Aplicaciones:** Videojuegos, control, robótica

## Archivos del Laboratorio
- `teoria_redes_neuronales.py` - Explicación detallada de los 12 tipos
- `ejemplos_supervisado.py` - 3 ejemplos de aprendizaje supervisado
- `ejemplos_no_supervisado.py` - 3 ejemplos de aprendizaje no supervisado
- `ejemplos_reforzado.py` - 3 ejemplos de aprendizaje por refuerzo
- `comparacion_redes.py` - Análisis comparativo de rendimiento

## Cómo ejecutar
```bash
# Teoría y explicaciones
python teoria_redes_neuronales.py

# Ejemplos por paradigma
python ejemplos_supervisado.py
python ejemplos_no_supervisado.py
python ejemplos_reforzado.py

# Comparación
python comparacion_redes.py
```

## Entregables
- **Fase 1: Teoría**
  - Resumen de las 12 redes neuronales
  - Tabla comparativa de características
  - Diagramas de arquitectura

- **Fase 2: Implementación**
  - 9 ejemplos funcionales (3 por paradigma)
  - Código comentado explicando cada componente
  - Resultados y visualizaciones

- **Fase 3: Análisis**
  - Comparación de rendimiento
  - Ventajas/desventajas de cada tipo
  - Recomendaciones de uso por problema

## Evaluación
- Comprensión teórica (30%)
- Implementación correcta (40%)
- Análisis crítico (30%)
