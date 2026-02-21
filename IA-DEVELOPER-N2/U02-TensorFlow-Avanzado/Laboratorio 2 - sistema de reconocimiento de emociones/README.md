# Laboratorio 2 - Sistema de Reconocimiento de Emociones

## Descripción
Implementación de un sistema de reconocimiento facial de emociones utilizando una red neuronal convolucional (CNN) con TensorFlow/Keras.

## Archivo principal
- `emociones.py`

## Cómo ejecutar
```bash
python emociones.py
```

## Prácticas sugeridas
- Ejecutar y entender el ejercicio `emociones.py`
- Ubicar en el código: generación/carga de datos, preprocesamiento de imágenes, arquitectura CNN, entrenamiento con callbacks y evaluación
- Modificar una parte del modelo (por ejemplo: `Dropout`, número de filtros o épocas) y comparar el `accuracy`

## Entregables
- Evidencia de ejecución (capturas o logs) mostrando:
  - División train/val/test
  - Entrenamiento (accuracy/loss)
  - Evaluación final (accuracy general y por clase)
- Breve reporte en Markdown (5-10 líneas) con:
  - Qué cambiaste
  - Qué impacto tuvo en el entrenamiento/validación

## Notas importantes
- Este ejercicio usa `tensorflow` y `opencv-python` (`cv2`)
- El script incluye una demo con datos sintéticos (en un caso real se usaría un dataset como FER2013)

## Conceptos que debes ubicar en el código
- Arquitectura CNN (capas `Conv2D`, `MaxPooling2D`, `BatchNormalization`, `Dropout`)
- Data augmentation con `ImageDataGenerator`
- Callbacks: `EarlyStopping` y `ReduceLROnPlateau`
- Predicción: `argmax` + confianza (`max(probabilidades)`)
