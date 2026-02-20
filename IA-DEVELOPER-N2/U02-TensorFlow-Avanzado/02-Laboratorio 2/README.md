# U02 - Laboratorio

## Prácticas sugeridas

- Ejecutar y entender el ejercicio `emociones.py` (clasificación de emociones con CNN).
- Ubicar en el código: generación/carga de datos, preprocesamiento de imágenes, arquitectura CNN, entrenamiento con callbacks y evaluación.
- Modificar una parte del modelo (por ejemplo: `Dropout`, número de filtros o épocas) y comparar el `accuracy`.

## Entregables

- Evidencia de ejecución (capturas o logs) mostrando:
  - División train/val/test
  - Entrenamiento (accuracy/loss)
  - Evaluación final (accuracy general y por clase)
- Breve reporte en Markdown (5-10 líneas) con:
  - Qué cambiaste
  - Qué impacto tuvo en el entrenamiento/validación

## Archivo principal

- `emociones.py`

## Cómo ejecutar

Desde esta carpeta:

```bash
python emociones.py
```

## Notas importantes

- Este ejercicio usa `tensorflow` y `opencv-python` (`cv2`).
- El script incluye una demo con datos sintéticos (en un caso real se usaría un dataset como FER2013).

## Conceptos que debes ubicar en el código

- Arquitectura CNN (capas `Conv2D`, `MaxPooling2D`, `BatchNormalization`, `Dropout`).
- Data augmentation con `ImageDataGenerator`.
- Callbacks: `EarlyStopping` y `ReduceLROnPlateau`.
- Predicción: `argmax` + confianza (`max(probabilidades)`).