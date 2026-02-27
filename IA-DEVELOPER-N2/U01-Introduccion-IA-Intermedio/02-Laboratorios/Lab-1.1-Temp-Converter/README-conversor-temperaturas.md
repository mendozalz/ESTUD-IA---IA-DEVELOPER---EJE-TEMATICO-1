# Laboratorio 1 - Conversor de Temperaturas con IA

## Descripción
Implementación de un conversor de temperaturas (Celsius a Fahrenheit) utilizando una red neuronal con TensorFlow/Keras.

## Archivo principal
- `conversor.py`

## Cómo ejecutar
```bash
python conversor.py
```

## Prácticas sugeridas
- Ejecutar y entender el ejercicio `conversor.py`
- Identificar en el código: preparación de datos, normalización, arquitectura, entrenamiento y evaluación
- Modificar hiperparámetros (épocas, `batch_size`, `learning_rate`) y observar impacto en métricas y curvas

## Entregables
- Evidencia de ejecución (capturas o logs) mostrando:
  - Entrenamiento (pérdida/MAE)
  - Evaluación (error absoluto/relativo)
  - Pruebas de conversión
- Breve reporte en Markdown (5-10 líneas) con conclusiones:
  - Qué tan bien aproximó la función de conversión
  - Qué cambio hiciste y qué mejoró/empeoró

## Conceptos que debes ubicar en el código
- Generación de dataset y etiquetas (aprendizaje supervisado)
- Normalización con `MinMaxScaler`
- Red neuronal densa (capas `Dense` + `Dropout`)
- Entrenamiento con `EarlyStopping`
- Comparación IA vs. fórmula tradicional
