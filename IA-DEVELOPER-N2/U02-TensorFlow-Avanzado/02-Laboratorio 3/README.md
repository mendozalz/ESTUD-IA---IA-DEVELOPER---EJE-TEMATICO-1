# U02 - Laboratorio

## Prácticas sugeridas

- Ejecutar los 3 ejercicios (`Ej1.py`, `Ej2.py`, `Ej3.py`) y comparar paradigmas:
  - Supervisado (clasificación)
  - No supervisado (clustering)
  - Refuerzo (Q-learning)
- Identificar en cada ejercicio:
  - Preparación de datos
  - Entrenamiento
  - Métricas/visualizaciones
- Realizar 1 ajuste por ejercicio (parámetro o parte del flujo) y registrar el efecto.

## Entregables

- Evidencia de ejecución (capturas o logs) de los 3 ejercicios.
- Un reporte corto en Markdown con:
  - Qué hace cada ejercicio
  - 1 cambio realizado por ejercicio
  - Resultado observado (métrica/visualización) y conclusión

## Archivos

- `Ej1.py` (aprendizaje supervisado: Iris + Random Forest)
- `Ej2.py` (aprendizaje no supervisado: K-Means + métricas de clustering)
- `Ej3.py` (aprendizaje por refuerzo: Q-learning en laberinto)

## Cómo ejecutar

Desde esta carpeta:

```bash
python Ej1.py
python Ej2.py
python Ej3.py
```

## Qué observar en cada ejercicio

### Ejercicio 1 (Supervisado)

- Dataset Iris, EDA y visualizaciones.
- Preprocesamiento (train/test + `StandardScaler`).
- Entrenamiento y evaluación (accuracy, matriz de confusión, `classification_report`).

### Ejercicio 2 (No supervisado)

- Estandarización y búsqueda de `k`.
- Métricas: inercia (codo), `silhouette_score`, `davies_bouldin_score`.
- Visualización: PCA 2D + análisis de perfiles de clusters.

### Ejercicio 3 (Refuerzo)

- Entorno: laberinto con obstáculos y recompensas.
- Política epsilon-greedy y actualización Q (ecuación de Bellman).
- Progreso: recompensas/pasos/exploración y política aprendida.