# Laboratorio 3 - Tres Paradigmas de Machine Learning

## Descripción
Tres ejercicios completos que demuestran los principales paradigmas de aprendizaje:
1. **Aprendizaje Supervisado** - Clasificación de flores Iris con Random Forest
2. **Aprendizaje No Supervisado** - Segmentación de clientes con K-Means
3. **Aprendizaje por Refuerzo** - Navegación en laberinto con Q-Learning

## Archivos principales
- `Ej1.py` - Clasificador de Iris (supervisado)
- `Ej2.py` - Segmentador de clientes (no supervisado)
- `Ej3.py` - Agente Q-Learning (refuerzo)

## Cómo ejecutar
```bash
python Ej1.py
python Ej2.py
python Ej3.py
```

## Prácticas sugeridas
- **Fase 1: Identificación y Ejecución**
  - Distinguir y reconocer cuál ejercicio corresponde a aprendizaje supervisado, no supervisado y reforzado
  - Ejecutar los 3 prototipos para entender su funcionamiento y diferencias
  - Identificar en cada ejercicio: preparación de datos, entrenamiento, métricas/visualizaciones
  - Realizar 1 ajuste por ejercicio (parámetro o parte del flujo) y registrar el efecto

- **Fase 2: Adaptación al Proyecto Integrador**
  - Relacionar cada paradigma con tu proyecto integrador
  - Generar 3 prototipos adaptados a tus propuestas específicas
  - Crear recursos que aporten directamente a tu proyecto final

## Entregables

### Fase 1: Prototipos Originales
- Evidencia de ejecución (capturas o logs) de los 3 ejercicios
- Un reporte corto en Markdown con:
  - Qué hace cada ejercicio
  - 1 cambio realizado por ejercicio
  - Resultado observado (métrica/visualización) y conclusión

### Fase 2: Adaptación al Proyecto Integrador
- **3 prototipos adaptados** a tu proyecto específico:
  - Prototipo supervisado adaptado a tu contexto
  - Prototipo no supervisado adaptado a tu contexto  
  - Prototipo por refuerzo adaptado a tu contexto
- **Documentación de cada adaptación**:
  - Cómo se relaciona con tu proyecto
  - Qué problema específico resuelve
  - Datos y parámetros adaptados
  - Resultados esperados en tu proyecto
- **Recursos generados** para tu proyecto final (código, datasets, modelos)

## Qué observar en cada ejercicio

### Ejercicio 1 (Supervisado)
- Dataset Iris, EDA y visualizaciones
- Preprocesamiento (train/test + `StandardScaler`)
- Entrenamiento y evaluación (accuracy, matriz de confusión, `classification_report`)

### Ejercicio 2 (No supervisado)
- Estandarización y búsqueda de `k`
- Métricas: inercia (codo), `silhouette_score`, `davies_bouldin_score`
- Visualización: PCA 2D + análisis de perfiles de clusters

### Ejercicio 3 (Refuerzo)
- Entorno: laberinto con obstáculos y recompensas
- Política epsilon-greedy y actualización Q (ecuación de Bellman)
- Progreso: recompensas/pasos/exploración y política aprendida
