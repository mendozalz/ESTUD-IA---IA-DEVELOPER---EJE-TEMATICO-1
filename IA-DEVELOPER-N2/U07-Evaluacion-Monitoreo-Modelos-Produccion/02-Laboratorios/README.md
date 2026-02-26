# Laboratorios Prácticos - Unidad 7

## 🧪 Descripción General

Esta sección contiene 5 laboratorios prácticos que cubren todos los aspectos del monitoreo y evaluación de modelos de IA en producción, utilizando las herramientas más avanzadas de 2026.

## 📋 Lista de Laboratorios

### 🔧 Laboratorio 7.1: Evaluación de Modelos con Métricas Avanzadas y TensorFlow Model Analysis
- **Contexto**: Modelo de clasificación de fraudes en producción
- **Objetivos**: Calcular ROC-AUC, PR-AUC, análisis por subgrupos
- **Herramientas**: TensorFlow Model Analysis, scikit-learn
- **Duración**: 8-10 horas
- **Dificultad**: Intermedia

### 🔧 Laboratorio 7.2: Detección de Drift con Evidently AI y Alertas en Grafana
- **Contexto**: Modelo de predicción de demanda en retail
- **Objetivos**: Detectar drift, generar alertas, automatizar reentrenamiento
- **Herramientas**: Evidently AI, Grafana, Airflow
- **Duración**: 10-12 horas
- **Dificultad**: Avanzada

### 🔧 Laboratorio 7.3: Monitoreo en Tiempo Real con OpenTelemetry y Prometheus
- **Contexto**: Modelo de clasificación de imágenes médicas
- **Objetivos**: Instrumentar API, configurar dashboards
- **Herramientas**: OpenTelemetry, Prometheus, Grafana
- **Duración**: 8-10 horas
- **Dificultad**: Intermedia

### 🔧 Laboratorio 7.4: Reentrenamiento Automático con MLflow y TFX
- **Contexto**: Modelo de recomendación de productos en e-commerce
- **Objetivos**: Pipeline automático, validación, despliegue seguro
- **Herramientas**: MLflow, TFX, Kubernetes
- **Duración**: 12-15 horas
- **Dificultad**: Avanzada

### 🔧 Laboratorio 7.5: Explicabilidad y Cumplimiento con SHAP y Arize
- **Contexto**: Modelo de aprobación de préstamos
- **Objetivos**: Explicaciones individuales, auditorías automáticas
- **Herramientas**: SHAP, Arize, documentación regulatoria
- **Duración**: 10-12 horas
- **Dificultad**: Avanzada

## 🎯 Aprendizajes Esperados

Al completar todos los laboratorios, los estudiantes serán capaces de:

1. **Evaluar Modelos en Producción**
   - Calcular métricas avanzadas (ROC-AUC, PR-AUC)
   - Analizar rendimiento por subgrupos
   - Generar reportes de evaluación completos

2. **Detectar y Responder al Drift**
   - Implementar sistemas de detección de drift
   - Configurar alertas automáticas
   - Desencadenar reentrenamiento cuando sea necesario

3. **Monitorear en Tiempo Real**
   - Instrumentar aplicaciones con OpenTelemetry
   - Configurar dashboards en Grafana
   - Establecer sistemas de alertas proactivas

4. **Automatizar el Ciclo de Vida**
   - Implementar pipelines de reentrenamiento
   - Gestionar versiones de modelos con MLflow
   - Desplegar actualizaciones de forma segura

5. **Garantizar Cumplimiento y Explicabilidad**
   - Generar explicaciones para cada predicción
   - Crear documentación para auditorías
   - Implementar sistemas de monitoreo de equidad

## 📁 Estructura de Archivos

Cada laboratorio sigue una estructura estándar:

```
Laboratorio 7.X - [Nombre]/
├── README.md                 # Guía detallada del laboratorio
├── requirements.txt           # Dependencias Python
├── [archivos principales]     # Scripts y configuraciones
├── data/                    # Datos de ejemplo
├── notebooks/               # Jupyter notebooks para análisis
├── configs/                 # Archivos de configuración
└── docs/                    # Documentación adicional
```

## 🚀 Flujo de Trabajo Recomendado

1. **Revisión Teórica**: Leer la guía conceptual (01-Guía)
2. **Configuración**: Instalar dependencias y configurar entorno
3. **Implementación**: Seguir los pasos del README del laboratorio
4. **Pruebas**: Validar que todo funcione correctamente
5. **Experimentación**: Probar diferentes configuraciones
6. **Documentación**: Registrar aprendizajes y resultados

## 📊 Evaluación y Entregables

### Criterios de Evaluación

- **Funcionalidad** (40%): El sistema funciona como se espera
- **Calidad del Código** (20%): Código limpio, documentado, mantenible
- **Monitoreo Efectivo** (20%): Detección oportuna de problemas
- **Automatización** (20%): Procesos automáticos funcionando correctamente

### Entregables por Laboratorio

Cada laboratorio incluye:

1. **Código Fuente Completo**
   - Scripts principales funcionando
   - Configuraciones y archivos de soporte
   - Tests de validación

2. **Documentación**
   - README detallado con pasos
   - Diagramas de arquitectura
   - Guías de configuración

3. **Resultados y Reportes**
   - Métricas de evaluación
   - Dashboards funcionales
   - Análisis de resultados

---

**Nota**: Se recomienda completar los laboratorios en orden secuencial, ya que cada uno construye sobre los conceptos del anterior.
