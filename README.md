<div align="center">

![Estud-IA Logo](Estud-IA_Logo.png)

# **Campus Estud-IA**

# IA DEVELOPER (Nivel 2 - Intermedio)

**Institución:** Estud-IA  
**Curso:** IA DEVELOPER (Nivel 2: Intermedio - Inteligencia Artificial - AI Developer - Habilidades digitales)  
**Profesor:** Developer Gabriel Hernandez

## Objetivo general

Capacitar en los fundamentos de la inteligencia artificial AI developer, diseñar e implementar aplicaciones prácticas con herramientas de machine learning, integrar herramientas digitales avanzadas para la automatización y optimización de modelos, reconocer el impacto de la IA en la creación de contenido innovador, desarrollar habilidades para la evaluación tecnológica y resolución de problemas intermedios, adaptar metodologías ágiles para proyectos de desarrollo de IA, y explorar estrategias de empleabilidad en entornos tecnológicos, con la finalidad de potenciar su capacidad técnica y profesional en el ámbito de la inteligencia artificial.

## Población objetivo y prerequisitos

Este curso está dirigido a personas del Distrito Especial de Ciencia Tecnología e Innovación de Medellin, con saberes previos de iniciación en habilidades digitales, que buscan avanzar en inteligencia artificial AI developer, aprobando una prueba técnica con más del 80% y registrándose en OPE.

## Estructura curricular (módulos, lecciones y proyectos)

Cada módulo incluye:

- **Descripción breve**
- **Objetivos de aprendizaje específicos** (mínimo 3)
- **Temas clave**
- **Lecciones sugeridas** (secuencia práctica)
- **Proyecto práctico sugerido**

---

## Módulo 1. Introducción al desarrollo de IA (Intermedio)

### Descripción breve

Revisión de conceptos de IA y ML, flujo de trabajo moderno (datos → entrenamiento → evaluación → despliegue), y primer contacto con frameworks de deep learning (TensorFlow/Keras) y automatización básica.

### Objetivos de aprendizaje específicos

- Entender el desarrollo de IA para aplicaciones prácticas.
- Diferenciar tareas de ML clásico vs. Deep Learning y escoger un enfoque adecuado según el problema.
- Estructurar un proyecto de IA reproducible (entorno, dependencias, semillas, configuración, experimentos).

### Temas clave a cubrir

- Tipos de aprendizaje (supervisado/no supervisado) y tareas comunes (clasificación, regresión, NLP, visión)
- Pipeline de ML: preparación de datos, partición, métricas, sobreajuste, generalización
- Ecosistema: TensorFlow/Keras, scikit-learn, notebooks vs. proyectos Python, gestión de entornos
- Automatización básica: scripts, argumentos por CLI, estructura de carpetas, convenciones

### Lecciones sugeridas

- Lección 1: Mapa mental de IA para developers (casos reales, trade-offs)
- Lección 2: Reproducibilidad (venv/conda, `requirements.txt`, seeds, config)
- Lección 3: Repaso de métricas y validación (train/val/test, leakage)
- Lección 4: Primer modelo con Keras (baseline) y comparación con scikit-learn

### Proyecto práctico sugerido

- Construir un baseline de clasificación con scikit-learn y un baseline con Keras para el mismo dataset, comparando métricas y costos.

---

## Módulo 2. Programación avanzada con TensorFlow

### Descripción breve

Construcción de modelos con TensorFlow/Keras a nivel intermedio: capas personalizadas, `tf.data`, entrenamiento controlado, callbacks y manejo de datasets complejos.

### Objetivos de aprendizaje específicos

- Desarrollar modelos de IA para automatizar procesos.
- Implementar arquitecturas con capas personalizadas y/o modelos subclasificados.
- Preparar pipelines eficientes de entrada usando `tf.data` para entrenamiento escalable.

### Temas clave a cubrir

- API funcional vs. subclases (`tf.keras.Model`), capas personalizadas (`tf.keras.layers.Layer`)
- `tf.data.Dataset`: lectura, `map`, `batch`, `prefetch`, `cache`
- Callbacks: EarlyStopping, ModelCheckpoint, TensorBoard
- Guardado y carga de modelos (SavedModel), serialización y consideraciones de compatibilidad

### Lecciones sugeridas

- Lección 1: Repaso de Keras avanzado (API funcional, modelos multi-input/output)
- Lección 2: `tf.data` para datos tabulares/texto/imágenes
- Lección 3: Capas personalizadas y métricas personalizadas
- Lección 4: Entrenamiento controlado (custom training loop básico con `tf.GradientTape`)

### Proyecto práctico sugerido

- Entrenar un modelo con `tf.data`, callbacks y exportación SavedModel, listo para inferencia.

---

## Módulo 3. Automatización de flujos de trabajo de IA

### Descripción breve

Creación de pipelines de datos y automatización del ciclo de vida de modelos con Python: scripts, integración con APIs, y empaquetado básico para ejecución repetible.

### Objetivos de aprendizaje específicos

- Automatizar el ciclo de vida de modelos de IA.
- Diseñar un pipeline (ingesta → validación → entrenamiento → evaluación → artefactos) ejecutable por script.
- Integrar datos/servicios externos mediante APIs (consumo, paginación, manejo de errores).

### Temas clave a cubrir

- Estructura de proyecto (src, tests, configs), logging, manejo de errores
- Automatización con scripts Python (argparse/typer), tareas repetibles
- Integración con APIs (requests, autenticación por token cuando aplique, rate limits)
- Versionado de datos/experimentos (buenas prácticas; introducción conceptual)

### Lecciones sugeridas

- Lección 1: Diseño de pipelines reproducibles (inputs/outputs bien definidos)
- Lección 2: Automatización con CLI (parámetros, rutas, perfiles)
- Lección 3: Consumo de APIs y normalización de respuestas
- Lección 4: Empaquetado mínimo y ejecución consistente (entrypoints)

### Proyecto práctico sugerido

- Pipeline por CLI que: descarga datos desde una API, entrena un modelo, guarda métricas y exporta el artefacto.

---

## Módulo 4. Desarrollo de aplicaciones prácticas de clasificación

### Descripción breve

Implementación de soluciones prácticas de clasificación con deep learning: CNNs para imágenes y/o modelos para texto, incluyendo preparación de datos y evaluación.

### Objetivos de aprendizaje específicos

- Implementar soluciones prácticas de IA para clasificación.
- Diseñar y entrenar una CNN para un problema de clasificación de imágenes.
- Evaluar y diagnosticar errores por clase para iterar el modelo.

### Temas clave a cubrir

- CNNs: convolución, pooling, data augmentation
- Transfer learning (MobileNet/EfficientNet), fine-tuning
- Clasificación de texto (tokenización, embeddings) (opcional según ruta)
- Análisis de errores: matriz de confusión, ejemplos mal clasificados

### Lecciones sugeridas

- Lección 1: Preparación de dataset de imágenes (split, labels, augmentations)
- Lección 2: CNN baseline vs. transfer learning
- Lección 3: Fine-tuning y evaluación por clase
- Lección 4: Empaquetar inferencia (script de predicción y formato de salida)

### Proyecto práctico sugerido

- Clasificador de imágenes (dataset pequeño/mediano) con transfer learning y reporte de resultados.

---

## Módulo 5. Optimización y ajuste de modelos

### Descripción breve

Optimización de modelos con estrategias de regularización, selección de hiperparámetros y técnicas de mejora de generalización.

### Objetivos de aprendizaje específicos

- Optimizar modelos de IA para mejorar su rendimiento.
- Aplicar regularización (dropout, L2, data augmentation) para mitigar overfitting.
- Implementar tuning de hiperparámetros y comparar experimentos de forma sistemática.

### Temas clave a cubrir

- Overfitting/underfitting y diagnósticos (curvas de aprendizaje)
- Regularización: dropout, weight decay/L2, batch normalization, early stopping
- Hyperparameter tuning (grid/random; introducción a enfoques más avanzados)
- Selección de umbrales y calibración básica (cuando aplique)

### Lecciones sugeridas

- Lección 1: Diagnóstico por curvas y señales típicas
- Lección 2: Regularización en Keras (capas, optimizadores, callbacks)
- Lección 3: Búsqueda de hiperparámetros y registro de resultados
- Lección 4: Comparación experimental y decisión final (trade-offs)

### Proyecto práctico sugerido

- Notebook/script de tuning con al menos 8 corridas, con tabla comparativa y decisión justificada.

---

## Módulo 6. Integración de IA en aplicaciones reales

### Descripción breve

Desarrollo de aplicaciones prácticas con IA integrada: despliegue como API (Flask/FastAPI), consumo desde un front-end básico y consideraciones de rendimiento.

### Objetivos de aprendizaje específicos

- Desarrollar aplicaciones prácticas con IA integrada.
- Exponer un modelo como servicio HTTP con endpoints de predicción.
- Integrar un cliente (front-end simple o script) para consumir el modelo.

### Temas clave a cubrir

- Serving: FastAPI/Flask, validación de entrada (Pydantic), manejo de errores
- Carga del modelo y optimización de inferencia (batching simple, warmup)
- Diseño de contratos (schema de request/response) y versionado
- Seguridad básica (no exponer secretos, límites de payload)

### Lecciones sugeridas

- Lección 1: Diseño del contrato de inferencia (JSON, validación)
- Lección 2: API de predicción con FastAPI
- Lección 3: Cliente de consumo (front-end básico o script)
- Lección 4: Empaquetado para ejecución (comandos, estructura)

### Proyecto práctico sugerido

- API de predicción con FastAPI + UI básica o script cliente para probar múltiples casos.

---

## Módulo 7. Evaluación y monitoreo de modelos en producción

### Descripción breve

Evaluación avanzada, monitoreo de drift y estrategias de reentrenamiento y mantenimiento de modelos desplegados.

### Objetivos de aprendizaje específicos

- Evaluar y mantener modelos de IA en entornos reales.
- Usar métricas avanzadas (ROC-AUC, PR-AUC, F1, calibration) según el caso.
- Detectar señales de drift y definir un plan de reentrenamiento.

### Temas clave a cubrir

- Métricas avanzadas y selección según objetivo del negocio
- Umbral de decisión, curvas ROC/PR, costos de falsos positivos/negativos
- Drift: de datos y de concepto (señales, triggers)
- Estrategias de reentrenamiento y validación post-deploy

### Lecciones sugeridas

- Lección 1: Métricas y umbrales (casos prácticos)
- Lección 2: Reportes de evaluación reproducibles
- Lección 3: Drift (detección conceptual y práctica)
- Lección 4: Diseño de un plan de monitoreo y reentrenamiento

### Proyecto práctico sugerido

- Dashboard/report (Markdown/Notebook) con evaluación avanzada + simulación simple de drift y plan de acción.

---

## Módulo 8. Proyecto integrador: Aplicación IA automatizada

### Descripción breve

Diseño, desarrollo, despliegue y presentación de una aplicación práctica. Integra automatización, entrenamiento, evaluación, serving y retroalimentación.

### Objetivos de aprendizaje específicos

- Aplicar conocimientos en un proyecto completo para automatización.
- Diseñar un backlog técnico y plan de entregables con enfoque ágil.
- Presentar resultados técnicos (métricas, decisiones, riesgos, mejoras futuras).

### Temas clave a cubrir

- Definición del problema y criterios de éxito
- Arquitectura: pipeline + modelo + servicio + cliente
- Calidad: pruebas básicas, validación, manejo de errores
- Presentación técnica: storytelling con métricas y demos

### Lecciones sugeridas

- Lección 1: Kickoff (alcance, dataset, métricas, riesgos)
- Lección 2: Implementación pipeline y baseline
- Lección 3: Mejoras, tuning y exportación de modelo
- Lección 4: Despliegue (API) y demo funcional
- Lección 5: Entrega final y retroalimentación

### Proyecto práctico sugerido

- Aplicación end-to-end (pipeline automatizado + modelo + API + cliente) con demo y reporte final.

---

## Módulo 9. English for Tech

### Descripción breve

Inglés técnico para TI enfocado en lectura de documentación, comunicación de decisiones técnicas y colaboración en entornos digitales.

### Objetivos de aprendizaje específicos

- Leer y comprender documentación técnica (APIs, frameworks, errores) con estrategias de escaneo.
- Escribir mensajes técnicos claros (issues, PRs, changelogs) y pedir/recibir feedback.
- Presentar un resumen técnico corto (arquitectura, resultados, próximos pasos).

### Temas clave a cubrir

- Vocabulario técnico y verbos comunes (build, deploy, debug, refactor, benchmark)
- Lectura de docs: estructura, ejemplos, referencias, troubleshooting
- Comunicación escrita: tickets, README, reportes de experimentos
- Comunicación oral: demos, daily updates, retroalimentación

### Proyecto práctico sugerido

- Redactar un README en inglés para el proyecto integrador + un “technical summary” (1 página).

---

## Módulo 10. Habilidades para el empleo

### Descripción breve

Habilidades socioemocionales y de empleabilidad para trabajo en equipo, liderazgo, ejecución y presentación de proyectos tecnológicos.

### Objetivos de aprendizaje específicos

- Gestionar trabajo colaborativo con roles, acuerdos y comunicación efectiva.
- Preparar portafolio técnico (GitHub) y evidencias del aprendizaje.
- Practicar entrevistas técnicas intermedias orientadas a IA aplicada.

### Temas clave a cubrir

- Trabajo en equipo: acuerdos, feedback, gestión de conflictos
- Metodologías ágiles adaptadas a IA (backlog, experimentación, definición de “hecho”)
- Portafolio: estructura, proyectos, README, métricas y demos
- Preparación laboral: CV, LinkedIn/GitHub, entrevista técnica, prueba práctica

### Proyecto práctico sugerido

- Portafolio (GitHub) con 2–3 proyectos curados + pitch (3–5 min) del proyecto integrador.

---

# 🌸 Clasificación de Flores Iris - Machine Learning

### Proyecto de la Universidad Estud-IA
**Ejercicio #1 - Clasificación Supervisada**

</div>

## 📋 Tabla de Contenidos
1. [Descripción del Proyecto](#descripción-del-proyecto)
2. [¿Qué se quiere conseguir?](#qué-se-quiere-conseguir)
3. [Requisitos Previos](#requisitos-previos)
4. [Configuración del Entorno Virtual](#configuración-del-entorno-virtual)
5. [Instalación de Dependencias](#instalación-de-dependencias)
6. [Cómo ejecutar el Proyecto](#cómo-ejecutar-el-proyecto)
7. [Entendiendo el Código](#entendiendo-el-código)

---

## 📚 Descripción del Proyecto

Este proyecto implementa un **modelo de Machine Learning de clasificación supervisada** usando el famoso dataset Iris. 

El objetivo es **predecir qué tipo de flor es** (Setosa, Versicolor o Virginica) basándose en características físicas como:
- Largo del sépalo
- Ancho del sépalo
- Largo del pétalo
- Ancho del pétalo

---

## 🎯 ¿Qué se quiere conseguir?

Este proyecto cumple con los siguientes objetivos educativos:

✅ **Aprender Machine Learning**: Entender cómo funcionan los modelos de clasificación

✅ **Procesamiento de datos**: Cargar, analizar y preparar datos

✅ **Visualización**: Crear gráficos para entender los datos

✅ **Entrenamiento de modelos**: Usar el algoritmo Random Forest

✅ **Evaluación**: Medir qué tan bien funciona el modelo

✅ **Predicciones**: Usar el modelo entrenado para predecir nuevas flores

---

## 🔧 Requisitos Previos

Antes de comenzar, necesitas tener instalado en tu computadora:

### Python
- **Python 3.8 o superior**
- Puedes descargarlo desde: https://www.python.org/downloads/

**¿Cómo verificar si tienes Python?**
```bash
python --version
```

Si ves algo como `Python 3.10.5`, significa que ya tienes Python instalado.

---

## 🚀 Configuración del Entorno Virtual

Un **entorno virtual** es como una carpeta especial que guarda todas las librerías de tu proyecto, separadas del resto de tu computadora. Esto es muy importante para no contaminar tu sistema.

### Paso 1: Abre la terminal/CMD

**En Windows:**
- Presiona `Win + R`
- Escribe `cmd` y presiona Enter

**En Mac/Linux:**
- Abre la Terminal (busca en Aplicaciones)

### Paso 2: Navega a la carpeta del proyecto

```bash
cd ruta/a/tu/proyecto
```

Por ejemplo:
```bash
cd /home/mendozalz/Escritorio/StudiaIA/Ejercicio#1-clasificacion
```

### Paso 3: Crea el entorno virtual

**En Windows:**
```bash
python -m venv venv
```

**En Mac/Linux:**
```bash
python3 -m venv venv
```

Este comando crea una carpeta llamada `venv` (es como un "mini Python" solo para tu proyecto).

### Paso 4: Activa el entorno virtual

**En Windows:**
```bash
venv\Scripts\activate
```

**En Mac/Linux:**
```bash
source venv/bin/activate
```

**¿Funcionó?** Si ves algo como esto en tu terminal, significa que los pasos anteriores fueron correctos:
```bash
(venv) $
```

Nota que ahora tu terminal muestra `(venv)` al inicio. Eso significa que estás dentro del entorno virtual.

---

## 📦 Instalación de Dependencias

Las **dependencias** son librerías externas (código escrito por otros colaboradores) que nuestro proyecto necesita para funcionar.

### Paso 1: Asegúrate de tener el archivo `requirements.txt`

El archivo `requirements.txt` debe estar en la carpeta de tu proyecto junto a `app.py`.

### Paso 2: Instala todas las dependencias

Con el entorno virtual activado (recuerda que debe mostrar `(venv)` en tu terminal), ejecuta:

```bash
pip install -r requirements.txt
```

### ¿Qué hace este comando?

- `pip` = el gestor de paquetes de Python (es como un "instalador" de librerías)
- `install` = instalar
- `-r requirements.txt` = lee el archivo requirements.txt y instala todo lo que dice

**Espera un momento** mientras todas las librerías se descargan e instalan. Verás algo como:
```
Successfully installed numpy-1.24.0 pandas-2.0.0 ...
```

### Actualizar una dependencia (opcional)

Si en el futuro necesitas actualizar una librería específica:

```bash
pip install --upgrade nombre_libreria
```

Por ejemplo:
```bash
pip install --upgrade pandas
```

---

## ▶️ Cómo ejecutar el Proyecto

### Paso 1: Asegúrate de estar en el entorno virtual

Tu terminal debe mostrar `(venv)` al inicio:
```bash
(venv) $
```

### Paso 2: Ejecuta el archivo principal

```bash
python app.py
```

### ¿Qué pasará?

El programa hará esto automáticamente:

1. 📊 **Carga el dataset Iris** - Descarga 150 flores con sus características
2. 📈 **Crea visualizaciones** - Muestra gráficos para entender los datos
3. 🧠 **Entrena el modelo** - Usa Random Forest para aprender patrones
4. ✅ **Evalúa el resultado** - Muestra qué tan preciso es el modelo
5. 🔮 **Hace predicciones** - Prueba el modelo con nuevas flores

Deberías ver algo como:
```
============================================================
DEMOSTRACION: CLASIFICACION SUPERVISADA - IRIS
============================================================
Cargando dataset Iris...
Dataset cargado: 150 muestras, 4 características
...
Accuracy general: 1.0000 (100.00%)
```

---

## 💡 Entendiendo el Código

### Estructura del Proyecto

```
Ejercicio#1-clasificacion/
│
├── app.py                  # Archivo principal (el código del proyecto)
├── requirements.txt        # Lista de dependencias
└── README.md              # Este archivo (instrucciones)
```

### Las Librerías Explicadas

| Librería | Para qué sirve |
|----------|---|
| **numpy** | Trabajo con números y arrays (listas de números) |
| **pandas** | Manejo de datos en tablas (como Excel en Python) |
| **matplotlib** | Crear gráficos y visualizaciones |
| **seaborn** | Gráficos más bonitos y complejos que matplotlib |
| **scikit-learn** | Modelos de Machine Learning listos para usar |

### Flujo del Programa

```
1. Cargar datos Iris (150 flores)
           ↓
2. Visualizar datos (crear gráficos)
           ↓
3. Dividir datos en entrenamiento (80%) y prueba (20%)
           ↓
4. Normalizar datos (hacer que todos tengan la misma escala)
           ↓
5. Entrenar modelo Random Forest
           ↓
6. Evaluar el modelo
           ↓
7. Hacer predicciones con nuevas flores
```

---

## 🆘 Solución de Problemas

### Problema: "Python no se reconoce"

**Solución:** Python no está en el PATH de tu sistema. Reinstálalo asegurándote de marcar la opción "Add Python to PATH".

### Problema: "No existe el archivo requirements.txt"

**Solución:** Asegúrate de que el archivo `requirements.txt` está en la misma carpeta que `app.py`.

### Problema: "ModuleNotFoundError: No module named 'sklearn'"

**Solución:** Las dependencias no se instalaron correctamente. Intenta:
```bash
pip install scikit-learn pandas numpy matplotlib seaborn
```

### Problema: "No puedo activar el entorno virtual"

**Solución:** Verifica que estés en la carpeta correcta del proyecto y usa el comando correcto para tu sistema operativo.

---

## 📖 Recursos Adicionales

- [Documentación oficial de scikit-learn](https://scikit-learn.org/)
- [Documentación de pandas](https://pandas.pydata.org/)
- [Dataset Iris en Kaggle](https://www.kaggle.com/datasets/uciml/iris)
- [Tutorial de Random Forest](https://medium.com/@evertongomede/understanding-random-forest-classifier-e066f2e0b2ef)

---

## 📝 Lo que Aprendiste

Después de completar este proyecto, ahora sabes:

✅ Cómo crear un entorno virtual
✅ Cómo instalar dependencias de Python
✅ Cómo cargar y explorar datasets
✅ Cómo preprocesar datos
✅ Cómo entrenar un modelo de Machine Learning
✅ Cómo evaluar modelos
✅ Cómo hacer predicciones

¡Felicidades! 🎉 Ya has completado tu primer proyecto de Machine Learning.

---

**Última actualización:** 19 de febrero de 2026