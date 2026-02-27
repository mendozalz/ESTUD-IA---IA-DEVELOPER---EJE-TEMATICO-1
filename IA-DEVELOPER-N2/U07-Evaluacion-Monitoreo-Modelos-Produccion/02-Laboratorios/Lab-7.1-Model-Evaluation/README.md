# Laboratorio 7.1: Evaluación de Modelos con Métricas Avanzadas y TensorFlow Model Analysis

## 🎯 Contexto
Un modelo de clasificación de fraudes en producción necesita ser evaluado con métricas avanzadas (ROC-AUC, PR-AUC) y analizado por subgrupos (ej: por región, tipo de transacción).

## 🎯 Objetivos

- Calcular ROC-AUC y PR-AUC para el modelo
- Analizar métricas por subgrupos (ej: transacciones por región)
- Generar un reporte de evaluación con TensorFlow Model Analysis (TFMA)

## 📋 Marco Lógico del Laboratorio

| Componente | Objetivo | Indicadores | Medios de Verificación |
|------------|----------|-------------|------------------------|
| **Evaluación Básica** | Calcular métricas estándar | Accuracy, Precision, Recall, F1-score | Scripts de evaluación con scikit-learn |
| **Métricas Avanzadas** | Calcular ROC-AUC y PR-AUC | ROC-AUC > 0.85, PR-AUC > 0.80 | Código con TFMA y scikit-learn |
| **Análisis por Subgrupos** | Evaluar rendimiento por segmentos | Métricas por región > 0.80 | Reporte TFMA con slicing |
| **Reporte Completo** | Generar documentación de evaluación | Reporte HTML con visualizaciones | Archivos HTML generados por TFMA |

## 🛠️ Tecnologías Utilizadas

- **TensorFlow Model Analysis (TFMA)**: Evaluación avanzada de modelos
- **scikit-learn**: Métricas estándar y avanzadas
- **TensorFlow**: Manejo de datos y modelos
- **Matplotlib/Seaborn**: Visualización de resultados

## 📁 Estructura del Laboratorio

```
Laboratorio 7.1 - Evaluación de Modelos con Métricas Avanzadas y TensorFlow Model Analysis/
├── README.md                           # Esta guía
├── requirements.txt                     # Dependencias Python
├── load_model_data.py                  # Cargar modelo y datos
├── evaluate_with_tfma.py               # Evaluación con TFMA
├── analyze_subgroups.py                # Análisis por subgrupos
├── data/                              # Datos de ejemplo
│   ├── train.tfrecord                  # Datos de entrenamiento
│   ├── eval.tfrecord                   # Datos de evaluación
│   └── reference_data.csv             # Datos de referencia
├── models/                            # Modelos entrenados
│   └── fraud_classifier.h5            # Modelo de clasificación
├── outputs/                           # Resultados generados
│   ├── tfma_output/                   # Salida de TFMA
│   ├── tfma_report.html               # Reporte principal
│   └── subgroup_analysis.html         # Análisis por subgrupos
└── notebooks/                         # Jupyter notebooks
    └── evaluation_analysis.ipynb      # Análisis interactivo
```

## 🚀 Implementación Paso a Paso

### Paso 1: Configuración del Entorno

```bash
pip install tensorflow==2.15.0 tensorflow-model-analysis==0.46.0 scikit-learn==1.3.0 matplotlib==3.8.0 pandas==2.0.0 seaborn==0.12.0
```

### Paso 2: Cargar Modelo y Datos

El script `load_model_data.py` genera datos de ejemplo de fraudes y los guarda en formato TFRecord:

```python
import tensorflow as tf
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Generar datos de ejemplo (fraudes)
X, y = make_classification(
    n_samples=10000,
    n_features=20,
    n_informative=10,
    n_redundant=5,
    n_classes=2,
    weights=[0.95, 0.05],  # Clases desbalanceadas (5% fraudes)
    random_state=42
)

# Dividir en train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Añadir metadatos (ej: región)
regions = np.random.choice(['NA', 'EU', 'ASIA'], size=len(y))
X_train = np.column_stack((X_train, regions[:len(X_train)]))
X_test = np.column_stack((X_test, regions[len(X_train):]))

# Guardar en TFRecords
def write_tfrecords(X, y, filename):
    with tf.io.TFRecordWriter(filename) as writer:
        for features, label in zip(X, y):
            example = tf.train.Example(features=tf.train.Features(feature={
                'features': tf.train.Feature(float_list=tf.train.FloatList(value=features[:-1])),
                'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
                'region': tf.train.Feature(bytes_list=tf.train.BytesList(value=[features[-1].encode('utf-8')]))
            }))
            writer.write(example.SerializeToString())

write_tfrecords(X_train, y_train, 'data/train.tfrecord')
write_tfrecords(X_test, y_test, 'data/eval.tfrecord')
```

### Paso 3: Evaluación con TFMA

El script `evaluate_with_tfma.py` configura y ejecuta la evaluación con TFMA:

```python
import tensorflow_model_analysis as tfma
from tensorflow_model_analysis.view.render_slicing_metrics import render_slicing_metrics

# Definir config de evaluación
eval_config = tfma.EvalConfig(
    model_specs=[tfma.ModelSpec(label_key='label')],
    metrics_specs=[
        tfma.MetricConfig(class_name='BinaryAccuracy'),
        tfma.MetricConfig(class_name='AUC'),
        tfma.MetricConfig(class_name='AUC', name='pr_auc', options={'curve': 'PR'}),
        tfma.MetricConfig(class_name='ExampleCount')
    ],
    slicing_specs=[
        tfma.SlicingSpec(),
        tfma.SlicingSpec(feature_keys=['region'])
    ]
)

# Cargar modelo (ejemplo: modelo dummy)
def model_fn():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(10, activation='relu', input_shape=(20,)),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Ejecutar evaluación
eval_result = tfma.run_model_analysis(
    tfma.load_model(model_fn),
    data_location='data/eval.tfrecord',
    eval_config=eval_config,
    output_path='outputs/tfma_output'
)

# Visualizar resultados
render_slicing_metrics(eval_result, output_file='outputs/tfma_report.html')
```

### Paso 4: Análisis por Subgrupos

El script `analyze_subgroups.py` extrae y analiza las métricas por región:

```python
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

# Cargar datos de evaluación
eval_data = tfma.load_eval_result('outputs/tfma_output')

# Convertir a DataFrame para análisis
metrics_df = pd.DataFrame()
for slice_key, slice_metrics in eval_data.slicing_metrics().items():
    if slice_key:
        region = slice_key[0][1].decode('utf-8') if slice_key[0][0] == 'region' else 'overall'
        metrics_df = metrics_df.append({
            'region': region,
            'accuracy': slice_metrics['binary_accuracy']['doubleValue'],
            'roc_auc': slice_metrics['auc']['doubleValue'],
            'pr_auc': slice_metrics['pr_auc']['doubleValue'],
            'example_count': slice_metrics['example_count']['doubleValue']
        }, ignore_index=True)

print(metrics_df)

# Graficar métricas por región
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 8))
plt.subplot(2, 2, 1)
sns.barplot(data=metrics_df, x='region', y='roc_auc')
plt.title('ROC-AUC por Región')

plt.subplot(2, 2, 2)
sns.barplot(data=metrics_df, x='region', y='pr_auc')
plt.title('PR-AUC por Región')

plt.subplot(2, 2, 3)
sns.barplot(data=metrics_df, x='region', y='accuracy')
plt.title('Accuracy por Región')

plt.subplot(2, 2, 4)
sns.barplot(data=metrics_df, x='region', y='example_count')
plt.title('Número de Ejemplos por Región')

plt.tight_layout()
plt.savefig('outputs/subgroup_analysis.png')
plt.show()
```

## 📊 Resultados Esperados

### Métricas Generales
- **ROC-AUC**: > 0.85 (indicador de buen rendimiento)
- **PR-AUC**: > 0.80 (importante para clases desbalanceadas)
- **Accuracy**: > 0.90
- **F1-Score**: > 0.75

### Análisis por Subgrupos
- **Región NA**: ROC-AUC > 0.85
- **Región EU**: ROC-AUC > 0.80
- **Región ASIA**: ROC-AUC > 0.75

### Visualizaciones Generadas
- **tfma_report.html**: Reporte interactivo de TFMA
- **subgroup_analysis.png**: Gráficos de métricas por región
- **confusion_matrix.png**: Matriz de confusión por subgrupo

## 🔍 Análisis y Interpretación

### Interpretación de Métricas

1. **ROC-AUC**: Mide la capacidad del modelo para distinguir entre clases
   - 0.5: Sin capacidad de discriminación
   - 0.7-0.8: Discriminación aceptable
   - 0.8-0.9: Buena discriminación
   - >0.9: Excelente discriminación

2. **PR-AUC**: Más informativo que ROC-AUC para clases desbalanceadas
   - Considera tanto precisión como recall
   - Sensible a la distribución de clases

3. **Análisis por Subgrupos**: Identifica sesgos y problemas de equidad
   - Diferencias significativas entre regiones pueden indicar sesgo
   - Bajo número de ejemplos en un subgrupo afecta la fiabilidad

### Decisiones Basadas en Resultados

1. **Si ROC-AUC < 0.80**: Revisar características y modelo
2. **Si hay diferencias > 0.10 entre regiones**: Investigar sesgos
3. **Si PR-AUC bajo**: Considerar técnicas para clases desbalanceadas
4. **Si accuracy alta pero ROC-AUC bajo**: Posible overfitting

## 📌 Entregables del Laboratorio

| Entregable | Descripción | Formato |
|-------------|-------------|-----------|
| Generación de Datos | Script para crear datos de fraudes con metadatos | load_model_data.py |
| Configuración TFMA | Evaluación con métricas avanzadas y slicing | evaluate_with_tfma.py |
| Análisis de Subgrupos | Comparación de métricas por región | analyze_subgroups.py |
| Reporte TFMA | Visualización de resultados | outputs/tfma_report.html |
| Análisis Visual | Gráficos de métricas por subgrupo | outputs/subgroup_analysis.png |
| Documentación | Guía para replicar la evaluación | README.md |

## 🎯 Criterios de Evaluación

- **Funcionalidad** (40%): Scripts ejecutan correctamente y generan resultados
- **Calidad del Código** (20%): Código limpio, documentado y mantenible
- **Análisis Completo** (25%): Todas las métricas calculadas correctamente
- **Visualizaciones** (15%): Gráficos claros e informativos

## 🚀 Extensión y Mejoras

### Mejoras Sugeridas
1. **Más Subgrupos**: Analizar por tipo de transacción, monto, hora del día
2. **Métricas Adicionales**: Include F-beta, Matthews correlation coefficient
3. **Análisis Temporal**: Evaluar rendimiento a lo largo del tiempo
4. **Comparación de Modelos**: Evaluar múltiples algoritmos

### Aplicaciones en Producción
1. **Monitoreo Continuo**: Integrar TFMA en pipelines de CI/CD
2. **Alertas Automáticas**: Notificar cuando las métricas caen below umbrales
3. **Dashboard en Tiempo Real**: Visualizar métricas en Grafana
4. **A/B Testing**: Comparar versiones del modelo con métricas avanzadas

---

**Duración Estimada**: 8-10 horas  
**Nivel de Dificultad**: Intermedia  
**Prerrequisitos**: Conocimientos de TensorFlow, scikit-learn, métricas de evaluación
