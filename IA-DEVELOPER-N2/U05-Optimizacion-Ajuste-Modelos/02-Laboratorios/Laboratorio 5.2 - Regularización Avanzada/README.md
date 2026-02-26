# Laboratorio 5.2: Regularización Avanzada y Normalización en Modelos Profundos

## 🎯 Objetivos del Laboratorio

### Objetivo General
Implementar y comparar técnicas avanzadas de regularización y normalización para reducir overfitting en modelos de detección de fraudes financieros.

### Objetivos Específicos
- Aplicar múltiples técnicas de regularización (L1/L2, Dropout, Stochastic Depth)
- Implementar diferentes métodos de normalización (BatchNorm, LayerNorm, GroupNorm)
- Realizar ablation studies para evaluar el impacto de cada técnica
- Comparar rendimiento con métricas como precision, recall, F1-score
- Visualizar el efecto de las técnicas en las curvas de aprendizaje

## 📋 Marco Lógico del Proyecto

| Componente | Objetivo | Indicadores | Medios de Verificación |
|------------|----------|-------------|------------------------|
| **Generación de Datos** | Crear dataset sintético de fraudes | 100K transacciones con 5% fraudes | Scripts de generación y estadísticas |
| **Modelo Base** | Establecer baseline sin regularización | Overfitting >15% entre train/val | Código baseline y métricas iniciales |
| **Regularización L1/L2** | Aplicar penalización de pesos | Reducción de overfitting <10% | Código con regularización y comparación |
| **Dropout Avanzado** | Implementar Dropout y SpatialDropout | Mejora en generalización | Implementación y análisis |
| **Stochastic Depth** | Aplicar dropout a nivel de capa | Reducción de overfitting <5% | Código custom y evaluación |
| **Normalización** | Implementar BatchNorm, LayerNorm, GroupNorm | Estabilización del entrenamiento | Comparación de métodos |
| **Comparación Final** | Análisis exhaustivo de técnicas | Tabla comparativa completa | Reporte de análisis y visualizaciones |

## 🛠️ Tecnologías y Herramientas

### Principales Tecnologías
- **TensorFlow 2.15**: Framework de deep learning
- **Scikit-learn 1.4**: Métricas de evaluación y preprocesamiento
- **NumPy 1.24**: Computación numérica
- **Pandas 2.1**: Análisis de datos
- **Matplotlib/Seaborn**: Visualización

### Dependencias Adicionales
- **Imbalanced-learn**: Para manejo de clases desbalanceadas
- **Plotly**: Visualizaciones interactivas
- **Jupyter Notebook**: Análisis exploratorio

## 📁 Estructura del Proyecto

```
Laboratorio 5.2 - Regularización Avanzada/
├── README.md                           # Guía del laboratorio
├── requirements.txt                    # Dependencias
├── src/
│   ├── generate_fraud_data.py         # Generación de datos sintéticos
│   ├── baseline_model.py              # Modelo sin regularización
│   ├── regularized_model.py           # Modelo con regularización completa
│   ├── stochastic_depth_model.py      # Modelo con stochastic depth
│   ├── compare_models.py              # Comparación y análisis
│   └── regularization_utils.py        # Utilidades de regularización
├── notebooks/
│   ├── exploratory_analysis.ipynb     # Análisis exploratorio
│   ├── model_comparison.ipynb         # Comparación de modelos
│   └── ablation_study.ipynb           # Estudio de ablación
├── data/
│   ├── fraud_transactions.csv        # Dataset generado
│   └── processed/                     # Datos preprocesados
├── models/
│   ├── baseline_model.h5              # Modelo baseline
│   ├── regularized_model.h5           # Modelo regularizado
│   └── stochastic_depth_model.h5      # Modelo con stochastic depth
├── results/
│   ├── training_curves/               # Curvas de aprendizaje
│   ├── comparisons/                   # Gráficos comparativos
│   └── metrics/                       # Tablas de métricas
└── docs/
    ├── methodology_report.md          # Reporte metodológico
    └── technical_documentation.md    # Documentación técnica
```

## 🔧 Implementación Detallada

### Fase 1: Generación de Datos Sintéticos de Fraudes

#### Dataset Realista de Transacciones Financieras
```python
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

def generate_fraud_data(n_samples=100000, n_features=20, fraud_ratio=0.05):
    """
    Genera datos sintéticos de transacciones financieras con patrones realistas
    """
    # Generar datos desbalanceados (5% fraudes)
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=15,
        n_redundant=5,
        n_classes=2,
        weights=[1-fraud_ratio, fraud_ratio],
        flip_y=0.01,  # Ruido en las etiquetas
        random_state=42
    )

    # Añadir características categóricas y metadatos
    df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(n_features)])
    df['amount'] = np.random.exponential(scale=100, size=n_samples)
    df['time'] = np.random.uniform(0, 24, size=n_samples)  # Hora del día
    df['day_of_week'] = np.random.randint(0, 7, size=n_samples)
    df['is_fraud'] = y

    # Añadir correlaciones realistas
    df.loc[df['is_fraud'] == 1, 'amount'] *= 3  # Fraudes tienen montos más altos
    df.loc[df['is_fraud'] == 1, 'time'] = np.random.uniform(22, 6, size=df[df['is_fraud'] == 1].shape[0])  # Fraudes más comunes de noche

    return df
```

### Fase 2: Modelo Base (Sin Regularización)

#### Arquitectura Simple sin Regularización
```python
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

def build_baseline_model(input_shape):
    """
    Construye modelo base sin técnicas de regularización
    """
    model = models.Sequential([
        layers.Dense(128, activation='relu', input_shape=(input_shape,)),
        layers.Dense(64, activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
            tf.keras.metrics.AUC(name='auc')
        ]
    )
    
    return model
```

### Fase 3: Modelo con Regularización Avanzada

#### Implementación Completa de Técnicas de Regularización
```python
from tensorflow.keras import regularizers, constraints
from tensorflow.keras.layers import LayerNormalization, GroupNormalization

def build_regularized_model(input_shape):
    """
    Construye modelo con múltiples técnicas de regularización
    """
    model = models.Sequential()

    # Capa de entrada con normalización
    model.add(layers.Input(shape=(input_shape,)))
    model.add(LayerNormalization())

    # Capas ocultas con regularización
    model.add(layers.Dense(
        128,
        activation='relu',
        kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
        kernel_constraint=constraints.max_norm(3.0)
    ))
    model.add(layers.Dropout(0.3))
    model.add(GroupNormalization(groups=8))
    
    model.add(layers.Dense(
        64,
        activation='relu',
        kernel_regularizer=regularizers.l2(1e-4)
    ))
    model.add(layers.Dropout(0.4))
    model.add(LayerNormalization())
    
    model.add(layers.Dense(
        32,
        activation='relu',
        kernel_regularizer=regularizers.l1(1e-5)
    ))
    model.add(layers.Dropout(0.2))

    # Capa de salida
    model.add(layers.Dense(1, activation='sigmoid'))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
            tf.keras.metrics.AUC(name='auc')
        ]
    )
    
    return model
```

### Fase 4: Stochastic Depth

#### Implementación de Stochastic Depth
```python
class StochasticDepth(layers.Layer):
    """
    Implementación de Stochastic Depth (Dropout a nivel de capa)
    """
    def __init__(self, drop_prob, **kwargs):
        super().__init__(**kwargs)
        self.drop_prob = drop_prob

    def call(self, inputs, training=None):
        if not training or self.drop_prob == 0.:
            return inputs
        
        batch_size = tf.shape(inputs)[0]
        random_tensor = tf.random.uniform([batch_size, 1, 1, 1], 0, 1)
        keep_prob = 1. - self.drop_prob
        binary_tensor = tf.floor(keep_prob + random_tensor)
        return tf.div(inputs, keep_prob) * binary_tensor

def build_stochastic_depth_model(input_shape):
    """
    Construye modelo con Stochastic Depth
    """
    inputs = layers.Input(shape=(input_shape,))

    # Bloque 1
    x = layers.Dense(128, activation='relu')(inputs)
    x = LayerNormalization()(x)
    x = StochasticDepth(0.2)(x)

    # Bloque 2
    x = layers.Dense(64, activation='relu')(x)
    x = GroupNormalization(groups=4)(x)
    x = StochasticDepth(0.2)(x)

    # Bloque 3
    x = layers.Dense(32, activation='relu')(x)
    x = LayerNormalization()(x)
    x = StochasticDepth(0.1)(x)

    outputs = layers.Dense(1, activation='sigmoid')(x)

    model = models.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
            tf.keras.metrics.AUC(name='auc')
        ]
    )
    
    return model
```

### Fase 5: Comparación de Modelos

#### Análisis Comparativo Completo
```python
import matplotlib.pyplot as plt
import seaborn as sns
import json

def compare_models(histories, model_names, save_dir='results'):
    """
    Compara múltiples modelos y genera visualizaciones
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Métricas a comparar
    metrics = ['accuracy', 'precision', 'recall', 'auc']
    
    # Graficar curvas de aprendizaje
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, metric in enumerate(metrics):
        for j, (history, name) in enumerate(zip(histories, model_names)):
            axes[i].plot(history[metric], label=f'{name} - Train', alpha=0.7)
            axes[i].plot(history[f'val_{metric}'], label=f'{name} - Val', alpha=0.7)
        
        axes[i].set_title(f'{metric.capitalize()} sobre Épocas')
        axes[i].set_xlabel('Época')
        axes[i].set_ylabel(metric)
        axes[i].legend()
        axes[i].grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'learning_curves.png'), dpi=300, bbox_inches='tight')
    
    # Tabla comparativa de métricas finales
    final_metrics = {}
    for name, history in zip(model_names, histories):
        final_metrics[name] = {
            'accuracy': history['accuracy'][-1],
            'val_accuracy': history['val_accuracy'][-1],
            'precision': history['precision'][-1],
            'val_precision': history['val_precision'][-1],
            'recall': history['recall'][-1],
            'val_recall': history['val_recall'][-1],
            'auc': history['auc'][-1],
            'val_auc': history['val_auc'][-1]
        }
    
    df_metrics = pd.DataFrame(final_metrics).T
    print("Métricas Finales:")
    print(df_metrics.round(4))
    
    # Guardar tabla
    df_metrics.to_csv(os.path.join(save_dir, 'final_metrics.csv'))
    
    # Heatmap de comparación
    plt.figure(figsize=(12, 8))
    sns.heatmap(df_metrics, annot=True, cmap='viridis', fmt='.3f')
    plt.title('Comparación de Métricas Finales')
    plt.savefig(os.path.join(save_dir, 'metrics_heatmap.png'), dpi=300, bbox_inches='tight')
    
    return df_metrics
```

## 📊 Entregables del Laboratorio

### 1. Código Fuente Completo
- **generate_fraud_data.py**: Generación de datos sintéticos realistas
- **baseline_model.py**: Modelo sin regularización
- **regularized_model.py**: Modelo con regularización completa
- **stochastic_depth_model.py**: Implementación de stochastic depth
- **compare_models.py**: Análisis comparativo
- **regularization_utils.py**: Utilidades y funciones auxiliares

### 2. Modelos Entrenados
- **baseline_model.h5**: Modelo baseline
- **regularized_model.h5**: Modelo regularizado
- **stochastic_depth_model.h5**: Modelo con stochastic depth

### 3. Dataset
- **fraud_transactions.csv**: Dataset generado con 100K transacciones
- **processed/**: Datos preprocesados y divididos

### 4. Resultados y Visualizaciones
- **learning_curves/**: Curvas de aprendizaje de todos los modelos
- **comparisons/**: Gráficos comparativos
- **metrics/**: Tablas de métricas y análisis
- **final_metrics.csv**: Resumen de métricas finales

### 5. Documentación
- **methodology_report.md**: Reporte metodológico completo
- **technical_documentation.md**: Documentación técnica
- **README.md**: Guía de uso y reproducción

## 🎯 Criterios de Evaluación

### Componente Técnico (60%)
- **Generación de Datos** (15%): Dataset realista y bien balanceado
- **Implementación de Técnicas** (25%): Correcta implementación de regularización
- **Comparación de Modelos** (20%): Análisis exhaustivo y visualizaciones

### Componente Analítico (40%)
- **Análisis de Resultados** (20%): Insights sobre el impacto de cada técnica
- **Documentación** (20%): Reportes completos y reproducibles

### Métricas de Éxito
- **Reducción de Overfitting**: <5% entre train/val accuracy
- **Mejora en F1-Score**: >10% vs baseline
- **Análisis Completo**: Todas las técnicas evaluadas y comparadas
- **Reproducibilidad**: 100% de resultados reproducibles

## 🚀 Extensiones y Mejoras

### Opcionales Avanzados
1. **Ensemble Methods**: Combinación de múltiples modelos regularizados
2. **Adversarial Training**: Entrenamiento con ejemplos adversariales
3. **Curriculum Learning**: Estrategia de entrenamiento progresivo
4. **Meta-Learning**: Optimización automática de hiperparámetros de regularización

### Aplicaciones en Producción
1. **Online Learning**: Actualización del modelo con nuevos datos
2. **Drift Detection**: Detección de cambios en la distribución de datos
3. **Explainability**: Interpretación de predicciones de fraude
4. **Real-time Monitoring**: Monitoreo continuo del rendimiento

## 📈 Análisis Esperado

### Impacto de las Técnicas de Regularización

| Técnica | Impacto Esperado en Overfitting | Impacto en Accuracy | Complejidad |
|---------|--------------------------------|-------------------|-------------|
| **Baseline** | Alto (>15%) | 85-88% | Baja |
| **L1/L2 Regularization** | Medio (8-12%) | 87-90% | Media |
| **Dropout** | Medio-Bajo (5-10%) | 88-91% | Media |
| **BatchNorm** | Bajo (3-8%) | 89-92% | Media |
| **LayerNorm** | Bajo (3-8%) | 89-92% | Media |
| **GroupNorm** | Bajo (3-8%) | 89-92% | Alta |
| **Stochastic Depth** | Muy Bajo (<5%) | 90-93% | Alta |
| **Combinación Completa** | Muy Bajo (<3%) | 91-94% | Muy Alta |

### Insights Esperados
1. **L1/L2**: Efectivo para reducir complejidad del modelo
2. **Dropout**: Excelente para prevenir co-adaptación de características
3. **Normalización**: Crucial para estabilizar entrenamiento profundo
4. **Stochastic Depth**: Superior para redes profundas
5. **Combinación**: Sinergia entre técnicas produce mejores resultados

---

**Duración Estimada**: 6-8 horas  
**Dificultad**: Intermedia  
**Prerrequisitos**: Conocimientos de deep learning y regularización
