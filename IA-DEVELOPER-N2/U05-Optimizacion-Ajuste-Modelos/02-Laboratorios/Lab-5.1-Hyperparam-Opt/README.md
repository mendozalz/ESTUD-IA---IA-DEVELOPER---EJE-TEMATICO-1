# Laboratorio 5.1: Hyperparameter Tuning Avanzado con Optuna y Weights & Biases

## 🎯 Objetivos del Laboratorio

### Objetivo General
Implementar un sistema completo de optimización de hiperparámetros para un modelo de clasificación de imágenes médicas usando técnicas avanzadas de búsqueda bayesiana y tracking de experimentos.

### Objetivos Específicos
- Configurar un estudio de Optuna con TPE + Hyperband pruning
- Integrar con Weights & Biases para tracking de métricas en tiempo real
- Analizar los hiperparámetros óptimos encontrados
- Visualizar correlaciones y patrones en los resultados
- Implementar early stopping para optimización eficiente

## 📋 Marco Lógico del Proyecto

| Componente | Objetivo | Indicadores | Medios de Verificación |
|------------|----------|-------------|------------------------|
| **Configuración del Entorno** | Preparar herramientas y dependencias | Entorno funcional con todas las librerías instaladas | Scripts de instalación y verificación de versiones |
| **Carga de Datos Médicos** | Preprocesar dataset de rayos X | Dataset cargado y preprocesado correctamente | Funciones de carga y estadísticas del dataset |
| **Espacio de Búsqueda** | Definir hiperparámetros a optimizar | 8+ hiperparámetros configurados | Código de definición del espacio de búsqueda |
| **Optimización con Optuna** | Encontrar mejores hiperparámetros | Mejora >5% en AUC vs baseline | Logs de Optuna y mejores trials |
| **Tracking con W&B** | Monitorear experimentos en tiempo real | Dashboard completo con métricas | Dashboard de W&B y visualizaciones |
| **Análisis de Resultados** | Extraer insights de la optimización | Reporte con análisis detallado | Notebook de análisis y visualizaciones |

## 🛠️ Tecnologías y Herramientas

### Principales Tecnologías
- **Optuna 4.0**: Framework de optimización bayesiana
- **Weights & Biases 2.15**: Tracking de experimentos
- **TensorFlow 2.15**: Framework de deep learning
- **TensorFlow Datasets 4.9.2**: Dataset de imágenes médicas

### Dependencias Adicionales
- **Matplotlib/Seaborn**: Visualización de resultados
- **Pandas**: Análisis de datos
- **NumPy**: Computación numérica
- **Jupyter Notebook**: Análisis interactivo

## 📁 Estructura del Proyecto

```
Laboratorio 5.1 - Hyperparameter Tuning Avanzado/
├── README.md                           # Guía del laboratorio
├── requirements.txt                    # Dependencias
├── src/
│   ├── load_medical_data.py          # Carga y preprocesamiento
│   ├── optuna_hypersearch.py         # Optimización con Optuna
│   ├── analyze_results.py            # Análisis de resultados
│   └── wandb_utils.py                # Utilidades para W&B
├── notebooks/
│   ├── exploratory_analysis.ipynb     # Análisis exploratorio
│   └── results_visualization.ipynb   # Visualización de resultados
├── data/
│   └── processed/                     # Datos preprocesados
├── models/
│   └── best_model.h5                  # Mejor modelo encontrado
├── results/
│   ├── optuna_study.db               # Base de datos de Optuna
│   └── plots/                        # Gráficos de resultados
└── docs/
    ├── methodology_report.md          # Reporte metodológico
    └── technical_documentation.md    # Documentación técnica
```

## 🔧 Implementación Detallada

### Fase 1: Configuración del Entorno

#### Instalación de Dependencias
```bash
pip install optuna==4.0.0 wandb==0.15.12 tensorflow==2.15.0 tensorflow-datasets==4.9.2
pip install matplotlib==3.8.0 seaborn==0.13.0 pandas==2.1.0 numpy==1.24.3
pip install jupyter==1.0.0 scikit-learn==1.4.0
```

#### Configuración de Weights & Biases
```python
import wandb

# Inicializar W&B
wandb.init(
    project="medical-image-tuning",
    config={
        "dataset": "CheXpert",
        "model_type": "EfficientNetB0",
        "optimization_method": "TPE + Hyperband"
    }
)
```

### Fase 2: Carga y Preprocesamiento de Datos

#### Dataset de Rayos X (CheXpert)
```python
import tensorflow as tf
import tensorflow_datasets as tfds

def load_medical_data():
    """
    Carga y preprocesa el dataset de rayos X para detección de neumonía
    """
    # Cargar dataset
    (train_ds, val_ds, test_ds), info = tfds.load(
        'chexpert',
        split=['train[:80%]', 'train[80%:90%]', 'train[90%:]'],
        shuffle_files=True,
        as_supervised=True,
        with_info=True
    )

    # Preprocesamiento
    def preprocess(image, label):
        # Resize y normalización
        image = tf.image.resize(image, (224, 224))
        image = tf.cast(image, tf.float32) / 255.0
        
        # Data augmentation
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_brightness(image, 0.1)
        image = tf.image.random_contrast(image, 0.9, 1.1)
        
        return image, label

    # Aplicar preprocesamiento
    train_ds = train_ds.map(preprocess).batch(32).prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.map(preprocess).batch(32).prefetch(tf.data.AUTOTUNE)
    test_ds = test_ds.map(preprocess).batch(32).prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds, test_ds, info
```

### Fase 3: Definición del Espacio de Búsqueda

#### Hiperparámetros a Optimizar
```python
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras import layers, models, optimizers, regularizers

def build_model(hp):
    """
    Construye modelo con hiperparámetros variables
    """
    # Hiperparámetros de arquitectura
    use_pretrained = hp.Boolean('use_pretrained')
    num_layers = hp.Int('num_layers', 1, 4)
    dropout_rate = hp.Float('dropout_rate', 0.1, 0.5, step=0.1)
    l2_reg = hp.Float('l2_reg', 1e-5, 1e-2, log=True)
    
    # Hiperparámetros de entrenamiento
    lr = hp.Float('lr', 1e-5, 1e-3, log=True)
    batch_size = hp.Choice('batch_size', [16, 32, 64])
    
    # Construir modelo
    inputs = layers.Input(shape=(224, 224, 3))
    
    if use_pretrained:
        base_model = EfficientNetB0(
            include_top=False,
            weights='imagenet',
            input_shape=(224, 224, 3)
        )
        base_model.trainable = False
    else:
        base_model = EfficientNetB0(
            include_top=False,
            weights=None,
            input_shape=(224, 224, 3)
        )
    
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    
    # Capas densas con regularización
    for _ in range(num_layers):
        x = layers.Dense(
            128,
            activation='relu',
            kernel_regularizer=regularizers.l2(l2_reg)
        )(x)
        x = layers.Dropout(dropout_rate)(x)
        x = layers.BatchNormalization()(x)
    
    outputs = layers.Dense(1, activation='sigmoid')(x)
    model = models.Model(inputs=inputs, outputs=outputs)
    
    # Compilar
    model.compile(
        optimizer=optimizers.Adam(learning_rate=lr),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )
    
    return model
```

### Fase 4: Optimización con Optuna

#### Configuración del Estudio
```python
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import HyperbandPruner
from optuna.integration.wandb import WeightsAndBiasesCallback

def objective(trial):
    """
    Función objetivo para Optuna
    """
    # Hiperparámetros
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    num_layers = trial.suggest_int("num_layers", 1, 4)
    dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5, step=0.1)
    l2_reg = trial.suggest_float("l2_reg", 1e-5, 1e-2, log=True)
    use_pretrained = trial.suggest_categorical("use_pretrained", [True, False])
    
    # Iniciar run de W&B
    wandb.init(
        project="medical-image-tuning",
        config={
            "lr": lr,
            "batch_size": batch_size,
            "num_layers": num_layers,
            "dropout_rate": dropout_rate,
            "l2_reg": l2_reg,
            "use_pretrained": use_pretrained
        },
        reinit=True
    )
    
    # Construir modelo
    model = build_model_from_params(lr, batch_size, num_layers, dropout_rate, l2_reg, use_pretrained)
    
    # Callbacks
    callbacks = [
        wandb.keras.WandbCallback(
            monitor='val_auc',
            mode='max',
            save_weights_only=True
        ),
        optuna.integration.TFKerasPruningCallback(trial, 'val_auc'),
        tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
    ]
    
    # Entrenar
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=30,
        callbacks=callbacks,
        verbose=0
    )
    
    # Reportar métrica para pruning
    val_auc = history.history['val_auc'][-1]
    trial.report(val_auc, step=30)
    
    # Manejar pruning
    if trial.should_prune():
        wandb.finish()
        raise optuna.TrialPruned()
    
    wandb.finish()
    return val_auc

# Configurar estudio
study = optuna.create_study(
    direction="maximize",
    sampler=TPESampler(n_startup_trials=10, n_ei_candidates=24),
    pruner=HyperbandPruner(
        min_resource=1,
        reduction_factor=3,
        min_early_stopping_rate=0
    )
)

# Ejecutar optimización
study.optimize(
    objective,
    n_trials=100,
    timeout=7200,  # 2 horas
    callbacks=[WeightsAndBiasesCallback()]
)
```

### Fase 5: Análisis de Resultados

#### Visualización en Weights & Biases
```python
import wandb
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_optuna_results(study):
    """
    Analiza y visualiza los resultados de Optuna
    """
    # Obtener mejores trials
    best_trials = sorted(
        study.trials,
        key=lambda t: t.value if t.value is not None else 0,
        reverse=True
    )[:10]
    
    # DataFrame con resultados
    results_df = pd.DataFrame([
        {
            'trial_id': trial.number,
            'auc': trial.value,
            **trial.params
        }
        for trial in best_trials
        if trial.value is not None
    ])
    
    # Visualizaciones
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. AUC vs Learning Rate
    sns.scatterplot(data=results_df, x='lr', y='auc', ax=axes[0, 0])
    axes[0, 0].set_xscale('log')
    axes[0, 0].set_title('AUC vs Learning Rate')
    
    # 2. AUC vs Dropout Rate
    sns.boxplot(data=results_df, x='dropout_rate', y='auc', ax=axes[0, 1])
    axes[0, 1].set_title('AUC vs Dropout Rate')
    
    # 3. AUC vs Number of Layers
    sns.boxplot(data=results_df, x='num_layers', y='auc', ax=axes[0, 2])
    axes[0, 2].set_title('AUC vs Number of Layers')
    
    # 4. Pretrained vs From Scratch
    sns.boxplot(data=results_df, x='use_pretrained', y='auc', ax=axes[1, 0])
    axes[1, 0].set_title('Pretrained vs From Scratch')
    
    # 5. Batch Size Impact
    sns.boxplot(data=results_df, x='batch_size', y='auc', ax=axes[1, 1])
    axes[1, 1].set_title('AUC vs Batch Size')
    
    # 6. Parallel Coordinates Plot
    from optuna.visualization import plot_parallel_coordinate
    plot_parallel_coordinate(study).show()
    
    plt.tight_layout()
    plt.savefig('results/hyperparameter_analysis.png', dpi=300, bbox_inches='tight')
    
    return results_df

# Ejecutar análisis
results_df = analyze_optuna_results(study)
```

## 📊 Entregables del Laboratorio

### 1. Código Fuente Completo
- **load_medical_data.py**: Carga y preprocesamiento del dataset
- **optuna_hypersearch.py**: Implementación completa de optimización
- **analyze_results.py**: Análisis y visualización de resultados
- **wandb_utils.py**: Utilidades para integración con W&B

### 2. Modelos Entrenados
- **best_model.h5**: Mejor modelo encontrado
- **top_5_models/**: Top 5 modelos con diferentes configuraciones

### 3. Resultados y Visualizaciones
- **optuna_study.db**: Base de datos completa de Optuna
- **plots/**: Gráficos de análisis de hiperparámetros
- **results_summary.csv**: Resumen de mejores trials

### 4. Documentación
- **methodology_report.md**: Reporte metodológico completo
- **technical_documentation.md**: Documentación técnica detallada
- **README.md**: Guía de uso y reproducción

### 5. Dashboard de W&B
- **Experiment Tracking**: Dashboard completo con todos los trials
- **Hyperparameter Analysis**: Visualizaciones interactivas
- **Model Comparison**: Comparación de diferentes configuraciones

## 🎯 Criterios de Evaluación

### Componente Técnico (60%)
- **Funcionalidad del Sistema** (20%): Sistema completo y funcional
- **Optimización Implementada** (20%): Correcta implementación de TPE + Hyperband
- **Integración con W&B** (20%): Tracking completo y visualizaciones

### Componente Analítico (40%)
- **Análisis de Resultados** (20%): Insights extraídos de la optimización
- **Documentación** (20%): Reportes completos y reproducibles

### Métricas de Éxito
- **Mejora en AUC**: >5% vs baseline
- **Eficiencia**: <2 horas para 100 trials
- **Reproducibilidad**: 100% de resultados reproducibles
- **Documentación**: Guías completas y ejemplos

## 🚀 Extensiones y Mejoras

### Opcionales Avanzados
1. **Multi-objective Optimization**: Optimizar AUC y latencia simultáneamente
2. **Distributed Tuning**: Usar Ray Tune para scaling horizontal
3. **Meta-learning**: Transfer learning de hiperparámetros entre datasets
4. **AutoML Integration**: Integración con AutoKeras para NAS completo

### Aplicaciones en Producción
1. **Continuous Optimization**: Sistema de optimización continua
2. **Model Registry**: Registro automático de mejores modelos
3. **A/B Testing**: Testing automático de nuevos modelos
4. **Monitoring**: Monitoreo de drift y rendimiento

---

**Duración Estimada**: 4-6 horas  
**Dificultad**: Intermedia-Avanzada  
**Prerrequisitos**: Conocimientos de deep learning y optimización
