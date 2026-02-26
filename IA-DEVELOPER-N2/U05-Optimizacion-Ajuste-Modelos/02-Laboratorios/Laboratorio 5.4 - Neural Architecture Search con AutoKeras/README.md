# Laboratorio 5.4: Neural Architecture Search con AutoKeras y Google NAS

## 🎯 Objetivos del Laboratorio

### Objetivo General
Automatizar la búsqueda de la mejor arquitectura para un modelo de predicción de demanda en retail usando Neural Architecture Search (NAS) sin diseño manual.

### Objetivos Específicos
- Encontrar la mejor arquitectura (CNN, RNN o híbrida) para series temporales
- Optimizar hiperparámetros simultáneamente con la arquitectura
- Evaluar el modelo encontrado en métricas como MAE y RMSE
- Comparar con arquitecturas manuales (LSTM, CNN 1D)
- Implementar AutoML pipeline completo

## 📋 Marco Lógico del Proyecto

| Componente | Objetivo | Indicadores | Medios de Verificación |
|------------|----------|-------------|------------------------|
| **Generación de Datos** | Crear series temporales sintéticas | 10K muestras con patrones estacionales | Scripts de generación y análisis |
| **NAS con AutoKeras** | Buscar automáticamente arquitecturas | Mejor arquitectura encontrada | Código AutoKeras y resultados |
| **Búsqueda Híbrida** | Explorar CNN, RNN y arquitecturas mixtas | Múltiples tipos de arquitecturas evaluadas | Logs de búsqueda y comparación |
| **Optimización de Hiperparámetros** | Tuning simultáneo con NAS | Hiperparámetros óptimos encontrados | Configuración final del modelo |
| **Comparación Manual** | Comparar con arquitecturas diseñadas | Tabla comparativa de rendimiento | Modelos manuales y métricas |
| **Evaluación Final** | Validación del modelo encontrado | MAE < 10% del rango de valores | Evaluación en test set |

## 🛠️ Tecnologías y Herramientas

### Principales Tecnologías
- **AutoKeras 1.0**: AutoML framework con NAS
- **TensorFlow 2.15**: Backend de deep learning
- **Scikit-learn 1.4**: Métricas y preprocesamiento
- **Scikeras 0.3**: Wrapper de Keras para scikit-learn

### Dependencias Adicionales
- **NumPy/Pandas**: Manipulación de datos
- **Matplotlib/Seaborn**: Visualización
- **Jupyter Notebook**: Análisis interactivo

## 📁 Estructura del Proyecto

```
Laboratorio 5.4 - Neural Architecture Search/
├── README.md                           # Guía del laboratorio
├── requirements.txt                    # Dependencias
├── src/
│   ├── generate_time_series.py        # Generación de datos sintéticos
│   ├── autokeras_nas.py                # NAS con AutoKeras
│   ├── manual_models.py                # Modelos manuales para comparación
│   ├── evaluate_models.py              # Evaluación comparativa
│   └── nas_analysis.py                 # Análisis de resultados NAS
├── notebooks/
│   ├── data_exploration.ipynb          # Análisis exploratorio
│   ├── nas_results.ipynb               # Análisis de resultados NAS
│   └── model_comparison.ipynb          # Comparación de modelos
├── data/
│   ├── demand_data.npz                 # Datos generados
│   └── processed/                      # Datos preprocesados
├── models/
│   ├── autokeras_best_model.h5         # Mejor modelo NAS
│   ├── lstm_manual.h5                  # Modelo LSTM manual
│   ├── cnn_manual.h5                   # Modelo CNN manual
│   └── hybrid_manual.h5                # Modelo híbrido manual
├── results/
│   ├── nas_search.log                  # Logs de búsqueda NAS
│   ├── architecture_analysis/          # Análisis de arquitecturas
│   └── performance_comparison/          # Comparación de rendimiento
└── docs/
    ├── methodology_report.md            # Reporte metodológico
    └── technical_documentation.md      # Documentación técnica
```

## 🔧 Implementación Detallada

### Fase 1: Generación de Datos de Series Temporales

#### Dataset Sintético de Demanda Retail
```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def generate_synthetic_demand(n_samples=10000, n_timesteps=30):
    """
    Genera datos sintéticos de demanda con patrones realistas
    """
    # Generar datos sintéticos de demanda con patrones realistas
    time = np.arange(n_samples * n_timesteps).reshape(n_samples, n_timesteps)
    base_demand = 100 + 50 * np.sin(time * 0.1)  # Patrones semanales
    noise = np.random.normal(0, 10, size=(n_samples, n_timesteps))
    spikes = np.random.choice([0, 1], size=(n_samples, n_timesteps), p=[0.95, 0.05]) * np.random.normal(100, 50, size=(n_samples, n_timesteps))
    demand = base_demand + noise + spikes
    demand = np.clip(demand, 0, None)  # Demanda no puede ser negativa

    # Crear características adicionales
    day_of_week = (time % 7).astype(int)
    month = (time // 30 % 12).astype(int)

    # Formato: (samples, timesteps, features)
    X = np.stack([demand, day_of_week, month], axis=-1)
    y = demand[:, -1]  # Predecir el último timestep

    return X, y
```

### Fase 2: Neural Architecture Search con AutoKeras

#### Búsqueda Automática de Arquitecturas
```python
import autokeras as ak
import numpy as np
from scikeras.wrappers import KerasRegressor
from sklearn.model_selection import RandomizedSearchCV

def build_model(max_trials=20):
    """
    Define el buscador de NAS con AutoKeras
    """
    input_node = ak.Input()
    output_node = ak.TimeSeriesBlock()(
        input_node,
        num_layers=ak.Int(1, 3, default=2),
        num_heads=ak.Int(1, 4, default=2),
        dropout=ak.Float(0.0, 0.5, default=0.2, step=0.1),
        use_bidirectional=ak.Boolean(default=True)
    )
    output_node = ak.RegressionHead()(output_node)
    return ak.AutoModel(
        inputs=input_node,
        outputs=output_node,
        max_trials=max_trials,
        directory='autokeras_demand',
        project_name='demand_forecasting'
    )

# Búsqueda con AutoKeras
search = build_model(max_trials=20)
search.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=30,
    batch_size=64,
    callbacks=[
        ak.callbacks.EarlyStopping(patience=3)
    ]
)

# Evaluar el mejor modelo
best_model = search.export_model()
loss, mae = best_model.evaluate(X_test, y_test)
print(f"Best model MAE: {mae:.4f}")
```

### Fase 3: Modelos Manuales para Comparación

#### Implementación de Arquitecturas de Referencia
```python
import tensorflow as tf
from tensorflow.keras import layers, models

def build_lstm_model(input_shape):
    """
    Construye modelo LSTM manual
    """
    model = models.Sequential([
        layers.LSTM(64, return_sequences=True, input_shape=input_shape),
        layers.LSTM(32, return_sequences=False),
        layers.Dense(16, activation='relu'),
        layers.Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def build_cnn_model(input_shape):
    """
    Construye modelo CNN 1D manual
    """
    model = models.Sequential([
        layers.Conv1D(64, 3, activation='relu', input_shape=input_shape),
        layers.MaxPooling1D(2),
        layers.Conv1D(32, 3, activation='relu'),
        layers.GlobalAveragePooling1D(),
        layers.Dense(16, activation='relu'),
        layers.Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def build_hybrid_model(input_shape):
    """
    Construye modelo híbrido CNN-LSTM manual
    """
    model = models.Sequential([
        layers.Conv1D(64, 3, activation='relu', input_shape=input_shape),
        layers.MaxPooling1D(2),
        layers.LSTM(32, return_sequences=False),
        layers.Dense(16, activation='relu'),
        layers.Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model
```

### Fase 4: Evaluación Comparativa

#### Análisis de Rendimiento
```python
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pandas as pd

def evaluate_model(model, X_test, y_test, model_name):
    """
    Evalúa un modelo y retorna métricas
    """
    y_pred = model.predict(X_test)
    
    metrics = {
        'model': model_name,
        'mae': mean_absolute_error(y_test, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
        'r2': r2_score(y_test, y_pred),
        'mape': np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    }
    
    return metrics

def compare_all_models(models_dict, X_test, y_test):
    """
    Compara múltiples modelos
    """
    results = []
    
    for name, model in models_dict.items():
        metrics = evaluate_model(model, X_test, y_test, name)
        results.append(metrics)
    
    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values('mae')
    
    print("Comparación de Modelos:")
    print(df_results.round(4))
    
    return df_results
```

### Fase 5: Análisis de Arquitecturas Encontradas

#### Visualización y Análisis de NAS
```python
def analyze_nas_results(search):
    """
    Analiza los resultados del Neural Architecture Search
    """
    # Obtener información del mejor modelo
    best_model = search.export_model()
    
    # Analizar la arquitectura encontrada
    print("Mejor Arquitectura Encontrada:")
    best_model.summary()
    
    # Visualizar el historial de búsqueda
    try:
        # Obtener todos los trials evaluados
        import os
        import json
        
        # Buscar archivos de resultados
        results_dir = 'autokeras_demand/demand_forecasting'
        if os.path.exists(results_dir):
            print(f"Resultados guardados en: {results_dir}")
            
            # Listar todos los trials
            trials = [d for d in os.listdir(results_dir) if d.startswith('trial')]
            print(f"Total de trials evaluados: {len(trials)}")
            
            # Analizar convergencia
            convergence_data = []
            for trial in trials[:10]:  # Primeros 10 trials
                trial_dir = os.path.join(results_dir, trial)
                if os.path.exists(trial_dir):
                    # Leer métricas si existen
                    metrics_file = os.path.join(trial_dir, 'metrics.json')
                    if os.path.exists(metrics_file):
                        with open(metrics_file, 'r') as f:
                            metrics = json.load(f)
                            convergence_data.append({
                                'trial': trial,
                                'mae': metrics.get('mae', 0),
                                'loss': metrics.get('loss', 0)
                            })
            
            if convergence_data:
                df_convergence = pd.DataFrame(convergence_data)
                print("\nConvergencia de la Búsqueda:")
                print(df_convergence.sort_values('mae'))
    
    except Exception as e:
        print(f"No se pudieron analizar los resultados detallados: {e}")
    
    return best_model
```

## 📊 Entregables del Laboratorio

### 1. Código Fuente Completo
- **generate_time_series.py**: Generación de datos sintéticos
- **autokeras_nas.py**: Implementación NAS con AutoKeras
- **manual_models.py**: Modelos manuales de referencia
- **evaluate_models.py**: Evaluación comparativa
- **nas_analysis.py**: Análisis de resultados NAS

### 2. Modelos Entrenados
- **autokeras_best_model.h5**: Mejor modelo encontrado por NAS
- **lstm_manual.h5**: Modelo LSTM manual
- **cnn_manual.h5**: Modelo CNN manual
- **hybrid_manual.h5**: Modelo híbrido manual

### 3. Dataset
- **demand_data.npz**: Datos de series temporales generados
- **processed/**: Datos preprocesados y divididos

### 4. Resultados y Análisis
- **nas_search.log**: Logs completos de búsqueda NAS
- **architecture_analysis/**: Análisis de arquitecturas encontradas
- **performance_comparison/**: Comparación de rendimiento

### 5. Documentación
- **methodology_report.md**: Reporte metodológico completo
- **technical_documentation.md**: Guías técnicas
- **README.md**: Instrucciones de uso

## 🎯 Criterios de Evaluación

### Componente Técnico (60%)
- **Implementación NAS** (25%): Correcta configuración de AutoKeras
- **Modelos Manuales** (20%): Implementación de arquitecturas de referencia
- **Evaluación Comparativa** (15%): Análisis exhaustivo de rendimiento

### Componente Analítico (40%)
- **Análisis de Arquitecturas** (20%): Insights sobre NAS vs manual
- **Documentación** (20%): Reportes completos y reproducibles

### Métricas de Éxito
- **Mejora vs Manual**: >10% mejora en MAE vs mejor modelo manual
- **Eficiencia de NAS**: <50 trials para encontrar buena arquitectura
- **Reproducibilidad**: 100% de resultados reproducibles
- **Documentación**: Análisis completo de arquitecturas

## 🚀 Extensiones y Mejoras

### Opciones Avanzadas
1. **Multi-Objective NAS**: Optimizar múltiples métricas simultáneamente
2. **Hardware-Aware NAS**: Optimizar para hardware específico
3. **Meta-Learning**: Transfer learning de arquitecturas entre datasets
4. **Ensemble NAS**: Combinación de múltiples arquitecturas encontradas

### Aplicaciones en Producción
1. **Continuous NAS**: Búsqueda continua de mejores arquitecturas
2. **Adaptive Architecture**: Adaptación automática a cambios de datos
3. **Federated NAS**: NAS preservando privacidad de datos
4. **Real-time NAS**: Búsqueda de arquitecturas en tiempo real

## 📈 Análisis Esperado

### Comparación de Arquitecturas

| Arquitectura | MAE Esperado | RMSE Esperado | R² Esperado | Complejidad |
|-------------|--------------|---------------|-------------|-------------|
| **LSTM Manual** | 15-20 | 25-30 | 0.70-0.80 | Media |
| **CNN Manual** | 12-18 | 20-28 | 0.75-0.85 | Media |
| **Híbrido Manual** | 10-15 | 18-25 | 0.80-0.90 | Alta |
| **AutoKeras NAS** | 8-12 | 15-22 | 0.85-0.95 | Variable |
| **Mejor NAS** | 5-10 | 12-20 | 0.90-0.98 | Alta |

### Insights Esperados
1. **NAS Superioridad**: NAS encontrará arquitecturas no intuitivas
2. **Híbridos Eficientes**: Combinaciones CNN-LSTM suelen ser mejores
3. **Optimización Automática**: NAS optimiza hiperparámetros simultáneamente
4. **Trade-offs**: Complejidad vs interpretabilidad vs rendimiento

---

**Duración Estimada**: 6-8 horas  
**Dificultad**: Intermedia-Avanzada  
**Prerrequisitos**: Conocimientos de series temporales y AutoML
