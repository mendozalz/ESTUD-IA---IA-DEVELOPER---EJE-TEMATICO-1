# Buenas Prácticas - Optimización y Ajuste de Modelos

## 📋 Introducción

Este documento establece las mejores prácticas y metodologías para la optimización y ajuste de modelos de IA, utilizando el marco lógico como metodología fundamental para garantizar el rendimiento, eficiencia y escalabilidad de los modelos.

## 🎯 Metodología de Marco Lógico

### **Definición del Marco Lógico**
El marco lógico es una herramienta de planificación y gestión que estructura los proyectos mediante una matriz de objetivos, indicadores, medios de verificación y supuestos críticos.

### **Componentes del Marco Lógico**

#### **1. Jerarquía de Objetivos**
```
Fin → Propósito → Componentes → Actividades
```

- **Fin**: Objetivo de desarrollo al que contribuye el proyecto
- **Propósito**: Efecto directo esperado al completar el proyecto
- **Componentes**: Resultados específicos que el proyecto debe producir
- **Actividades**: Tareas necesarias para producir los componentes

#### **2. Matriz del Marco Lógico**
| Nivel | Objetivos | Indicadores Verificables | Medios de Verificación | Supuestos Críticos |
|-------|-----------|-------------------------|----------------------|-------------------|
| **Fin** | Impacto a largo plazo | KPIs de negocio | Reportes ejecutivos | Condiciones externas |
| **Propósito** | Efecto directo | Métricas de éxito | Dashboards | Factores internos |
| **Componentes** | Resultados entregables | Especificaciones técnicas | Documentación | Recursos disponibles |
| **Actividades** | Tareas ejecutadas | Cronograma cumplido | Logs y reports | Capacitación equipo |

## 🏗️ Aplicación a Proyectos de Optimización

### **Ejemplo: Optimización de Modelo CNN**

#### **Marco Lógico - Optimización CNN**

| Nivel | Objetivo | Indicador | Verificación | Supuestos |
|-------|-----------|------------|--------------|-----------|
| **Fin** | Reducir costos computacionales | Speedup >2x | Benchmarks | Hardware disponible |
| **Propósito** | Optimizar rendimiento modelo | Latency <50ms | Dashboard de métricas | Técnicas aplicadas |
| **Componentes** | Modelo optimizado funcional | Size reducido 70% | Tests de rendimiento | Frameworks optimizados |
| **Actividades** | Aplicar técnicas de optimización | Código optimizado | Repositorio | Conocimientos técnicos |

### **Ejemplo: Ajuste de Hiperparámetros**

#### **Marco Lógico - Hiperparámetros**

| Nivel | Objetivo | Indicador | Verificación | Supuestos |
|-------|-----------|------------|--------------|-----------|
| **Fin** | Mejorar accuracy de modelos | Accuracy +15% | Validación final | Datos disponibles |
| **Propósito** | Encontrar mejores hiperparámetros | Best params encontrados | MLflow tracking | Tiempo computo |
| **Componentes** | Sistema de tuning funcional | Optimización completada | Logs de búsqueda | Herramientas tuning |
| **Actividades** | Implementar búsqueda automática | Código de tuning | Repositorio | Frameworks disponibles |

## 📋 Buenas Prácticas por Componente

### **1. Técnicas de Optimización de Modelos**

#### **✅ Qué Hacer**
- **Aplicar pruning** para eliminar conexiones innecesarias
- **Implementar cuantización** para reducir tamaño y mejorar velocidad
- **Usar knowledge distillation** para crear modelos más pequeños
- **Aplicar técnicas** de compresión avanzadas

#### **🔧 Cómo Hacerlo**
```python
# Ejemplo: Optimización completa de modelo
class ModelOptimizer:
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        self.metrics = MetricsCollector()
    
    def optimize_model(self, model, dataset):
        """Optimiza modelo usando múltiples técnicas"""
        self.logger.info("Starting comprehensive model optimization")
        
        # 1. Pruning
        pruned_model = self._apply_pruning(model, dataset)
        
        # 2. Cuantización
        quantized_model = self._apply_quantization(pruned_model, dataset)
        
        # 3. Knowledge Distillation
        distilled_model = self._apply_knowledge_distillation(
            quantized_model, dataset
        )
        
        # 4. Optimización de arquitectura
        optimized_model = self._optimize_architecture(distilled_model)
        
        # 5. Compilación optimizada
        final_model = self._compile_optimized(optimized_model)
        
        # Medir mejoras
        improvements = self._measure_improvements(model, final_model)
        
        self.logger.info(f"Model optimization completed. Improvements: {improvements}")
        return final_model, improvements
    
    def _apply_pruning(self, model, dataset):
        """Aplica pruning al modelo"""
        import tensorflow_model_optimization as tfmot
        
        # Definir política de pruning
        prune_low_magnitude = tfmot.sparsity.keras.ConstantSparsity(
            0.5, begin_step=0, frequency=100
        )
        
        # Aplicar pruning a capas densas y convolucionales
        pruned_model = tfmot.sparsity.keras.prune_low_magnitude(
            model, pruning_schedule=prune_low_magnitude
        )
        
        # Re-compilar modelo pruned
        pruned_model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.logger.info("Pruning applied: 50% sparsity")
        return pruned_model
    
    def _apply_quantization(self, model, dataset):
        """Aplica cuantización post-training"""
        # Convertir a TensorFlow Lite
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        
        # Cuantización dinámica
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        quantized_model = converter.convert()
        
        # Guardar modelo cuantizado
        with open('quantized_model.tflite', 'wb') as f:
            f.write(quantized_model)
        
        self.logger.info("Quantization applied: Dynamic range")
        return quantized_model
    
    def _apply_knowledge_distillation(self, teacher_model, dataset):
        """Aplica knowledge distillation"""
        # Crear modelo estudiante más pequeño
        student_model = self._create_student_model()
        
        # Configurar distillation
        distiller = Distiller(student=student_model, teacher=teacher_model)
        
        # Entrenar con distillation
        distiller.compile(
            optimizer='adam',
            metrics=['accuracy'],
            student_loss_fn=tf.keras.losses.categorical_crossentropy,
            distillation_loss_fn=tf.keras.losses.KLDivergence()
        )
        
        # Entrenar modelo estudiante
        distiller.fit(dataset, epochs=50, batch_size=32)
        
        self.logger.info("Knowledge distillation completed")
        return student_model
    
    def _measure_improvements(self, original_model, optimized_model):
        """Mide mejoras en rendimiento"""
        improvements = {}
        
        # Medir tamaño
        original_size = self._get_model_size(original_model)
        optimized_size = self._get_model_size(optimized_model)
        improvements['size_reduction'] = (original_size - optimized_size) / original_size
        
        # Medir latency
        original_latency = self._measure_inference_time(original_model)
        optimized_latency = self._measure_inference_time(optimized_model)
        improvements['speedup'] = original_latency / optimized_latency
        
        # Medir accuracy
        original_acc = self._evaluate_model(original_model)
        optimized_acc = self._evaluate_model(optimized_model)
        improvements['accuracy_change'] = optimized_acc - original_acc
        
        return improvements
```

#### **📊 Aplicación al Proyecto Integral**
- **Implementar técnicas** de optimización avanzadas
- **Medir impacto** en rendimiento y accuracy
- **Seleccionar técnicas** apropiadas para cada caso
- **Documentar trade-offs** entre tamaño, velocidad y accuracy

### **2. Ajuste de Hiperparámetros**

#### **✅ Qué Hacer**
- **Implementar búsqueda automática** de hiperparámetros
- **Usar técnicas avanzadas** (Bayesian optimization, TPE)
- **Paralelizar búsquedas** para acelerar proceso
- **Implementar early stopping** para búsquedas ineficientes

#### **🔧 Cómo Hacerlo**
```python
# Ejemplo: Sistema avanzado de tuning
class HyperparameterTuner:
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        self.study = None
        self.trials = []
    
    def create_study(self):
        """Crea estudio de optimización"""
        import optuna
        
        # Crear estudio con sampler apropiado
        self.study = optuna.create_study(
            direction='maximize',
            sampler=optuna.samplers.TPESampler(),
            pruner=optuna.pruners.HyperbandPruner()
        )
        
        self.logger.info("Optuna study created with TPE sampler")
        return self.study
    
    def define_search_space(self):
        """Define espacio de búsqueda de hiperparámetros"""
        search_space = {
            'learning_rate': ('loguniform', 1e-5, 1e-1),
            'batch_size': ('categorical', [16, 32, 64, 128]),
            'dropout_rate': ('uniform', 0.1, 0.5),
            'num_layers': ('int', 2, 6),
            'units_per_layer': ('categorical', [32, 64, 128, 256]),
            'activation': ('categorical', ['relu', 'gelu', 'swish']),
            'optimizer': ('categorical', ['adam', 'rmsprop', 'sgd']),
            'weight_decay': ('loguniform', 1e-6, 1e-3)
        }
        
        return search_space
    
    def objective_function(self, trial):
        """Función objetivo para optimización"""
        # Sugerir hiperparámetros
        params = trial.suggest_categorical('optimizer', ['adam', 'rmsprop'])
        lr = trial.suggest_loguniform('learning_rate', 1e-5, 1e-1)
        dropout = trial.suggest_uniform('dropout_rate', 0.1, 0.5)
        
        # Crear y entrenar modelo
        model = self._create_model(params)
        history = model.fit(
            train_data,
            validation_data=val_data,
            epochs=50,
            batch_size=32,
            callbacks=[self._create_pruning_callback(trial)],
            verbose=0
        )
        
        # Obtener mejor métrica de validación
        best_val_accuracy = max(history.history['val_accuracy'])
        
        # Reportar a Optuna
        trial.report(best_val_accuracy, epoch)
        
        # Handle pruning
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
        
        return best_val_accuracy
    
    def run_optimization(self, n_trials=100):
        """Ejecuta optimización de hiperparámetros"""
        self.logger.info(f"Starting hyperparameter optimization with {n_trials} trials")
        
        # Crear estudio
        study = self.create_study()
        
        # Ejecutar optimización
        study.optimize(
            self.objective_function,
            n_trials=n_trials,
            timeout=3600,  # 1 hora timeout
            n_jobs=-1  # Paralelizar
        )
        
        # Obtener mejores parámetros
        best_params = study.best_params
        best_value = study.best_value
        
        # Guardar resultados
        self._save_optimization_results(study)
        
        self.logger.info(f"Optimization completed. Best accuracy: {best_value:.4f}")
        self.logger.info(f"Best parameters: {best_params}")
        
        return best_params, best_value
    
    def _create_pruning_callback(self, trial):
        """Crea callback para pruning de trials"""
        import optuna
        
        class OptunaPruningCallback(tf.keras.callbacks.Callback):
            def __init__(self, trial):
                super().__init__()
                self.trial = trial
            
            def on_epoch_end(self, epoch, logs=None):
                self.trial.report(logs['val_accuracy'], epoch)
                if self.trial.should_prune():
                    self.model.stop_training = True
        
        return OptunaPruningCallback(trial)
```

#### **📊 Aplicación al Proyecto Integral**
- **Implementar búsqueda** automática eficiente
- **Usar técnicas** avanzadas de optimización
- **Paralelizar procesos** para acelerar búsqueda
- **Visualizar y analizar** resultados de tuning

### **3. Arquitectura Neural (NAS)**

#### **✅ Qué Hacer**
- **Implementar Neural Architecture Search**
- **Usar algoritmos evolutivos** para encontrar arquitecturas
- **Definir espacio de búsqueda** restringido pero flexible
- **Evaluar arquitecturas** de manera eficiente

#### **🔧 Cómo Hacerlo**
```python
# Ejemplo: Neural Architecture Search
class NeuralArchitectureSearch:
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        self.search_space = self._define_search_space()
    
    def _define_search_space(self):
        """Define espacio de búsqueda de arquitecturas"""
        search_space = {
            'num_conv_layers': (2, 8),
            'filters_per_layer': [(16, 32, 64, 128, 256)],
            'kernel_sizes': [(3, 5, 7)],
            'pooling': ['max', 'avg'],
            'dropout_rates': (0.0, 0.5),
            'dense_layers': (1, 4),
            'dense_units': [(64, 128, 256, 512)],
            'activations': ['relu', 'gelu', 'swish'],
            'batch_norm': [True, False]
        }
        
        return search_space
    
    def create_model_from_architecture(self, architecture):
        """Crea modelo a partir de arquitectura"""
        model = tf.keras.Sequential()
        
        # Capas convolucionales
        input_shape = self.config['input_shape']
        for i in range(architecture['num_conv_layers']):
            if i == 0:
                model.add(tf.keras.layers.Conv2D(
                    architecture['filters_per_layer'][i],
                    architecture['kernel_sizes'][i],
                    activation=architecture['activations'],
                    input_shape=input_shape,
                    padding='same'
                ))
            else:
                model.add(tf.keras.layers.Conv2D(
                    architecture['filters_per_layer'][i],
                    architecture['kernel_sizes'][i],
                    activation=architecture['activations'],
                    padding='same'
                ))
            
            # Batch normalization
            if architecture['batch_norm']:
                model.add(tf.keras.layers.BatchNormalization())
            
            # Dropout
            model.add(tf.keras.layers.Dropout(architecture['dropout_rates'][i]))
            
            # Pooling
            if architecture['pooling'][i] == 'max':
                model.add(tf.keras.layers.MaxPooling2D((2, 2)))
            else:
                model.add(tf.keras.layers.AveragePooling2D((2, 2)))
        
        # Flatten y capas densas
        model.add(tf.keras.layers.Flatten())
        
        for i in range(architecture['dense_layers']):
            model.add(tf.keras.layers.Dense(
                architecture['dense_units'][i],
                activation=architecture['activations']
            ))
            model.add(tf.keras.layers.Dropout(architecture['dropout_rates'][-1]))
        
        # Capa de salida
        model.add(tf.keras.layers.Dense(
            self.config['num_classes'],
            activation='softmax'
        ))
        
        return model
    
    def evolutionary_search(self, generations=10, population_size=20):
        """Búsqueda evolutiva de arquitecturas"""
        import random
        from deap import base, creator, tools, algorithms
        
        # Definir fitness function
        def evaluate_architecture(architecture):
            model = self.create_model_from_architecture(architecture)
            model.compile(
                optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            
            # Entrenamiento rápido para evaluación
            history = model.fit(
                train_data,
                validation_data=val_data,
                epochs=10,
                batch_size=32,
                verbose=0
            )
            
            # Fitness: accuracy - complejidad
            accuracy = max(history.history['val_accuracy'])
            complexity = self._calculate_complexity(architecture)
            fitness = accuracy - 0.001 * complexity
            
            return fitness
        
        # Configurar algoritmo genético
        creator.createFitness("FitnessMax", evaluate_architecture)
        
        # Ejecutar búsqueda evolutiva
        best_architecture = algorithms.eaSimple(
            population,
            toolbox,
            cxpb=0.7,
            mutpb=0.3,
            ngen=generations,
            stats=True,
            halloffame="best_architecture"
        )
        
        return best_architecture
```

#### **📊 Aplicación al Proyecto Integral**
- **Implementar NAS** para encontrar arquitecturas óptimas
- **Usar algoritmos evolutivos** para búsqueda eficiente
- **Balancear complejidad** y rendimiento
- **Documentar arquitecturas** descubiertas

## 🎯 Aplicación a Proyectos Integrales

### **Metodología de Implementación**

#### **Paso 1: Definición del Marco Lógico**
```python
# Plantilla para definir marco lógico
def define_optimization_logical_framework(project_name):
    return {
        'project_name': project_name,
        'fin': {
            'objetivo': 'Maximizar eficiencia de modelos de IA',
            'indicadores': ['Speedup >2x', 'Size reduction >70%', 'Accuracy mantenido'],
            'verificacion': ['Benchmarks', 'Profile reports', 'Validation metrics'],
            'supuestos': ['Hardware disponible', 'Tiempo computo', 'Datos calidad']
        },
        'propósito': {
            'objetivo': 'Optimizar rendimiento y reducir recursos',
            'indicadores': ['Latency <50ms', 'Memory <100MB', 'Accuracy >95%'],
            'verificacion': ['Performance dashboard', 'Resource monitoring', 'Accuracy tests'],
            'supuestos': ['Modelos base disponibles', 'Técnicas conocidas', 'Herramientas instaladas']
        },
        'componentes': {
            'objetivo': 'Sistema de optimización funcional',
            'indicadores': ['Modelo optimizado', 'Hiperparámetros ajustados', 'NAS completado'],
            'verificacion': ['Repositorio', 'MLflow tracking', 'Benchmark reports'],
            'supuestos': ['Frameworks optimizados', 'Recursos computo', 'Tiempo disponible']
        },
        'actividades': {
            'objetivo': 'Implementar técnicas de optimización',
            'indicadores': ['Pruning aplicado', 'Cuantización completada', 'Tuning finalizado'],
            'verificacion': ['Git commits', 'Optimization logs', 'Performance metrics'],
            'supuestos': ['Conocimientos técnicos', 'Herramientas disponibles', 'Tiempo dedicado']
        }
    }
```

### **Ejemplo Completo: Proyecto Integral de Optimización**

#### **Marco Lógico del Proyecto**
```yaml
project_name: "Sistema Integral de Optimización de Modelos"

fin:
  objetivo: "Maximizar eficiencia computacional de modelos de IA"
  indicadores:
    - name: "Speedup computacional"
      target: ">3x"
      current: "1x"
    - name: "Reducción de tamaño"
      target: ">80%"
      current: "0%"
    - name: "Mantenimiento de accuracy"
      target: ">95%"
      current: "100%"
  verificacion:
    - "Benchmarks sistemáticos"
    - "Reportes de perfilado"
    - "Métricas de validación"
  supuestos:
    - "Hardware de alto rendimiento"
    - "Tiempo de computo disponible"
    - "Modelos base de calidad"

propósito:
  objetivo: "Optimizar rendimiento y reducir uso de recursos"
  indicadores:
    - name: "Latencia de inferencia"
      target: "<30ms"
      current: "100ms"
    - name: "Uso de memoria"
      target: "<50MB"
      current: "200MB"
    - name: "Throughput"
      target: ">1000 inferences/sec"
      current: "100 inferences/sec"
  verificacion:
    - "Dashboard de rendimiento"
    - "Monitoreo de recursos"
    - "Tests de carga"
  supuestos:
    - "Modelos base optimizados"
    - "Técnicas de optimización conocidas"
    - "Frameworks de alto rendimiento"

componentes:
  objetivo: "Sistema de optimización completo"
  indicadores:
    - name: "Modelo pruning aplicado"
      target: "100%"
      current: "0%"
    - name: "Cuantización implementada"
      target: "100%"
      current: "0%"
    - name: "Hiperparámetros optimizados"
      target: "Best params encontrados"
      current: "Default"
    - name: "NAS completado"
      target: "Top 5 arquitecturas"
      current: "0"
  verificacion:
    - "Repositorio Git"
    - "MLflow tracking"
    - "Reportes de benchmark"
  supuestos:
    - "Frameworks de optimización"
    - "Recursos computacionales"
    - "Tiempo para experimentación"

actividades:
  objetivo: "Implementar pipeline de optimización"
  indicadores:
    - name: "Pruning implementado"
      target: "100%"
      current: "0%"
    - name: "Cuantización completada"
      target: "100%"
      current: "0%"
    - name: "Búsqueda de hiperparámetros"
      target: "100 trials"
      current: "0 trials"
    - name: "NAS ejecutado"
      target: "10 generaciones"
      current: "0 generaciones"
  verificacion:
    - "Commits de optimización"
    - "Logs de experimentos"
    - "Métricas de rendimiento"
  supuestos:
    - "Conocimientos de optimización"
    - "Herramientas disponibles"
    - "Tiempo dedicado al proyecto"
```

## 📊 Métricas y Verificación

### **Dashboard de Seguimiento del Marco Lógico**

```python
# Ejemplo: Dashboard de optimización
class OptimizationDashboard:
    def create_dashboard(self, framework):
        """Crear dashboard para seguimiento del marco lógico"""
        
        dashboard = {
            'fin': self.create_fin_section(framework['fin']),
            'propósito': self.create_purpose_section(framework['propósito']),
            'componentes': self.create_components_section(framework['componentes']),
            'actividades': self.create_activities_section(framework['actividades'])
        }
        
        return dashboard
    
    def track_optimization_metrics(self, framework):
        """Seguimiento de métricas de optimización"""
        
        progress = {}
        
        for level in ['fin', 'propósito', 'componentes', 'actividades']:
            indicators = framework[level]['indicadores']
            progress[level] = self.calculate_progress(indicators)
        
        return progress
```

## 🚀 Conclusión

La aplicación del marco lógico a proyectos de optimización de modelos proporciona:

- **Claridad estratégica** en objetivos y resultados esperados
- **Medición objetiva** del progreso y éxito
- **Verificación sistemática** de logros
- **Gestión de riesgos** mediante identificación de supuestos
- **Mejora continua** basada en evidencia

Al aplicar esta metodología, los estudiantes desarrollarán habilidades para:

- **Optimizar modelos** para producción
- **Ajustar hiperparámetros** sistemáticamente
- **Implementar NAS** para descubrir arquitecturas
- **Balancear trade-offs** entre rendimiento y recursos
- **Documentar resultados** de manera profesional

---

**Esta guía proporciona el marco metodológico para que los estudiantes apliquen sistemáticamente las mejores prácticas en todos sus proyectos de optimización, garantizando la calidad y el impacto real de sus soluciones.**
