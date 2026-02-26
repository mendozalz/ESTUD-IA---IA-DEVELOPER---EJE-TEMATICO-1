"""
Optimización de Hiperparámetros con Optuna y Weights & Biases
Laboratorio 5.1 - Hyperparameter Tuning Avanzado con Optuna y Weights & Biases
"""

import optuna
from optuna.samplers import TPESampler
from optuna.pruners import HyperbandPruner
from optuna.integration.wandb import WeightsAndBiasesCallback
from optuna.visualization import plot_optimization_history, plot_parallel_coordinate, plot_param_importances

import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0, ResNet50V2
from tensorflow.keras import layers, models, optimizers, regularizers, callbacks
import wandb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from typing import Dict, Any, Tuple
import os
import json
from datetime import datetime

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HyperparameterOptimizer:
    """
    Clase para optimización de hiperparámetros con Optuna y W&B
    """
    
    def __init__(self, 
                 project_name: str = "medical-image-tuning",
                 study_name: str = None,
                 direction: str = "maximize",
                 metric_name: str = "val_auc"):
        
        self.project_name = project_name
        self.study_name = study_name or f"study_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.direction = direction
        self.metric_name = metric_name
        
        # Configuración de Optuna
        self.sampler = TPESampler(n_startup_trials=10, n_ei_candidates=24)
        self.pruner = HyperbandPruner(
            min_resource=1,
            reduction_factor=3,
            min_early_stopping_rate=0
        )
        
        # Callbacks
        self.wandb_callback = WeightsAndBiasesCallback(
            metric_name=self.metric_name,
            wandb_kwargs={
                "project": self.project_name,
                "config": {
                    "framework": "TensorFlow",
                    "dataset": "CheXpert",
                    "optimization_method": "TPE + Hyperband"
                }
            }
        )
        
        # Estudio de Optuna
        self.study = None
        self.best_params = None
        self.best_value = None
        
        logger.info(f"HyperparameterOptimizer inicializado: {self.study_name}")
    
    def build_model(self, trial: optuna.Trial) -> tf.keras.Model:
        """
        Construye un modelo con hiperparámetros variables
        """
        # Hiperparámetros de arquitectura
        model_type = trial.suggest_categorical('model_type', ['EfficientNetB0', 'ResNet50V2', 'CustomCNN'])
        use_pretrained = trial.suggest_categorical('use_pretrained', [True, False])
        num_layers = trial.suggest_int('num_layers', 1, 4)
        dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5, step=0.05)
        l2_reg = trial.suggest_float('l2_reg', 1e-5, 1e-2, log=True)
        use_batch_norm = trial.suggest_categorical('use_batch_norm', [True, False])
        
        # Hiperparámetros de entrenamiento
        lr = trial.suggest_float('lr', 1e-5, 1e-3, log=True)
        optimizer_type = trial.suggest_categorical('optimizer_type', ['Adam', 'RMSprop', 'SGD'])
        
        # Construir modelo base
        inputs = layers.Input(shape=(224, 224, 3))
        
        if model_type == 'EfficientNetB0':
            base_model = EfficientNetB0(
                include_top=False,
                weights='imagenet' if use_pretrained else None,
                input_shape=(224, 224, 3)
            )
        elif model_type == 'ResNet50V2':
            base_model = ResNet50V2(
                include_top=False,
                weights='imagenet' if use_pretrained else None,
                input_shape=(224, 224, 3)
            )
        else:  # CustomCNN
            base_model = self._build_custom_cnn(trial)
        
        # Congelar capas base si es pretrained
        if use_pretrained and hasattr(base_model, 'trainable'):
            base_model.trainable = False
        
        # Construir modelo completo
        x = base_model(inputs, training=False)
        x = layers.GlobalAveragePooling2D()(x)
        
        # Capas densas con regularización
        for i in range(num_layers):
            x = layers.Dense(
                128 // (i + 1),  # Reducir unidades progresivamente
                activation='relu',
                kernel_regularizer=regularizers.l2(l2_reg)
            )(x)
            
            if use_batch_norm:
                x = layers.BatchNormalization()(x)
            
            x = layers.Dropout(dropout_rate)(x)
        
        outputs = layers.Dense(1, activation='sigmoid')(x)
        model = models.Model(inputs=inputs, outputs=outputs)
        
        # Configurar optimizador
        if optimizer_type == 'Adam':
            optimizer = optimizers.Adam(learning_rate=lr)
        elif optimizer_type == 'RMSprop':
            optimizer = optimizers.RMSprop(learning_rate=lr)
        else:  # SGD
            optimizer = optimizers.SGD(learning_rate=lr, momentum=0.9)
        
        # Compilar modelo
        model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=[
                'accuracy',
                tf.keras.metrics.AUC(name=self.metric_name),
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall')
            ]
        )
        
        return model
    
    def _build_custom_cnn(self, trial: optuna.Trial) -> tf.keras.Model:
        """
        Construye una CNN personalizada
        """
        filters = trial.suggest_categorical('filters', [32, 64, 128])
        kernel_size = trial.suggest_categorical('kernel_size', [3, 5])
        
        model = models.Sequential([
            layers.Conv2D(filters, (kernel_size, kernel_size), activation='relu', input_shape=(224, 224, 3)),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(filters * 2, (kernel_size, kernel_size), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(filters * 4, (kernel_size, kernel_size), activation='relu'),
            layers.GlobalAveragePooling2D()
        ])
        
        return model
    
    def objective(self, trial: optuna.Trial, train_ds, val_ds) -> float:
        """
        Función objetivo para Optuna
        """
        # Hiperparámetros de entrenamiento
        batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
        epochs = trial.suggest_int('epochs', 10, 50)
        patience = trial.suggest_int('patience', 3, 10)
        
        # Construir modelo
        model = self.build_model(trial)
        
        # Callbacks
        callbacks_list = [
            optuna.integration.TFKerasPruningCallback(trial, self.metric_name),
            callbacks.EarlyStopping(
                monitor=self.metric_name,
                patience=patience,
                restore_best_weights=True,
                mode='max'
            ),
            callbacks.ReduceLROnPlateau(
                monitor=self.metric_name,
                factor=0.5,
                patience=patience // 2,
                min_lr=1e-7,
                mode='max'
            )
        ]
        
        # Entrenar modelo
        try:
            history = model.fit(
                train_ds,
                validation_data=val_ds,
                epochs=epochs,
                callbacks=callbacks_list,
                verbose=0
            )
            
            # Obtener mejor métrica
            best_metric = max(history.history[self.metric_name])
            
            # Reportar métrica para pruning
            trial.report(best_metric, step=epochs)
            
            # Manejar pruning
            if trial.should_prune():
                raise optuna.TrialPruned()
            
            return best_metric
            
        except Exception as e:
            logger.error(f"Error en trial {trial.number}: {e}")
            raise optuna.TrialPruned()
    
    def optimize(self, 
                 train_ds, 
                 val_ds, 
                 n_trials: int = 100, 
                 timeout: int = 7200,
                 save_study: bool = True) -> Dict[str, Any]:
        """
        Ejecuta la optimización de hiperparámetros
        """
        logger.info(f"Iniciando optimización: {n_trials} trials, {timeout}s timeout")
        
        # Crear estudio
        self.study = optuna.create_study(
            direction=self.direction,
            sampler=self.sampler,
            pruner=self.pruner,
            study_name=self.study_name
        )
        
        # Ejecutar optimización
        self.study.optimize(
            lambda trial: self.objective(trial, train_ds, val_ds),
            n_trials=n_trials,
            timeout=timeout,
            callbacks=[self.wandb_callback]
        )
        
        # Guardar mejores resultados
        self.best_params = self.study.best_params
        self.best_value = self.study.best_value
        
        logger.info(f"Optimización completada:")
        logger.info(f"  Mejor valor: {self.best_value:.4f}")
        logger.info(f"  Mejores parámetros: {self.best_params}")
        
        # Guardar estudio
        if save_study:
            self.save_study()
        
        return {
            'best_params': self.best_params,
            'best_value': self.best_value,
            'n_trials_completed': len(self.study.trials),
            'n_trials_pruned': sum(1 for t in self.study.trials if t.state == optuna.trial.TrialState.PRUNED)
        }
    
    def save_study(self, save_dir: str = 'optuna_results'):
        """
        Guarda el estudio de Optuna
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # Guardar estudio
        study_path = os.path.join(save_dir, f'{self.study_name}.db')
        optuna.storage.RDBStorage(f'sqlite:///{study_path}')
        
        # Guardar resultados en JSON
        results = {
            'study_name': self.study_name,
            'best_params': self.best_params,
            'best_value': self.best_value,
            'trials': []
        }
        
        for trial in self.study.trials:
            trial_data = {
                'number': trial.number,
                'value': trial.value,
                'params': trial.params,
                'state': trial.state.name,
                'datetime': trial.datetime_start.isoformat() if trial.datetime_start else None
            }
            results['trials'].append(trial_data)
        
        with open(os.path.join(save_dir, 'optimization_results.json'), 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Estudio guardado en: {save_dir}")
    
    def visualize_results(self, save_dir: str = 'optuna_results'):
        """
        Genera visualizaciones de los resultados
        """
        if not self.study:
            logger.error("No hay estudio para visualizar")
            return
        
        os.makedirs(save_dir, exist_ok=True)
        
        # Historial de optimización
        fig = plot_optimization_history(self.study)
        fig.write_image(os.path.join(save_dir, 'optimization_history.png'))
        
        # Gráfico de coordenadas paralelas
        fig = plot_parallel_coordinate(self.study)
        fig.write_image(os.path.join(save_dir, 'parallel_coordinate.png'))
        
        # Importancia de parámetros
        fig = plot_param_importances(self.study)
        fig.write_image(os.path.join(save_dir, 'param_importances.png'))
        
        # Análisis adicional
        self._create_custom_visualizations(save_dir)
        
        logger.info(f"Visualizaciones guardadas en: {save_dir}")
    
    def _create_custom_visualizations(self, save_dir: str):
        """
        Crea visualizaciones personalizadas
        """
        # Extraer datos de trials
        trials_data = []
        for trial in self.study.trials:
            if trial.value is not None:
                trial_data = {
                    'trial_id': trial.number,
                    'value': trial.value,
                    **trial.params
                }
                trials_data.append(trial_data)
        
        if not trials_data:
            return
        
        df = pd.DataFrame(trials_data)
        
        # Gráficos de análisis
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. Distribución de valores
        axes[0, 0].hist(df['value'], bins=20, alpha=0.7)
        axes[0, 0].set_title('Distribución de Valores')
        axes[0, 0].set_xlabel('Valor')
        axes[0, 0].set_ylabel('Frecuencia')
        
        # 2. Valor vs Learning Rate
        if 'lr' in df.columns:
            axes[0, 1].scatter(df['lr'], df['value'], alpha=0.6)
            axes[0, 1].set_xscale('log')
            axes[0, 1].set_title('Valor vs Learning Rate')
            axes[0, 1].set_xlabel('Learning Rate')
            axes[0, 1].set_ylabel('Valor')
        
        # 3. Valor vs Dropout Rate
        if 'dropout_rate' in df.columns:
            sns.boxplot(data=df, x='dropout_rate', y='value', ax=axes[0, 2])
            axes[0, 2].set_title('Valor vs Dropout Rate')
        
        # 4. Valor vs Number of Layers
        if 'num_layers' in df.columns:
            sns.boxplot(data=df, x='num_layers', y='value', ax=axes[1, 0])
            axes[1, 0].set_title('Valor vs Number of Layers')
        
        # 5. Valor vs Model Type
        if 'model_type' in df.columns:
            sns.boxplot(data=df, x='model_type', y='value', ax=axes[1, 1])
            axes[1, 1].set_title('Valor vs Model Type')
        
        # 6. Valor vs Batch Size
        if 'batch_size' in df.columns:
            sns.boxplot(data=df, x='batch_size', y='value', ax=axes[1, 2])
            axes[1, 2].set_title('Valor vs Batch Size')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'custom_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def train_best_model(self, train_ds, val_ds, save_path: str = 'best_model.h5'):
        """
        Entrena el mejor modelo con todos los datos
        """
        if not self.best_params:
            logger.error("No hay mejores parámetros disponibles")
            return None
        
        logger.info("Entrenando mejor modelo con todos los datos...")
        
        # Crear trial dummy para construir el modelo
        class DummyTrial:
            def __init__(self, params):
                self.params = params
            
            def suggest_categorical(self, name, choices):
                return self.params[name]
            
            def suggest_float(self, name, low, high, **kwargs):
                return self.params[name]
            
            def suggest_int(self, name, low, high):
                return self.params[name]
        
        dummy_trial = DummyTrial(self.best_params)
        model = self.build_model(dummy_trial)
        
        # Entrenar con más épocas
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=50,
            callbacks=[
                callbacks.EarlyStopping(
                    monitor=self.metric_name,
                    patience=10,
                    restore_best_weights=True,
                    mode='max'
                ),
                callbacks.ReduceLROnPlateau(
                    monitor=self.metric_name,
                    factor=0.5,
                    patience=5,
                    min_lr=1e-7,
                    mode='max'
                )
            ],
            verbose=1
        )
        
        # Guardar modelo
        model.save(save_path)
        logger.info(f"Mejor modelo guardado en: {save_path}")
        
        return model, history


def main():
    """
    Función principal para demostrar el uso del HyperparameterOptimizer
    """
    try:
        # Importar datos de ejemplo
        from load_medical_data import create_synthetic_medical_data, MedicalDataLoader
        
        # Crear datos sintéticos para demostración
        images, labels = create_synthetic_medical_data(num_samples=1000)
        
        # Crear datasets de TensorFlow
        train_ds = tf.data.Dataset.from_tensor_slices((images[:800], labels[:800]))
        val_ds = tf.data.Dataset.from_tensor_slices((images[800:900], labels[800:900]))
        test_ds = tf.data.Dataset.from_tensor_slices((images[900:], labels[900:]))
        
        # Preprocesamiento
        def preprocess(image, label):
            image = tf.image.resize(image, (224, 224))
            return image, label
        
        train_ds = train_ds.map(preprocess).batch(32).prefetch(tf.data.AUTOTUNE)
        val_ds = val_ds.map(preprocess).batch(32).prefetch(tf.data.AUTOTUNE)
        test_ds = test_ds.map(preprocess).batch(32).prefetch(tf.data.AUTOTUNE)
        
        # Inicializar optimizador
        optimizer = HyperparameterOptimizer(
            project_name="demo-tuning",
            study_name="demo_study"
        )
        
        # Ejecutar optimización (menos trials para demostración)
        results = optimizer.optimize(
            train_ds=train_ds,
            val_ds=val_ds,
            n_trials=10,  # Reducido para demostración
            timeout=600   # 10 minutos
        )
        
        # Visualizar resultados
        optimizer.visualize_results()
        
        # Entrenar mejor modelo
        best_model, history = optimizer.train_best_model(train_ds, val_ds)
        
        logger.info("✅ Optimización completada exitosamente")
        
    except Exception as e:
        logger.error(f"Error en la demostración: {e}")


if __name__ == "__main__":
    main()
