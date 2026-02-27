"""
Caso de Uso 4 - Reentrenamiento Automático con MLflow y TFX
Fase 2: Módulo de Entrenamiento para Modelo de Recomendación
"""

import tensorflow as tf
from tfx.components.trainer.fn_args_utils import FnArgs
from tfx.types import artifact_utils
from tfx.utils import telemetry
import pandas as pd
import numpy as np
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def _get_tf_examples(dataset):
    """Convertir dataset a ejemplos de TensorFlow"""
    def _parse_function(example_proto):
        feature_description = {
            'user_id': tf.io.FixedLenFeature([], tf.int64),
            'item_id': tf.io.FixedLenFeature([], tf.int64),
            'rating': tf.io.FixedLenFeature([], tf.float32),
            'timestamp': tf.io.FixedLenFeature([], tf.int64)
        }
        parsed_features = tf.io.parse_single_example(example_proto, feature_description)
        return parsed_features
    
    return dataset.map(_parse_function)

def _create_model(input_shape, num_users, num_items, embedding_dim=32):
    """Crear modelo de recomendación con embeddings"""
    # Inputs
    user_input = tf.keras.layers.Input(shape=(1,), name='user_id')
    item_input = tf.keras.layers.Input(shape=(1,), name='item_id')
    
    # Embeddings
    user_embedding = tf.keras.layers.Embedding(
        input_dim=num_users,
        output_dim=embedding_dim,
        name='user_embedding'
    )(user_input)
    
    item_embedding = tf.keras.layers.Embedding(
        input_dim=num_items,
        output_dim=embedding_dim,
        name='item_embedding'
    )(item_input)
    
    # Flatten embeddings
    user_vec = tf.keras.layers.Flatten()(user_embedding)
    item_vec = tf.keras.layers.Flatten()(item_embedding)
    
    # Concatenate and dense layers
    concat = tf.keras.layers.Concatenate()([user_vec, item_vec])
    
    dense1 = tf.keras.layers.Dense(128, activation='relu')(concat)
    dropout1 = tf.keras.layers.Dropout(0.2)(dense1)
    
    dense2 = tf.keras.layers.Dense(64, activation='relu')(dropout1)
    dropout2 = tf.keras.layers.Dropout(0.2)(dense2)
    
    # Output layer
    output = tf.keras.layers.Dense(1, activation='sigmoid')(dropout2)
    
    # Create model
    model = tf.keras.Model(
        inputs=[user_input, item_input],
        outputs=output,
        name='recommendation_model'
    )
    
    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )
    
    return model

def _get_dataset_size(tfrecord_path):
    """Obtener tamaño del dataset"""
    count = 0
    for _ in tf.data.TFRecordDataset(tfrecord_path):
        count += 1
    return count

def run_fn(fn_args: FnArgs):
    """
    Función principal de entrenamiento para TFX Trainer
    """
    logger.info("Iniciando entrenamiento del modelo de recomendación...")
    
    # Registrar telemetría
    telemetry_label = telemetry.LABEL_PIPELINE
    telemetry.set_namespace(telemetry_label)
    
    try:
        # Cargar datos transformados
        logger.info(f"Cargando datos desde: {fn_args.transformed_examples}")
        train_dataset = tf.data.TFRecordDataset(fn_args.transformed_examples)
        
        # Parsear ejemplos
        train_dataset = _get_tf_examples(train_dataset)
        
        # Obtener información del dataset
        dataset_size = _get_dataset_size(fn_args.transformed_examples)
        logger.info(f"Tamaño del dataset: {dataset_size}")
        
        # Dividir en entrenamiento y validación
        train_size = int(0.8 * dataset_size)
        train_dataset = train_dataset.take(train_size)
        val_dataset = train_dataset.skip(train_size)
        
        # Preparar datos para el modelo
        def _prepare_features(features):
            return {
                'user_id': features['user_id'],
                'item_id': features['item_id']
            }, features['rating']
        
        train_dataset = train_dataset.map(_prepare_features)
        val_dataset = val_dataset.map(_prepare_features)
        
        # Batch y prefetch
        batch_size = fn_args.custom_config.get('batch_size', 256)
        train_dataset = train_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        val_dataset = val_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        
        # Obtener número de usuarios e items (en producción, esto vendría de los datos)
        num_users = 1000  # Ejemplo
        num_items = 500   # Ejemplo
        
        # Crear modelo
        logger.info("Creando modelo de recomendación...")
        model = _create_model(
            input_shape=(2,),
            num_users=num_users,
            num_items=num_items,
            embedding_dim=32
        )
        
        # Callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=1e-7
            ),
            tf.keras.callbacks.TensorBoard(
                log_dir=os.path.join(fn_args.model_run_dir, 'logs'),
                histogram_freq=1
            )
        ]
        
        # Entrenar modelo
        logger.info(f"Entrenando modelo por {fn_args.train_args.num_steps} pasos...")
        history = model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=10,
            steps_per_epoch=min(fn_args.train_args.num_steps // 1000, 100),
            validation_steps=min(fn_args.eval_args.num_steps // 1000, 20),
            callbacks=callbacks,
            verbose=1
        )
        
        # Evaluar modelo
        logger.info("Evaluando modelo...")
        eval_results = model.evaluate(val_dataset, verbose=1)
        
        # Imprimir métricas
        metric_names = ['loss', 'accuracy', 'auc']
        for i, metric_name in enumerate(metric_names):
            logger.info(f"{metric_name}: {eval_results[i]:.4f}")
        
        # Guardar modelo
        logger.info(f"Guardando modelo en: {fn_args.serving_model_dir}")
        model.save(fn_args.serving_model_dir, save_format='tf')
        
        # Guardar métricas para MLflow
        metrics_dict = {
            'train_loss': history.history['loss'][-1],
            'val_loss': history.history['val_loss'][-1],
            'train_accuracy': history.history['accuracy'][-1],
            'val_accuracy': history.history['val_accuracy'][-1],
            'train_auc': history.history['auc'][-1],
            'val_auc': history.history['val_auc'][-1],
            'dataset_size': dataset_size,
            'num_users': num_users,
            'num_items': num_items
        }
        
        # Guardar métricas en archivo para MLflow
        metrics_file = os.path.join(fn_args.model_run_dir, 'metrics.json')
        import json
        with open(metrics_file, 'w') as f:
            json.dump(metrics_dict, f, indent=2)
        
        logger.info("Entrenamiento completado exitosamente")
        logger.info(f"Métricas finales: {metrics_dict}")
        
    except Exception as e:
        logger.error(f"Error durante el entrenamiento: {e}")
        raise

def _create_dummy_data(output_path, num_samples=10000):
    """
    Crear datos dummy para pruebas (solo para desarrollo)
    """
    logger.info(f"Creando datos dummy con {num_samples} muestras...")
    
    # Generar datos aleatorios
    np.random.seed(42)
    
    data = {
        'user_id': np.random.randint(1, 1000, num_samples),
        'item_id': np.random.randint(1, 500, num_samples),
        'rating': np.random.choice([0, 1], num_samples, p=[0.7, 0.3]),
        'timestamp': np.random.randint(1600000000, 1700000000, num_samples)
    }
    
    df = pd.DataFrame(data)
    
    # Guardar como TFRecord
    def _serialize_example(user_id, item_id, rating, timestamp):
        feature = {
            'user_id': tf.train.Feature(int64_list=tf.train.Int64List(value=[user_id])),
            'item_id': tf.train.Feature(int64_list=tf.train.Int64List(value=[item_id])),
            'rating': tf.train.Feature(float_list=tf.train.FloatList(value=[rating])),
            'timestamp': tf.train.Feature(int64_list=tf.train.Int64List(value=[timestamp]))
        }
        example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
        return example_proto.SerializeToString()
    
    with tf.io.TFRecordWriter(output_path) as writer:
        for _, row in df.iterrows():
            example = _serialize_example(
                row['user_id'], row['item_id'], row['rating'], row['timestamp']
            )
            writer.write(example)
    
    logger.info(f"Datos dummy guardados en: {output_path}")

if __name__ == "__main__":
    # Para pruebas locales
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--create-dummy-data", action="store_true")
    parser.add_argument("--output-path", default="dummy_data.tfrecord")
    args = parser.parse_args()
    
    if args.create_dummy_data:
        _create_dummy_data(args.output_path)
