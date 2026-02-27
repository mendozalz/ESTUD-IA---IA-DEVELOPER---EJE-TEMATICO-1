"""
Laboratorio 1 - Pipeline de Datos Automatizado
Módulo de transformación de features para ventas retail con TFX
"""

import tensorflow as tf
import tensorflow_transform as tft
from tensorflow_transform.tf_metadata import schema_utils

# Constantes para el pipeline
SALES_FEATURE_KEY = 'sales'
INVENTORY_FEATURE_KEY = 'inventory'
PRICE_FEATURE_KEY = 'price'
PROMOTION_FEATURE_KEY = 'promotion'
TEMPERATURE_FEATURE_KEY = 'temperature'
DAY_OF_WEEK_FEATURE_KEY = 'day_of_week'
IS_WEEKEND_FEATURE_KEY = 'is_weekend'
MONTH_FEATURE_KEY = 'month'

# Features transformadas
SALES_PER_INVENTORY_KEY = 'sales_per_inventory'
PRICE_CATEGORY_KEY = 'price_category'
TEMPERATURE_BUCKET_KEY = 'temperature_bucket'
SALES_LAG_1_KEY = 'sales_lag_1'
SALES_MA_7_KEY = 'sales_ma_7'

def preprocessing_fn(inputs):
    """
    Función de preprocesamiento para datos de ventas retail
    
    Args:
        inputs: Diccionario de tensores con features crudas
        
    Returns:
        outputs: Diccionario de tensores con features transformadas
    """
    
    logger = tf.get_logger()
    logger.info("Iniciando preprocesamiento de datos de ventas retail")
    
    # Extraer features crudas
    sales = inputs[SALES_FEATURE_KEY]
    inventory = inputs[INVENTORY_FEATURE_KEY]
    price = inputs[PRICE_FEATURE_KEY]
    promotion = inputs[PROMOTION_FEATURE_KEY]
    temperature = inputs[TEMPERATURE_FEATURE_KEY]
    day_of_week = inputs[DAY_OF_WEEK_FEATURE_KEY]
    is_weekend = inputs[IS_WEEKEND_FEATURE_KEY]
    month = inputs[MONTH_FEATURE_KEY]
    
    outputs = {}
    
    # 1. Features numéricas básicas (normalización)
    logger.info("Normalizando features numéricas básicas")
    
    # Normalizar ventas (log transform para manejar skewness)
    sales_log = tf.math.log1p(sales)
    outputs['sales_normalized'] = tft.scale_to_z_score(sales_log)
    
    # Normalizar inventario
    outputs['inventory_normalized'] = tft.scale_to_z_score(inventory)
    
    # Normalizar precio
    outputs['price_normalized'] = tft.scale_to_z_score(price)
    
    # Normalizar temperatura
    outputs['temperature_normalized'] = tft.scale_to_z_score(temperature)
    
    # 2. Features categóricas
    logger.info("Procesando features categóricas")
    
    # Promoción (binaria)
    outputs['promotion_binary'] = promotion
    
    # Fin de semana (binaria)
    outputs['is_weekend_binary'] = is_weekend
    
    # Día de la semana (one-hot encoding)
    day_of_week_vocab = tf.range(7, dtype=tf.int64)
    outputs['day_of_week_onehot'] = tft.compute_and_apply_vocabulary(
        day_of_week,
        day_of_week_vocab,
        vocab_filename='day_of_week_vocab',
        frequency_threshold=1,
        num_oov_buckets=1
    )
    
    # Mes (cíclico encoding para capturar estacionalidad)
    month_sin = tf.sin(2.0 * tf.constant(np.pi) * tf.cast(month, tf.float32) / 12.0)
    month_cos = tf.cos(2.0 * tf.constant(np.pi) * tf.cast(month, tf.float32) / 12.0)
    outputs['month_sin'] = month_sin
    outputs['month_cos'] = month_cos
    
    # 3. Features derivadas
    logger.info("Creando features derivadas")
    
    # Ratio ventas/inventario
    sales_per_inventory = tf.math.divide_no_nan(sales, inventory)
    outputs[SALES_PER_INVENTORY_KEY] = sales_per_inventory
    
    # Categorización de precio
    price_categories = tf.where(
        price < 25,
        tf.fill(tf.shape(price), 'low'),
        tf.where(
            price < 50,
            tf.fill(tf.shape(price), 'medium'),
            tf.fill(tf.shape(price), 'high')
        )
    )
    price_vocab = tf.constant(['low', 'medium', 'high'], dtype=tf.string)
    outputs[PRICE_CATEGORY_KEY] = tft.compute_and_apply_vocabulary(
        price_categories,
        price_vocab,
        vocab_filename='price_category_vocab',
        frequency_threshold=1,
        num_oov_buckets=0
    )
    
    # Bucketing de temperatura
    temperature_buckets = tft.bucketize(
        temperature,
        num_buckets=5,
        epsilon=0.01,
        name='temperature_bucket'
    )
    outputs[TEMPERATURE_BUCKET_KEY] = temperature_buckets
    
    # 4. Features temporales avanzadas
    logger.info("Creando features temporales")
    
    # Nota: En un pipeline real, estas features requerirían ordenamiento por tiempo
    # Para simplificar, usaremos aproximaciones
    
    # Simulación de lag features (requeriría windowing en producción)
    # Aquí creamos una versión simplificada
    sales_lag_1 = tf.roll(sales, shift=1, axis=0)
    outputs[SALES_LAG_1_KEY] = sales_lag_1
    
    # Media móvil de 7 días (simplificada)
    # En producción, esto requeriría window functions
    sales_ma_7 = tft.mean(sales, window_size=7, shift=0, reduce_window_size=True)
    outputs[SALES_MA_7_KEY] = sales_ma_7
    
    # 5. Features de interacción
    logger.info("Creando features de interacción")
    
    # Interacción precio * promoción
    price_promotion_interaction = price * tf.cast(promotion, tf.float32)
    outputs['price_promotion_interaction'] = price_promotion_interaction
    
    # Interacción temperatura * fin de semana
    temp_weekend_interaction = temperature * tf.cast(is_weekend, tf.float32)
    outputs['temp_weekend_interaction'] = temp_weekend_interaction
    
    # 6. Features de agregación (simuladas)
    logger.info("Creando features de agregación")
    
    # En producción, estas features se calcularían por tienda/producto
    # Aquí usamos aproximaciones simples
    
    # Ventas totales por tienda (simulado)
    store_sales_total = tft.mean(sales, reduce_window_size=True)
    outputs['store_sales_total'] = store_sales_total
    
    # Ratio de ventas vs promedio de tienda
    sales_vs_store_avg = sales / (store_sales_total + 1e-6)
    outputs['sales_vs_store_avg'] = sales_vs_store_avg
    
    # 7. Feature target (para entrenamiento supervisado)
    logger.info("Preparando feature target")
    
    # Target: ventas del siguiente día (simulado)
    # En producción, esto sería el lag -1 de ventas
    target_sales = tf.roll(sales, shift=-1, axis=0)
    outputs['target_sales'] = target_sales
    
    # Log del target para mejor distribución
    target_sales_log = tf.math.log1p(target_sales)
    outputs['target_sales_log'] = target_sales_log
    
    logger.info("Preprocesamiento completado exitosamente")
    
    return outputs

def _get_raw_feature_spec(schema):
    """
    Obtiene especificación de features crudas desde schema
    """
    return schema_utils.schema_as_feature_spec(schema)

def _transformed_name(key):
    """
    Genera nombre para feature transformada
    """
    return key + '_xf'

def build_transform_fn(module_file):
    """
    Construye la función de transformación para TFX
    """
    
    def transform_fn(tf_transform_output, schema):
        """
        Función wrapper para TFX
        """
        # Obtener especificación de features
        raw_feature_spec = _get_raw_feature_spec(schema)
        
        # Parsear datos crudos
        raw_data = tf.io.parse_example(
            tf_transform_output,
            raw_feature_spec
        )
        
        # Aplicar preprocesamiento
        transformed_data = preprocessing_fn(raw_data)
        
        return transformed_data
    
    return transform_fn

# Funciones auxiliares para validación y debugging
def validate_transformed_features(transformed_features):
    """
    Valida features transformadas
    """
    logger = tf.get_logger()
    logger.info("Validando features transformadas")
    
    validation_results = {}
    
    # Verificar que no haya NaNs en features clave
    key_features = ['sales_normalized', 'inventory_normalized', 'price_normalized']
    
    for feature in key_features:
        if feature in transformed_features:
            feature_tensor = transformed_features[feature]
            has_nan = tf.reduce_any(tf.math.is_nan(feature_tensor))
            validation_results[f'{feature}_has_nan'] = has_nan
    
    # Verificar rangos de features normalizadas
    normalized_features = ['sales_normalized', 'inventory_normalized', 'price_normalized']
    
    for feature in normalized_features:
        if feature in transformed_features:
            feature_tensor = transformed_features[feature]
            min_val = tf.reduce_min(feature_tensor)
            max_val = tf.reduce_max(feature_tensor)
            
            validation_results[f'{feature}_min'] = min_val
            validation_results[f'{feature}_max'] = max_val
            validation_results[f'{feature}_range_ok'] = tf.logical_and(
                min_val > -5.0, max_val < 5.0
            )
    
    return validation_results

def get_feature_importance():
    """
    Retorna diccionario de importancia de features (basado en conocimiento del dominio)
    """
    return {
        'sales_normalized': 1.0,
        'inventory_normalized': 0.8,
        'price_normalized': 0.7,
        'promotion_binary': 0.6,
        'is_weekend_binary': 0.5,
        'month_sin': 0.4,
        'month_cos': 0.4,
        'sales_per_inventory': 0.9,
        'price_category': 0.6,
        'temperature_bucket': 0.3,
        'price_promotion_interaction': 0.7,
        'temp_weekend_interaction': 0.5,
        'sales_lag_1': 0.8,
        'sales_ma_7': 0.9
    }

# Configuración de logging
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    logger.info("Módulo de transformación retail cargado")
    
    # Ejemplo de uso para testing
    try:
        # Crear datos de ejemplo
        sample_data = {
            'sales': tf.constant([100.0, 150.0, 80.0, 200.0]),
            'inventory': tf.constant([500.0, 450.0, 600.0, 400.0]),
            'price': tf.constant([25.0, 30.0, 20.0, 40.0]),
            'promotion': tf.constant([1, 0, 1, 0]),
            'temperature': tf.constant([22.0, 25.0, 18.0, 28.0]),
            'day_of_week': tf.constant([1, 2, 3, 4]),
            'is_weekend': tf.constant([0, 0, 0, 1]),
            'month': tf.constant([1, 1, 1, 1])
        }
        
        # Aplicar transformación
        transformed = preprocessing_fn(sample_data)
        
        # Validar resultados
        validation_results = validate_transformed_features(transformed)
        
        logger.info("Transformación de ejemplo completada exitosamente")
        logger.info(f"Features transformadas: {list(transformed.keys())}")
        
    except Exception as e:
        logger.error(f"Error en ejemplo de transformación: {str(e)}")
        raise
