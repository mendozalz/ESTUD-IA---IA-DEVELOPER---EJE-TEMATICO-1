"""
Caso de Uso 1 - Evaluación de Modelos con TFMA
Fase 1: Carga de Datos y Modelo
"""

import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

class FraudDataGenerator:
    """
    Clase para generar y preparar datos de fraude para evaluación con TFMA
    """
    
    def __init__(self, n_samples=10000, fraud_ratio=0.05):
        self.n_samples = n_samples
        self.fraud_ratio = fraud_ratio
        self.regions = ['NA', 'EU', 'ASIA', 'LATAM']
        self.transaction_types = ['online', 'in_store', 'mobile', 'atm']
        
    def generate_synthetic_data(self):
        """Generar datos sintéticos de transacciones con fraudes"""
        print(f"Generando {self.n_samples} muestras con {self.fraud_ratio*100}% de fraudes...")
        
        # Generar características base
        X, y = make_classification(
            n_samples=self.n_samples,
            n_features=20,
            n_informative=10,
            n_redundant=5,
            n_classes=2,
            weights=[1-self.fraud_ratio, self.fraud_ratio],
            random_state=42,
            flip_y=0.01  # 1% de ruido
        )
        
        # Crear DataFrame para mejor manipulación
        feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        df = pd.DataFrame(X, columns=feature_names)
        df['is_fraud'] = y
        
        # Añadir características categóricas realistas
        df['region'] = np.random.choice(self.regions, size=len(df))
        df['transaction_type'] = np.random.choice(self.transaction_types, size=len(df))
        
        # Añadir características temporales
        df['hour_of_day'] = np.random.randint(0, 24, size=len(df))
        df['day_of_week'] = np.random.randint(0, 7, size=len(df))
        
        # Añadir características monetarias
        df['amount'] = np.random.exponential(scale=100, size=len(df))
        df['amount'] = np.clip(df['amount'], 1, 10000)
        
        # Añadir ID de transacción
        df['transaction_id'] = [f'tx_{i:06d}' for i in range(len(df))]
        
        # Ajustar características para fraudes (patrones realistas)
        fraud_mask = df['is_fraud'] == 1
        df.loc[fraud_mask, 'amount'] *= np.random.uniform(1.5, 5.0, size=fraud_mask.sum())
        df.loc[fraud_mask, 'hour_of_day'] = np.random.choice([2, 3, 4, 22, 23], size=fraud_mask.sum())
        
        print(f"Datos generados: {len(df)} transacciones")
        print(f"Fraudes detectados: {fraud_mask.sum()} ({fraud_mask.mean()*100:.2f}%)")
        
        return df
    
    def preprocess_data(self, df):
        """Preprocesar datos para el modelo"""
        # Codificar variables categóricas
        df_encoded = pd.get_dummies(df, columns=['region', 'transaction_type'])
        
        # Normalizar características numéricas
        scaler = StandardScaler()
        numeric_features = ['amount', 'hour_of_day', 'day_of_week'] + \
                        [f'feature_{i}' for i in range(20)]
        
        df_encoded[numeric_features] = scaler.fit_transform(df_encoded[numeric_features])
        
        return df_encoded, scaler
    
    def split_data(self, df):
        """Dividir datos en entrenamiento y evaluación"""
        # Separar características y etiqueta
        feature_cols = [col for col in df.columns if col not in ['is_fraud', 'transaction_id']]
        X = df[feature_cols]
        y = df['is_fraud']
        
        # Dividir en train/test manteniendo proporción de fraudes
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Añadir metadatos para TFMA
        X_train['region_original'] = df.loc[X_train.index, 'region']
        X_test['region_original'] = df.loc[X_test.index, 'region']
        
        print(f"División completada:")
        print(f"  Entrenamiento: {len(X_train)} muestras ({y_train.mean()*100:.2f}% fraudes)")
        print(f"  Evaluación: {len(X_test)} muestras ({y_test.mean()*100:.2f}% fraudes)")
        
        return X_train, X_test, y_train, y_test
    
    def write_tfrecords(self, X, y, filename, metadata=None):
        """Escribir datos en formato TFRecord para TFMA"""
        print(f"Escribiendo {len(X)} muestras en {filename}...")
        
        # Crear directorio si no existe
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        with tf.io.TFRecordWriter(filename) as writer:
            for i, (features, label) in enumerate(zip(X.values, y)):
                # Extraer características numéricas (excluyendo metadatos)
                numeric_features = features[:-1] if metadata else features
                
                # Crear ejemplo TF
                example = tf.train.Example(features=tf.train.Features(feature={
                    'features': tf.train.Feature(
                        float_list=tf.train.FloatList(value=numeric_features.astype(float))
                    ),
                    'label': tf.train.Feature(
                        int64_list=tf.train.Int64List(value=[int(label)])
                    )
                }))
                
                # Añadir metadatos si están disponibles
                if metadata and 'region_original' in X.columns:
                    region_value = X.iloc[i]['region_original']
                    example.features.feature['region'] = tf.train.Feature(
                        bytes_list=tf.train.BytesList(value=[region_value.encode('utf-8')])
                    )
                
                writer.write(example.SerializeToString())
        
        print(f"Datos guardados exitosamente en {filename}")
    
    def generate_and_save_data(self):
        """Generar y guardar todos los datos necesarios"""
        print("=" * 60)
        print("GENERACIÓN DE DATOS DE FRAUDE PARA EVALUACIÓN CON TFMA")
        print("=" * 60)
        
        # Generar datos sintéticos
        df = self.generate_synthetic_data()
        
        # Guardar datos originales para referencia
        os.makedirs('data', exist_ok=True)
        df.to_csv('data/reference_data.csv', index=False)
        print("Datos de referencia guardados en data/reference_data.csv")
        
        # Preprocesar datos
        df_processed, scaler = self.preprocess_data(df)
        
        # Dividir datos
        X_train, X_test, y_train, y_test = self.split_data(df_processed)
        
        # Guardar en formato TFRecord
        self.write_tfrecords(X_train, y_train, 'data/train.tfrecord', metadata=True)
        self.write_tfrecords(X_test, y_test, 'data/eval.tfrecord', metadata=True)
        
        # Guardar scaler para uso futuro
        import joblib
        joblib.dump(scaler, 'data/scaler.pkl')
        print("Scaler guardado en data/scaler.pkl")
        
        # Generar estadísticas descriptivas
        self.generate_statistics(df)
        
        print("\n" + "=" * 60)
        print("DATOS GENERADOS EXITOSAMENTE")
        print("=" * 60)
        
        return df, X_train, X_test, y_train, y_test
    
    def generate_statistics(self, df):
        """Generar estadísticas descriptivas de los datos"""
        print("\nGenerando estadísticas descriptivas...")
        
        stats = {
            'total_transactions': len(df),
            'fraud_transactions': df['is_fraud'].sum(),
            'fraud_rate': df['is_fraud'].mean(),
            'regions': df['region'].value_counts().to_dict(),
            'transaction_types': df['transaction_type'].value_counts().to_dict(),
            'avg_amount': df['amount'].mean(),
            'median_amount': df['amount'].median(),
            'max_amount': df['amount'].max(),
            'min_amount': df['amount'].min()
        }
        
        # Guardar estadísticas
        import json
        with open('data/data_statistics.json', 'w') as f:
            json.dump(stats, f, indent=2)
        
        print("Estadísticas guardadas en data/data_statistics.json")
        
        # Mostrar estadísticas principales
        print(f"\nEstadísticas Principales:")
        print(f"  Total de transacciones: {stats['total_transactions']:,}")
        print(f"  Transacciones fraudulentas: {stats['fraud_transactions']:,}")
        print(f"  Tasa de fraude: {stats['fraud_rate']*100:.3f}%")
        print(f"  Monto promedio: ${stats['avg_amount']:.2f}")
        print(f"  Monto máximo: ${stats['max_amount']:.2f}")
        print(f"  Distribución por región: {stats['regions']}")

def main():
    """Función principal para generar datos de evaluación"""
    # Crear generador de datos
    data_generator = FraudDataGenerator(n_samples=10000, fraud_ratio=0.05)
    
    # Generar y guardar datos
    df, X_train, X_test, y_train, y_test = data_generator.generate_and_save_data()
    
    # Mostrar ejemplo de datos procesados
    print("\nEjemplo de datos procesados:")
    print(X_train.head())
    print(f"\nForma de características de entrenamiento: {X_train.shape}")
    print(f"Forma de etiquetas de entrenamiento: {y_train.shape}")
    
    return df, X_train, X_test, y_train, y_test

if __name__ == "__main__":
    main()
