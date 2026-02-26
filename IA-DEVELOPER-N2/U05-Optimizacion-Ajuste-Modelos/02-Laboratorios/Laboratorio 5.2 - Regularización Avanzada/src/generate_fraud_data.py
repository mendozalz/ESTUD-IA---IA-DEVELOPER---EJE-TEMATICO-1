"""
Generación de Datos Sintéticos de Fraudes Financieros
Laboratorio 5.2 - Regularización Avanzada y Normalización en Modelos Profundos
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from typing import Tuple, Dict, Any
import os

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FraudDataGenerator:
    """
    Clase para generar datos sintéticos realistas de transacciones financieras
    """
    
    def __init__(self, n_samples: int = 100000, fraud_ratio: float = 0.05, random_state: int = 42):
        self.n_samples = n_samples
        self.fraud_ratio = fraud_ratio
        self.random_state = random_state
        np.random.seed(random_state)
        
        logger.info(f"FraudDataGenerator inicializado: {n_samples} muestras, {fraud_ratio*100:.1f}% fraudes")
    
    def generate_base_features(self, n_features: int = 20) -> Tuple[np.ndarray, np.ndarray]:
        """
        Genera características base usando make_classification
        """
        logger.info("Generando características base...")
        
        X, y = make_classification(
            n_samples=self.n_samples,
            n_features=n_features,
            n_informative=15,
            n_redundant=5,
            n_classes=2,
            weights=[1-self.fraud_ratio, self.fraud_ratio],
            flip_y=0.01,  # 1% de ruido en etiquetas
            random_state=self.random_state
        )
        
        return X, y
    
    def add_transaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Añade características realistas de transacciones
        """
        logger.info("Añadiendo características de transacción...")
        
        # Monto de transacción (distribución exponencial)
        df['amount'] = np.random.exponential(scale=100, size=self.n_samples)
        
        # Hora del día (0-24 horas)
        df['time'] = np.random.uniform(0, 24, size=self.n_samples)
        
        # Día de la semana (0-6)
        df['day_of_week'] = np.random.randint(0, 7, size=self.n_samples)
        
        # Mes del año (1-12)
        df['month'] = np.random.randint(1, 13, size=self.n_samples)
        
        # Tipo de transacción (categorías)
        transaction_types = ['purchase', 'transfer', 'withdrawal', 'deposit', 'payment']
        df['transaction_type'] = np.random.choice(transaction_types, size=self.n_samples)
        
        # Método de pago
        payment_methods = ['credit_card', 'debit_card', 'bank_transfer', 'digital_wallet', 'cash']
        df['payment_method'] = np.random.choice(payment_methods, size=self.n_samples)
        
        # Ubicación geográfica (simulada)
        countries = ['US', 'UK', 'CA', 'AU', 'DE', 'FR', 'JP', 'CN', 'IN', 'BR']
        df['country'] = np.random.choice(countries, size=self.n_samples)
        
        # Dispositivo usado
        devices = ['mobile', 'desktop', 'tablet', 'atm', 'pos']
        df['device'] = np.random.choice(devices, size=self.n_samples)
        
        return df
    
    def add_fraud_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Añade patrones realistas específicos para transacciones fraudulentas
        """
        logger.info("Añadiendo patrones de fraude...")
        
        fraud_mask = df['is_fraud'] == 1
        n_fraud = fraud_mask.sum()
        
        if n_fraud > 0:
            # Patrones de fraude:
            
            # 1. Montos más altos en fraudes
            df.loc[fraud_mask, 'amount'] *= np.random.uniform(2, 5, n_fraud)
            
            # 2. Fraudes más comunes en horarios inusuales (noche/madrugada)
            df.loc[fraud_mask, 'time'] = np.random.choice(
                [np.random.uniform(22, 24), np.random.uniform(0, 6)], 
                size=n_fraud
            )
            
            # 3. Mayor probabilidad de fraude en ciertos países
            high_risk_countries = ['CN', 'IN', 'BR', 'RU']
            high_risk_prob = 0.3
            mask = (fraud_mask) & (np.random.random(n_fraud) < high_risk_prob)
            df.loc[mask, 'country'] = np.random.choice(high_risk_countries, size=mask.sum())
            
            # 4. Preferencia por ciertos métodos de pago en fraudes
            risky_methods = ['digital_wallet', 'bank_transfer']
            risky_method_prob = 0.4
            mask = (fraud_mask) & (np.random.random(n_fraud) < risky_method_prob)
            df.loc[mask, 'payment_method'] = np.random.choice(risky_methods, size=mask.sum())
            
            # 5. Patrones de dispositivo (más fraudes en móviles)
            mobile_fraud_prob = 0.6
            mask = (fraud_mask) & (np.random.random(n_fraud) < mobile_fraud_prob)
            df.loc[mask, 'device'] = 'mobile'
            
            # 6. Secuencias sospechosas (múltiples transacciones pequeñas)
            if n_fraud > 100:
                # Añadir características de secuencia
                df.loc[fraud_mask, 'recent_transactions'] = np.random.poisson(5, n_fraud)
                df.loc[~fraud_mask, 'recent_transactions'] = np.random.poisson(2, self.n_samples - n_fraud)
            
            # 7. Velocidad de transacción (tiempo desde última transacción)
            df.loc[fraud_mask, 'time_since_last'] = np.random.exponential(0.1, n_fraud)  # Más rápidas
            df.loc[~fraud_mask, 'time_since_last'] = np.random.exponential(1.0, self.n_samples - n_fraud)
        
        return df
    
    def add_behavioral_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Añade características de comportamiento del usuario
        """
        logger.info("Añadiendo características de comportamiento...")
        
        # Historial del usuario (simulado)
        df['customer_age_months'] = np.random.exponential(24, self.n_samples)  # Meses como cliente
        
        # Score de crédito (simulado)
        df['credit_score'] = np.random.normal(650, 100, self.n_samples)
        df['credit_score'] = np.clip(df['credit_score'], 300, 850)
        
        # Ingresos mensuales (simulado)
        df['monthly_income'] = np.random.lognormal(8, 0.5, self.n_samples)
        
        # Número de transacciones previas (últimos 30 días)
        df['prev_transactions_30d'] = np.random.poisson(10, self.n_samples)
        
        # Monto promedio de transacciones previas
        df['avg_amount_prev'] = np.random.exponential(80, self.n_samples)
        
        # Ratio de chargebacks previos
        df['chargeback_ratio'] = np.random.beta(1, 50, self.n_samples)  # Generalmente bajo
        
        # Ajustar características para fraudes
        fraud_mask = df['is_fraud'] == 1
        if fraud_mask.sum() > 0:
            # Clientes más nuevos tienen más probabilidad de fraude
            df.loc[fraud_mask, 'customer_age_months'] *= 0.5
            
            # Scores de crédito más bajos en fraudes
            df.loc[fraud_mask, 'credit_score'] -= np.random.normal(50, 20, fraud_mask.sum())
            
            # Ratios de chargeback más altos en fraudes
            df.loc[fraud_mask, 'chargeback_ratio'] *= 5
        
        return df
    
    def encode_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Codifica características categóricas
        """
        logger.info("Codificando características categóricas...")
        
        # One-hot encoding para variables categóricas
        categorical_columns = ['transaction_type', 'payment_method', 'country', 'device']
        
        for col in categorical_columns:
            if col in df.columns:
                dummies = pd.get_dummies(df[col], prefix=col)
                df = pd.concat([df, dummies], axis=1)
                df.drop(col, axis=1, inplace=True)
        
        return df
    
    def generate_dataset(self, include_categorical: bool = True) -> pd.DataFrame:
        """
        Genera el dataset completo de fraudes
        """
        logger.info("Generando dataset completo de fraudes...")
        
        # Generar características base
        X, y = self.generate_base_features()
        
        # Crear DataFrame inicial
        feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        df = pd.DataFrame(X, columns=feature_names)
        df['is_fraud'] = y
        
        # Añadir características de transacción
        df = self.add_transaction_features(df)
        
        # Añadir patrones de fraude
        df = self.add_fraud_patterns(df)
        
        # Añadir características de comportamiento
        df = self.add_behavioral_features(df)
        
        # Codificar variables categóricas
        if include_categorical:
            df = self.encode_categorical_features(df)
        else:
            # Eliminar columnas categóricas sin codificar
            categorical_columns = ['transaction_type', 'payment_method', 'country', 'device']
            df.drop(columns=[col for col in categorical_columns if col in df.columns], inplace=True)
        
        # Eliminar posibles valores NaN
        df = df.fillna(0)
        
        logger.info(f"Dataset generado: {df.shape}")
        logger.info(f"Distribución de clases: {df['is_fraud'].value_counts().to_dict()}")
        
        return df
    
    def analyze_dataset(self, df: pd.DataFrame, save_dir: str = 'analysis') -> Dict[str, Any]:
        """
        Analiza y visualiza el dataset generado
        """
        logger.info("Analizando dataset...")
        
        os.makedirs(save_dir, exist_ok=True)
        
        analysis = {}
        
        # Estadísticas básicas
        analysis['basic_stats'] = {
            'total_samples': len(df),
            'total_features': df.shape[1] - 1,  # Excluyendo target
            'fraud_cases': df['is_fraud'].sum(),
            'fraud_ratio': df['is_fraud'].mean(),
            'non_fraud_cases': (df['is_fraud'] == 0).sum()
        }
        
        # Correlación con target
        correlations = df.corr()['is_fraud'].sort_values(ascending=False)
        analysis['top_correlations'] = correlations.head(10).to_dict()
        
        # Visualizaciones
        self._create_visualizations(df, save_dir)
        
        # Guardar análisis
        import json
        with open(os.path.join(save_dir, 'dataset_analysis.json'), 'w') as f:
            json.dump(analysis, f, indent=2, default=str)
        
        logger.info(f"Análisis guardado en: {save_dir}")
        
        return analysis
    
    def _create_visualizations(self, df: pd.DataFrame, save_dir: str):
        """
        Crea visualizaciones del dataset
        """
        logger.info("Creando visualizaciones...")
        
        # 1. Distribución de clases
        plt.figure(figsize=(8, 6))
        df['is_fraud'].value_counts().plot(kind='bar', color=['blue', 'red'])
        plt.title('Distribución de Clases (Fraude vs No Fraude)')
        plt.xlabel('Clase')
        plt.ylabel('Cantidad')
        plt.xticks([0, 1], ['No Fraude', 'Fraude'], rotation=0)
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(save_dir, 'class_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Distribución de montos por clase
        if 'amount' in df.columns:
            plt.figure(figsize=(10, 6))
            fraud_amounts = df[df['is_fraud'] == 1]['amount']
            legit_amounts = df[df['is_fraud'] == 0]['amount']
            
            plt.hist([legit_amounts, fraud_amounts], bins=50, alpha=0.7, 
                    label=['No Fraude', 'Fraude'], color=['blue', 'red'])
            plt.title('Distribución de Montos por Clase')
            plt.xlabel('Monto')
            plt.ylabel('Frecuencia')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(save_dir, 'amount_distribution.png'), dpi=300, bbox_inches='tight')
            plt.close()
        
        # 3. Distribución temporal de fraudes
        if 'time' in df.columns:
            plt.figure(figsize=(10, 6))
            fraud_times = df[df['is_fraud'] == 1]['time']
            legit_times = df[df['is_fraud'] == 0]['time']
            
            plt.hist([legit_times, fraud_times], bins=24, alpha=0.7,
                    label=['No Fraude', 'Fraude'], color=['blue', 'red'])
            plt.title('Distribución de Transacciones por Hora del Día')
            plt.xlabel('Hora del Día')
            plt.ylabel('Frecuencia')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(save_dir, 'time_distribution.png'), dpi=300, bbox_inches='tight')
            plt.close()
        
        # 4. Correlaciones
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        correlations = df[numeric_cols].corr()['is_fraud'].sort_values(ascending=False)
        
        plt.figure(figsize=(12, 8))
        top_features = correlations.head(15).index
        correlation_matrix = df[top_features].corr()
        
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                   fmt='.2f', annot_kws={'size': 8})
        plt.title('Matriz de Correlación - Top 15 Features')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'correlation_matrix.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 5. Feature importance (correlación absoluta)
        plt.figure(figsize=(10, 8))
        abs_correlations = correlations.abs().sort_values(ascending=False).head(15)
        
        plt.barh(range(len(abs_correlations)), abs_correlations.values)
        plt.yticks(range(len(abs_correlations)), abs_correlations.index)
        plt.xlabel('Correlación Absoluta con Fraude')
        plt.title('Top 15 Features - Importancia por Correlación')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'feature_importance.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_dataset(self, df: pd.DataFrame, filename: str = 'fraud_transactions.csv'):
        """
        Guarda el dataset en archivo CSV
        """
        df.to_csv(filename, index=False)
        logger.info(f"Dataset guardado en: {filename}")
    
    def create_train_test_split(self, df: pd.DataFrame, test_size: float = 0.2, 
                               val_size: float = 0.2, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Divide el dataset en train/validation/test
        """
        from sklearn.model_selection import train_test_split
        
        # Separar features y target
        X = df.drop('is_fraud', axis=1)
        y = df['is_fraud']
        
        # Dividir en train+val y test
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Dividir train+val en train y val
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=val_size_adjusted, 
            random_state=random_state, stratify=y_train_val
        )
        
        # Crear DataFrames
        train_df = pd.concat([X_train, y_train], axis=1)
        val_df = pd.concat([X_val, y_val], axis=1)
        test_df = pd.concat([X_test, y_test], axis=1)
        
        logger.info(f"División completada:")
        logger.info(f"  Train: {len(train_df)} muestras ({len(train_df)/len(df)*100:.1f}%)")
        logger.info(f"  Val: {len(val_df)} muestras ({len(val_df)/len(df)*100:.1f}%)")
        logger.info(f"  Test: {len(test_df)} muestras ({len(test_df)/len(df)*100:.1f}%)")
        
        return train_df, val_df, test_df


def main():
    """
    Función principal para demostrar el uso del FraudDataGenerator
    """
    # Crear generador
    generator = FraudDataGenerator(
        n_samples=100000,
        fraud_ratio=0.05,
        random_state=42
    )
    
    # Generar dataset
    df = generator.generate_dataset(include_categorical=False)  # Sin categóricas para simplicidad
    
    # Analizar dataset
    analysis = generator.analyze_dataset(df)
    
    # Guardar dataset
    generator.save_dataset(df)
    
    # Crear splits
    train_df, val_df, test_df = generator.create_train_test_split(df)
    
    # Guardar splits
    train_df.to_csv('train_fraud_data.csv', index=False)
    val_df.to_csv('val_fraud_data.csv', index=False)
    test_df.to_csv('test_fraud_data.csv', index=False)
    
    logger.info("✅ Dataset de fraudes generado y procesado exitosamente")
    logger.info(f"Estadísticas finales: {analysis['basic_stats']}")


if __name__ == "__main__":
    main()
