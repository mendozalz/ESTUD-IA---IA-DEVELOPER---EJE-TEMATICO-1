"""
Laboratorio 5.4 - Generación de Series Temporales Sintéticas
==========================================================

Este script genera datos sintéticos de series temporales para
predicción de demanda en retail, con patrones estacionales.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple, Dict, Any, List
import datetime
from sklearn.preprocessing import MinMaxScaler

class TimeSeriesGenerator:
    """
    Clase para generar series temporales sintéticas de demanda retail
    """
    
    def __init__(self, num_samples: int = 10000, 
                 start_date: str = '2020-01-01'):
        """
        Inicializa el generador de series temporales
        
        Args:
            num_samples: Número de muestras a generar
            start_date: Fecha inicial de la serie
        """
        self.num_samples = num_samples
        self.start_date = pd.to_datetime(start_date)
        self.data = None
        self.scaler = MinMaxScaler()
        
    def generate_base_demand(self) -> np.ndarray:
        """
        Genera demanda base con tendencia y estacionalidad
        
        Returns:
            Array con demanda base
        """
        print("📊 Generando demanda base...")
        
        # Crear fechas
        dates = pd.date_range(start=self.start_date, 
                            periods=self.num_samples, 
                            freq='H')
        
        # Componentes de demanda
        demand = np.zeros(self.num_samples)
        
        # 1. Tendencia (crecimiento anual del 5%)
        trend = np.linspace(100, 100 * 1.05, self.num_samples)
        
        # 2. Estacionalidad anual
        annual_seasonality = 20 * np.sin(2 * np.pi * np.arange(self.num_samples) / (365 * 24))
        
        # 3. Estacionalidad semanal
        weekly_seasonality = 10 * np.sin(2 * np.pi * np.arange(self.num_samples) / (7 * 24))
        
        # 4. Estacionalidad diaria
        daily_seasonality = 15 * np.sin(2 * np.pi * np.arange(self.num_samples) / 24)
        
        # 5. Patrones de días laborables vs fines de semana
        weekday_pattern = np.where(dates.dayofweek < 5, 1.0, 0.7)  # 30% menos fines de semana
        
        # 6. Patrones de horas pico
        hour_pattern = np.ones(self.num_samples)
        # Horas pico: 8-10am y 6-8pm
        morning_peak = ((dates.hour >= 8) & (dates.hour <= 10)).astype(float) * 0.3
        evening_peak = ((dates.hour >= 18) & (dates.hour <= 20)).astype(float) * 0.4
        hour_pattern += morning_peak + evening_peak
        
        # Combinar componentes
        demand = trend * weekday_pattern * hour_pattern
        demand += annual_seasonality + weekly_seasonality + daily_seasonality
        
        # Asegurar valores positivos
        demand = np.maximum(demand, 10)
        
        print(f"✅ Demanda base generada: rango [{demand.min():.1f}, {demand.max():.1f}]")
        
        return demand
    
    def add_external_factors(self, demand: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Agrega factores externos que afectan la demanda
        
        Args:
            demand: Demanda base
            
        Returns:
            Diccionario con factores externos
        """
        print("🌡️ Agregando factores externos...")
        
        # Crear fechas para referencia
        dates = pd.date_range(start=self.start_date, 
                            periods=self.num_samples, 
                            freq='H')
        
        factors = {}
        
        # 1. Temperatura (afecta demanda de ciertos productos)
        temp_base = 20 + 10 * np.sin(2 * np.pi * np.arange(self.num_samples) / (365 * 24))
        temp_noise = np.random.normal(0, 2, self.num_samples)
        temperature = temp_base + temp_noise
        factors['temperature'] = temperature
        
        # Impacto de temperatura en demanda
        temp_impact = 1 + 0.02 * (temperature - 20)  # 2% por grado sobre 20°C
        
        # 2. Precipitación (reduce demanda de productos al aire libre)
        precipitation = np.random.exponential(0.5, self.num_samples)
        precipitation = np.clip(precipitation, 0, 10)  # Máximo 10mm/hora
        
        # Más lluvia en invierno
        winter_mask = (dates.month.isin([12, 1, 2])).astype(float)
        precipitation[winter_mask > 0] *= 2
        
        factors['precipitation'] = precipitation
        
        # Impacto de precipitación en demanda
        rain_impact = 1 - 0.05 * np.minimum(precipitation, 5) / 5  # Hasta 5% de reducción
        
        # 3. Eventos especiales (feriados, promociones)
        events = np.zeros(self.num_samples)
        
        # Feriados principales (Navidad, Año Nuevo, etc.)
        holidays = [
            (12, 25),  # Navidad
            (1, 1),    # Año Nuevo
            (5, 1),    # Día del trabajo
            (9, 15),   # Independencia
            (11, 25),  # Acción de Gracias
        ]
        
        for month, day in holidays:
            holiday_mask = ((dates.month == month) & (dates.day == day)).astype(float)
            # Efecto de 3 días alrededor del feriado
            for offset in [-1, 0, 1]:
                holiday_offset = np.roll(holiday_mask, offset * 24)
                events += holiday_offset * 0.3  # 30% de aumento
        
        # Promociones aleatorias (cada 2-4 semanas)
        promo_interval = np.random.randint(14*24, 28*24, 20)  # Cada 2-4 semanas
        for start in promo_interval:
            if start + 72 < self.num_samples:  # Promoción de 3 días
                events[start:start+72] += 0.2  # 20% de aumento
        
        factors['events'] = events
        
        # Aplicar factores a la demanda
        demand_adjusted = demand * temp_impact * rain_impact * (1 + events)
        
        # Agregar ruido aleatorio
        noise = np.random.normal(0, 0.05, self.num_samples)
        demand_adjusted *= (1 + noise)
        
        # Asegurar valores positivos
        demand_adjusted = np.maximum(demand_adjusted, 5)
        
        factors['demand_adjusted'] = demand_adjusted
        
        print(f"✅ Factores externos agregados")
        print(f"   - Temperatura: [{temperature.min():.1f}, {temperature.max():.1f}]°C")
        print(f"   - Precipitación: [{precipitation.min():.1f}, {precipitation.max():.1f}]mm/h")
        print(f"   - Eventos: [{events.min():.2f}, {events.max():.2f}]")
        
        return factors
    
    def add_product_categories(self, demand: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Agrega demanda por categorías de productos
        
        Args:
            demand: Demanda total
            
        Returns:
            Diccionario con demanda por categorías
        """
        print("🛍️ Agregando categorías de productos...")
        
        categories = {
            'electronics': 0.25,    # 25% de demanda total
            'clothing': 0.20,      # 20% de demanda total
            'food': 0.30,          # 30% de demanda total
            'home': 0.15,          # 15% de demanda total
            'sports': 0.10         # 10% de demanda total
        }
        
        category_demand = {}
        
        for category, proportion in categories.items():
            # Base proporcional
            cat_demand = demand * proportion
            
            # Patrones específicos por categoría
            if category == 'electronics':
                # Más demanda en temporada de navidad y black friday
                dates = pd.date_range(start=self.start_date, 
                                    periods=self.num_samples, 
                                    freq='H')
                christmas_boost = ((dates.month == 12) & (dates.day >= 20)).astype(float) * 0.5
                cat_demand *= (1 + christmas_boost)
                
            elif category == 'clothing':
                # Estacionalidad más marcada (verano vs invierno)
                seasonal_pattern = 1 + 0.3 * np.sin(2 * np.pi * np.arange(self.num_samples) / (365 * 24) - np.pi/2)
                cat_demand *= seasonal_pattern
                
            elif category == 'food':
                # Menor variación estacional, más patrino semanal
                weekly_pattern = 1 + 0.2 * np.sin(2 * np.pi * np.arange(self.num_samples) / (7 * 24))
                cat_demand *= weekly_pattern
                
            elif category == 'home':
                # Más demanda en fines de semana
                dates = pd.date_range(start=self.start_date, 
                                    periods=self.num_samples, 
                                    freq='H')
                weekend_boost = (dates.dayofweek >= 5).astype(float) * 0.3
                cat_demand *= (1 + weekend_boost)
                
            elif category == 'sports':
                # Más demanda en primavera y verano
                spring_summer_boost = ((dates.month.isin([3, 4, 5, 6, 7, 8]))).astype(float) * 0.4
                cat_demand *= (1 + spring_summer_boost)
            
            # Agregar ruido específico por categoría
            category_noise = np.random.normal(0, 0.03, self.num_samples)
            cat_demand *= (1 + category_noise)
            
            category_demand[category] = np.maximum(cat_demand, 1)
        
        print(f"✅ Categorías generadas:")
        for category, demand_array in category_demand.items():
            print(f"   - {category}: demanda promedio {demand_array.mean():.1f}")
        
        return category_demand
    
    def create_sequences(self, data: np.ndarray, 
                        sequence_length: int = 24,
                        prediction_horizon: int = 6) -> Tuple[np.ndarray, np.ndarray]:
        """
        Crea secuencias para entrenamiento de modelos de series temporales
        
        Args:
            data: Datos de series temporales
            sequence_length: Longitud de secuencia de entrada
            prediction_horizon: Horizonte de predicción
            
        Returns:
            Tupla con secuencias (X, y)
        """
        print(f"🔄 Creando secuencias (seq_len={sequence_length}, horizon={prediction_horizon})...")
        
        X, y = [], []
        
        for i in range(len(data) - sequence_length - prediction_horizon):
            # Secuencia de entrada
            seq_x = data[i:i + sequence_length]
            # Secuencia de salida (predicción)
            seq_y = data[i + sequence_length:i + sequence_length + prediction_horizon]
            
            X.append(seq_x)
            y.append(seq_y)
        
        X = np.array(X)
        y = np.array(y)
        
        print(f"✅ Secuencias creadas: {X.shape[0]} muestras")
        
        return X, y
    
    def normalize_data(self, data: np.ndarray) -> np.ndarray:
        """
        Normaliza datos al rango [0, 1]
        
        Args:
            data: Datos a normalizar
            
        Returns:
            Datos normalizados
        """
        print("📏 Normalizando datos...")
        
        # Reshape para scaler
        data_reshaped = data.reshape(-1, 1)
        
        # Fit y transform
        normalized = self.scaler.fit_transform(data_reshaped)
        
        # Volver a shape original
        normalized = normalized.flatten()
        
        print(f"✅ Datos normalizados: rango [{normalized.min():.3f}, {normalized.max():.3f}]")
        
        return normalized
    
    def inverse_normalize(self, data: np.ndarray) -> np.ndarray:
        """
        Invierte la normalización
        
        Args:
            data: Datos normalizados
            
        Returns:
            Datos en escala original
        """
        data_reshaped = data.reshape(-1, 1)
        original = self.scaler.inverse_transform(data_reshaped)
        return original.flatten()
    
    def generate_complete_dataset(self) -> Dict[str, Any]:
        """
        Genera el dataset completo con todos los componentes
        
        Returns:
            Diccionario con dataset completo
        """
        print("🚀 Generando dataset completo...")
        
        # 1. Generar demanda base
        base_demand = self.generate_base_demand()
        
        # 2. Agregar factores externos
        factors = self.add_external_factors(base_demand)
        demand_adjusted = factors['demand_adjusted']
        
        # 3. Agregar categorías
        categories = self.add_product_categories(demand_adjusted)
        
        # 4. Normalizar datos
        normalized_demand = self.normalize_data(demand_adjusted)
        
        # 5. Crear secuencias para diferentes horizontes
        sequences = {}
        
        # Para predicción de 6 horas
        X_6h, y_6h = self.create_sequences(normalized_demand, 24, 6)
        sequences['6h'] = {'X': X_6h, 'y': y_6h}
        
        # Para predicción de 24 horas
        X_24h, y_24h = self.create_sequences(normalized_demand, 48, 24)
        sequences['24h'] = {'X': X_24h, 'y': y_24h}
        
        # Para predicción de 7 días
        X_7d, y_7d = self.create_sequences(normalized_demand, 168, 168)
        sequences['7d'] = {'X': X_7d, 'y': y_7d}
        
        # 6. Crear DataFrame principal
        dates = pd.date_range(start=self.start_date, 
                            periods=self.num_samples, 
                            freq='H')
        
        main_df = pd.DataFrame({
            'timestamp': dates,
            'demand': demand_adjusted,
            'demand_normalized': normalized_demand,
            'temperature': factors['temperature'],
            'precipitation': factors['precipitation'],
            'events': factors['events']
        })
        
        # Agregar categorías
        for category, demand_array in categories.items():
            main_df[f'demand_{category}'] = demand_array
        
        # Agregar features temporales
        main_df['hour'] = dates.hour
        main_df['day_of_week'] = dates.dayofweek
        main_df['month'] = dates.month
        main_df['is_weekend'] = (dates.dayofweek >= 5).astype(int)
        main_df['is_holiday'] = (factors['events'] > 0.2).astype(int)
        
        dataset = {
            'main_dataframe': main_df,
            'sequences': sequences,
            'categories': categories,
            'factors': factors,
            'scaler': self.scaler,
            'metadata': {
                'num_samples': self.num_samples,
                'start_date': self.start_date,
                'end_date': dates[-1],
                'frequency': 'H'
            }
        }
        
        print("✅ Dataset completo generado")
        print(f"   - Período: {dataset['metadata']['start_date']} a {dataset['metadata']['end_date']}")
        print(f"   - Secuencias 6h: {sequences['6h']['X'].shape}")
        print(f"   - Secuencias 24h: {sequences['24h']['X'].shape}")
        print(f"   - Secuencias 7d: {sequences['7d']['X'].shape}")
        
        return dataset
    
    def visualize_dataset(self, dataset: Dict[str, Any], 
                          sample_days: int = 30):
        """
        Visualiza el dataset generado
        
        Args:
            dataset: Dataset generado
            sample_days: Número de días a visualizar
        """
        print(f"📊 Visualizando {sample_days} días del dataset...")
        
        df = dataset['main_dataframe']
        
        # Limitar a sample_days
        sample_df = df.head(sample_days * 24)
        
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        
        # 1. Demanda principal
        axes[0, 0].plot(sample_df['timestamp'], sample_df['demand'], 'b-', linewidth=1)
        axes[0, 0].set_title('Demanda Principal')
        axes[0, 0].set_ylabel('Demanda')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Demanda por categorías
        category_cols = [col for col in sample_df.columns if col.startswith('demand_') and col != 'demand_normalized']
        for col in category_cols:
            axes[0, 1].plot(sample_df['timestamp'], sample_df[col], label=col.replace('demand_', ''), alpha=0.7)
        axes[0, 1].set_title('Demanda por Categoría')
        axes[0, 1].set_ylabel('Demanda')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Temperatura
        axes[1, 0].plot(sample_df['timestamp'], sample_df['temperature'], 'r-', linewidth=1)
        axes[1, 0].set_title('Temperatura')
        axes[1, 0].set_ylabel('Temperatura (°C)')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Precipitación
        axes[1, 1].plot(sample_df['timestamp'], sample_df['precipitation'], 'g-', linewidth=1)
        axes[1, 1].set_title('Precipitación')
        axes[1, 1].set_ylabel('Precipitación (mm/h)')
        axes[1, 1].grid(True, alpha=0.3)
        
        # 5. Eventos
        axes[2, 0].plot(sample_df['timestamp'], sample_df['events'], 'purple', linewidth=1)
        axes[2, 0].set_title('Eventos Especiales')
        axes[2, 0].set_ylabel('Factor de Evento')
        axes[2, 0].grid(True, alpha=0.3)
        
        # 6. Patrones diarios promedio
        hourly_avg = sample_df.groupby('hour')['demand'].mean()
        axes[2, 1].plot(hourly_avg.index, hourly_avg.values, 'o-', color='orange')
        axes[2, 1].set_title('Patrón Diario Promedio')
        axes[2, 1].set_xlabel('Hora del Día')
        axes[2, 1].set_ylabel('Demanda Promedio')
        axes[2, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def save_dataset(self, dataset: Dict[str, Any], 
                    filepath: str = 'retail_demand_dataset.csv'):
        """
        Guarda el dataset en formato CSV
        
        Args:
            dataset: Dataset a guardar
            filepath: Ruta del archivo CSV
        """
        print(f"💾 Guardando dataset en {filepath}...")
        
        dataset['main_dataframe'].to_csv(filepath, index=False)
        
        # Guardar secuencias en formato NPZ
        np.savez('sequences.npz', **{k: v for seq_dict in dataset['sequences'].values() for k, v in seq_dict.items()})
        
        print("✅ Dataset guardado exitosamente")
    
    def get_statistics(self, dataset: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calcula estadísticas del dataset
        
        Args:
            dataset: Dataset generado
            
        Returns:
            Diccionario con estadísticas
        """
        print("📈 Calculando estadísticas del dataset...")
        
        df = dataset['main_dataframe']
        
        stats = {
            'demand_stats': {
                'mean': df['demand'].mean(),
                'std': df['demand'].std(),
                'min': df['demand'].min(),
                'max': df['demand'].max(),
                'median': df['demand'].median()
            },
            'category_stats': {},
            'temporal_stats': {
                'peak_hour': df.groupby('hour')['demand'].mean().idxmax(),
                'peak_day': df.groupby('day_of_week')['demand'].mean().idxmax(),
                'peak_month': df.groupby('month')['demand'].mean().idxmax()
            },
            'correlation_matrix': df[['demand', 'temperature', 'precipitation', 'events']].corr().to_dict()
        }
        
        # Estadísticas por categoría
        category_cols = [col for col in df.columns if col.startswith('demand_') and col != 'demand_normalized']
        for col in category_cols:
            stats['category_stats'][col.replace('demand_', '')] = {
                'mean': df[col].mean(),
                'std': df[col].std(),
                'proportion_of_total': df[col].mean() / df['demand'].mean()
            }
        
        print("✅ Estadísticas calculadas:")
        print(f"   - Demanda media: {stats['demand_stats']['mean']:.1f}")
        print(f"   - Hora pico: {stats['temporal_stats']['peak_hour']}:00")
        print(f"   - Día pico: {stats['temporal_stats']['peak_day']} (0=Lunes, 6=Domingo)")
        
        return stats


def main():
    """
    Función principal para probar el generador de series temporales
    """
    print("🚀 Iniciando generación de series temporales sintéticas")
    
    # Crear generador
    generator = TimeSeriesGenerator(num_samples=10000, start_date='2020-01-01')
    
    # Generar dataset completo
    dataset = generator.generate_complete_dataset()
    
    # Visualizar dataset
    generator.visualize_dataset(dataset, sample_days=30)
    
    # Obtener estadísticas
    stats = generator.get_statistics(dataset)
    
    # Guardar dataset
    generator.save_dataset(dataset)
    
    print("✅ Generación de series temporales completada exitosamente")


if __name__ == "__main__":
    main()
