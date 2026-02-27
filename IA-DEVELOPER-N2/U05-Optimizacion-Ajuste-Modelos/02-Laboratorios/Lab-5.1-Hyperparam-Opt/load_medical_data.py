"""
Carga y Preprocesamiento de Datos Médicos
Laboratorio 5.1 - Hyperparameter Tuning Avanzado con Optuna y Weights & Biases
"""

import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import pandas as pd
import logging
from typing import Tuple, Dict, Any
import os

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MedicalDataLoader:
    """
    Clase para cargar y preprocesar datasets médicos de imágenes
    """
    
    def __init__(self, dataset_name: str = 'chexpert', image_size: Tuple[int, int] = (224, 224)):
        self.dataset_name = dataset_name
        self.image_size = image_size
        self.info = None
        self.class_names = None
        
        logger.info(f"MedicalDataLoader inicializado para dataset: {dataset_name}")
    
    def load_dataset(self, split_ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15)) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
        """
        Carga el dataset médico y lo divide en train/val/test
        """
        logger.info(f"Cargando dataset {self.dataset_name}...")
        
        try:
            # Cargar dataset completo
            full_dataset, self.info = tfds.load(
                self.dataset_name,
                split='train',
                shuffle_files=True,
                as_supervised=True,
                with_info=True
            )
            
            # Obtener tamaño total
            total_size = self.info.splits['train'].num_examples
            logger.info(f"Dataset cargado: {total_size} muestras")
            
            # Calcular tamaños de splits
            train_size = int(total_size * split_ratios[0])
            val_size = int(total_size * split_ratios[1])
            test_size = total_size - train_size - val_size
            
            # Dividir dataset
            train_dataset = full_dataset.take(train_size)
            remaining = full_dataset.skip(train_size)
            val_dataset = remaining.take(val_size)
            test_dataset = remaining.skip(val_size)
            
            logger.info(f"División completada:")
            logger.info(f"  Train: {train_size} muestras ({split_ratios[0]*100:.1f}%)")
            logger.info(f"  Val: {val_size} muestras ({split_ratios[1]*100:.1f}%)")
            logger.info(f"  Test: {test_size} muestras ({split_ratios[2]*100:.1f}%)")
            
            return train_dataset, val_dataset, test_dataset
            
        except Exception as e:
            logger.error(f"Error cargando dataset: {e}")
            raise
    
    def preprocess_image(self, image: tf.Tensor, label: tf.Tensor, training: bool = True) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Preprocesa una imagen individual
        """
        # Resize a tamaño objetivo
        image = tf.image.resize(image, self.image_size)
        
        # Normalización a [0, 1]
        image = tf.cast(image, tf.float32) / 255.0
        
        if training:
            # Data augmentation solo para entrenamiento
            image = tf.image.random_flip_left_right(image)
            image = tf.image.random_flip_up_down(image)
            image = tf.image.random_brightness(image, 0.1)
            image = tf.image.random_contrast(image, 0.9, 1.1)
            image = tf.image.random_saturation(image, 0.9, 1.1)
            
            # Rotación aleatoria
            angle = tf.random.uniform([], -0.1, 0.1)  # ±10 grados
            image = tfa.image.rotate(image, angle)
            
            # Zoom aleatorio
            scales = tf.random.uniform([], 0.9, 1.1)
            new_size = tf.cast(tf.cast(tf.shape(image)[:2], tf.float32) * scales, tf.int32)
            image = tf.image.resize(image, new_size)
            image = tf.image.resize_with_crop_or_pad(image, self.image_size[0], self.image_size[1])
        
        return image, label
    
    def create_datasets(self, train_dataset: tf.data.Dataset, val_dataset: tf.data.Dataset, 
                        test_dataset: tf.data.Dataset, batch_size: int = 32) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
        """
        Crea los datasets finales con preprocesamiento y batching
        """
        logger.info(f"Creando datasets con batch_size={batch_size}...")
        
        # Aplicar preprocesamiento
        train_ds = train_dataset.map(
            lambda x, y: self.preprocess_image(x, y, training=True),
            num_parallel_calls=tf.data.AUTOTUNE
        )
        
        val_ds = val_dataset.map(
            lambda x, y: self.preprocess_image(x, y, training=False),
            num_parallel_calls=tf.data.AUTOTUNE
        )
        
        test_ds = test_dataset.map(
            lambda x, y: self.preprocess_image(x, y, training=False),
            num_parallel_calls=tf.data.AUTOTUNE
        )
        
        # Configurar batching y prefetching
        train_ds = train_ds.shuffle(1000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
        val_ds = val_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        test_ds = test_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        
        logger.info("Datasets creados exitosamente")
        
        return train_ds, val_ds, test_ds
    
    def analyze_dataset(self, train_dataset: tf.data.Dataset, num_samples: int = 1000) -> Dict[str, Any]:
        """
        Analiza estadísticas del dataset
        """
        logger.info("Analizando estadísticas del dataset...")
        
        # Extraer muestras para análisis
        samples = list(train_dataset.take(num_samples))
        images = np.array([sample[0].numpy() for sample in samples])
        labels = np.array([sample[1].numpy() for sample in samples])
        
        # Estadísticas de imágenes
        image_stats = {
            'mean': np.mean(images),
            'std': np.std(images),
            'min': np.min(images),
            'max': np.max(images),
            'shape': images.shape[1:],
            'dtype': images.dtype
        }
        
        # Estadísticas de etiquetas
        unique_labels, counts = np.unique(labels, return_counts=True)
        label_distribution = dict(zip(unique_labels.tolist(), counts.tolist()))
        
        # Calcular balance de clases
        total_samples = len(labels)
        class_balance = {k: v/total_samples for k, v in label_distribution.items()}
        
        stats = {
            'total_samples': len(samples),
            'image_stats': image_stats,
            'label_distribution': label_distribution,
            'class_balance': class_balance,
            'num_classes': len(unique_labels)
        }
        
        logger.info(f"Análisis completado:")
        logger.info(f"  Muestras analizadas: {stats['total_samples']}")
        logger.info(f"  Clases: {stats['num_classes']}")
        logger.info(f"  Balance de clases: {class_balance}")
        
        return stats
    
    def visualize_samples(self, dataset: tf.data.Dataset, num_samples: int = 9, save_path: str = None):
        """
        Visualiza muestras del dataset
        """
        logger.info(f"Visualizando {num_samples} muestras...")
        
        # Extraer muestras
        samples = list(dataset.take(num_samples))
        images = [sample[0].numpy() for sample in samples]
        labels = [sample[1].numpy() for sample in samples]
        
        # Crear grid de visualización
        fig, axes = plt.subplots(3, 3, figsize=(12, 12))
        axes = axes.flatten()
        
        for i in range(min(num_samples, len(images))):
            ax = axes[i]
            
            # Desnormalizar para visualización
            img = images[i]
            if img.max() <= 1.0:
                img = (img * 255).astype(np.uint8)
            
            ax.imshow(img)
            ax.set_title(f'Label: {labels[i]}')
            ax.axis('off')
        
        # Ocultar ejes extra
        for i in range(num_samples, len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Visualización guardada en: {save_path}")
        
        plt.show()
    
    def create_class_weights(self, train_dataset: tf.data.Dataset) -> Dict[int, float]:
        """
        Calcula pesos para clases desbalanceadas
        """
        logger.info("Calculando pesos de clases...")
        
        # Extraer todas las etiquetas
        labels = []
        for batch in train_dataset:
            labels.extend(batch[1].numpy().tolist())
        
        labels = np.array(labels)
        unique_labels, counts = np.unique(labels, return_counts=True)
        
        # Calcular pesos inversamente proporcionales
        total_samples = len(labels)
        num_classes = len(unique_labels)
        
        class_weights = {}
        for label, count in zip(unique_labels, counts):
            weight = total_samples / (num_classes * count)
            class_weights[int(label)] = weight
        
        logger.info(f"Pesos de clases: {class_weights}")
        
        return class_weights
    
    def save_dataset_info(self, save_dir: str):
        """
        Guarda información del dataset
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # Guardar información del dataset
        info_dict = {
            'dataset_name': self.dataset_name,
            'image_size': self.image_size,
            'total_samples': self.info.splits['train'].num_examples if self.info else None,
            'features': self.info.features if self.info else None,
            'supervised_keys': self.info.supervised_keys if self.info else None
        }
        
        import json
        with open(os.path.join(save_dir, 'dataset_info.json'), 'w') as f:
            json.dump(info_dict, f, indent=2, default=str)
        
        logger.info(f"Información del dataset guardada en: {save_dir}")


def create_synthetic_medical_data(num_samples: int = 1000, image_size: Tuple[int, int] = (224, 224)) -> Tuple[np.ndarray, np.ndarray]:
    """
    Crea datos médicos sintéticos para pruebas cuando el dataset real no está disponible
    """
    logger.info(f"Creando dataset sintético con {num_samples} muestras...")
    
    # Generar imágenes sintéticas con patrones médicos
    images = []
    labels = []
    
    for i in range(num_samples):
        # Crear imagen base con ruido
        image = np.random.randn(*image_size, 3) * 0.1
        
        # Añadir patrones simulados
        label = np.random.randint(0, 2)
        
        if label == 1:  # Caso positivo (ej: neumonía)
            # Añadir opacidades simuladas
            center_x = np.random.randint(50, image_size[0] - 50)
            center_y = np.random.randint(50, image_size[1] - 50)
            radius = np.random.randint(20, 60)
            
            y, x = np.ogrid[:image_size[0], :image_size[1]]
            mask = (x - center_x)**2 + (y - center_y)**2 <= radius**2
            
            # Hacer la región más brillante (opacidad)
            for c in range(3):
                image[mask, c] += np.random.uniform(0.3, 0.7)
        
        # Normalizar a [0, 1]
        image = np.clip(image, 0, 1)
        
        images.append(image)
        labels.append(label)
    
    images = np.array(images, dtype=np.float32)
    labels = np.array(labels, dtype=np.int32)
    
    logger.info(f"Dataset sintético creado: {images.shape}, {labels.shape}")
    
    return images, labels


def main():
    """
    Función principal para demostrar el uso del MedicalDataLoader
    """
    try:
        # Intentar cargar dataset real
        loader = MedicalDataLoader('chexpert', image_size=(224, 224))
        
        # Cargar datasets
        train_ds_raw, val_ds_raw, test_ds_raw = loader.load_dataset()
        
        # Crear datasets procesados
        train_ds, val_ds, test_ds = loader.create_datasets(
            train_ds_raw, val_ds_raw, test_ds_raw, batch_size=32
        )
        
        # Analizar dataset
        stats = loader.analyze_dataset(train_ds)
        
        # Visualizar muestras
        loader.visualize_samples(train_ds, num_samples=9, save_path='sample_visualization.png')
        
        # Calcular pesos de clases
        class_weights = loader.create_class_weights(train_ds)
        
        # Guardar información
        loader.save_dataset_info('data_info')
        
        logger.info("✅ Dataset médico cargado y procesado exitosamente")
        
    except Exception as e:
        logger.warning(f"No se pudo cargar el dataset real: {e}")
        logger.info("Creando dataset sintético para demostración...")
        
        # Crear dataset sintético
        images, labels = create_synthetic_medical_data(num_samples=1000)
        
        # Visualizar muestras sintéticas
        fig, axes = plt.subplots(2, 5, figsize=(15, 6))
        axes = axes.flatten()
        
        for i in range(10):
            ax = axes[i]
            ax.imshow(images[i])
            ax.set_title(f'Label: {labels[i]}')
            ax.axis('off')
        
        plt.tight_layout()
        plt.savefig('synthetic_samples.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        logger.info("✅ Dataset sintético creado para demostración")


if __name__ == "__main__":
    main()
