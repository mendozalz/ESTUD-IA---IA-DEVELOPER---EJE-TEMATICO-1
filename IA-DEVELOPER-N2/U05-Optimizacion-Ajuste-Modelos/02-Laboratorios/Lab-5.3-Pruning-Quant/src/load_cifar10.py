"""
Laboratorio 5.3 - Carga y Preprocesamiento de CIFAR-10
========================================================

Este script carga y preprocesa el dataset CIFAR-10 para el laboratorio
de optimización de modelos con pruning, quantization y distillation.
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Dict, Any

class CIFAR10DataLoader:
    """
    Clase para cargar y preprocesar el dataset CIFAR-10
    """
    
    def __init__(self, batch_size: int = 32):
        """
        Inicializa el cargador de datos
        
        Args:
            batch_size: Tamaño del batch para entrenamiento
        """
        self.batch_size = batch_size
        self.class_names = [
            'airplane', 'automobile', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck'
        ]
        self.num_classes = 10
        self.input_shape = (32, 32, 3)
        
    def load_data(self) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
        """
        Carga el dataset CIFAR-10
        
        Returns:
            Tuple con datos de entrenamiento y prueba
        """
        print("📦 Cargando dataset CIFAR-10...")
        
        # Cargar datos
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
        
        # Convertir labels a one-hot encoding
        y_train = tf.keras.utils.to_categorical(y_train, self.num_classes)
        y_test = tf.keras.utils.to_categorical(y_test, self.num_classes)
        
        print(f"✅ Datos cargados:")
        print(f"   - Entrenamiento: {x_train.shape} imágenes")
        print(f"   - Prueba: {x_test.shape} imágenes")
        print(f"   - Clases: {self.num_classes}")
        
        return (x_train, y_train), (x_test, y_test)
    
    def preprocess_data(self, x: np.ndarray, y: np.ndarray, 
                       augment: bool = False) -> Tuple[tf.data.Dataset, int]:
        """
        Preprocesa los datos y crea dataset de TensorFlow
        
        Args:
            x: Imágenes
            y: Labels
            augment: Si aplicar data augmentation
            
        Returns:
            Dataset de TensorFlow y número de muestras
        """
        print(f"🔧 Preprocesando datos (augmentation={augment})...")
        
        # Normalizar a [0, 1]
        x = x.astype('float32') / 255.0
        
        # Crear dataset
        dataset = tf.data.Dataset.from_tensor_slices((x, y))
        
        if augment:
            # Data augmentation para entrenamiento
            def augment_image(image, label):
                # Flip horizontal
                image = tf.image.random_flip_left_right(image)
                # Random brightness
                image = tf.image.random_brightness(image, max_delta=0.1)
                # Random contrast
                image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
                # Random crop
                image = tf.image.random_crop(image, size=[28, 28, 3])
                # Resize back
                image = tf.image.resize(image, size=[32, 32])
                return image, label
            
            dataset = dataset.map(augment_image, num_parallel_calls=tf.data.AUTOTUNE)
        
        # Shuffle, batch y prefetch
        dataset = dataset.shuffle(buffer_size=10000)
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        num_samples = len(x)
        print(f"✅ Dataset creado: {num_samples} muestras")
        
        return dataset, num_samples
    
    def create_datasets(self) -> Dict[str, Any]:
        """
        Crea datasets completos para entrenamiento y prueba
        
        Returns:
            Diccionario con datasets y metadata
        """
        # Cargar datos
        (x_train, y_train), (x_test, y_test) = self.load_data()
        
        # Crear datasets
        train_dataset, train_samples = self.preprocess_data(x_train, y_train, augment=True)
        test_dataset, test_samples = self.preprocess_data(x_test, y_test, augment=False)
        
        # Dataset de validación (20% del entrenamiento)
        val_size = int(train_samples * 0.2)
        val_dataset = train_dataset.take(val_size)
        train_dataset = train_dataset.skip(val_size)
        
        return {
            'train_dataset': train_dataset,
            'val_dataset': val_dataset,
            'test_dataset': test_dataset,
            'train_samples': train_samples - val_size,
            'val_samples': val_size,
            'test_samples': test_samples,
            'class_names': self.class_names,
            'input_shape': self.input_shape,
            'num_classes': self.num_classes
        }
    
    def visualize_samples(self, dataset: tf.data.Dataset, num_samples: int = 9):
        """
        Visualiza muestras del dataset
        
        Args:
            dataset: Dataset de TensorFlow
            num_samples: Número de muestras a visualizar
        """
        print(f"🖼️ Visualizando {num_samples} muestras...")
        
        # Obtener un batch
        images, labels = next(iter(dataset))
        
        # Crear figura
        plt.figure(figsize=(10, 10))
        
        for i in range(min(num_samples, len(images))):
            plt.subplot(3, 3, i + 1)
            plt.imshow(images[i])
            label_idx = tf.argmax(labels[i]).numpy()
            plt.title(f"{self.class_names[label_idx]}")
            plt.axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def get_data_statistics(self, dataset: tf.data.Dataset) -> Dict[str, Any]:
        """
        Calcula estadísticas del dataset
        
        Args:
            dataset: Dataset de TensorFlow
            
        Returns:
            Diccionario con estadísticas
        """
        print("📊 Calculando estadísticas del dataset...")
        
        # Calcular media y desviación estándar
        all_images = []
        all_labels = []
        
        for batch_images, batch_labels in dataset.take(100):  # Limitar para eficiencia
            all_images.append(batch_images.numpy())
            all_labels.append(batch_labels.numpy())
        
        all_images = np.concatenate(all_images)
        all_labels = np.concatenate(all_labels)
        
        # Estadísticas
        stats = {
            'mean': np.mean(all_images, axis=(0, 1, 2)),
            'std': np.std(all_images, axis=(0, 1, 2)),
            'min': np.min(all_images),
            'max': np.max(all_images),
            'shape': all_images.shape,
            'label_distribution': np.sum(all_labels, axis=0)
        }
        
        print(f"✅ Estadísticas calculadas:")
        print(f"   - Forma: {stats['shape']}")
        print(f"   - Rango: [{stats['min']:.3f}, {stats['max']:.3f}]")
        print(f"   - Media RGB: {stats['mean']}")
        print(f"   - Std RGB: {stats['std']}")
        
        return stats


def main():
    """
    Función principal para probar el cargador de datos
    """
    print("🚀 Iniciando prueba del cargador de CIFAR-10")
    
    # Crear cargador
    loader = CIFAR10DataLoader(batch_size=32)
    
    # Crear datasets
    datasets = loader.create_datasets()
    
    # Visualizar muestras
    loader.visualize_samples(datasets['train_dataset'])
    
    # Obtener estadísticas
    stats = loader.get_data_statistics(datasets['train_dataset'])
    
    print("✅ Prueba completada exitosamente")


if __name__ == "__main__":
    main()
