"""
Data Loader para Clasificación de Imágenes Médicas
Laboratorio 4.1 - Clasificación de Imágenes Médicas con EfficientNetV3 y ViT
"""

import os
import cv2
import numpy as np
import albumentations as A
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
import tensorflow as tf

class MedicalImageDataLoader:
    """
    Clase para cargar y preprocesar imágenes médicas con aumento de datos
    """
    
    def __init__(self, image_size=(224, 224), batch_size=32):
        self.image_size = image_size
        self.batch_size = batch_size
        
        # Transformaciones de aumento de datos específicas para imágenes médicas
        self.train_transform = A.Compose([
            A.RandomRotate90(p=0.5),
            A.Flip(p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            A.GaussNoise(p=0.2),
            A.ElasticTransform(p=0.2),
            A.Resize(image_size[0], image_size[1]),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.val_transform = A.Compose([
            A.Resize(image_size[0], image_size[1]),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def load_and_augment(self, image_path):
        """
        Carga y aplica aumento de datos a una imagen médica
        """
        try:
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"No se pudo cargar la imagen: {image_path}")
            
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            augmented = self.train_transform(image=image)['image']
            return augmented
        except Exception as e:
            print(f"Error al procesar {image_path}: {str(e)}")
            return None
    
    def create_data_generators(self, data_dir, validation_split=0.2):
        """
        Crea generadores de datos para entrenamiento y validación
        """
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            validation_split=validation_split,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )
        
        train_generator = train_datagen.flow_from_directory(
            data_dir,
            target_size=self.image_size,
            batch_size=self.batch_size,
            class_mode='binary',
            subset='training',
            shuffle=True
        )
        
        validation_generator = train_datagen.flow_from_directory(
            data_dir,
            target_size=self.image_size,
            batch_size=self.batch_size,
            class_mode='binary',
            subset='validation',
            shuffle=False
        )
        
        return train_generator, validation_generator
    
    def preprocess_for_prediction(self, image_path):
        """
        Preprocesa una imagen para predicción
        """
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"No se pudo cargar la imagen: {image_path}")
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, self.image_size)
        image = image / 255.0
        image = np.expand_dims(image, axis=0)
        
        return image
    
    def get_class_weights(self, data_dir):
        """
        Calcula pesos de clases para datasets desbalanceados
        """
        from sklearn.utils.class_weight import compute_class_weight
        import pandas as pd
        
        # Contar muestras por clase
        class_counts = {}
        for class_name in os.listdir(data_dir):
            class_path = os.path.join(data_dir, class_name)
            if os.path.isdir(class_path):
                class_counts[class_name] = len(os.listdir(class_path))
        
        # Calcular pesos
        classes = list(class_counts.keys())
        counts = list(class_counts.values())
        weights = compute_class_weight('balanced', classes=np.unique(classes), y=classes)
        
        return dict(zip(classes, weights))


if __name__ == "__main__":
    # Ejemplo de uso
    loader = MedicalImageDataLoader()
    
    # Crear generadores de datos
    train_gen, val_gen = loader.create_data_generators("data/train")
    
    print(f"Clases encontradas: {train_gen.class_indices}")
    print(f"Batch shape: {next(train_gen)[0].shape}")
    
    # Probar preprocesamiento
    try:
        sample_image = loader.preprocess_for_prediction("data/test/sample.jpg")
        print(f"Imagen preprocesada shape: {sample_image.shape}")
    except Exception as e:
        print(f"Error: {e}")
