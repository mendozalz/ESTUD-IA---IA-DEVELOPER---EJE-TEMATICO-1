"""
Preprocesamiento de Datos Multimodales (Imagen + Texto)
Laboratorio 4.3 - Clasificación Multimodal Retail
"""

import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import json
import logging
from typing import List, Tuple, Dict, Any
import albumentations as A

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MultimodalDataset(Dataset):
    """
    Dataset personalizado para datos multimodales (imagen + texto)
    """
    
    def __init__(self, image_paths: List[str], texts: List[str], labels: List[int],
                 image_transform=None, tokenizer=None, max_length=128):
        self.image_paths = image_paths
        self.texts = texts
        self.labels = labels
        self.image_transform = image_transform
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Validar longitudes
        assert len(image_paths) == len(texts) == len(labels), \
            "Todas las listas deben tener la misma longitud"
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Cargar imagen
        image_path = self.image_paths[idx]
        try:
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"No se pudo cargar la imagen: {image_path}")
            
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            if self.image_transform:
                transformed = self.image_transform(image=image)
                image = transformed['image']
            else:
                # Resize por defecto
                image = cv2.resize(image, (300, 300))
                image = image / 255.0
                
        except Exception as e:
            logger.error(f"Error cargando imagen {image_path}: {e}")
            # Imagen por defecto
            image = np.zeros((300, 300, 3), dtype=np.float32)
        
        # Procesar texto
        text = str(self.texts[idx])
        if self.tokenizer:
            encoding = self.tokenizer(
                text,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            input_ids = encoding['input_ids'].flatten()
            attention_mask = encoding['attention_mask'].flatten()
        else:
            input_ids = torch.zeros(self.max_length, dtype=torch.long)
            attention_mask = torch.zeros(self.max_length, dtype=torch.long)
        
        return {
            'image': torch.FloatTensor(image).permute(2, 0, 1),  # CHW format
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'label': torch.tensor(self.labels[idx], dtype=torch.long)
        }

class MultimodalDataPreprocessor:
    """
    Clase para preprocesar datos multimodales
    """
    
    def __init__(self, image_size=(300, 300), max_text_length=128):
        self.image_size = image_size
        self.max_text_length = max_text_length
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.label_encoder = LabelEncoder()
        
        # Transformaciones para imágenes
        self.train_transform = A.Compose([
            A.Resize(image_size[0], image_size[1]),
            A.RandomRotate90(p=0.5),
            A.Flip(p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            A.GaussNoise(p=0.2),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.val_transform = A.Compose([
            A.Resize(image_size[0], image_size[1]),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def load_data_from_csv(self, csv_path: str, image_dir: str, 
                          image_column: str = 'image_path',
                          text_column: str = 'description',
                          label_column: str = 'category') -> pd.DataFrame:
        """
        Carga datos desde archivo CSV
        """
        try:
            df = pd.read_csv(csv_path)
            
            # Validar columnas
            required_columns = [image_column, text_column, label_column]
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise ValueError(f"Columnas faltantes: {missing_columns}")
            
            # Construir rutas completas de imágenes
            df['full_image_path'] = df[image_column].apply(
                lambda x: os.path.join(image_dir, x) if pd.notna(x) else ''
            )
            
            # Filtrar imágenes que existen
            df['image_exists'] = df['full_image_path'].apply(
                lambda x: os.path.exists(x) if x else False
            )
            
            initial_count = len(df)
            df = df[df['image_exists']]
            filtered_count = len(df)
            
            logger.info(f"Datos cargados: {filtered_count}/{initial_count} muestras con imágenes válidas")
            
            # Limpiar texto
            df['clean_text'] = df[text_column].fillna('').astype(str).apply(self.clean_text)
            
            # Filtrar textos vacíos
            df = df[df['clean_text'].str.len() > 0]
            
            logger.info(f"Datos después de limpieza de texto: {len(df)} muestras")
            
            return df
            
        except Exception as e:
            logger.error(f"Error cargando datos: {e}")
            raise
    
    def clean_text(self, text: str) -> str:
        """
        Limpia y preprocesa el texto
        """
        if not text or pd.isna(text):
            return ""
        
        # Convertir a minúsculas
        text = str(text).lower()
        
        # Eliminar caracteres especiales excepto espacios y letras básicas
        text = ''.join(c if c.isalnum() or c.isspace() else ' ' for c in text)
        
        # Eliminar espacios extra
        text = ' '.join(text.split())
        
        return text.strip()
    
    def encode_labels(self, labels: List[str]) -> Tuple[List[int], List[str]]:
        """
        Codifica etiquetas a números
        """
        encoded_labels = self.label_encoder.fit_transform(labels)
        class_names = self.label_encoder.classes_.tolist()
        
        logger.info(f"Clases encontradas: {class_names}")
        
        return encoded_labels.tolist(), class_names
    
    def create_datasets(self, df: pd.DataFrame, 
                       image_column: str = 'full_image_path',
                       text_column: str = 'clean_text',
                       label_column: str = 'category',
                       test_size: float = 0.2,
                       val_size: float = 0.1,
                       random_state: int = 42) -> Tuple:
        """
        Crea datasets de entrenamiento, validación y prueba
        """
        # Codificar etiquetas
        labels, class_names = self.encode_labels(df[label_column].tolist())
        
        # Extraer características
        image_paths = df[image_column].tolist()
        texts = df[text_column].tolist()
        
        # Dividir datos
        # Primero: train+val vs test
        train_val_indices, test_indices = train_test_split(
            range(len(image_paths)),
            test_size=test_size,
            random_state=random_state,
            stratify=labels
        )
        
        # Segundo: train vs val
        train_indices, val_indices = train_test_split(
            train_val_indices,
            test_size=val_size/(1-test_size),
            random_state=random_state,
            stratify=[labels[i] for i in train_val_indices]
        )
        
        # Crear datasets
        train_dataset = MultimodalDataset(
            [image_paths[i] for i in train_indices],
            [texts[i] for i in train_indices],
            [labels[i] for i in train_indices],
            image_transform=self.train_transform,
            tokenizer=self.tokenizer,
            max_length=self.max_text_length
        )
        
        val_dataset = MultimodalDataset(
            [image_paths[i] for i in val_indices],
            [texts[i] for i in val_indices],
            [labels[i] for i in val_indices],
            image_transform=self.val_transform,
            tokenizer=self.tokenizer,
            max_length=self.max_text_length
        )
        
        test_dataset = MultimodalDataset(
            [image_paths[i] for i in test_indices],
            [texts[i] for i in test_indices],
            [labels[i] for i in test_indices],
            image_transform=self.val_transform,
            tokenizer=self.tokenizer,
            max_length=self.max_text_length
        )
        
        logger.info(f"Datasets creados:")
        logger.info(f"Train: {len(train_dataset)} muestras")
        logger.info(f"Val: {len(val_dataset)} muestras")
        logger.info(f"Test: {len(test_dataset)} muestras")
        
        return train_dataset, val_dataset, test_dataset, class_names
    
    def create_dataloaders(self, train_dataset, val_dataset, test_dataset,
                          batch_size=16, num_workers=4):
        """
        Carga DataLoaders para PyTorch
        """
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
        
        return train_loader, val_loader, test_loader
    
    def preprocess_single_sample(self, image_path: str, text: str) -> Dict[str, Any]:
        """
        Preprocesa una muestra individual para predicción
        """
        # Procesar imagen
        try:
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"No se pudo cargar la imagen: {image_path}")
            
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Aplicar transformación de validación
            transformed = self.val_transform(image=image)
            image = transformed['image']
            
            # Convertir a tensor CHW
            image = torch.FloatTensor(image).permute(2, 0, 1)
            
        except Exception as e:
            logger.error(f"Error procesando imagen: {e}")
            image = torch.zeros(3, self.image_size[0], self.image_size[1])
        
        # Procesar texto
        clean_text = self.clean_text(text)
        encoding = self.tokenizer(
            clean_text,
            max_length=self.max_text_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'image': image,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'clean_text': clean_text
        }
    
    def save_preprocessor(self, path: str):
        """
        Guarda el estado del preprocesador
        """
        state = {
            'image_size': self.image_size,
            'max_text_length': self.max_text_length,
            'label_encoder_classes': self.label_encoder.classes_.tolist()
        }
        
        with open(path, 'w') as f:
            json.dump(state, f, indent=2)
        
        logger.info(f"Preprocesador guardado en: {path}")
    
    def load_preprocessor(self, path: str):
        """
        Carga el estado del preprocesador
        """
        with open(path, 'r') as f:
            state = json.load(f)
        
        self.image_size = tuple(state['image_size'])
        self.max_text_length = state['max_text_length']
        self.label_encoder.classes_ = np.array(state['label_encoder_classes'])
        
        logger.info(f"Preprocesador cargado desde: {path}")
    
    def analyze_dataset(self, df: pd.DataFrame, 
                       image_column: str = 'full_image_path',
                       text_column: str = 'clean_text',
                       label_column: str = 'category') -> Dict[str, Any]:
        """
        Analiza estadísticas del dataset
        """
        print("\n📊 Análisis del Dataset Multimodal:")
        print(f"Total de muestras: {len(df)}")
        
        # Distribución de clases
        print("\n📈 Distribución de clases:")
        class_dist = df[label_column].value_counts()
        for label, count in class_dist.items():
            print(f"{label}: {count} muestras ({count/len(df)*100:.1f}%)")
        
        # Estadísticas de texto
        text_lengths = df[text_column].str.len()
        print(f"\n📝 Estadísticas de texto:")
        print(f"Longitud promedio: {text_lengths.mean():.1f} caracteres")
        print(f"Longitud máxima: {text_lengths.max()} caracteres")
        print(f"Longitud mínima: {text_lengths.min()} caracteres")
        
        # Estadísticas de imágenes
        image_sizes = []
        sample_images = df[image_column].dropna().head(100).tolist()
        
        for img_path in sample_images:
            try:
                img = cv2.imread(img_path)
                if img is not None:
                    h, w = img.shape[:2]
                    image_sizes.append((w, h))
            except:
                continue
        
        if image_sizes:
            widths, heights = zip(*image_sizes)
            print(f"\n🖼️ Estadísticas de imágenes (muestra de {len(image_sizes)}):")
            print(f"Dimensiones promedio: {np.mean(widths):.0f}x{np.mean(heights):.0f}")
            print(f"Dimensiones máximas: {max(widths)}x{max(heights)}")
            print(f"Dimensiones mínimas: {min(widths)}x{min(heights)}")
        
        return {
            'total_samples': len(df),
            'class_distribution': class_dist.to_dict(),
            'text_stats': {
                'mean_length': text_lengths.mean(),
                'max_length': text_lengths.max(),
                'min_length': text_lengths.min()
            },
            'image_stats': {
                'sample_count': len(image_sizes),
                'avg_dimensions': (np.mean(widths), np.mean(heights)) if image_sizes else None
            }
        }


def main():
    """
    Ejemplo de uso
    """
    preprocessor = MultimodalDataPreprocessor()
    
    # Ejemplo de carga de datos
    try:
        df = preprocessor.load_data_from_csv(
            'data/products.csv',
            'data/images/',
            image_column='image_filename',
            text_column='description',
            label_column='category'
        )
        
        # Analizar dataset
        stats = preprocessor.analyze_dataset(df)
        
        # Crear datasets
        train_ds, val_ds, test_ds, class_names = preprocessor.create_datasets(df)
        
        # Crear dataloaders
        train_loader, val_loader, test_loader = preprocessor.create_dataloaders(
            train_ds, val_ds, test_ds, batch_size=8
        )
        
        # Probar batch
        batch = next(iter(train_loader))
        print(f"\nBatch shapes:")
        print(f"Images: {batch['image'].shape}")
        print(f"Input IDs: {batch['input_ids'].shape}")
        print(f"Attention Mask: {batch['attention_mask'].shape}")
        print(f"Labels: {batch['label'].shape}")
        
        # Guardar preprocesador
        preprocessor.save_preprocessor('preprocessor_state.json')
        
    except FileNotFoundError:
        print("Archivos de datos no encontrados. Este es un ejemplo de uso.")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
