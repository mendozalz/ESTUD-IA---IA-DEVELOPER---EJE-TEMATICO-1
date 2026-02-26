"""
Data Loader para Clasificación de Texto con BERT
Laboratorio 4.2 - Clasificación de Texto con BERT y Adaptadores
"""

import pandas as pd
import numpy as np
from transformers import BertTokenizer
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import logging

# Descargar recursos de NLTK
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

class TextDataset(Dataset):
    """
    Dataset personalizado para texto con BERT
    """
    
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        # Tokenizar
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

class TextDataLoader:
    """
    Clase para cargar y preprocesar datos de texto
    """
    
    def __init__(self, model_name='bert-base-uncased', max_length=128):
        self.model_name = model_name
        self.max_length = max_length
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.stop_words = set(stopwords.words('english'))
        
        # Configurar logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def clean_text(self, text):
        """
        Limpia y preprocesa el texto
        """
        if pd.isna(text):
            return ""
        
        # Convertir a minúsculas
        text = str(text).lower()
        
        # Eliminar URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Eliminar menciones y hashtags
        text = re.sub(r'@\w+|#\w+', '', text)
        
        # Eliminar caracteres especiales y números
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Eliminar espacios extra
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Tokenizar y eliminar stopwords
        tokens = word_tokenize(text)
        tokens = [token for token in tokens if token not in self.stop_words]
        
        return ' '.join(tokens)
    
    def load_data(self, file_path, text_column='text', label_column='label'):
        """
        Carga datos desde archivo CSV
        """
        try:
            df = pd.read_csv(file_path)
            
            # Validar columnas
            if text_column not in df.columns or label_column not in df.columns:
                raise ValueError(f"Columnas requeridas no encontradas: {text_column}, {label_column}")
            
            # Limpiar texto
            self.logger.info("Limpiando texto...")
            df['clean_text'] = df[text_column].apply(self.clean_text)
            
            # Eliminar textos vacíos
            df = df[df['clean_text'].str.len() > 0]
            
            # Mapear etiquetas a números
            unique_labels = df[label_column].unique()
            label_map = {label: idx for idx, label in enumerate(unique_labels)}
            df['label_id'] = df[label_column].map(label_map)
            
            self.logger.info(f"Datos cargados: {len(df)} muestras")
            self.logger.info(f"Clases: {label_map}")
            
            return df, label_map
            
        except Exception as e:
            self.logger.error(f"Error cargando datos: {str(e)}")
            raise
    
    def create_datasets(self, df, text_column='clean_text', label_column='label_id', 
                       test_size=0.2, val_size=0.1, random_state=42):
        """
        Crea datasets de entrenamiento, validación y prueba
        """
        # Dividir en train+val y test
        train_val_df, test_df = train_test_split(
            df, 
            test_size=test_size, 
            random_state=random_state,
            stratify=df[label_column]
        )
        
        # Dividir train en train y val
        train_df, val_df = train_test_split(
            train_val_df,
            test_size=val_size/(1-test_size),
            random_state=random_state,
            stratify=train_val_df[label_column]
        )
        
        # Crear datasets
        train_dataset = TextDataset(
            train_df[text_column].values,
            train_df[label_column].values,
            self.tokenizer,
            self.max_length
        )
        
        val_dataset = TextDataset(
            val_df[text_column].values,
            val_df[label_column].values,
            self.tokenizer,
            self.max_length
        )
        
        test_dataset = TextDataset(
            test_df[text_column].values,
            test_df[label_column].values,
            self.tokenizer,
            self.max_length
        )
        
        self.logger.info(f"Datasets creados:")
        self.logger.info(f"Train: {len(train_dataset)} muestras")
        self.logger.info(f"Val: {len(val_dataset)} muestras")
        self.logger.info(f"Test: {len(test_dataset)} muestras")
        
        return train_dataset, val_dataset, test_dataset
    
    def create_dataloaders(self, train_dataset, val_dataset, test_dataset, 
                          batch_size=16, num_workers=4):
        """
        Crea DataLoaders para PyTorch
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
    
    def tokenize_text(self, text, return_tensors='pt'):
        """
        Tokeniza un texto individual
        """
        return self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors=return_tensors
        )
    
    def get_class_weights(self, labels):
        """
        Calcula pesos para clases desbalanceadas
        """
        from sklearn.utils.class_weight import compute_class_weight
        
        class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(labels),
            y=labels
        )
        
        return torch.tensor(class_weights, dtype=torch.float)
    
    def analyze_dataset(self, df, text_column='clean_text', label_column='label_id'):
        """
        Analiza estadísticas del dataset
        """
        print("\n📊 Análisis del Dataset:")
        print(f"Total de muestras: {len(df)}")
        
        # Distribución de clases
        print("\n📈 Distribución de clases:")
        class_dist = df[label_column].value_counts().sort_index()
        for label, count in class_dist.items():
            print(f"Clase {label}: {count} muestras ({count/len(df)*100:.1f}%)")
        
        # Estadísticas de texto
        text_lengths = df[text_column].str.len()
        print(f"\n📝 Estadísticas de texto:")
        print(f"Longitud promedio: {text_lengths.mean():.1f} caracteres")
        print(f"Longitud máxima: {text_lengths.max()} caracteres")
        print(f"Longitud mínima: {text_lengths.min()} caracteres")
        
        return {
            'total_samples': len(df),
            'class_distribution': class_dist.to_dict(),
            'text_stats': {
                'mean_length': text_lengths.mean(),
                'max_length': text_lengths.max(),
                'min_length': text_lengths.min()
            }
        }


if __name__ == "__main__":
    # Ejemplo de uso
    loader = TextDataLoader()
    
    # Cargar datos (ejemplo)
    try:
        df, label_map = loader.load_data("data/reviews.csv")
        
        # Analizar dataset
        stats = loader.analyze_dataset(df)
        
        # Crear datasets
        train_ds, val_ds, test_ds = loader.create_datasets(df)
        
        # Crear dataloaders
        train_loader, val_loader, test_loader = loader.create_dataloaders(
            train_ds, val_ds, test_ds, batch_size=16
        )
        
        # Probar batch
        batch = next(iter(train_loader))
        print(f"\nBatch shape:")
        print(f"Input IDs: {batch['input_ids'].shape}")
        print(f"Attention Mask: {batch['attention_mask'].shape}")
        print(f"Labels: {batch['labels'].shape}")
        
    except FileNotFoundError:
        print("Archivo de datos no encontrado. Este es un ejemplo de uso.")
    except Exception as e:
        print(f"Error: {e}")
