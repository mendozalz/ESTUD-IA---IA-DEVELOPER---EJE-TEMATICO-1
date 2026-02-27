"""
Modelo BERT con Adaptadores para Clasificación de Texto
Laboratorio 4.2 - Clasificación de Texto con BERT y Adaptadores
"""

import torch
import torch.nn as nn
from transformers import BertModel, BertConfig, AdapterConfig
from transformers.adapters import BERTAdapterModel
import torch.nn.functional as F
from typing import Optional, Dict, Any
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BERTWithAdapters(nn.Module):
    """
    Modelo BERT con adaptadores para clasificación eficiente
    """
    
    def __init__(self, 
                 model_name: str = 'bert-base-uncased',
                 num_classes: int = 2,
                 adapter_config: Optional[Dict] = None,
                 dropout_rate: float = 0.1):
        super().__init__()
        
        self.model_name = model_name
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        
        # Cargar BERT base
        self.bert = BertModel.from_pretrained(model_name)
        
        # Congelar parámetros de BERT
        for param in self.bert.parameters():
            param.requires_grad = False
        
        # Configurar adaptadores
        if adapter_config is None:
            adapter_config = {
                'mh_adapter': True,
                'output_adapter': True,
                'reduction_factor': 16,
                'non_linearity': 'relu'
            }
        
        # Añadir adaptadores (simulado con capas lineales)
        self.adapter_config = adapter_config
        self.adapters = nn.ModuleDict()
        
        # Añadir adaptador para cada capa transformer
        for i, layer in enumerate(self.bert.encoder.layer):
            adapter_name = f'adapter_{i}'
            hidden_size = layer.output.dense.out_features
            
            self.adapters[adapter_name] = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // adapter_config['reduction_factor']),
                nn.ReLU() if adapter_config['non_linearity'] == 'relu' else nn.GELU(),
                nn.Linear(hidden_size // adapter_config['reduction_factor'], hidden_size),
                nn.Dropout(dropout_rate)
            )
        
        # Capa de clasificación
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(self.bert.config.hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
        )
        
        logger.info(f"Modelo BERT con adaptadores inicializado: {model_name}")
        logger.info(f"Clases de salida: {num_classes}")
        logger.info(f"Adaptadores añadidos: {len(self.adapters)}")
    
    def forward(self, 
                input_ids: torch.Tensor,
                attention_mask: torch.Tensor,
                token_type_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass del modelo
        """
        # Pasar por BERT
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_hidden_states=True
        )
        
        # Obtener hidden states
        hidden_states = outputs.hidden_states
        
        # Aplicar adaptadores a cada capa
        adapted_states = []
        for i, (layer_output, adapter) in enumerate(zip(hidden_states[1:], self.adapters.values())):
            # Aplicar adaptador
            adapted_output = adapter(layer_output)
            # Residual connection
            adapted_output = layer_output + adapted_output
            adapted_states.append(adapted_output)
        
        # Usar la última capa adaptada
        last_hidden_state = adapted_states[-1]
        
        # Pooling (usar token [CLS])
        cls_output = last_hidden_state[:, 0, :]
        
        # Clasificación
        logits = self.classifier(cls_output)
        
        return logits
    
    def unfreeze_adapters(self):
        """
        Descongelar solo los adaptadores para fine-tuning
        """
        for adapter in self.adapters.parameters():
            adapter.requires_grad = True
        
        for param in self.classifier.parameters():
            param.requires_grad = True
        
        logger.info("Adaptadores y clasificador descongelados para fine-tuning")
    
    def unfreeze_top_layers(self, num_layers: int = 2):
        """
        Descongelar las últimas capas de BERT
        """
        total_layers = len(self.bert.encoder.layer)
        start_layer = total_layers - num_layers
        
        for i in range(start_layer, total_layers):
            for param in self.bert.encoder.layer[i].parameters():
                param.requires_grad = True
        
        logger.info(f"Descongeladas las últimas {num_layers} capas de BERT")
    
    def get_trainable_parameters(self):
        """
        Retorna el número de parámetros entrenables
        """
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.parameters())
        
        logger.info(f"Parámetros entrenables: {trainable_params:,}")
        logger.info(f"Parámetros totales: {total_params:,}")
        logger.info(f"Porcentaje entrenable: {trainable_params/total_params*100:.2f}%")
        
        return trainable_params, total_params
    
    def save_model(self, path: str):
        """
        Guardar el modelo
        """
        torch.save({
            'model_state_dict': self.state_dict(),
            'model_config': {
                'model_name': self.model_name,
                'num_classes': self.num_classes,
                'adapter_config': self.adapter_config,
                'dropout_rate': self.dropout_rate
            }
        }, path)
        logger.info(f"Modelo guardado en: {path}")
    
    @classmethod
    def load_model(cls, path: str):
        """
        Cargar el modelo
        """
        checkpoint = torch.load(path, map_location='cpu')
        config = checkpoint['model_config']
        
        model = cls(**config)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        logger.info(f"Modelo cargado desde: {path}")
        return model


class AdapterTrainer:
    """
    Clase para entrenar el modelo con adaptadores
    """
    
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.training_history = []
    
    def train_epoch(self, dataloader, optimizer, criterion, class_weights=None):
        """
        Entrena una época
        """
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch in dataloader:
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            optimizer.zero_grad()
            
            outputs = self.model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            
            if class_weights is not None:
                loss = loss * class_weights[labels]
                loss = loss.mean()
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        avg_loss = total_loss / len(dataloader)
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def validate(self, dataloader, criterion):
        """
        Valida el modelo
        """
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(input_ids, attention_mask)
                loss = criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        avg_loss = total_loss / len(dataloader)
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def train(self, train_loader, val_loader, epochs=10, lr=2e-5, class_weights=None):
        """
        Entrena el modelo completo
        """
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        
        best_val_accuracy = 0
        
        for epoch in range(epochs):
            # Entrenamiento
            train_loss, train_acc = self.train_epoch(train_loader, optimizer, criterion, class_weights)
            
            # Validación
            val_loss, val_acc = self.validate(val_loader, criterion)
            
            # Guardar historial
            self.training_history.append({
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'train_accuracy': train_acc,
                'val_loss': val_loss,
                'val_accuracy': val_acc
            })
            
            # Guardar mejor modelo
            if val_acc > best_val_accuracy:
                best_val_accuracy = val_acc
                self.model.save_model('best_bert_adapters.pth')
            
            logger.info(f"Epoch {epoch + 1}/{epochs}")
            logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            logger.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            logger.info("-" * 50)
        
        return self.training_history


if __name__ == "__main__":
    # Ejemplo de uso
    model = BERTWithAdapters(num_classes=2)
    model.get_trainable_parameters()
    
    # Descongelar adaptadores
    model.unfreeze_adapters()
    model.get_trainable_parameters()
    
    # Probar forward pass
    batch_size = 2
    seq_length = 128
    
    input_ids = torch.randint(0, 30000, (batch_size, seq_length))
    attention_mask = torch.ones(batch_size, seq_length)
    
    try:
        outputs = model(input_ids, attention_mask)
        print(f"Output shape: {outputs.shape}")
        print("✅ Forward pass exitoso")
    except Exception as e:
        print(f"❌ Error en forward pass: {e}")
