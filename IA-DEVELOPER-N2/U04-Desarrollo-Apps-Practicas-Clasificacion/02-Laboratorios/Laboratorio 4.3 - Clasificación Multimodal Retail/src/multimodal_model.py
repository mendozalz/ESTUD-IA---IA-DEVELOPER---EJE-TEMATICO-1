"""
Modelo Multimodal (CNN + Transformer) para Retail
Laboratorio 4.3 - Clasificación Multimodal Retail
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import DistilBertModel, DistilBertConfig
from torchvision import models
import logging
from typing import Tuple, Optional
import numpy as np

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MultimodalFusionLayer(nn.Module):
    """
    Capa de fusión multimodal con atención cruzada
    """
    
    def __init__(self, image_dim: int, text_dim: int, fusion_dim: int, num_heads: int = 8):
        super().__init__()
        self.image_dim = image_dim
        self.text_dim = text_dim
        self.fusion_dim = fusion_dim
        self.num_heads = num_heads
        
        # Proyecciones a dimensión común
        self.image_proj = nn.Linear(image_dim, fusion_dim)
        self.text_proj = nn.Linear(text_dim, fusion_dim)
        
        # Multi-head attention
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=fusion_dim,
            num_heads=num_heads,
            batch_first=True
        )
        
        # Capas de normalización
        self.norm1 = nn.LayerNorm(fusion_dim)
        self.norm2 = nn.LayerNorm(fusion_dim)
        
        # Feed-forward
        self.ffn = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim * 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(fusion_dim * 4, fusion_dim)
        )
        
        # Gate para control de fusión
        self.gate = nn.Sequential(
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.Sigmoid()
        )
    
    def forward(self, image_features: torch.Tensor, text_features: torch.Tensor) -> torch.Tensor:
        """
        Fusiona características de imagen y texto
        """
        # Proyectar a dimensión común
        image_proj = self.image_proj(image_features)  # [batch, fusion_dim]
        text_proj = self.text_proj(text_features)     # [batch, fusion_dim]
        
        # Añadir dimensión de secuencia para attention
        image_seq = image_proj.unsqueeze(1)  # [batch, 1, fusion_dim]
        text_seq = text_proj.unsqueeze(1)   # [batch, 1, fusion_dim]
        
        # Concatenar secuencias
        combined_seq = torch.cat([image_seq, text_seq], dim=1)  # [batch, 2, fusion_dim]
        
        # Cross-attention
        attended, _ = self.cross_attention(combined_seq, combined_seq, combined_seq)
        attended = self.norm1(combined_seq + attended)
        
        # Feed-forward
        ffn_out = self.ffn(attended)
        fused = self.norm2(attended + ffn_out)
        
        # Pooling sobre la dimensión de secuencia
        fused = fused.mean(dim=1)  # [batch, fusion_dim]
        
        # Gate para controlar la fusión
        gate_input = torch.cat([image_proj, text_proj], dim=-1)
        gate_weights = self.gate(gate_input)
        
        # Aplicar gate
        final_features = fused * gate_weights
        
        return final_features


class MultimodalRetailClassifier(nn.Module):
    """
    Clasificador multimodal para productos de retail
    """
    
    def __init__(self, 
                 num_classes: int = 10,
                 image_model: str = 'efficientnet_b3',
                 text_model: str = 'distilbert-base-uncased',
                 fusion_dim: int = 512,
                 dropout_rate: float = 0.3):
        super().__init__()
        
        self.num_classes = num_classes
        self.fusion_dim = fusion_dim
        self.dropout_rate = dropout_rate
        
        # Inicializar modelo de imágenes
        self.image_encoder = self._build_image_encoder(image_model)
        
        # Inicializar modelo de texto
        self.text_encoder = self._build_text_encoder(text_model)
        
        # Capa de fusión
        self.fusion_layer = MultimodalFusionLayer(
            image_dim=self.image_encoder_output_dim,
            text_dim=self.text_encoder_output_dim,
            fusion_dim=fusion_dim
        )
        
        # Clasificador final
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(fusion_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes)
        )
        
        # Inicializar pesos
        self._initialize_weights()
        
        logger.info(f"Modelo multimodal inicializado: {num_classes} clases")
        logger.info(f"Encoder imagen: {image_model}")
        logger.info(f"Encoder texto: {text_model}")
    
    def _build_image_encoder(self, model_name: str) -> nn.Module:
        """
        Construye el encoder de imágenes
        """
        if model_name == 'efficientnet_b3':
            backbone = models.efficientnet_b3(pretrained=True)
            # Remover capa de clasificación
            backbone.classifier = nn.Identity()
            self.image_encoder_output_dim = 1536  # EfficientNet-B3 output dim
        elif model_name == 'resnet50':
            backbone = models.resnet50(pretrained=True)
            backbone.fc = nn.Identity()
            self.image_encoder_output_dim = 2048
        else:
            raise ValueError(f"Modelo de imagen no soportado: {model_name}")
        
        return backbone
    
    def _build_text_encoder(self, model_name: str) -> nn.Module:
        """
        Construye el encoder de texto
        """
        if model_name == 'distilbert-base-uncased':
            backbone = DistilBertModel.from_pretrained(model_name)
            self.text_encoder_output_dim = 768  # DistilBERT hidden size
        else:
            raise ValueError(f"Modelo de texto no soportado: {model_name}")
        
        return backbone
    
    def _initialize_weights(self):
        """
        Inicializa pesos de las capas nuevas
        """
        for module in [self.fusion_layer, self.classifier]:
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.LayerNorm):
                    nn.init.constant_(m.bias, 0)
                    nn.init.constant_(m.weight, 1.0)
    
    def forward(self, 
                images: torch.Tensor, 
                input_ids: torch.Tensor, 
                attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass del modelo multimodal
        """
        # Extraer características de imagen
        image_features = self.image_encoder(images)  # [batch, image_dim]
        
        # Extraer características de texto
        text_outputs = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Usar token [CLS] para clasificación
        text_features = text_outputs.last_hidden_state[:, 0, :]  # [batch, text_dim]
        
        # Fusionar características
        fused_features = self.fusion_layer(image_features, text_features)
        
        # Clasificación final
        logits = self.classifier(fused_features)
        
        return logits
    
    def freeze_encoders(self):
        """
        Congela los encoders para fine-tuning
        """
        for param in self.image_encoder.parameters():
            param.requires_grad = False
        
        for param in self.text_encoder.parameters():
            param.requires_grad = False
        
        logger.info("Encoders congelados")
    
    def unfreeze_encoders(self):
        """
        Descongela los encoders
        """
        for param in self.image_encoder.parameters():
            param.requires_grad = True
        
        for param in self.text_encoder.parameters():
            param.requires_grad = True
        
        logger.info("Encoders descongelados")
    
    def unfreeze_top_layers(self, num_image_layers: int = 2, num_text_layers: int = 2):
        """
        Descongola las últimas capas de los encoders
        """
        # Descongelar últimas capas del encoder de imágenes
        if hasattr(self.image_encoder, 'classifier'):
            for param in self.image_encoder.classifier.parameters():
                param.requires_grad = True
        
        # Descongelar últimas capas del encoder de texto
        text_layers = self.text_encoder.transformer.layer
        for i in range(len(text_layers) - num_text_layers, len(text_layers)):
            for param in text_layers[i].parameters():
                param.requires_grad = True
        
        logger.info(f"Descongeladas últimas {num_image_layers} capas de imagen y {num_text_layers} de texto")
    
    def get_trainable_parameters(self) -> Tuple[int, int]:
        """
        Retorna el número de parámetros entrenables y totales
        """
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.parameters())
        
        logger.info(f"Parámetros entrenables: {trainable_params:,}")
        logger.info(f"Parámetros totales: {total_params:,}")
        logger.info(f"Porcentaje entrenable: {trainable_params/total_params*100:.2f}%")
        
        return trainable_params, total_params
    
    def save_model(self, path: str):
        """
        Guarda el modelo
        """
        torch.save({
            'model_state_dict': self.state_dict(),
            'model_config': {
                'num_classes': self.num_classes,
                'fusion_dim': self.fusion_dim,
                'dropout_rate': self.dropout_rate
            }
        }, path)
        logger.info(f"Modelo guardado en: {path}")
    
    @classmethod
    def load_model(cls, path: str):
        """
        Carga el modelo
        """
        checkpoint = torch.load(path, map_location='cpu')
        config = checkpoint['model_config']
        
        model = cls(**config)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        logger.info(f"Modelo cargado desde: {path}")
        return model


class MultimodalTrainer:
    """
    Clase para entrenar el modelo multimodal
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
            images = batch['image'].to(self.device)
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['label'].to(self.device)
            
            optimizer.zero_grad()
            
            outputs = self.model(images, input_ids, attention_mask)
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
                images = batch['image'].to(self.device)
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                
                outputs = self.model(images, input_ids, attention_mask)
                loss = criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        avg_loss = total_loss / len(dataloader)
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def train(self, train_loader, val_loader, epochs=20, lr=1e-4, class_weights=None):
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
                self.model.save_model('best_multimodal_model.pth')
            
            logger.info(f"Epoch {epoch + 1}/{epochs}")
            logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            logger.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            logger.info("-" * 50)
        
        return self.training_history


def main():
    """
    Ejemplo de uso
    """
    # Crear modelo
    model = MultimodalRetailClassifier(num_classes=10)
    model.get_trainable_parameters()
    
    # Congelar encoders inicialmente
    model.freeze_encoders()
    model.get_trainable_parameters()
    
    # Probar forward pass
    batch_size = 2
    
    # Datos de ejemplo
    images = torch.randn(batch_size, 3, 300, 300)
    input_ids = torch.randint(0, 30000, (batch_size, 128))
    attention_mask = torch.ones(batch_size, 128)
    
    try:
        outputs = model(images, input_ids, attention_mask)
        print(f"✅ Forward pass exitoso")
        print(f"Output shape: {outputs.shape}")
        
        # Probar predicción
        probabilities = F.softmax(outputs, dim=-1)
        predictions = torch.argmax(probabilities, dim=-1)
        print(f"Predicciones: {predictions}")
        print(f"Probabilidades: {probabilities}")
        
    except Exception as e:
        print(f"❌ Error en forward pass: {e}")


if __name__ == "__main__":
    main()
