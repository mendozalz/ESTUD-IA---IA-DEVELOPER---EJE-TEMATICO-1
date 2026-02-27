import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import cv2
import logging
from typing import Tuple, Optional, Union
import io

logger = logging.getLogger(__name__)

class ImagePreprocessor:
    def __init__(self, target_size: Tuple[int, int] = (224, 224)):
        self.target_size = target_size
        self.mean = [0.485, 0.456, 0.406]  # ImageNet normalization
        self.std = [0.229, 0.224, 0.225]
    
    def preprocess(self, image: Union[np.ndarray, Image.Image]) -> np.ndarray:
        """Preprocesar una imagen para el modelo"""
        try:
            # Convertir a PIL Image si es numpy array
            if isinstance(image, np.ndarray):
                if len(image.shape) == 3 and image.shape[2] == 3:
                    image = Image.fromarray(image.astype(np.uint8))
                else:
                    image = Image.fromarray(image.astype(np.uint8), mode='L')
                    image = image.convert('RGB')
            
            # Aplicar preprocesamiento
            processed_image = self._apply_preprocessing_pipeline(image)
            
            return processed_image
            
        except Exception as e:
            logger.error(f"Error preprocessing image: {e}")
            raise
    
    def _apply_preprocessing_pipeline(self, image: Image.Image) -> np.ndarray:
        """Aplicar pipeline completo de preprocesamiento"""
        
        # 1. Redimensionar
        image = self._resize_image(image)
        
        # 2. Mejorar calidad
        image = self._enhance_image(image)
        
        # 3. Normalizar
        image_array = self._normalize_image(image)
        
        return image_array
    
    def _resize_image(self, image: Image.Image) -> Image.Image:
        """Redimensionar imagen manteniendo aspect ratio"""
        try:
            # Obtener dimensiones originales
            width, height = image.size
            target_width, target_height = self.target_size
            
            # Calcular ratio
            ratio = min(target_width / width, target_height / height)
            new_width = int(width * ratio)
            new_height = int(height * ratio)
            
            # Redimensionar
            image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # Crear imagen cuadrada con padding
            new_image = Image.new('RGB', self.target_size, (0, 0, 0))
            
            # Calcular posición para centrar
            x_offset = (target_width - new_width) // 2
            y_offset = (target_height - new_height) // 2
            
            # Pegar imagen centrada
            new_image.paste(image, (x_offset, y_offset))
            
            return new_image
            
        except Exception as e:
            logger.error(f"Error resizing image: {e}")
            raise
    
    def _enhance_image(self, image: Image.Image) -> Image.Image:
        """Mejorar calidad de la imagen"""
        try:
            # Mejorar contraste
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(1.1)
            
            # Mejorar nitidez
            enhancer = ImageEnhance.Sharpness(image)
            image = enhancer.enhance(1.1)
            
            # Reducir ruido
            image = image.filter(ImageFilter.UnsharpMask(radius=1, percent=120, threshold=3))
            
            return image
            
        except Exception as e:
            logger.error(f"Error enhancing image: {e}")
            return image
    
    def _normalize_image(self, image: Image.Image) -> np.ndarray:
        """Normalizar valores de píxeles"""
        try:
            # Convertir a numpy array
            image_array = np.array(image, dtype=np.float32)
            
            # Normalizar a [0, 1]
            image_array = image_array / 255.0
            
            # Aplicar normalización ImageNet
            for i in range(3):
                image_array[:, :, i] = (image_array[:, :, i] - self.mean[i]) / self.std[i]
            
            return image_array
            
        except Exception as e:
            logger.error(f"Error normalizing image: {e}")
            raise
    
    def preprocess_batch(self, images: list) -> np.ndarray:
        """Preprocesar un lote de imágenes"""
        try:
            processed_images = []
            
            for image in images:
                processed_image = self.preprocess(image)
                processed_images.append(processed_image)
            
            return np.array(processed_images)
            
        except Exception as e:
            logger.error(f"Error preprocessing batch: {e}")
            raise
    
    def augment_image(self, image: Image.Image) -> Image.Image:
        """Aplicar aumentación de datos"""
        try:
            # Rotación aleatoria
            if np.random.random() > 0.5:
                angle = np.random.uniform(-15, 15)
                image = image.rotate(angle, expand=False, fillcolor=(0, 0, 0))
            
            # Flip horizontal
            if np.random.random() > 0.5:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
            
            # Ajuste de brillo
            if np.random.random() > 0.5:
                enhancer = ImageEnhance.Brightness(image)
                image = enhancer.enhance(np.random.uniform(0.8, 1.2))
            
            # Ajuste de contraste
            if np.random.random() > 0.5:
                enhancer = ImageEnhance.Contrast(image)
                image = enhancer.enhance(np.random.uniform(0.8, 1.2))
            
            return image
            
        except Exception as e:
            logger.error(f"Error augmenting image: {e}")
            return image
    
    def extract_features(self, image: Image.Image) -> dict:
        """Extraer características básicas de la imagen"""
        try:
            image_array = np.array(image)
            
            features = {
                "width": image.size[0],
                "height": image.size[1],
                "channels": len(image.getbands()),
                "mode": image.mode,
                "has_transparency": image.mode in ('RGBA', 'LA') or 'transparency' in image.info
            }
            
            # Calcular estadísticas de color
            if len(image_array.shape) == 3:
                features["color_stats"] = {
                    "mean_rgb": np.mean(image_array, axis=(0, 1)).tolist(),
                    "std_rgb": np.std(image_array, axis=(0, 1)).tolist(),
                    "brightness": np.mean(image_array)
                }
            
            # Calcular histograma
            if len(image_array.shape) == 3:
                hist_r = np.histogram(image_array[:, :, 0], bins=256, range=(0, 256))[0]
                hist_g = np.histogram(image_array[:, :, 1], bins=256, range=(0, 256))[0]
                hist_b = np.histogram(image_array[:, :, 2], bins=256, range=(0, 256))[0]
                
                features["histogram"] = {
                    "red": hist_r.tolist()[:50],  # Limitar para eficiencia
                    "green": hist_g.tolist()[:50],
                    "blue": hist_b.tolist()[:50]
                }
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting features: {e}")
            return {}
    
    def validate_image(self, image: Union[np.ndarray, Image.Image]) -> dict:
        """Validar que la imagen sea adecuada para el modelo"""
        try:
            validation_result = {
                "is_valid": True,
                "errors": [],
                "warnings": []
            }
            
            # Convertir a PIL si es necesario
            if isinstance(image, np.ndarray):
                image = Image.fromarray(image.astype(np.uint8))
            
            # Verificar dimensiones mínimas
            if image.size[0] < 32 or image.size[1] < 32:
                validation_result["is_valid"] = False
                validation_result["errors"].append("Image too small (minimum 32x32)")
            
            # Verificar formato
            if image.mode not in ['RGB', 'RGBA']:
                validation_result["warnings"].append("Image mode not RGB, will be converted")
            
            # Verificar si está demasiado oscuro
            image_array = np.array(image)
            if np.mean(image_array) < 30:
                validation_result["warnings"].append("Image appears very dark")
            
            # Verificar si está saturada
            if np.mean(image_array) > 225:
                validation_result["warnings"].append("Image appears saturated")
            
            return validation_result
            
        except Exception as e:
            logger.error(f"Error validating image: {e}")
            return {"is_valid": False, "errors": [str(e)]}
