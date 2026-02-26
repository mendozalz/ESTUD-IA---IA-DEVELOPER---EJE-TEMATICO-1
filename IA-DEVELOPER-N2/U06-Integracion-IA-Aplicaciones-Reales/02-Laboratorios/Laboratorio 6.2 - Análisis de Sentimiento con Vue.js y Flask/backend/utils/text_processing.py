import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class TextProcessor:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
        # Descargar recursos NLTK si no están disponibles
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')
        
        try:
            nltk.data.find('corpora/wordnet')
        except LookupError:
            nltk.download('wordnet')
        
        # Patrones de limpieza
        self.url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        self.email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
        self.mention_pattern = re.compile(r'@[A-Za-z0-9_]+')
        self.hashtag_pattern = re.compile(r'#[A-Za-z0-9_]+')
        self.number_pattern = re.compile(r'\b\d+\.?\d*\b')
        self.punctuation_pattern = re.compile(r'[^\w\s]')
        self.whitespace_pattern = re.compile(r'\s+')
        
    def preprocess(self, text: str) -> str:
        """Preprocesar texto completo"""
        try:
            if not text or not isinstance(text, str):
                return ""
            
            # 1. Convertir a minúsculas
            text = text.lower()
            
            # 2. Eliminar URLs
            text = self.url_pattern.sub(' ', text)
            
            # 3. Eliminar emails
            text = self.email_pattern.sub(' ', text)
            
            # 4. Eliminar menciones y hashtags (mantener el texto)
            text = self.mention_pattern.sub(' ', text)
            text = self.hashtag_pattern.sub(' ', text)
            
            # 5. Eliminar números
            text = self.number_pattern.sub(' ', text)
            
            # 6. Eliminar puntuación
            text = self.punctuation_pattern.sub(' ', text)
            
            # 7. Normalizar whitespace
            text = self.whitespace_pattern.sub(' ', text)
            
            # 8. Tokenizar
            tokens = word_tokenize(text.strip())
            
            # 9. Eliminar stopwords y lematizar
            processed_tokens = []
            for token in tokens:
                if token not in self.stop_words and len(token) > 1:
                    lemmatized = self.lemmatizer.lemmatize(token)
                    processed_tokens.append(lemmatized)
            
            # 10. Unir tokens
            processed_text = ' '.join(processed_tokens)
            
            return processed_text.strip()
            
        except Exception as e:
            logger.error(f"Error preprocessing text: {e}")
            return text
    
    def extract_features(self, text: str) -> Dict[str, Any]:
        """Extraer características del texto"""
        try:
            features = {
                "original_length": len(text),
                "word_count": len(text.split()),
                "char_count": len(text),
                "avg_word_length": 0,
                "has_urls": bool(self.url_pattern.search(text)),
                "has_emails": bool(self.email_pattern.search(text)),
                "has_mentions": bool(self.mention_pattern.search(text)),
                "has_hashtags": bool(self.hashtag_pattern.search(text)),
                "has_numbers": bool(self.number_pattern.search(text)),
                "punctuation_count": len(self.punctuation_pattern.findall(text)),
                "uppercase_count": sum(1 for c in text if c.isupper()),
                "lowercase_count": sum(1 for c in text if c.islower()),
                "digit_count": sum(1 for c in text if c.isdigit()),
                "sentiment_indicators": self._extract_sentiment_indicators(text)
            }
            
            # Calcular longitud promedio de palabra
            words = text.split()
            if words:
                features["avg_word_length"] = sum(len(word) for word in words) / len(words)
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting features: {e}")
            return {}
    
    def _extract_sentiment_indicators(self, text: str) -> Dict[str, int]:
        """Extraer indicadores de sentimiento"""
        positive_words = [
            'good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic',
            'love', 'like', 'awesome', 'perfect', 'best', 'happy', 'joy',
            'beautiful', 'nice', 'brilliant', 'outstanding', 'superb'
        ]
        
        negative_words = [
            'bad', 'terrible', 'awful', 'horrible', 'hate', 'dislike',
            'worst', 'poor', 'sad', 'angry', 'frustrated', 'disappointed',
            'ugly', 'disgusting', 'useless', 'broken', 'failed', 'wrong'
        ]
        
        intensifiers = [
            'very', 'extremely', 'really', 'absolutely', 'completely',
            'totally', 'utterly', 'highly', 'deeply', 'truly'
        ]
        
        negations = [
            'not', 'no', 'never', 'none', 'nothing', 'nowhere',
            'neither', 'nor', 'cannot', "can't", "won't", "don't"
        ]
        
        text_lower = text.lower()
        words = text_lower.split()
        
        indicators = {
            "positive_count": 0,
            "negative_count": 0,
            "intensifier_count": 0,
            "negation_count": 0,
            "exclamation_count": text.count('!'),
            "question_count": text.count('?')
        }
        
        for word in words:
            if word in positive_words:
                indicators["positive_count"] += 1
            elif word in negative_words:
                indicators["negative_count"] += 1
            elif word in intensifiers:
                indicators["intensifier_count"] += 1
            elif word in negations:
                indicators["negation_count"] += 1
        
        return indicators
    
    def clean_for_display(self, text: str) -> str:
        """Limpiar texto para visualización"""
        try:
            # Eliminar caracteres especiales pero mantener estructura básica
            text = re.sub(r'[^\w\s\.\!\?\,\;\:\-]', ' ', text)
            text = re.sub(r'\s+', ' ', text)
            return text.strip()
            
        except Exception as e:
            logger.error(f"Error cleaning text for display: {e}")
            return text
    
    def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """Extraer entidades básicas del texto"""
        try:
            entities = []
            
            # Extraer emails
            emails = self.email_pattern.findall(text)
            for email in emails:
                entities.append({
                    "text": email,
                    "type": "email",
                    "start": text.find(email),
                    "end": text.find(email) + len(email)
                })
            
            # Extraer URLs
            urls = self.url_pattern.findall(text)
            for url in urls:
                entities.append({
                    "text": url,
                    "type": "url",
                    "start": text.find(url),
                    "end": text.find(url) + len(url)
                })
            
            # Extraer menciones
            mentions = self.mention_pattern.findall(text)
            for mention in mentions:
                entities.append({
                    "text": mention,
                    "type": "mention",
                    "start": text.find(mention),
                    "end": text.find(mention) + len(mention)
                })
            
            # Extraer hashtags
            hashtags = self.hashtag_pattern.findall(text)
            for hashtag in hashtags:
                entities.append({
                    "text": hashtag,
                    "type": "hashtag",
                    "start": text.find(hashtag),
                    "end": text.find(hashtag) + len(hashtag)
                })
            
            return entities
            
        except Exception as e:
            logger.error(f"Error extracting entities: {e}")
            return []
    
    def validate_text(self, text: str) -> Dict[str, Any]:
        """Validar texto para análisis"""
        try:
            validation = {
                "is_valid": True,
                "errors": [],
                "warnings": []
            }
            
            if not text or not isinstance(text, str):
                validation["is_valid"] = False
                validation["errors"].append("Text must be a non-empty string")
                return validation
            
            if len(text.strip()) == 0:
                validation["is_valid"] = False
                validation["errors"].append("Text cannot be empty")
                return validation
            
            if len(text) > 1000:
                validation["warnings"].append("Text is very long, may affect processing time")
            
            # Verificar si tiene contenido significativo
            processed = self.preprocess(text)
            if len(processed.split()) < 2:
                validation["warnings"].append("Text has very few meaningful words")
            
            # Verificar si es solo caracteres especiales
            if len(re.sub(r'[^\w\s]', '', text)) < len(text) * 0.3:
                validation["warnings"].append("Text contains many special characters")
            
            return validation
            
        except Exception as e:
            logger.error(f"Error validating text: {e}")
            return {"is_valid": False, "errors": [str(e)]}
    
    def batch_preprocess(self, texts: List[str]) -> List[str]:
        """Preprocesar un lote de textos"""
        try:
            processed_texts = []
            for text in texts:
                processed = self.preprocess(text)
                processed_texts.append(processed)
            return processed_texts
            
        except Exception as e:
            logger.error(f"Error in batch preprocessing: {e}")
            return texts
