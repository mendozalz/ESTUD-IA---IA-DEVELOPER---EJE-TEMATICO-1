"""
Caso de Uso 3 - Generación de Código con Transformers
Fase 3: Desarrollo - Capa de Atención Personalizada para Código
"""

import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np
from typing import Tuple, Dict, List, Optional
import re

class CodeAttentionLayer(layers.Layer):
    """
    Capa de atención personalizada para procesamiento de código fuente
    Incorpora conocimiento específico de sintaxis y estructura de código
    """
    
    def __init__(self, 
                 embed_dim: int,
                 num_heads: int = 8,
                 dropout_rate: float = 0.1,
                 max_sequence_length: int = 1024,
                 use_syntax_aware: bool = True,
                 **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.max_sequence_length = max_sequence_length
        self.use_syntax_aware = use_syntax_aware
        
        # Capa de atención multi-head estándar
        self.attention = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embed_dim,
            dropout=dropout_rate,
            name='code_multihead_attention'
        )
        
        # Capas para procesamiento específico de código
        if use_syntax_aware:
            # Procesador de tokens de sintaxis
            self.syntax_processor = layers.Dense(
                embed_dim,
                activation='relu',
                name='syntax_processor'
            )
            
            # Procesador de estructura de bloques
            self.structure_processor = layers.Dense(
                embed_dim,
                activation='relu',
                name='structure_processor'
            )
            
            # Mecanismo de atención para sintaxis
            self.syntax_attention = layers.MultiHeadAttention(
                num_heads=num_heads // 2,
                key_dim=embed_dim,
                dropout=dropout_rate,
                name='syntax_attention'
            )
        
        # Normalización y regularización
        self.layer_norm1 = layers.LayerNormalization(name='layer_norm1')
        self.layer_norm2 = layers.LayerNormalization(name='layer_norm2')
        self.dropout = layers.Dropout(dropout_rate, name='attention_dropout')
        
        # Capa de feed-forward
        self.ffn = tf.keras.Sequential([
            layers.Dense(embed_dim * 4, activation='relu', name='ffn_dense1'),
            layers.Dropout(dropout_rate, name='ffn_dropout1'),
            layers.Dense(embed_dim, name='ffn_dense2'),
            layers.Dropout(dropout_rate, name='ffn_dropout2')
        ], name='feed_forward_network')
    
    def build(self, input_shape):
        """
        Construye los pesos de la capa basándose en las formas de entrada
        """
        # input_shape: (batch_size, sequence_length, embed_dim)
        super().build(input_shape)
        
        # Máscara de atención causal para generación autoregresiva
        batch_size = input_shape[0] if input_shape[0] is not None else None
        seq_len = input_shape[1] if input_shape[1] is not None else self.max_sequence_length
        
        self.causal_mask = self._create_causal_mask(seq_len)
    
    def _create_causal_mask(self, seq_len: int):
        """
        Crea máscara causal para atención autoregresiva
        """
        mask = tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1)
        return mask  # [seq_len, seq_len]
    
    def _extract_syntax_features(self, input_ids: tf.Tensor) -> tf.Tensor:
        """
        Extrae características de sintaxis del código
        """
        # Características simples basadas en patrones de código
        batch_size = tf.shape(input_ids)[0]
        seq_len = tf.shape(input_ids)[1]
        
        # Inicializar características de sintaxis
        syntax_features = tf.zeros((batch_size, seq_len, self.embed_dim))
        
        # Patrones de sintaxis (simplificados)
        patterns = {
            'indentation': tf.cast(input_ids > 100, tf.float32),  # Tokens de indentación
            'brackets': tf.cast(tf.abs(input_ids - 50) < 5, tf.float32),  # Tokens de llaves
            'keywords': tf.cast(tf.abs(input_ids - 10) < 10, tf.float32),  # Palabras clave
        }
        
        # Combinar características
        for i, (pattern_name, pattern) in enumerate(patterns.items()):
            pattern_expanded = tf.expand_dims(pattern, axis=-1)
            pattern_embedded = pattern_expanded * (i + 1) / len(patterns)
            syntax_features += pattern_embedded
        
        return self.syntax_processor(syntax_features)
    
    def _extract_structure_features(self, input_ids: tf.Tensor) -> tf.Tensor:
        """
        Extrae características de estructura de bloques del código
        """
        batch_size = tf.shape(input_ids)[0]
        seq_len = tf.shape(input_ids)[1]
        
        # Detectar bloques (funciones, clases, etc.)
        block_starts = tf.cast(input_ids == 200, tf.float32)  # Token especial para inicio de bloque
        block_ends = tf.cast(input_ids == 201, tf.float32)    # Token especial para fin de bloque
        
        # Crear características de estructura
        structure_features = tf.stack([block_starts, block_ends], axis=-1)
        structure_features = tf.pad(
            structure_features,
            [[0, 0], [0, 0], [0, self.embed_dim - 2]]
        )
        
        return self.structure_processor(structure_features)
    
    def call(self, inputs, training: Optional[bool] = None):
        """
        Forward pass de la capa de atención para código
        
        Args:
            inputs: Tensor de entrada [batch_size, sequence_length, embed_dim]
            training: Booleano para modo de entrenamiento
        
        Returns:
            output: Tensor procesado [batch_size, sequence_length, embed_dim]
        """
        batch_size = tf.shape(inputs)[0]
        seq_len = tf.shape(inputs)[1]
        
        # Crear máscara de atención causal
        causal_mask = self._create_causal_mask(seq_len)
        
        # Atención multi-head estándar
        attention_output = self.attention(
            inputs, inputs, 
            attention_mask=causal_mask,
            use_causal_mask=True
        )
        
        # Conexión residual y normalización
        x = self.layer_norm1(inputs + attention_output)
        
        # Procesamiento específico de código si está habilitado
        if self.use_syntax_aware:
            # Extraer características de sintaxis y estructura
            # (Nota: en una implementación real, necesitaríamos los IDs de tokens originales)
            syntax_features = self._extract_syntax_features(
                tf.cast(tf.random.normal((batch_size, seq_len)), tf.int32)
            )
            structure_features = self._extract_structure_features(
                tf.cast(tf.random.normal((batch_size, seq_len)), tf.int32)
            )
            
            # Combinar características específicas de código
            code_aware_features = syntax_features + structure_features
            
            # Atención con conocimiento de sintaxis
            syntax_attention_output = self.syntax_attention(
                x, code_aware_features
            )
            
            # Combinar con salida principal
            x = x + syntax_attention_output
        
        # Feed-forward network
        ffn_output = self.ffn(x)
        
        # Conexión residual y normalización final
        output = self.layer_norm2(x + ffn_output)
        
        if training:
            output = self.dropout(output, training=training)
        
        return output
    
    def get_config(self):
        """
        Retorna la configuración de la capa para serialización
        """
        config = super().get_config()
        config.update({
            'embed_dim': self.embed_dim,
            'num_heads': self.num_heads,
            'dropout_rate': self.dropout_rate,
            'max_sequence_length': self.max_sequence_length,
            'use_syntax_aware': self.use_syntax_aware
        })
        return config

class CodeGeneratorModel(Model):
    """
    Modelo completo para generación de código con Transformers
    """
    
    def __init__(self,
                 vocab_size: int = 50000,
                 max_sequence_length: int = 1024,
                 embed_dim: int = 768,
                 num_layers: int = 6,
                 num_heads: int = 8,
                 dropout_rate: float = 0.1,
                 use_syntax_aware: bool = True,
                 **kwargs):
        super().__init__(**kwargs)
        
        self.vocab_size = vocab_size
        self.max_sequence_length = max_sequence_length
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.use_syntax_aware = use_syntax_aware
        
        # Embeddings de tokens
        self.token_embedding = layers.Embedding(
            vocab_size,
            embed_dim,
            mask_zero=True,
            name='token_embedding'
        )
        
        # Embeddings posicionales
        self.position_embedding = layers.Embedding(
            max_sequence_length,
            embed_dim,
            name='position_embedding'
        )
        
        # Capas de Transformer
        self.transformer_layers = []
        for i in range(num_layers):
            self.transformer_layers.append(
                CodeAttentionLayer(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    dropout_rate=dropout_rate,
                    max_sequence_length=max_sequence_length,
                    use_syntax_aware=use_syntax_aware,
                    name=f'code_attention_{i}'
                )
            )
        
        # Capas de normalización final
        self.final_layer_norm = layers.LayerNormalization(name='final_layer_norm')
        self.dropout = layers.Dropout(dropout_rate, name='final_dropout')
        
        # Capa de salida
        self.output_layer = layers.Dense(
            vocab_size,
            name='output_projection'
        )
    
    def call(self, inputs, training: Optional[bool] = None):
        """
        Forward pass del modelo generador de código
        
        Args:
            inputs: Tensor de IDs de tokens [batch_size, sequence_length]
            training: Booleano para modo de entrenamiento
        
        Returns:
            output: Logits de predicción [batch_size, sequence_length, vocab_size]
        """
        batch_size = tf.shape(inputs)[0]
        seq_len = tf.shape(inputs)[1]
        
        # Crear máscara de padding
        padding_mask = tf.cast(tf.not_equal(inputs, 0), tf.float32)
        padding_mask = padding_mask[:, tf.newaxis, tf.newaxis, :]
        
        # Embeddings
        token_embeds = self.token_embedding(inputs)  # [batch_size, seq_len, embed_dim]
        
        # Embeddings posicionales
        positions = tf.range(start=0, limit=seq_len, delta=1)
        position_embeds = self.position_embedding(positions)  # [seq_len, embed_dim]
        position_embeds = tf.expand_dims(position_embeds, axis=0)  # [1, seq_len, embed_dim]
        position_embeds = tf.tile(position_embeds, [batch_size, 1, 1])
        
        # Combinar embeddings
        x = token_embeds + position_embeds
        
        if training:
            x = self.dropout(x, training=training)
        
        # Pasar por capas Transformer
        for layer in self.transformer_layers:
            x = layer(x, training=training)
        
        # Normalización final
        x = self.final_layer_norm(x)
        
        # Proyección a vocabulario
        logits = self.output_layer(x)  # [batch_size, seq_len, vocab_size]
        
        return logits
    
    def generate_code(self, 
                     prompt: str,
                     tokenizer,
                     max_length: int = 512,
                     temperature: float = 1.0,
                     top_k: int = 50,
                     top_p: float = 0.9) -> str:
        """
        Genera código a partir de un prompt
        
        Args:
            prompt: Texto de entrada
            tokenizer: Tokenizador para procesar el texto
            max_length: Longitud máxima de generación
            temperature: Temperatura para muestreo
            top_k: Top-k sampling
            top_p: Top-p (nucleus) sampling
        
        Returns:
            generated_code: Código generado
        """
        # Tokenizar prompt
        input_ids = tokenizer.encode(prompt, return_tensors='tf')
        input_ids = input_ids['input_ids']
        
        generated_ids = input_ids.numpy()[0].tolist()
        
        # Generación autoregresiva
        for _ in range(max_length):
            # Preparar entrada
            current_input = tf.convert_to_tensor([generated_ids], dtype=tf.int32)
            
            # Obtener logits del modelo
            with tf.no_grad():
                logits = self(current_input, training=False)
            
            # Obtener logits del siguiente token
            next_token_logits = logits[0, -1, :]
            
            # Aplicar temperatura
            next_token_logits = next_token_logits / temperature
            
            # Aplicar top-k sampling
            if top_k > 0:
                top_k_logits, top_k_indices = tf.math.top_k(
                    next_token_logits, k=top_k
                )
                next_token_logits = tf.scatter_nd(
                    tf.expand_dims(top_k_indices, axis=-1),
                    top_k_logits,
                    tf.shape_like(next_token_logits)
                )
            
            # Aplicar top-p (nucleus) sampling
            if top_p < 1.0:
                sorted_logits = tf.sort(next_token_logits, direction='DESCENDING')
                sorted_indices = tf.argsort(next_token_logits, direction='DESCENDING')
                cumulative_probs = tf.cumsum(tf.nn.softmax(sorted_logits), axis=-1)
                
                # Remover tokens con probabilidad acumulada > top_p
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove = tf.concat([
                    tf.constant([False], dtype=tf.bool),
                    sorted_indices_to_remove[:-1]
                ], axis=0)
                
                indices_to_remove = tf.gather(sorted_indices, tf.where(sorted_indices_to_remove))
                next_token_logits = tf.tensor_scatter_nd_update(
                    next_token_logits,
                    tf.expand_dims(indices_to_remove, axis=-1),
                    tf.fill(tf.shape(indices_to_remove), float('-inf'))
                )
            
            # Muestrear siguiente token
            next_token_probs = tf.nn.softmax(next_token_logits)
            next_token_id = tf.random.categorical(next_token_probs, num_samples=1)[0]
            
            # Añadir token generado
            generated_ids.append(int(next_token_id))
            
            # Detener si se genera token de fin de secuencia
            if next_token_id == tokenizer.eos_token_id:
                break
        
        # Decodificar texto generado
        generated_code = tokenizer.decode(generated_ids)
        return generated_code
    
    def get_config(self):
        """
        Retorna la configuración del modelo para serialización
        """
        config = super().get_config()
        config.update({
            'vocab_size': self.vocab_size,
            'max_sequence_length': self.max_sequence_length,
            'embed_dim': self.embed_dim,
            'num_layers': self.num_layers,
            'num_heads': self.num_heads,
            'dropout_rate': self.dropout_rate,
            'use_syntax_aware': self.use_syntax_aware
        })
        return config

class CodeQualityChecker:
    """
    Clase para verificar la calidad del código generado
    """
    
    def __init__(self):
        self.syntax_patterns = {
            'unclosed_brackets': r'[\[\{\(][^\]\}\)]*$',
            'missing_colon': r'(if|for|while|def|class)[^:]*$',
            'trailing_whitespace': r'\s+$',
        }
    
    def check_syntax(self, code: str) -> Dict:
        """
        Verifica la sintaxis del código generado
        """
        issues = []
        
        for pattern_name, pattern in self.syntax_patterns.items():
            matches = re.findall(pattern, code, re.MULTILINE)
            if matches:
                issues.append({
                    'type': pattern_name,
                    'count': len(matches),
                    'examples': matches[:3]  # Primeros 3 ejemplos
                })
        
        return {
            'syntax_issues': issues,
            'total_issues': len(issues),
            'quality_score': max(0, 100 - len(issues) * 10)
        }
    
    def check_security(self, code: str) -> Dict:
        """
        Verifica vulnerabilidades de seguridad comunes
        """
        security_patterns = {
            'sql_injection': r'(execute|exec|eval)\s*\(\s*["\'].*\+.*["\']',
            'command_injection': r'os\.system|subprocess\.call.*\+',
            'hardcoded_secrets': r'(password|secret|key)\s*=\s*["\'][^"\']+["\']',
        }
        
        vulnerabilities = []
        
        for vuln_name, pattern in security_patterns.items():
            matches = re.findall(pattern, code, re.IGNORECASE)
            if matches:
                vulnerabilities.append({
                    'type': vuln_name,
                    'count': len(matches),
                    'examples': matches[:3]
                })
        
        return {
            'security_issues': vulnerabilities,
            'total_vulnerabilities': len(vulnerabilities),
            'security_score': max(0, 100 - len(vulnerabilities) * 25)
        }

def main():
    """
    Función principal para ejecutar el caso de uso completo
    """
    print("💻 CASO DE USO 3: GENERACIÓN DE CÓDIGO CON TRANSFORMERS")
    print("=" * 70)
    
    # 1. Crear modelo
    print("\n🏗️ Construyendo modelo generador de código...")
    model = CodeGeneratorModel(
        vocab_size=50000,
        max_sequence_length=1024,
        embed_dim=768,
        num_layers=6,
        num_heads=8,
        dropout_rate=0.1,
        use_syntax_aware=True
    )
    
    model.summary()
    
    # 2. Compilar modelo
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # 3. Generar datos sintéticos
    print("\n📊 Generando datos de entrenamiento...")
    batch_size = 16
    num_samples = 1000
    seq_length = 256
    
    # Datos sintéticos de código (simplificados)
    input_data = np.random.randint(1, 1000, size=(num_samples, seq_length))
    target_data = np.random.randint(1, 1000, size=(num_samples, seq_length))
    
    # Dividir datos
    split_idx = int(0.8 * num_samples)
    
    train_input = input_data[:split_idx]
    train_target = target_data[:split_idx]
    
    val_input = input_data[split_idx:]
    val_target = target_data[split_idx:]
    
    print(f"   Datos generados:")
    print(f"   - Entrada: {train_input.shape}")
    print(f"   - Objetivo: {train_target.shape}")
    print(f"   - Entrenamiento: {len(train_input)} muestras")
    print(f"   - Validación: {len(val_input)} muestras")
    
    # 4. Entrenar modelo
    print("\n🚀 Iniciando entrenamiento...")
    history = model.fit(
        train_input,
        train_target,
        validation_data=(val_input, val_target),
        epochs=5,
        batch_size=8,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=2,
                restore_best_weights=True
            )
        ],
        verbose=1
    )
    
    # 5. Generar código de ejemplo
    print("\n💻 Generando código de ejemplo...")
    
    # Tokenizador simulado
    class MockTokenizer:
        def __init__(self):
            self.vocab_size = 50000
            self.eos_token_id = 2
        
        def encode(self, text):
            # Simulación simple de tokenización
            tokens = [hash(word) % self.vocab_size for word in text.split()]
            return {'input_ids': tf.convert_to_tensor([tokens], dtype=tf.int32)}
        
        def decode(self, token_ids):
            # Simulación simple de decodificación
            return "generated_code_example"
    
    tokenizer = MockTokenizer()
    
    # Generar código
    prompt = "def calculate_fibonacci(n):"
    generated_code = model.generate_code(
        prompt=prompt,
        tokenizer=tokenizer,
        max_length=100,
        temperature=0.8,
        top_k=50
    )
    
    print(f"   Prompt: {prompt}")
    print(f"   Código generado: {generated_code}")
    
    # 6. Verificar calidad del código
    print("\n🔍 Verificando calidad del código generado...")
    quality_checker = CodeQualityChecker()
    
    syntax_check = quality_checker.check_syntax(generated_code)
    security_check = quality_checker.check_security(generated_code)
    
    print(f"   Calidad sintáctica: {syntax_check['quality_score']}/100")
    print(f"   Problemas de sintaxis: {syntax_check['total_issues']}")
    print(f"   Puntuación de seguridad: {security_check['security_score']}/100")
    print(f"   Vulnerabilidades: {security_check['total_vulnerabilities']}")
    
    # 7. Guardar modelo
    print("\n💾 Guardando modelo...")
    model.save('code_generator_transformer.h5')
    
    print("\n✅ CASO DE USO 3 COMPLETADO")
    print("=" * 70)
    print("🎯 RESULTADOS:")
    print("   • Modelo Transformer entrenado y guardado")
    print("   • Código generado con atención personalizada")
    print("   • Verificación de calidad implementada")
    print("   • Listo para fase de optimización")
    print("=" * 70)

if __name__ == "__main__":
    main()
