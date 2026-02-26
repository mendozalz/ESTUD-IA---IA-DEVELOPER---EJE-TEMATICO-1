import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import warnings
warnings.filterwarnings('ignore')

class ExplicadorRedesNeuronales:
    def __init__(self):
        self.redes = {
            'Dense': self.explicar_dense,
            'CNN': self.explicar_cnn,
            'RNN': self.explicar_rnn,
            'LSTM': self.explicar_lstm,
            'GRU': self.explicar_gru,
            'Autoencoder': self.explicar_autoencoder,
            'GAN': self.explicar_gan,
            'GNN': self.explicar_gnn,
            'Transformer': self.explicar_transformer,
            'ResNet': self.explicar_resnet,
            'UNet': self.explicar_unet,
            'DQN': self.explicar_dqn
        }
    
    def explicar_dense(self):
        """Red Neuronal Densa (Fully Connected)"""
        print("=" * 60)
        print("🔗 RED NEURONAL DENSA (FULLY CONNECTED)")
        print("=" * 60)
        
        print("\n📋 CARACTERÍSTICAS:")
        print("- Capas totalmente conectadas")
        print("- Cada neurona conectada con todas las anteriores")
        print("- Arquitectura básica del deep learning")
        
        print("\n🏗️  COMPOSICIÓN:")
        print("Input Layer → Hidden Layers → Output Layer")
        
        print("\n📦 COMPONENTES:")
        print("- Neuronas: Unidades básicas de procesamiento")
        print("- Pesos: Conexiones entre neuronas")
        print("- Sesgos: Términos de ajuste")
        print("- Funciones de activación: ReLU, Sigmoid, Tanh")
        
        print("\n🔧 LIBRERÍAS:")
        print("- TensorFlow/Keras: Dense layers")
        print("- PyTorch: nn.Linear")
        print("- Scikit-learn: MLPClassifier")
        
        print("\n🎯 APLICACIONES:")
        print("- Clasificación de datos tabulares")
        print("- Regresión")
        print("- Problemas generales de ML")
        
        # Ejemplo simple
        print("\n💻 EJEMPLO KERAS:")
        modelo = keras.Sequential([
            keras.layers.Dense(64, activation='relu', input_shape=(10,)),
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dense(1, activation='sigmoid')
        ])
        print("model.summary()")
        modelo.summary()
        
    def explicar_cnn(self):
        """Red Neuronal Convolucional"""
        print("\n" + "=" * 60)
        print("🖼️  RED NEURONAL CONVOLUCIONAL (CNN)")
        print("=" * 60)
        
        print("\n📋 CARACTERÍSTICAS:")
        print("- Convoluciones espaciales")
        print("- Jerarquía de características")
        print("- Compartición de pesos")
        
        print("\n🏗️  COMPOSICIÓN:")
        print("Conv → Pool → Conv → Pool → FC")
        
        print("\n📦 COMPONENTES:")
        print("- Convolución: Filtros/kernels")
        print("- Pooling: Reducción dimensional")
        print("- Padding: Manejo de bordes")
        print("- Strides: Movimiento de filtros")
        
        print("\n🔧 LIBRERÍAS:")
        print("- TensorFlow/Keras: Conv2D, MaxPooling2D")
        print("- PyTorch: nn.Conv2d, nn.MaxPool2d")
        print("- OpenCV: Procesamiento de imágenes")
        
        print("\n🎯 APLICACIONES:")
        print("- Clasificación de imágenes")
        print("- Detección de objetos")
        print("- Segmentación")
        
        # Ejemplo simple
        print("\n💻 EJEMPLO KERAS:")
        modelo = keras.Sequential([
            keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
            keras.layers.MaxPooling2D((2,2)),
            keras.layers.Conv2D(64, (3,3), activation='relu'),
            keras.layers.MaxPooling2D((2,2)),
            keras.layers.Flatten(),
            keras.layers.Dense(10, activation='softmax')
        ])
        print("model.summary()")
        modelo.summary()
    
    def explicar_rnn(self):
        """Red Neuronal Recurrente"""
        print("\n" + "=" * 60)
        print("🔄 RED NEURONAL RECURRENTE (RNN)")
        print("=" * 60)
        
        print("\n📋 CARACTERÍSTICAS:")
        print("- Memoria temporal")
        print("- Procesamiento secuencial")
        print("- Retroalimentación temporal")
        
        print("\n🏗️  COMPOSICIÓN:")
        print("Input → RNN Cell → Hidden State → Output")
        
        print("\n📦 COMPONENTES:")
        print("- Estado oculto: Memoria temporal")
        print("- Pesos compartidos: Misma célula en cada paso")
        print("- Secuencia: Procesamiento paso a paso")
        
        print("\n🔧 LIBRERÍAS:")
        print("- TensorFlow/Keras: SimpleRNN, LSTM, GRU")
        print("- PyTorch: nn.RNN, nn.LSTM, nn.GRU")
        
        print("\n🎯 APLICACIONES:")
        print("- Series temporales")
        print("- Procesamiento de texto")
        print("- Speech recognition")
        
    def explicar_lstm(self):
        """Long Short-Term Memory"""
        print("\n" + "=" * 60)
        print("🧠 LONG SHORT-TERM MEMORY (LSTM)")
        print("=" * 60)
        
        print("\n📋 CARACTERÍSTICAS:")
        print("- Memoria a largo plazo")
        print("- Mecanismo de puertas")
        print("- Soluciona vanishing gradient")
        
        print("\n🏗️  COMPOSICIÓN:")
        print("Input → Forget Gate → Input Gate → Output Gate → Cell State")
        
        print("\n📦 COMPONENTES:")
        print("- Forget Gate: Qué olvidar")
        print("- Input Gate: Qué almacenar")
        print("- Output Gate: Qué output")
        print("- Cell State: Memoria principal")
        
        print("\n🔧 LIBRERÍAS:")
        print("- TensorFlow/Keras: LSTM layer")
        print("- PyTorch: nn.LSTM")
        
        print("\n🎯 APLICACIONES:")
        print("- Traducción automática")
        print("- Análisis de sentimientos")
        print("- Predicción de series")
    
    def explicar_gru(self):
        """Gated Recurrent Units"""
        print("\n" + "=" * 60)
        print("🚪 GATED RECURRENT UNITS (GRU)")
        print("=" * 60)
        
        print("\n📋 CARACTERÍSTICAS:")
        print("- Simplificación de LSTM")
        print("- Menos parámetros")
        print("- Más rápido de entrenar")
        
        print("\n🏗️  COMPOSICIÓN:")
        print("Input → Reset Gate → Update Gate → Hidden State")
        
        print("\n📦 COMPONENTES:")
        print("- Reset Gate: Qué resetear")
        print("- Update Gate: Qué actualizar")
        print("- Hidden State: Memoria combinada")
        
        print("\n🔧 LIBRERÍAS:")
        print("- TensorFlow/Keras: GRU layer")
        print("- PyTorch: nn.GRU")
        
        print("\n🎯 APLICACIONES:")
        print("- Procesamiento de lenguaje")
        print("- Series temporales")
        print("- Speech recognition")
    
    def explicar_autoencoder(self):
        """Autoencoder"""
        print("\n" + "=" * 60)
        print("🗜️  AUTOENCODER")
        print("=" * 60)
        
        print("\n📋 CARACTERÍSTICAS:")
        print("- Aprendizaje no supervisado")
        print("- Compresión-descompresión")
        print("- Reducción dimensionalidad")
        
        print("\n🏗️  COMPOSICIÓN:")
        print("Input → Encoder → Latent Space → Decoder → Output")
        
        print("\n📦 COMPONENTES:")
        print("- Encoder: Compresión de datos")
        print("- Latent Space: Representación compacta")
        print("- Decoder: Reconstrucción")
        
        print("\n🔧 LIBRERÍAS:")
        print("- TensorFlow/Keras: Autoencoder models")
        print("- PyTorch: Autoencoder implementations")
        
        print("\n🎯 APLICACIONES:")
        print("- Reducción dimensionalidad")
        print("- Detección de anomalías")
        print("- Denoising de datos")
    
    def explicar_gan(self):
        """Generative Adversarial Networks"""
        print("\n" + "=" * 60)
        print("⚔️  GENERATIVE ADVERSARIAL NETWORKS (GAN)")
        print("=" * 60)
        
        print("\n📋 CARACTERÍSTICAS:")
        print("- Dos redes en competencia")
        print("- Aprendizaje adversarial")
        print("- Generación de datos sintéticos")
        
        print("\n🏗️  COMPOSICIÓN:")
        print("Generator ↔ Discriminator")
        
        print("\n📦 COMPONENTES:")
        print("- Generator: Crea datos falsos")
        print("- Discriminator: Distingue real/falso")
        print("- Loss Function: Minimax game")
        
        print("\n🔧 LIBRERÍAS:")
        print("- TensorFlow/Keras: GAN implementations")
        print("- PyTorch: GAN models")
        
        print("\n🎯 APLICACIONES:")
        print("- Generación de imágenes")
        print("- Data augmentation")
        print("- Style transfer")
    
    def explicar_gnn(self):
        """Redes Neuronales de Grafos"""
        print("\n" + "=" * 60)
        print("🕸️  REDES NEURONALES DE GRAFOS (GNN)")
        print("=" * 60)
        
        print("\n📋 CARACTERÍSTICAS:")
        print("- Procesamiento de grafos")
        print("- Operaciones en nodos/aristas")
        print("- Aprendizaje estructural")
        
        print("\n🏗️  COMPOSICIÓN:")
        print("Graph Conv → Message Passing → Readout")
        
        print("\n📦 COMPONENTES:")
        print("- Graph Convolution: Operación en grafos")
        print("- Message Passing: Comunicación entre nodos")
        print("- Readout: Predicción global")
        
        print("\n🔧 LIBRERÍAS:")
        print("- PyTorch Geometric: Principal librería")
        print("- DGL: Deep Graph Library")
        print("- Spektral: TensorFlow/Keras")
        
        print("\n🎯 APLICACIONES:")
        print("- Redes sociales")
        print("- Química molecular")
        print("- Sistemas de recomendación")
    
    def explicar_transformer(self):
        """Transformer Networks"""
        print("\n" + "=" * 60)
        print("🤖 TRANSFORMER NETWORKS")
        print("=" * 60)
        
        print("\n📋 CARACTERÍSTICAS:")
        print("- Mecanismo de atención")
        print("- Procesamiento paralelo")
        print("- Sin recurrencia")
        
        print("\n🏗️  COMPOSICIÓN:")
        print("Embedding → Multi-Head Attention → FFN → Output")
        
        print("\n📦 COMPONENTES:")
        print("- Self-Attention: Relaciones entre tokens")
        print("- Multi-Head: Múltiples cabezas de atención")
        print("- Positional Encoding: Información de posición")
        print("- Feed Forward: Procesamiento no lineal")
        
        print("\n🔧 LIBRERÍAS:")
        print("- Hugging Face: Transformers pre-entrenados")
        print("- TensorFlow/Keras: Transformer layers")
        print("- PyTorch: nn.Transformer")
        
        print("\n🎯 APLICACIONES:")
        print("- NLP (BERT, GPT)")
        print("- Vision (ViT)")
        print("- Multimodal")
    
    def explicar_resnet(self):
        """Redes Neuronales Residuales"""
        print("\n" + "=" * 60)
        print("🔀 REDES NEURONALES RESIDUALES (RESNET)")
        print("=" * 60)
        
        print("\n📋 CARACTERÍSTICAS:")
        print("- Skip connections")
        print("- Resuelve vanishing gradients")
        print("- Permite redes muy profundas")
        
        print("\n🏗️  COMPOSICIÓN:")
        print("Input → Conv Block + Skip → Output")
        
        print("\n📦 COMPONENTES:")
        print("- Skip Connections: Conexiones directas")
        print("- Residual Blocks: Bloques residuales")
        print("- Identity Mapping: Mapeo identidad")
        
        print("\n🔧 LIBRERÍAS:")
        print("- TensorFlow/Keras: ResNet implementations")
        print("- PyTorch: torchvision.models.resnet")
        
        print("\n🎯 APLICACIONES:")
        print("- Clasificación de imágenes")
        print("- Deep learning profundo")
        print("- Transfer learning")
    
    def explicar_unet(self):
        """U-Net"""
        print("\n" + "=" * 60)
        print("🏥 U-NET")
        print("=" * 60)
        
        print("\n📋 CARACTERÍSTICAS:")
        print("- Encoder-decoder simétrico")
        print("- Skip connections entre niveles")
        print("- Segmentación precisa")
        
        print("\n🏗️  COMPOSICIÓN:")
        print("Encoder → Bottleneck → Decoder")
        
        print("\n📦 COMPONENTES:")
        print("- Encoder: Extracción de características")
        print("- Bottleneck: Representación compacta")
        print("- Decoder: Reconstrucción espacial")
        print("- Skip Connections: Información espacial")
        
        print("\n🔧 LIBRERÍAS:")
        print("- TensorFlow/Keras: U-Net implementations")
        print("- PyTorch: U-Net models")
        
        print("\n🎯 APLICACIONES:")
        print("- Segmentación semántica")
        print("- Imágenes médicas")
        print("- Segmentación de objetos")
    
    def explicar_dqn(self):
        """Deep Q-Networks"""
        print("\n" + "=" * 60)
        print("🎮 DEEP Q-NETWORKS (DQN)")
        print("=" * 60)
        
        print("\n📋 CARACTERÍSTICAS:")
        print("- Q-learning + redes profundas")
        print("- Aproximación de función Q")
        print("- Experience replay")
        
        print("\n🏗️  COMPOSICIÓN:")
        print("CNN → FC Layers → Q-Values")
        
        print("\n📦 COMPONENTES:")
        print("- Q-Network: Aproxima función Q")
        print("- Target Network: Red objetivo estable")
        print("- Experience Replay: Buffer de experiencias")
        print("- Epsilon-greedy: Exploración-explotación")
        
        print("\n🔧 LIBRERÍAS:")
        print("- TensorFlow/Keras: DQN implementations")
        print("- PyTorch: DQN models")
        print("- Gym/OpenAI Gym: Entornos de prueba")
        
        print("\n🎯 APLICACIONES:")
        print("- Videojuegos (Atari)")
        print("- Robótica")
        print("- Control de sistemas")
    
    def mostrar_todas(self):
        """Muestra explicación de todas las redes"""
        print("🧠 EXPLICACIÓN COMPLETA DE 12 TIPOS DE REDES NEURONALES")
        print("=" * 80)
        
        for nombre, funcion in self.redes.items():
            funcion()
            print("\n" + "="*80)
    
    def crear_tabla_comparativa(self):
        """Crea tabla comparativa de características"""
        print("\n📊 TABLA COMPARATIVA DE REDES NEURONALES")
        print("=" * 80)
        
        caracteristicas = {
            'Tipo': ['Dense', 'CNN', 'RNN', 'LSTM', 'GRU', 'Autoencoder', 
                    'GAN', 'GNN', 'Transformer', 'ResNet', 'U-Net', 'DQN'],
            'Paradigma': ['Sup/NoSup', 'Sup', 'Sup', 'Sup', 'Sup', 'NoSup', 
                         'NoSup', 'Sup/NoSup', 'Sup', 'Sup', 'Sup', 'Ref'],
            'Datos': ['Tabulares', 'Imágenes', 'Secuencias', 'Secuencias', 
                     'Secuencias', 'Cualquiera', 'Imágenes', 'Grafos', 
                     'Secuencias', 'Imágenes', 'Imágenes', 'Estados'],
            'Complejidad': ['Baja', 'Media', 'Media', 'Alta', 'Media', 'Media', 
                          'Alta', 'Alta', 'Muy Alta', 'Alta', 'Alta', 'Alta'],
            'Parámetros': ['Medios', 'Altos', 'Medios', 'Altos', 'Medios', 
                          'Medios', 'Muy Altos', 'Variables', 'Muy Altos', 
                          'Altos', 'Altos', 'Altos']
        }
        
        df = pd.DataFrame(caracteristicas)
        print(df.to_string(index=False))
        
        return df

def demo_teoria_redes():
    """Demostración completa de teoría de redes neuronales"""
    print("=" * 80)
    print("🧠 LABORATORIO 4 - TEORÍA DE REDES NEURONALES")
    print("=" * 80)
    
    # Crear explicador
    explicador = ExplicadorRedesNeuronales()
    
    # Mostrar tabla comparativa
    tabla = explicador.crear_tabla_comparativa()
    
    # Explicar cada tipo (opcional - descomentar para ver todas)
    # explicador.mostrar_todas()
    
    print("\n" + "=" * 80)
    print("🎉 TEORÍA COMPLETADA")
    print("=" * 80)

if __name__ == "__main__":
    demo_teoria_redes()
