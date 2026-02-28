"""
Caso de Uso 1 - Optimización de Rutas de Entrega con GNNs
Fase 3: Desarrollo - Modelo GNN con Capas Personalizadas
"""

import tensorflow as tf
import spektral as sp
from spektral.layers import GATConv, GCNConv
from spektral.utils import normalized_adjacency
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from typing import Tuple, Dict, List, Optional

class DynamicGATLayer(tf.keras.layers.Layer):
    """
    Capa de Graph Attention Network personalizada para manejar grafos dinámicos
    con atributos variables en el tiempo (tráfico, clima, etc.)
    """
    
    def __init__(self, 
                 channels: int, 
                 num_heads: int = 4,
                 dropout_rate: float = 0.1,
                 use_edge_weights: bool = True,
                 **kwargs):
        super().__init__(**kwargs)
        self.channels = channels
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.use_edge_weights = use_edge_weights
        
        # Capa GAT principal
        self.gat = GATConv(
            channels=channels,
            attn_heads=num_heads,
            concat_heads=True,
            dropout_rate=dropout_rate,
            activation='elu'
        )
        
        # Normalización y regularización
        self.layer_norm = tf.keras.layers.LayerNormalization()
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        
        # Procesamiento de pesos dinámicos de aristas
        if use_edge_weights:
            self.edge_processor = tf.keras.layers.Dense(
                channels, 
                activation='sigmoid',
                name='edge_processor'
            )
    
    def build(self, input_shape):
        """
        Construye los pesos de la capa basándose en las formas de entrada
        """
        # input_shape: [(batch_size, num_nodes, node_features), 
        #                (batch_size, num_nodes, num_nodes), 
        #                (batch_size, num_nodes, num_nodes, edge_features)]
        super().build(input_shape)
    
    def call(self, inputs, training: Optional[bool] = None):
        """
        Forward pass de la capa GAT dinámica
        
        Args:
            inputs: Tupla de (X, A, E) donde:
                X: Features de nodos [batch_size, num_nodes, node_features]
                A: Matriz de adyacencia [batch_size, num_nodes, num_nodes]
                E: Features de aristas dinámicas [batch_size, num_nodes, num_nodes, edge_features]
            training: Booleano para modo de entrenamiento
        
        Returns:
            output: Features de nodos procesados [batch_size, num_nodes, channels * num_heads]
        """
        X, A, E = inputs
        
        # Procesar pesos dinámicos de aristas si están disponibles
        if self.use_edge_weights and E is not None:
            # E tiene forma [batch_size, num_nodes, num_nodes, edge_features]
            # Procesamos para obtener un peso por arista
            edge_weights = self.edge_processor(E)  # [batch_size, num_nodes, num_nodes, 1]
            edge_weights = tf.squeeze(edge_weights, axis=-1)  # [batch_size, num_nodes, num_nodes]
            
            # Combinar con matriz de adyacencia original
            A = tf.multiply(A, edge_weights)
        
        # Aplicar GAT
        node_output = self.gat([X, A])
        
        # Normalización y dropout
        if training:
            node_output = self.dropout(node_output, training=training)
        
        node_output = self.layer_norm(node_output)
        
        return node_output
    
    def get_config(self):
        """
        Retorna la configuración de la capa para serialización
        """
        config = super().get_config()
        config.update({
            'channels': self.channels,
            'num_heads': self.num_heads,
            'dropout_rate': self.dropout_rate,
            'use_edge_weights': self.use_edge_weights
        })
        return config

class TemporalGNNModel(tf.keras.Model):
    """
    Modelo completo de GNN para optimización de rutas con componentes temporales
    """
    
    def __init__(self, 
                 node_features: int,
                 edge_features: int,
                 hidden_units: List[int] = [128, 64, 32],
                 num_heads: int = 4,
                 dropout_rate: float = 0.1,
                 num_timesteps: int = 24,
                 **kwargs):
        super().__init__(**kwargs)
        
        self.node_features = node_features
        self.edge_features = edge_features
        self.hidden_units = hidden_units
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.num_timesteps = num_timesteps
        
        # Capas de entrada y preprocesamiento
        self.node_input_processor = tf.keras.layers.Dense(hidden_units[0])
        self.edge_input_processor = tf.keras.layers.Dense(hidden_units[0])
        
        # Capas GNN dinámicas
        self.gnn_layers = []
        for i, units in enumerate(hidden_units):
            self.gnn_layers.append(
                DynamicGATLayer(
                    channels=units,
                    num_heads=num_heads,
                    dropout_rate=dropout_rate,
                    use_edge_weights=True,
                    name=f'dynamic_gat_{i}'
                )
            )
        
        # Capa LSTM para procesamiento temporal
        self.temporal_processor = tf.keras.layers.LSTM(
            hidden_units[-1],
            return_sequences=True,
            dropout=dropout_rate,
            name='temporal_lstm'
        )
        
        # Capas de salida
        self.output_layers = [
            tf.keras.layers.Dense(1, activation='sigmoid', name='route_probability'),
            tf.keras.layers.Dense(1, activation='linear', name='estimated_time'),
            tf.keras.layers.Dense(1, activation='linear', name='fuel_cost')
        ]
        
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.layer_norm = tf.keras.layers.LayerNormalization()
    
    def build(self, input_shape):
        """
        Construye el modelo basándose en las formas de entrada
        """
        # input_shape: {
        #     'nodes': (batch_size, num_nodes, node_features),
        #     'edges': (batch_size, num_nodes, num_nodes, edge_features),
        #     'adjacency': (batch_size, num_nodes, num_nodes),
        #     'temporal': (batch_size, num_timesteps, num_nodes, edge_features)
        # }
        super().build(input_shape)
    
    def call(self, inputs, training: Optional[bool] = None):
        """
        Forward pass del modelo completo
        
        Args:
            inputs: Diccionario con todas las entradas del modelo
            training: Booleano para modo de entrenamiento
        
        Returns:
            outputs: Diccionario con predicciones múltiples
        """
        X = inputs['nodes']          # [batch_size, num_nodes, node_features]
        E = inputs['edges']          # [batch_size, num_nodes, num_nodes, edge_features]
        A = inputs['adjacency']      # [batch_size, num_nodes, num_nodes]
        T = inputs['temporal']       # [batch_size, num_timesteps, num_nodes, edge_features]
        
        batch_size = tf.shape(X)[0]
        num_nodes = tf.shape(X)[1]
        
        # Preprocesar entradas
        X_processed = self.node_input_processor(X)
        E_processed = self.edge_input_processor(E)
        
        # Procesar información temporal
        # Promediar características temporales de aristas
        E_temporal_avg = tf.reduce_mean(T, axis=1)  # [batch_size, num_nodes, num_nodes, edge_features]
        E_temporal_processed = self.edge_input_processor(E_temporal_avg)
        
        # Combinar características estáticas y temporales de aristas
        E_combined = (E_processed + E_temporal_processed) / 2.0
        
        # Pasar por capas GNN
        gnn_output = X_processed
        for gnn_layer in self.gnn_layers:
            gnn_output = gnn_layer([gnn_output, A, E_combined], training=training)
            if training:
                gnn_output = self.dropout(gnn_output, training=training)
        
        # Normalización final
        gnn_output = self.layer_norm(gnn_output)
        
        # Procesamiento temporal para predicciones dinámicas
        # Expandir dimensión temporal para LSTM
        gnn_output_expanded = tf.expand_dims(gnn_output, axis=1)  # [batch_size, 1, num_nodes, features]
        gnn_output_expanded = tf.tile(gnn_output_expanded, [1, self.num_timesteps, 1, 1])
        
        # Combinar con información temporal de aristas
        temporal_input = tf.concat([gnn_output_expanded, T], axis=-1)
        temporal_output = self.temporal_processor(temporal_input)
        
        # Obtener el último timestep para predicciones
        final_features = temporal_output[:, -1, :, :]  # [batch_size, num_nodes, features]
        
        # Generar múltiples salidas
        outputs = {}
        for i, output_layer in enumerate(self.output_layers):
            outputs[output_layer.name] = tf.reduce_mean(
                output_layer(final_features), 
                axis=1
            )  # [batch_size, 1] - promedio sobre todos los nodos
        
        return outputs
    
    def get_config(self):
        """
        Retorna la configuración del modelo para serialización
        """
        config = super().get_config()
        config.update({
            'node_features': self.node_features,
            'edge_features': self.edge_features,
            'hidden_units': self.hidden_units,
            'num_heads': self.num_heads,
            'dropout_rate': self.dropout_rate,
            'num_timesteps': self.num_timesteps
        })
        return config

class RouteOptimizer:
    """
    Clase principal para optimización de rutas usando el modelo GNN
    """
    
    def __init__(self, model: TemporalGNNModel):
        self.model = model
        self.scaler_nodes = None
        self.scaler_edges = None
        self.graph_structure = None
    
    def prepare_synthetic_data(self, 
                           num_nodes: int = 50, 
                           num_samples: int = 1000,
                           num_timesteps: int = 24) -> Dict:
        """
        Genera datos sintéticos para entrenamiento y prueba
        """
        print(f"📊 Generando datos sintéticos: {num_nodes} nodos, {num_samples} muestras")
        
        # Generar grafo base
        G = nx.erdos_renyi_graph(num_nodes, 0.1, seed=42)
        adjacency_matrix = nx.adjacency_matrix(G).todense()
        
        # Features de nodos (capacidad, tipo, ubicación, etc.)
        node_features = np.random.randn(num_samples, num_nodes, 8)
        
        # Features de aristas (distancia, tipo de carretera, tráfico base)
        edge_features = np.random.randn(num_samples, num_nodes, num_nodes, 4)
        
        # Features temporales de aristas (tráfico variable, clima, etc.)
        temporal_features = np.random.randn(num_samples, num_timesteps, num_nodes, num_nodes, 4)
        
        # Generar etiquetas (probabilidad de ruta óptima, tiempo estimado, costo)
        route_prob = np.random.rand(num_samples, 1)
        estimated_time = np.random.exponential(scale=30, size=(num_samples, 1))  # minutos
        fuel_cost = np.random.gamma(shape=2, scale=5, size=(num_samples, 1))  # dólares
        
        data = {
            'nodes': node_features.astype(np.float32),
            'edges': edge_features.astype(np.float32),
            'adjacency': adjacency_matrix.astype(np.float32),
            'temporal': temporal_features.astype(np.float32),
            'route_probability': route_prob.astype(np.float32),
            'estimated_time': estimated_time.astype(np.float32),
            'fuel_cost': fuel_cost.astype(np.float32)
        }
        
        self.graph_structure = G
        return data
    
    def train_model(self, 
                   data: Dict, 
                   validation_split: float = 0.2,
                   epochs: int = 50,
                   batch_size: int = 32) -> Dict:
        """
        Entrena el modelo GNN
        """
        print("🚀 Iniciando entrenamiento del modelo GNN...")
        
        # Dividir datos
        num_samples = data['nodes'].shape[0]
        val_size = int(num_samples * validation_split)
        
        train_data = {k: v[:-val_size] for k, v in data.items()}
        val_data = {k: v[-val_size:] for k, v in data.items()}
        
        # Compilar modelo
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss={
                'route_probability': 'binary_crossentropy',
                'estimated_time': 'mse',
                'fuel_cost': 'mse'
            },
            loss_weights={
                'route_probability': 1.0,
                'estimated_time': 0.5,
                'fuel_cost': 0.3
            },
            metrics={
                'route_probability': ['accuracy', 'auc'],
                'estimated_time': ['mae'],
                'fuel_cost': ['mae']
            }
        )
        
        # Callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6
            ),
            tf.keras.callbacks.ModelCheckpoint(
                'best_gnn_model.h5',
                monitor='val_loss',
                save_best_only=True
            )
        ]
        
        # Entrenar
        history = self.model.fit(
            {
                'nodes': train_data['nodes'],
                'edges': train_data['edges'],
                'adjacency': train_data['adjacency'],
                'temporal': train_data['temporal']
            },
            {
                'route_probability': train_data['route_probability'],
                'estimated_time': train_data['estimated_time'],
                'fuel_cost': train_data['fuel_cost']
            },
            validation_data=(
                {
                    'nodes': val_data['nodes'],
                    'edges': val_data['edges'],
                    'adjacency': val_data['adjacency'],
                    'temporal': val_data['temporal']
                },
                {
                    'route_probability': val_data['route_probability'],
                    'estimated_time': val_data['estimated_time'],
                    'fuel_cost': val_data['fuel_cost']
                }
            ),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        return history
    
    def predict_optimal_route(self, 
                           node_features: np.ndarray,
                           edge_features: np.ndarray,
                           temporal_features: np.ndarray) -> Dict:
        """
        Realiza predicción de ruta óptima para un caso específico
        """
        # Preparar entrada
        inputs = {
            'nodes': np.expand_dims(node_features, axis=0),
            'edges': np.expand_dims(edge_features, axis=0),
            'adjacency': np.expand_dims(nx.adjacency_matrix(self.graph_structure).todense(), axis=0),
            'temporal': np.expand_dims(temporal_features, axis=0)
        }
        
        # Realizar predicción
        predictions = self.model.predict(inputs)
        
        return {
            'route_probability': float(predictions['route_probability'][0][0]),
            'estimated_time': float(predictions['estimated_time'][0][0]),
            'fuel_cost': float(predictions['fuel_cost'][0][0])
        }
    
    def visualize_graph_with_predictions(self, 
                                    predictions: Dict,
                                    save_path: str = 'graph_predictions.png'):
        """
        Visualiza el grafo con las predicciones del modelo
        """
        plt.figure(figsize=(12, 8))
        
        # Dibujar grafo base
        pos = nx.spring_layout(self.graph_structure, k=2, iterations=50)
        
        # Colorear nodos según características
        node_colors = [self.graph_structure.degree(n) for n in self.graph_structure.nodes()]
        
        nx.draw(self.graph_structure, pos, 
                node_color=node_colors, 
                node_size=300,
                cmap='viridis',
                with_labels=True,
                font_size=8)
        
        # Añadir información de predicciones
        plt.title(f'Predicciones de Ruta Óptima\n'
                 f'Probabilidad: {predictions["route_probability"]:.3f}\n'
                 f'Tiempo Estimado: {predictions["estimated_time"]:.1f} min\n'
                 f'Costo Combustible: ${predictions["fuel_cost"]:.2f}',
                 fontsize=12, fontweight='bold')
        
        plt.colorbar(plt.cm.ScalarMappable(cmap='viridis'), 
                   label='Grado del Nodo')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

def main():
    """
    Función principal para ejecutar el caso de uso completo
    """
    print("CASO DE USO 1: OPTIMIZACION DE RUTAS CON GNNS")
    print("=" * 70)
    
    # 1. Crear modelo
    print("\nConstruyendo modelo GNN dinamico...")
    model = TemporalGNNModel(
        node_features=8,
        edge_features=4,
        hidden_units=[128, 64, 32],
        num_heads=4,
        dropout_rate=0.1,
        num_timesteps=24
    )
    
    model.summary()
    
    # 2. Crear optimizador
    optimizer = RouteOptimizer(model)
    
    # 3. Generar datos sintéticos
    print("\n📊 Generando datos de entrenamiento...")
    data = optimizer.prepare_synthetic_data(
        num_nodes=50,
        num_samples=1000,
        num_timesteps=24
    )
    
    print(f"   Datos generados:")
    print(f"   - Nodos: {data['nodes'].shape}")
    print(f"   - Aristas: {data['edges'].shape}")
    print(f"   - Temporal: {data['temporal'].shape}")
    
    # 4. Entrenar modelo
    print("\n🚀 Iniciando entrenamiento...")
    history = optimizer.train_model(
        data=data,
        validation_split=0.2,
        epochs=30,
        batch_size=16
    )
    
    # 5. Evaluar modelo
    print("\n📈 Evaluando modelo...")
    
    # Generar caso de prueba
    test_nodes = np.random.randn(50, 8).astype(np.float32)
    test_edges = np.random.randn(50, 50, 4).astype(np.float32)
    test_temporal = np.random.randn(24, 50, 50, 4).astype(np.float32)
    
    # Realizar predicción
    predictions = optimizer.predict_optimal_route(
        test_nodes, test_edges, test_temporal
    )
    
    print(f"   Resultados de predicción:")
    print(f"   - Probabilidad de ruta óptima: {predictions['route_probability']:.3f}")
    print(f"   - Tiempo estimado: {predictions['estimated_time']:.1f} minutos")
    print(f"   - Costo de combustible: ${predictions['fuel_cost']:.2f}")
    
    # 6. Visualizar resultados
    print("\n📊 Generando visualizaciones...")
    optimizer.visualize_graph_with_predictions(predictions)
    
    # 7. Guardar modelo
    print("\n💾 Guardando modelo...")
    model.save('dynamic_gnn_route_optimizer.h5')
    
    print("\n✅ CASO DE USO 1 COMPLETADO")
    print("=" * 70)
    print("🎯 RESULTADOS:")
    print("   • Modelo GNN dinámico entrenado y guardado")
    print("   • Predicciones de ruta óptima generadas")
    print("   • Visualizaciones del grafo creadas")
    print("   • Listo para fase de optimización")
    print("=" * 70)

if __name__ == "__main__":
    main()
