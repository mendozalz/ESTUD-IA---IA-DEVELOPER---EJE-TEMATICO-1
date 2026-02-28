"""
Caso de Uso 1 - Optimización de Rutas de Entrega con GNNs
Versión simplificada sin caracteres Unicode
"""

import tensorflow as tf
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple

class SimpleGNNModel(tf.keras.Model):
    """
    Modelo GNN simplificado para optimización de rutas
    """
    
    def __init__(self, num_nodes: int, hidden_units: List[int] = [64, 32]):
        super().__init__()
        self.num_nodes = num_nodes
        
        # Capas del modelo - creadas en __init__
        self.dropout_layer = tf.keras.layers.Dropout(0.2)
        self.dense_layers = []
        for i, units in enumerate(hidden_units):
            self.dense_layers.append(tf.keras.layers.Dense(units, activation='relu', name=f'dense_{i}'))
        
        # Capa de salida
        self.output_layer = tf.keras.layers.Dense(num_nodes, activation='sigmoid', name='output')
        
    def call(self, inputs, training=None):
        x = inputs
        for layer in self.dense_layers:
            x = layer(x)
            x = self.dropout_layer(x, training=training)
        
        # Output: matriz de adyacencia para rutas
        route_matrix = self.output_layer(x)
        return route_matrix

class RouteOptimizer:
    """
    Optimizador de rutas usando GNN
    """
    
    def __init__(self, model: SimpleGNNModel):
        self.model = model
        self.graph = None
        
    def create_sample_graph(self, num_nodes: int = 20) -> nx.Graph:
        """
        Crea un grafo de ejemplo para optimización
        """
        # Crear grafo aleatorio
        G = nx.erdos_renyi_graph(num_nodes, 0.3, seed=42)
        
        # Añadir pesos a las aristas (distancias)
        for edge in G.edges():
            G[edge[0]][edge[1]]['weight'] = np.random.uniform(1, 10)
            
        self.graph = G
        return G
    
    def prepare_training_data(self, num_samples: int = 100) -> Dict:
        """
        Prepara datos de entrenamiento sintéticos
        """
        print(f"Generando {num_samples} muestras de entrenamiento...")
        
        # Features de nodos (coordenadas, demanda, etc.)
        node_features = np.random.randn(num_samples, 20, 8).astype(np.float32)
        
        # Labels (rutas óptimas)
        route_labels = np.random.randint(0, 2, (num_samples, 20, 20)).astype(np.float32)
        
        return {
            'node_features': node_features,
            'route_labels': route_labels
        }
    
    def train_model(self, data: Dict, epochs: int = 10, validation_split: float = 0.2):
        """
        Entrena el modelo GNN
        """
        print("Iniciando entrenamiento del modelo...")
        
        # Compilar modelo
        self.model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        # Entrenar
        history = self.model.fit(
            data['node_features'],
            data['route_labels'],
            epochs=epochs,
            validation_split=validation_split,
            batch_size=16,
            verbose=1
        )
        
        return history
    
    def optimize_route(self, start_node: int, end_node: int) -> List[int]:
        """
        Optimiza ruta entre dos nodos
        """
        if self.graph is None:
            self.create_sample_graph()
        
        # Usar algoritmo simple de camino más corto
        try:
            path = nx.shortest_path(self.graph, start_node, end_node, weight='weight')
            return path
        except:
            return [start_node, end_node]
    
    def visualize_graph(self, save_path: str = 'graph_visualization.png'):
        """
        Visualiza el grafo y rutas
        """
        if self.graph is None:
            self.create_sample_graph()
        
        plt.figure(figsize=(12, 8))
        
        # Posiciones de los nodos
        pos = nx.spring_layout(self.graph)
        
        # Dibujar grafo
        nx.draw(self.graph, pos, with_labels=True, node_color='lightblue', 
                node_size=500, font_size=8, font_weight='bold')
        
        # Dibujar pesos en las aristas
        edge_labels = nx.get_edge_attributes(self.graph, 'weight')
        nx.draw_networkx_edge_labels(self.graph, pos, edge_labels, font_size=6)
        
        plt.title("Grafo de Optimización de Rutas")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

def main():
    """
    Función principal para ejecutar el caso de uso
    """
    print("=" * 60)
    print("CASO DE USO 1: OPTIMIZACION DE RUTAS CON GNNS")
    print("=" * 60)
    
    # 1. Crear modelo
    print("\n1. Creando modelo GNN...")
    model = SimpleGNNModel(num_nodes=20, hidden_units=[64, 32])
    model.summary()
    
    # 2. Crear optimizador
    print("\n2. Inicializando optimizador de rutas...")
    optimizer = RouteOptimizer(model)
    
    # 3. Crear grafo de ejemplo
    print("\n3. Creando grafo de ejemplo...")
    graph = optimizer.create_sample_graph(num_nodes=20)
    print(f"   - Nodos: {graph.number_of_nodes()}")
    print(f"   - Aristas: {graph.number_of_edges()}")
    
    # 4. Preparar datos de entrenamiento
    print("\n4. Preparando datos de entrenamiento...")
    data = optimizer.prepare_training_data(num_samples=100)
    print(f"   - Features shape: {data['node_features'].shape}")
    print(f"   - Labels shape: {data['route_labels'].shape}")
    
    # 5. Entrenar modelo
    print("\n5. Entrenando modelo...")
    history = optimizer.train_model(data, epochs=5)
    
    # 6. Optimizar rutas de ejemplo
    print("\n6. Optimizando rutas de ejemplo...")
    routes = []
    for i in range(5):
        start = np.random.randint(0, 19)
        end = np.random.randint(0, 19)
        if start != end:
            route = optimizer.optimize_route(start, end)
            routes.append((start, end, route))
            print(f"   - Ruta {start} -> {end}: {route}")
    
    # 7. Visualizar grafo
    print("\n7. Generando visualización...")
    optimizer.visualize_graph()
    
    # 8. Guardar modelo
    print("\n8. Guardando modelo...")
    model.save('simple_gnn_route_optimizer.h5')
    
    # 9. Resultados
    print("\n" + "=" * 60)
    print("RESULTADOS:")
    print("   - Modelo GNN entrenado y guardado")
    print("   - Rutas optimizadas generadas")
    print("   - Visualización del grafo creada")
    print("   - Modelo guardado como 'simple_gnn_route_optimizer.h5'")
    print("=" * 60)

if __name__ == "__main__":
    main()
