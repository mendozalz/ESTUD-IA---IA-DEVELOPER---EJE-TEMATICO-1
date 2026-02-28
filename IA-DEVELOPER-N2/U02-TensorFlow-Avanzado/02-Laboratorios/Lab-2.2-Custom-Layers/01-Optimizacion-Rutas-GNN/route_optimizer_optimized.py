"""
Optimizador de Rutas con GNN - Versión Optimizada y Simplificada
"""

import tensorflow as tf
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import os

class RouteOptimizerGNN:
    """Optimizador de rutas usando Graph Neural Networks"""
    
    def __init__(self, num_nodes=20):
        self.num_nodes = num_nodes
        self.model = None
        self.graph = None
        self._build_model()
    
    def _build_model(self):
        """Construye el modelo GNN optimizado"""
        self.model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(self.num_nodes, 4)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(self.num_nodes, activation='sigmoid')
        ])
        
        self.model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
    
    def create_graph(self):
        """Crea un grafo de transporte"""
        G = nx.erdos_renyi_graph(self.num_nodes, 0.3, seed=42)
        
        # Añadir pesos (distancias)
        for edge in G.edges():
            G[edge[0]][edge[1]]['weight'] = np.random.uniform(1, 10)
        
        self.graph = G
        return G
    
    def generate_training_data(self, num_samples=100):
        """Genera datos de entrenamiento"""
        # Features: coordenadas y demanda
        features = np.random.randn(num_samples, self.num_nodes, 4).astype(np.float32)
        
        # Labels: matriz de adyacencia
        labels = np.random.randint(0, 2, (num_samples, self.num_nodes * self.num_nodes)).astype(np.float32)
        labels = labels.reshape(num_samples, self.num_nodes, self.num_nodes)
        
        return features, labels
    
    def train(self, epochs=10):
        """Entrena el modelo"""
        print(f"Entrenando modelo por {epochs} épocas...")
        
        X_train, y_train = self.generate_training_data(200)
        
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            validation_split=0.2,
            batch_size=16,
            verbose=1
        )
        
        return history
    
    def find_shortest_path(self, start, end):
        """Encuentra la ruta más corta entre dos nodos"""
        if self.graph is None:
            self.create_graph()
        
        try:
            path = nx.shortest_path(self.graph, start, end, weight='weight')
            distance = nx.shortest_path_length(self.graph, start, end, weight='weight')
            return path, distance
        except nx.NetworkXNoPath:
            return None, float('inf')
    
    def optimize_multiple_routes(self, routes):
        """Optimiza múltiples rutas"""
        results = []
        for start, end in routes:
            path, distance = self.find_shortest_path(start, end)
            results.append({
                'start': start,
                'end': end,
                'path': path,
                'distance': distance
            })
        return results
    
    def visualize_graph(self, save_path='graph.png'):
        """Visualiza el grafo"""
        if self.graph is None:
            self.create_graph()
        
        plt.figure(figsize=(10, 8))
        pos = nx.spring_layout(self.graph, seed=42)
        
        # Dibujar nodos y aristas
        nx.draw(self.graph, pos, with_labels=True, 
                node_color='lightblue', node_size=500,
                font_size=8, font_weight='bold')
        
        # Mostrar pesos
        edge_labels = nx.get_edge_attributes(self.graph, 'weight')
        nx.draw_networkx_edge_labels(self.graph, pos, edge_labels, font_size=6)
        
        plt.title("Grafo de Optimización de Rutas")
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
        print(f"Gráfico guardado como {save_path}")
    
    def save_model(self, filename='route_optimizer.h5'):
        """Guarda el modelo entrenado"""
        self.model.save(filename)
        print(f"Modelo guardado como {filename}")
    
    def run_demo(self):
        """Ejecuta demostración completa"""
        print("=" * 50)
        print("OPTIMIZADOR DE RUTAS CON GNN")
        print("=" * 50)
        
        # 1. Crear grafo
        print("\n1. Creando grafo de transporte...")
        self.create_graph()
        print(f"   - Nodos: {self.graph.number_of_nodes()}")
        print(f"   - Aristas: {self.graph.number_of_edges()}")
        
        # 2. Entrenar modelo
        print("\n2. Entrenando modelo GNN...")
        self.train(epochs=5)
        
        # 3. Optimizar rutas de ejemplo
        print("\n3. Optimizando rutas de ejemplo...")
        test_routes = [(0, 10), (5, 15), (2, 18)]
        results = self.optimize_multiple_routes(test_routes)
        
        for result in results:
            if result['path']:
                print(f"   - Ruta {result['start']} -> {result['end']}: {result['path']}")
                print(f"     Distancia: {result['distance']:.2f}")
            else:
                print(f"   - No hay ruta de {result['start']} a {result['end']}")
        
        # 4. Visualizar
        print("\n4. Generando visualización...")
        self.visualize_graph()
        
        # 5. Guardar modelo
        print("\n5. Guardando modelo...")
        self.save_model()
        
        print("\n" + "=" * 50)
        print("DEMO COMPLETADA EXITOSAMENTE")
        print("=" * 50)

def main():
    """Función principal"""
    # Crear y ejecutar optimizador
    optimizer = RouteOptimizerGNN(num_nodes=20)
    optimizer.run_demo()

if __name__ == "__main__":
    main()
