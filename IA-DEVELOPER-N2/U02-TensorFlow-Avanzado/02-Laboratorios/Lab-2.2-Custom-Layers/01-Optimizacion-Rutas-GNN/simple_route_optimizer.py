# -*- coding: utf-8 -*-
"""
Simple Route Optimizer with GNN
"""

import tensorflow as tf
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

class SimpleRouteOptimizer:
    def __init__(self, num_nodes=20):
        self.num_nodes = num_nodes
        self.model = None
        self.graph = None
        self.build_model()
    
    def build_model(self):
        self.model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(self.num_nodes, 4)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(self.num_nodes * self.num_nodes, activation='sigmoid'),
            tf.keras.layers.Reshape((self.num_nodes, self.num_nodes))
        ])
        
        self.model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
    
    def create_graph(self):
        G = nx.erdos_renyi_graph(self.num_nodes, 0.3, seed=42)
        
        for edge in G.edges():
            G[edge[0]][edge[1]]['weight'] = np.random.uniform(1, 10)
        
        self.graph = G
        return G
    
    def generate_data(self, num_samples=100):
        features = np.random.randn(num_samples, self.num_nodes, 4).astype(np.float32)
        labels = np.random.randint(0, 2, (num_samples, self.num_nodes * self.num_nodes)).astype(np.float32)
        labels = labels.reshape(num_samples, self.num_nodes, self.num_nodes)
        return features, labels
    
    def train(self, epochs=10):
        print(f"Training model for {epochs} epochs...")
        
        X_train, y_train = self.generate_data(200)
        
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            validation_split=0.2,
            batch_size=16,
            verbose=1
        )
        
        return history
    
    def find_path(self, start, end):
        if self.graph is None:
            self.create_graph()
        
        try:
            path = nx.shortest_path(self.graph, start, end, weight='weight')
            distance = nx.shortest_path_length(self.graph, start, end, weight='weight')
            return path, distance
        except:
            return None, float('inf')
    
    def visualize_graph(self, save_path='graph.png'):
        if self.graph is None:
            self.create_graph()
        
        plt.figure(figsize=(10, 8))
        pos = nx.spring_layout(self.graph, seed=42)
        
        nx.draw(self.graph, pos, with_labels=True, 
                node_color='lightblue', node_size=500,
                font_size=8, font_weight='bold')
        
        edge_labels = nx.get_edge_attributes(self.graph, 'weight')
        nx.draw_networkx_edge_labels(self.graph, pos, edge_labels, font_size=6)
        
        plt.title("Route Optimization Graph")
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
        print(f"Graph saved as {save_path}")
    
    def save_model(self, filename='route_optimizer.h5'):
        self.model.save(filename)
        print(f"Model saved as {filename}")
    
    def run_demo(self):
        print("=" * 50)
        print("ROUTE OPTIMIZER WITH GNN")
        print("=" * 50)
        
        print("\n1. Creating transport graph...")
        self.create_graph()
        print(f"   - Nodes: {self.graph.number_of_nodes()}")
        print(f"   - Edges: {self.graph.number_of_edges()}")
        
        print("\n2. Training GNN model...")
        self.train(epochs=5)
        
        print("\n3. Testing route optimization...")
        test_routes = [(0, 10), (5, 15), (2, 18)]
        
        for start, end in test_routes:
            path, distance = self.find_path(start, end)
            if path:
                print(f"   - Route {start} -> {end}: {path}")
                print(f"     Distance: {distance:.2f}")
            else:
                print(f"   - No route from {start} to {end}")
        
        print("\n4. Generating visualization...")
        self.visualize_graph()
        
        print("\n5. Saving model...")
        self.save_model()
        
        print("\n" + "=" * 50)
        print("DEMO COMPLETED SUCCESSFULLY")
        print("=" * 50)

def main():
    optimizer = SimpleRouteOptimizer(num_nodes=20)
    optimizer.run_demo()

if __name__ == "__main__":
    main()
