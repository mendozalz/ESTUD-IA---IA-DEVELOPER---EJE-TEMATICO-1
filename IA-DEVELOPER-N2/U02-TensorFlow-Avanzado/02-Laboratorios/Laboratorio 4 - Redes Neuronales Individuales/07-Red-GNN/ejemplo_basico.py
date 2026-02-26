import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

print("RED GNN - CLASIFICACION DE NODOS EN RED SOCIAL")
print("=" * 60)

# --- 1. Crear dataset de grafo ---
print("Creando dataset de red social...")
# Crear grafo de red social simplificado
num_nodes = 50
num_features = 8
num_classes = 3

# Generar características de nodos (edad, intereses, etc.)
np.random.seed(42)
node_features = np.random.randn(num_nodes, num_features)

# Generar etiquetas (0: estudiante, 1: profesional, 2: empresario)
labels = np.random.choice(num_classes, num_nodes, p=[0.4, 0.4, 0.2])

# Crear matriz de adyacencia (conexiones en red social)
adjacency = np.zeros((num_nodes, num_nodes))

# Conectar nodos basado en similitud y probabilidad
for i in range(num_nodes):
    for j in range(i+1, num_nodes):
        # Probabilidad de conexión basada en similitud de características
        similarity = np.dot(node_features[i], node_features[j]) / (np.linalg.norm(node_features[i]) * np.linalg.norm(node_features[j]) + 1e-8)
        prob = 0.3 + 0.5 * max(0, similarity)  # Mayor probabilidad si son similares
        
        if np.random.random() < prob:
            adjacency[i, j] = 1
            adjacency[j, i] = 1

# Añadir auto-conexiones
np.fill_diagonal(adjacency, 1)

print(f"   Nodos: {num_nodes}")
print(f"   Características por nodo: {num_features}")
print(f"   Clases: {num_classes}")
print(f"   Conexiones totales: {np.sum(adjacency) // 2}")

# --- 2. Preparar datos para entrenamiento ---
print("\nPreparando datos para entrenamiento...")
# Dividir en entrenamiento y prueba
train_mask = np.zeros(num_nodes, dtype=bool)
test_mask = np.zeros(num_nodes, dtype=bool)

# 70% para entrenamiento, 30% para prueba
train_indices = np.random.choice(num_nodes, int(0.7 * num_nodes), replace=False)
train_mask[train_indices] = True
test_mask[~train_mask] = True

X_train = node_features[train_mask]
y_train = labels[train_mask]
X_test = node_features[test_mask]
y_test = labels[test_mask]

print(f"   Nodos de entrenamiento: {np.sum(train_mask)}")
print(f"   Nodos de prueba: {np.sum(test_mask)}")

# --- 3. Definir capa GNN personalizada ---
print("\nConstruyendo capa GNN...")

class GraphConvLayer(layers.Layer):
    def __init__(self, output_dim, activation='relu', **kwargs):
        super(GraphConvLayer, self).__init__(**kwargs)
        self.output_dim = output_dim
        self.activation = tf.keras.activations.get(activation)
    
    def build(self, input_shape):
        # input_shape: [batch_size, num_nodes, num_features]
        num_features = input_shape[-1]
        
        # Pesos para transformación de características
        self.kernel = self.add_weight(
            name='kernel',
            shape=(num_features, self.output_dim),
            initializer='glorot_uniform',
            trainable=True
        )
        
        # Bias
        self.bias = self.add_weight(
            name='bias',
            shape=(self.output_dim,),
            initializer='zeros',
            trainable=True
        )
        
        super(GraphConvLayer, self).build(input_shape)
    
    def call(self, inputs, adjacency):
        # inputs: [batch_size, num_nodes, num_features]
        # adjacency: [num_nodes, num_nodes]
        
        batch_size = tf.shape(inputs)[0]
        num_nodes = tf.shape(inputs)[1]
        
        # Transformación lineal
        transformed = tf.matmul(inputs, self.kernel) + self.bias
        
        # Propagación en grafo: A * X * W
        # Expandir adjacency para batch
        adj_expanded = tf.expand_dims(adjacency, axis=0)
        adj_expanded = tf.tile(adj_expanded, [batch_size, 1, 1])
        
        # Normalización de grado
        degree = tf.reduce_sum(adjacency, axis=1, keepdims=True)
        degree_inv_sqrt = tf.pow(degree + 1e-8, -0.5)
        norm_adj = tf.matmul(tf.matmul(degree_inv_sqrt, adjacency), degree_inv_sqrt)
        norm_adj_expanded = tf.expand_dims(norm_adj, axis=0)
        norm_adj_expanded = tf.tile(norm_adj_expanded, [batch_size, 1, 1])
        
        # Convolución en grafo
        propagated = tf.matmul(norm_adj_expanded, transformed)
        
        return self.activation(propagated)
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.output_dim)

# --- 4. Definir el modelo GNN ---
print("Construyendo el modelo GNN...")

def build_gnn_model(num_nodes, num_features, num_classes, hidden_dims=[64, 32]):
    inputs = layers.Input(shape=(num_nodes, num_features))
    
    # Capas GNN
    x = inputs
    for hidden_dim in hidden_dims:
        x = GraphConvLayer(hidden_dim, activation='relu')(x, adjacency)
        x = layers.Dropout(0.5)(x)
    
    # Capa de clasificación
    x = GraphConvLayer(num_classes, activation='softmax')(x, adjacency)
    
    model = models.Model(inputs=inputs, outputs=x)
    return model

# Crear modelo
model = build_gnn_model(num_nodes, num_features, num_classes)

# Mostrar arquitectura
model.summary()

# --- 5. Compilar y entrenar ---
print("\nCompilando y entrenando el modelo...")
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Preparar datos para entrenamiento
X_batch = np.expand_dims(node_features, axis=0)  # Añadir dimensión batch
y_batch = tf.keras.utils.to_categorical(labels, num_classes)
y_batch = np.expand_dims(y_batch, axis=0)  # Añadir dimensión batch

# Máscara de entrenamiento
train_mask_batch = np.expand_dims(train_mask, axis=0)

# Callbacks
callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=10)
]

# Función de pérdida personalizada para nodos específicos
def masked_loss(y_true, y_pred):
    mask = tf.cast(train_mask_batch, dtype=tf.float32)
    loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
    masked_loss = loss * mask
    return tf.reduce_sum(masked_loss) / tf.reduce_sum(mask)

def masked_accuracy(y_true, y_pred):
    mask = tf.cast(train_mask_batch, dtype=tf.float32)
    correct = tf.equal(tf.argmax(y_true, axis=-1), tf.argmax(y_pred, axis=-1))
    correct = tf.cast(correct, dtype=tf.float32)
    masked_correct = correct * mask
    return tf.reduce_sum(masked_correct) / tf.reduce_sum(mask)

model.compile(
    optimizer='adam',
    loss=masked_loss,
    metrics=[masked_accuracy]
)

history = model.fit(
    X_batch, y_batch,
    epochs=100,
    batch_size=1,
    validation_split=0.0,
    callbacks=callbacks,
    verbose=1
)

# --- 6. Evaluar el modelo ---
print("\nEvaluando el modelo...")
predictions = model.predict(X_batch, verbose=0)[0]  # Quitar dimensión batch

# Evaluar en conjunto de prueba
test_predictions = np.argmax(predictions[test_mask], axis=1)
test_labels = labels[test_mask]
test_accuracy = np.mean(test_predictions == test_labels)

print(f"   Precisión en prueba: {test_accuracy:.4f}")

# Evaluar en conjunto de entrenamiento
train_predictions = np.argmax(predictions[train_mask], axis=1)
train_labels = labels[train_mask]
train_accuracy = np.mean(train_predictions == train_labels)

print(f"   Precisión en entrenamiento: {train_accuracy:.4f}")

# --- 7. Visualizar el grafo ---
print("\nVisualizando el grafo y predicciones...")

# Crear grafo con NetworkX para visualización
G = nx.from_numpy_array(adjacency)

# Preparar colores para nodos según predicciones
node_colors = []
for i in range(num_nodes):
    if test_mask[i]:
        # Nodos de prueba: colorear según predicción
        node_colors.append(predictions[i].argmax())
    else:
        # Nodos de entrenamiento: colorear según etiqueta real
        node_colors.append(labels[i])

# Visualizar grafo
plt.figure(figsize=(15, 10))

plt.subplot(2, 2, 1)
pos = nx.spring_layout(G, seed=42)
nx.draw(G, pos, node_color=node_colors, node_size=300, cmap='viridis', with_labels=True)
plt.title(f'Grafo de Red Social (Precision Test: {test_accuracy:.3f})')

# --- 8. Visualizar características aprendidas ---
plt.subplot(2, 2, 2)
# Extraer características de una capa intermedia
intermediate_model = models.Model(
    inputs=model.input,
    outputs=model.layers[1].output  # Primera capa GNN
)
intermediate_features = intermediate_model.predict(X_batch, verbose=0)[0]

# Reducir dimensionalidad para visualización (PCA)
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
features_2d = pca.fit_transform(intermediate_features)

# Graficar características 2D
for class_idx in range(num_classes):
    mask = labels == class_idx
    plt.scatter(features_2d[mask, 0], features_2d[mask, 1], 
               label=f'Clase {class_idx}', alpha=0.7)
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.title('Caracteristicas Aprendidas (PCA)')
plt.legend()
plt.grid(True)

# --- 9. Visualizar entrenamiento ---
plt.subplot(2, 2, 3)
plt.plot(history.history['masked_accuracy'], label='Entrenamiento')
plt.title('Precision durante Entrenamiento')
plt.xlabel('Epoca')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

plt.subplot(2, 2, 4)
plt.plot(history.history['loss'], label='Entrenamiento')
plt.title('Perdida durante Entrenamiento')
plt.xlabel('Epoca')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# --- 10. Análisis de nodos mal clasificados ---
print("\nAnalizando nodos mal clasificados...")
misclassified = test_mask & (test_predictions != test_labels)
misclassified_indices = np.where(misclassified)[0]

print(f"   Nodos mal clasificados: {len(misclassified_indices)} de {np.sum(test_mask)}")

if len(misclassified_indices) > 0:
    print("   Ejemplos de nodos mal clasificados:")
    for i in misclassified_indices[:5]:  # Mostrar primeros 5
        print(f"   • Nodo {i}: Real={test_labels[i]}, Predicho={test_predictions[i]}")

# --- 11. Guardar el modelo ---
model.save('gnn_social_network.h5')
print("\nModelo guardado como 'gnn_social_network.h5'")

print("\n✅ EJERCICIO COMPLETADO")
print("=" * 60)
print("🎯 RESULTADOS:")
print(f"   • Precisión en entrenamiento: {train_accuracy:.4f}")
print(f"   • Precisión en prueba: {test_accuracy:.4f}")
print(f"   • Total de parámetros: {model.count_params():,}")
print(f"   • Épocas entrenadas: {len(history.history['loss'])}")
print("=" * 60)
