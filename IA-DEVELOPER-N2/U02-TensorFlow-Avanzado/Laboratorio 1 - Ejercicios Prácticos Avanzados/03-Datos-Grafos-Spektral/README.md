# Ejercicio 3: Datos de Grafos con Spektral

## 🕸️ Caso de Uso: Detección de Fraudes en Transacciones Bancarias

### Contexto Empresarial
En el sector financiero, detectar fraudes en transacciones requiere analizar patrones relacionales entre cuentas, comercios y dispositivos. Las transacciones fraudulentas a menudo forman patrones complejos en el grafo de transacciones que no son evidentes en análisis individuales.

### Problema a Resolver
- **Detección**: Identificar transacciones fraudulentas en tiempo real
- **Patrones**: Descubrir redes de fraude coordinadas
- **Velocidad**: Procesamiento de miles de transacciones por segundo
- **ROI**: Reducción del 25% en pérdidas por fraude

## 🎯 Objetivos de Aprendizaje

1. **Procesamiento de grafos**:
   - Construir grafos a partir de datos transaccionales
   - Usar GraphConv y GraphAttention para propagar información

2. **Detección de anomalías**:
   - Entrenar un modelo para clasificar transacciones como fraudulentas o legítimas
   - Implementar técnicas de detección de anomalías en grafos

3. **Integración con sistemas de pagos**:
   - Desplegar el modelo en un microservicio que procese transacciones en tiempo real
   - Implementar API RESTful para integración con sistemas bancarios

## 📊 Dataset

### Opción 1: Dataset Público - IEEE-CIS Fraud Detection
- **Descripción**: Transacciones con etiquetas de fraude
- **Descarga**: [Kaggle IEEE-CIS Fraud Detection](https://www.kaggle.com/c/ieee-fraud-detection)
- **Características**: 394 características, transacciones de tarjeta

### Opción 2: Datos Sintéticos (para desarrollo)
- **Generación**: Simular grafo de transacciones con patrones de fraude
- **Ventajas**: Control sobre tipos y patrones de fraude

## 🛠️ Implementación

### Paso 1: Configuración del Entorno
```python
import tensorflow as tf
import spektral as sp
from spektral.layers import GraphConv, GlobalAvgPool
import networkx as nx
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from fastapi import FastAPI
import uvicorn
import json
```

### Paso 2: Generación de Datos Sintéticos
```python
def generate_transaction_graph(num_accounts=1000, num_transactions=5000, fraud_ratio=0.05):
    """
    Generar grafo de transacciones sintéticas con patrones de fraude
    
    Args:
        num_accounts: Número de cuentas bancarias
        num_transactions: Número total de transacciones
        fraud_ratio: Proporción de transacciones fraudulentas
    """
    # Crear grafo vacío
    G = nx.DiGraph()
    
    # Añadir nodos (cuentas)
    for i in range(num_accounts):
        account_type = np.random.choice(['personal', 'business', 'merchant'])
        account_age = np.random.randint(30, 3650)  # días
        credit_score = np.random.randint(300, 850)
        
        G.add_node(i, 
                  account_type=account_type,
                  account_age=account_age,
                  credit_score=credit_score,
                  is_merchant=account_type == 'merchant')
    
    # Generar transacciones legítimas
    legitimate_transactions = int(num_transactions * (1 - fraud_ratio))
    for i in range(legitimate_transactions):
        # Seleccionar cuentas aleatorias
        sender = np.random.randint(0, num_accounts)
        receiver = np.random.randint(0, num_accounts)
        
        # Evitar auto-transacciones
        while receiver == sender:
            receiver = np.random.randint(0, num_accounts)
        
        # Generar características de transacción
        amount = np.random.lognormal(mean=3, sigma=1)  # Distribución log-normal
        amount = np.clip(amount, 1, 10000)
        
        time_of_day = np.random.randint(0, 24)
        day_of_week = np.random.randint(0, 7)
        
        # Añadir arista de transacción
        transaction_id = f"legit_{i}"
        G.add_edge(sender, receiver,
                  transaction_id=transaction_id,
                  amount=amount,
                  time_of_day=time_of_day,
                  day_of_week=day_of_week,
                  is_fraud=0,
                  device_id=f"device_{np.random.randint(0, 100)}")
    
    # Generar transacciones fraudulentas con patrones específicos
    fraud_transactions = num_transactions - legitimate_transactions
    
    # Patrón 1: Transacciones rápidas entre cuentas relacionadas
    fraud_ring_size = 10
    fraud_accounts = np.random.choice(num_accounts, fraud_ring_size, replace=False)
    
    for i in range(fraud_transactions // 2):
        sender = np.random.choice(fraud_accounts)
        receiver = np.random.choice(fraud_accounts)
        
        while receiver == sender:
            receiver = np.random.choice(fraud_accounts)
        
        # Transacciones fraudulentas suelen tener montos específicos
        amount = np.random.choice([999.99, 1999.99, 4999.99])
        
        transaction_id = f"fraud_ring_{i}"
        G.add_edge(sender, receiver,
                  transaction_id=transaction_id,
                  amount=amount,
                  time_of_day=np.random.randint(2, 6),  # Horas inusuales
                  day_of_week=np.random.randint(0, 2),  # Fines de semana
                  is_fraud=1,
                  device_id=f"device_{np.random.randint(0, 5)}")  # Mismos dispositivos
    
    # Patrón 2: Transacciones de alto monto desde cuentas nuevas
    new_accounts = np.random.choice(num_accounts, 20, replace=False)
    
    for i in range(fraud_transactions // 2):
        sender = np.random.choice(new_accounts)
        receiver = np.random.randint(0, num_accounts)
        
        while receiver == sender:
            receiver = np.random.randint(0, num_accounts)
        
        # Altos montos desde cuentas nuevas
        amount = np.random.uniform(5000, 15000)
        
        transaction_id = f"fraud_new_{i}"
        G.add_edge(sender, receiver,
                  transaction_id=transaction_id,
                  amount=amount,
                  time_of_day=np.random.randint(0, 24),
                  day_of_week=np.random.randint(0, 7),
                  is_fraud=1,
                  device_id=f"device_{np.random.randint(0, 10)}")
    
    return G

def extract_graph_features(G):
    """Extraer características de nodos y aristas del grafo"""
    
    # Características de nodos
    node_features = []
    node_labels = []
    
    for node in G.nodes():
        features = [
            G.nodes[node]['account_age'],
            G.nodes[node]['credit_score'],
            1 if G.nodes[node]['account_type'] == 'business' else 0,
            1 if G.nodes[node]['account_type'] == 'merchant' else 0,
            G.degree(node),  # Número de transacciones
            nx.degree_centrality(G)[node],  # Centralidad de grado
        ]
        node_features.append(features)
    
    # Características de aristas (transacciones)
    edge_features = []
    edge_labels = []
    
    for edge in G.edges(data=True):
        features = [
            edge[2]['amount'],
            edge[2]['time_of_day'],
            edge[2]['day_of_week'],
            1 if edge[2]['time_of_day'] < 6 or edge[2]['time_of_day'] > 22 else 0,  # Horario inusual
            edge[2]['amount'] / 1000,  # Monto normalizado
        ]
        edge_features.append(features)
        edge_labels.append(edge[2]['is_fraud'])
    
    return np.array(node_features), np.array(edge_features), np.array(edge_labels)
```

### Paso 3: Construcción del Modelo GNN
```python
class FraudDetectionGNN(tf.keras.Model):
    """Modelo GNN para detección de fraudes en transacciones"""
    
    def __init__(self, hidden_units=[64, 32], dropout_rate=0.3):
        super().__init__()
        self.hidden_units = hidden_units
        self.dropout_rate = dropout_rate
        
        # Capas de convolución de grafos
        self.gcn_layers = []
        for i, units in enumerate(hidden_units):
            self.gcn_layers.append(GraphConv(units, activation='relu'))
        
        # Capas de clasificación
        self.global_pool = GlobalAvgPool()
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.classifier = tf.keras.layers.Dense(1, activation='sigmoid')
    
    def call(self, inputs, training=False):
        """
        Forward pass del modelo
        
        Args:
            inputs: Tupla (X, A) donde X son features de nodos y A es matriz de adyacencia
            training: Booleano para modo de entrenamiento
        """
        X, A = inputs
        
        # Pasar por capas GCN
        for i, gcn_layer in enumerate(self.gcn_layers):
            X = gcn_layer([X, A])
            if training:
                X = self.dropout(X)
        
        # Global pooling y clasificación
        X = self.global_pool(X)
        output = self.classifier(X)
        
        return output
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "hidden_units": self.hidden_units,
            "dropout_rate": self.dropout_rate
        })
        return config

def build_advanced_gnn_model(input_dim, hidden_units=[128, 64, 32], dropout_rate=0.3):
    """Construir modelo GNN avanzado con atención"""
    
    # Inputs
    X_input = tf.keras.layers.Input(shape=(input_dim,), name='node_features')
    A_input = tf.keras.layers.Input(shape=(None,), sparse=True, name='adjacency_matrix')
    
    # Primera capa GCN
    x = GraphConv(hidden_units[0], activation='relu')([X_input, A_input])
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    
    # Capas adicionales
    for units in hidden_units[1:]:
        x = GraphConv(units, activation='relu')([x, A_input])
        x = tf.keras.layers.Dropout(dropout_rate)(x)
    
    # Global pooling
    x = GlobalAvgPool()(x)
    
    # Capas densas finales
    x = tf.keras.layers.Dense(16, activation='relu')(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    output = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    
    model = tf.keras.Model(inputs=[X_input, A_input], outputs=output)
    return model
```

### Paso 4: Preparación de Datos
```python
def prepare_graph_data(G):
    """Preparar datos del grafo para entrenamiento"""
    
    # Extraer características
    node_features, edge_features, edge_labels = extract_graph_features(G)
    
    # Normalizar características
    scaler = StandardScaler()
    node_features_scaled = scaler.fit_transform(node_features)
    edge_features_scaled = scaler.fit_transform(edge_features)
    
    # Construir matriz de adyacencia
    A = nx.adjacency_matrix(G).toarray()
    
    # Normalizar matriz de adyacencia
    A = sp.utils.normalized_adjacency_matrix(A)
    
    # Dividir datos en entrenamiento/prueba
    # Para grafos, dividimos las aristas (transacciones)
    train_edges, test_edges = train_test_split(
        range(len(edge_labels)), 
        test_size=0.2, 
        stratify=edge_labels,
        random_state=42
    )
    
    # Crear datasets de TensorFlow
    def create_dataset(edge_indices):
        edge_features_subset = edge_features_scaled[edge_indices]
        edge_labels_subset = edge_labels[edge_indices]
        
        return tf.data.Dataset.from_tensor_slices((
            (node_features_scaled, A),  # Features del grafo (compartidas)
            edge_features_subset,           # Features de la arista específica
            edge_labels_subset             # Etiqueta de la arista
        ))
    
    train_dataset = create_dataset(train_edges).batch(32).prefetch(tf.data.AUTOTUNE)
    test_dataset = create_dataset(test_edges).batch(32).prefetch(tf.data.AUTOTUNE)
    
    return (node_features_scaled, A), edge_features_scaled, edge_labels, train_dataset, test_dataset, scaler
```

### Paso 5: Entrenamiento y Evaluación
```python
def train_fraud_detection_model(train_dataset, test_dataset, graph_data, epochs=50):
    """Entrenar modelo de detección de fraudes"""
    
    (node_features, A), _, _, _, _, _ = graph_data
    
    # Construir modelo
    model = build_advanced_gnn_model(
        input_dim=node_features.shape[1],
        hidden_units=[128, 64, 32],
        dropout_rate=0.3
    )
    
    # Compilar modelo
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), tf.keras.metrics.AUC()]
    )
    
    # Callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True, monitor='val_auc'),
        tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5, monitor='val_loss'),
        tf.keras.callbacks.ModelCheckpoint('best_fraud_model.h5', save_best_only=True, monitor='val_auc')
    ]
    
    # Entrenar modelo
    history = model.fit(
        train_dataset,
        validation_data=test_dataset,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )
    
    return model, history

def evaluate_model(model, test_dataset):
    """Evaluar modelo de detección de fraudes"""
    
    # Realizar predicciones
    predictions = []
    true_labels = []
    
    for batch in test_dataset:
        (X, A), edge_features, labels = batch
        pred = model([X, A], training=False)
        predictions.extend(pred.numpy().flatten())
        true_labels.extend(labels.numpy())
    
    predictions = np.array(predictions)
    true_labels = np.array(true_labels)
    
    # Calcular métricas
    from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
    
    # Convertir a clases binarias
    pred_classes = (predictions > 0.5).astype(int)
    
    print("📊 MÉTRICAS DE EVALUACIÓN")
    print("=" * 50)
    print(f"AUC-ROC: {roc_auc_score(true_labels, predictions):.4f}")
    print("\nReporte de Clasificación:")
    print(classification_report(true_labels, pred_classes, target_names=['Legítimo', 'Fraude']))
    
    # Matriz de confusión
    cm = confusion_matrix(true_labels, pred_classes)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Legítimo', 'Fraude'], 
                yticklabels=['Legítimo', 'Fraude'])
    plt.title('Matriz de Confusión')
    plt.ylabel('Verdadero')
    plt.xlabel('Predicho')
    plt.show()
    
    return predictions, true_labels
```

### Paso 6: API RESTful con FastAPI
```python
from pydantic import BaseModel
from typing import List, Optional

class Transaction(BaseModel):
    """Modelo de datos para transacción"""
    sender_id: int
    receiver_id: int
    amount: float
    time_of_day: int
    day_of_week: int
    device_id: str

class FraudPredictionResponse(BaseModel):
    """Respuesta de predicción de fraude"""
    is_fraud: bool
    fraud_probability: float
    risk_score: float
    explanation: str

class FraudDetectionAPI:
    """API para detección de fraudes en tiempo real"""
    
    def __init__(self, model_path, scaler_path):
        self.model = tf.keras.models.load_model(model_path)
        self.scaler = joblib.load(scaler_path)
        self.transaction_history = []
        self.graph_cache = {}
    
    def load_model(self, model_path, scaler_path):
        """Cargar modelo y scaler"""
        self.model = tf.keras.models.load_model(model_path)
        self.scaler = joblib.load(scaler_path)
        print("Modelo y scaler cargados exitosamente")
    
    def predict_transaction(self, transaction: Transaction) -> FraudPredictionResponse:
        """Predecir si una transacción es fraudulenta"""
        
        # Extraer características de la transacción
        features = np.array([[
            transaction.amount,
            transaction.time_of_day,
            transaction.day_of_week,
            1 if transaction.time_of_day < 6 or transaction.time_of_day > 22 else 0,
            transaction.amount / 1000,
        ]])
        
        # Normalizar características
        features_scaled = self.scaler.transform(features)
        
        # Preparar datos del grafo (simplificado para demostración)
        # En producción, esto incluiría el estado actual del grafo
        node_features = np.random.rand(100, 6)  # Placeholder
        A = np.eye(100)  # Placeholder
        
        # Realizar predicción
        prediction = self.model.predict([node_features, A], verbose=0)[0][0]
        
        # Calcular risk score
        risk_score = self.calculate_risk_score(transaction, prediction)
        
        # Generar explicación
        explanation = self.generate_explanation(transaction, prediction, risk_score)
        
        return FraudPredictionResponse(
            is_fraud=prediction > 0.5,
            fraud_probability=float(prediction),
            risk_score=risk_score,
            explanation=explanation
        )
    
    def calculate_risk_score(self, transaction: Transaction, fraud_probability: float) -> float:
        """Calcular score de riesgo basado en múltiples factores"""
        
        risk_factors = []
        
        # Factor 1: Probabilidad del modelo
        risk_factors.append(fraud_probability)
        
        # Factor 2: Monto de la transacción
        if transaction.amount > 5000:
            risk_factors.append(0.8)
        elif transaction.amount > 1000:
            risk_factors.append(0.5)
        else:
            risk_factors.append(0.2)
        
        # Factor 3: Hora del día
        if transaction.time_of_day < 6 or transaction.time_of_day > 22:
            risk_factors.append(0.7)
        else:
            risk_factors.append(0.3)
        
        # Factor 4: Día de la semana
        if transaction.day_of_week in [0, 6]:  # Fin de semana
            risk_factors.append(0.6)
        else:
            risk_factors.append(0.4)
        
        # Combinar factores
        return np.mean(risk_factors)
    
    def generate_explanation(self, transaction: Transaction, fraud_probability: float, risk_score: float) -> str:
        """Generar explicación de la predicción"""
        
        explanations = []
        
        if fraud_probability > 0.7:
            explanations.append("Alta probabilidad de fraude detectada por el modelo")
        elif fraud_probability > 0.4:
            explanations.append("Probabilidad moderada de actividad sospechosa")
        else:
            explanations.append("Transacción parece legítima")
        
        if transaction.amount > 5000:
            explanations.append("Monto elevado para transacción estándar")
        
        if transaction.time_of_day < 6 or transaction.time_of_day > 22:
            explanations.append("Transacción en horario inusual")
        
        if transaction.day_of_week in [0, 6]:
            explanations.append("Transacción en fin de semana")
        
        return " | ".join(explanations)

# Crear aplicación FastAPI
app = FastAPI(title="Fraud Detection API", version="1.0.0")

# Inicializar sistema de detección
fraud_detector = FraudDetectionAPI("best_fraud_model.h5", "fraud_scaler.pkl")

@app.post("/predict", response_model=FraudPredictionResponse)
async def predict_fraud(transaction: Transaction):
    """Endpoint para predecir fraude en transacción"""
    try:
        prediction = fraud_detector.predict_transaction(transaction)
        return prediction
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en predicción: {str(e)}")

@app.get("/health")
async def health_check():
    """Endpoint de verificación de salud"""
    return {"status": "healthy", "model_loaded": fraud_detector.model is not None}

@app.get("/stats")
async def get_stats():
    """Obtener estadísticas del sistema"""
    return {
        "total_transactions": len(fraud_detector.transaction_history),
        "fraud_rate": 0.05,  # Placeholder
        "model_version": "1.0.0"
    }
```

## 🧪 Ejecución Completa

```python
def main():
    """Función principal para ejecutar el ejercicio completo"""
    
    print("🕸️ EJERCICIO 3: DETECCIÓN DE FRAUDES CON GRAFOS")
    print("=" * 60)
    
    # 1. Generar grafo de transacciones
    print("📊 Generando grafo de transacciones...")
    G = generate_transaction_graph(num_accounts=1000, num_transactions=5000, fraud_ratio=0.05)
    
    print(f"   Nodos (cuentas): {G.number_of_nodes()}")
    print(f"   Aristas (transacciones): {G.number_of_edges()}")
    print(f"   Transacciones fraudulentas: {sum(1 for _, _, data in G.edges(data=True) if data['is_fraud'])}")
    
    # 2. Visualizar grafo
    print("\n🎨 Visualizando estructura del grafo...")
    plt.figure(figsize=(12, 8))
    
    # Subgrafo para visualización (primeros 50 nodos)
    subgraph_nodes = list(G.nodes())[:50]
    subgraph = G.subgraph(subgraph_nodes)
    
    pos = nx.spring_layout(subgraph, k=1, iterations=50)
    
    # Colorear nodos por tipo
    node_colors = []
    for node in subgraph.nodes():
        if G.nodes[node]['account_type'] == 'merchant':
            node_colors.append('red')
        elif G.nodes[node]['account_type'] == 'business':
            node_colors.append('blue')
        else:
            node_colors.append('green')
    
    # Colorear aristas por tipo de transacción
    edge_colors = []
    edge_widths = []
    for edge in subgraph.edges(data=True):
        if edge[2]['is_fraud']:
            edge_colors.append('red')
            edge_widths.append(2)
        else:
            edge_colors.append('gray')
            edge_widths.append(0.5)
    
    nx.draw(subgraph, pos, node_color=node_colors, edge_color=edge_colors, 
            width=edge_widths, with_labels=False, node_size=50, alpha=0.7)
    plt.title('Subgrafo de Transacciones (50 nodos)')
    plt.show()
    
    # 3. Preparar datos
    print("\n🔄 Preparando datos del grafo...")
    graph_data = prepare_graph_data(G)
    (node_features, A), edge_features, edge_labels, train_dataset, test_dataset, scaler = graph_data
    
    print(f"   Features de nodos: {node_features.shape}")
    print(f"   Features de aristas: {edge_features.shape}")
    print(f"   Transacciones de entrenamiento: {len(train_dataset) * 32}")
    print(f"   Transacciones de prueba: {len(test_dataset) * 32}")
    
    # 4. Entrenar modelo
    print("\n🚀 Entrenando modelo GNN...")
    model, history = train_fraud_detection_model(train_dataset, test_dataset, graph_data, epochs=30)
    
    # 5. Evaluar modelo
    print("\n📈 Evaluando modelo...")
    predictions, true_labels = evaluate_model(model, test_dataset)
    
    # 6. Probar API
    print("\n🌐 Probando API REST...")
    
    # Crear transacción de prueba
    test_transaction = Transaction(
        sender_id=123,
        receiver_id=456,
        amount=9999.99,
        time_of_day=3,
        day_of_week=0,
        device_id="device_001"
    )
    
    # Realizar predicción
    prediction_result = fraud_detector.predict_transaction(test_transaction)
    
    print("   Resultado de la API:")
    print(f"     ¿Es fraude?: {prediction_result.is_fraud}")
    print(f"     Probabilidad: {prediction_result.fraud_probability:.4f}")
    print(f"     Score de riesgo: {prediction_result.risk_score:.4f}")
    print(f"     Explicación: {prediction_result.explanation}")
    
    # 7. Guardar modelo y scaler
    print("\n💾 Guardando modelo y scaler...")
    model.save('fraud_detection_gnn.h5')
    joblib.dump(scaler, 'fraud_scaler.pkl')
    
    print("\n✅ EJERCICIO COMPLETADO")
    print("=" * 60)
    print("🎯 RESULTADOS:")
    print(f"   • AUC-ROC: {roc_auc_score(true_labels, predictions):.4f}")
    print(f"   • Modelo guardado: fraud_detection_gnn.h5")
    print(f"   • API lista en: http://localhost:8000/docs")
    print(f"   • ROI estimado: 25% reducción en pérdidas por fraude")
    print("=" * 60)
    
    # Instrucciones para ejecutar API
    print("\n🚀 Para ejecutar la API:")
    print("   uvicorn ejercicio_completo:app --host 0.0.0.0 --port 8000")

if __name__ == "__main__":
    main()
```

## 📊 Métricas de Evaluación

### Métricas de Clasificación
- **AUC-ROC**: Área bajo la curva ROC (>0.95)
- **Precision**: Verdaderos positivos / (Verdaderos positivos + Falsos positivos)
- **Recall**: Verdaderos positivos / (Verdaderos positivos + Falsos negativos)
- **F1-Score**: Media armónica de precision y recall

### Métricas de Producción
- **Latencia**: Tiempo de predicción (<100ms)
- **Throughput**: Transacciones procesadas por segundo (>1000/s)
- **False Positive Rate**: Tasa de falsos positivos (<1%)

## 🚀 Desafíos Adicionales

### Desafío 1: Grafos Dinámicos
- Implementar actualización incremental del grafo
- Usar GraphSAGE para grafos muy grandes
- Manejar streaming de transacciones en tiempo real

### Desafío 2: Explicabilidad
- Implementar GNNExplainer para explicaciones
- Visualizar subgrafos que contribuyen a la predicción
- Generar reportes de auditoría

### Desafío 3: Integración Real
- Conectar con base de datos transaccional real
- Implementar sistema de alertas en tiempo real
- Añadir dashboard con Grafana

## 📝 Entrega Requerida

1. **Código fuente** completo y funcional
2. **Informe técnico** con:
   - Análisis del grafo de transacciones
   - Arquitectura del modelo GNN
   - Resultados experimentales
3. **API RESTful** documentada y funcionando
4. **Dashboard** de monitoreo de fraudes

## 🔗 Recursos Adicionales

- [Spektral Documentation](https://graphneural.network/)
- [Graph Neural Networks Review](https://arxiv.org/abs/1812.08434)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [NetworkX Graph Analysis](https://networkx.org/)
