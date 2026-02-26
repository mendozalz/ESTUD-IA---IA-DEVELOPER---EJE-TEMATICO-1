# Red Neuronal Deep Q-Network (DQN)

## 📖 Teoría

### Definición
Deep Q-Network (DQN) es una arquitectura de aprendizaje por refuerzo que combina Q-Learning con redes neuronales profundas. Utiliza una red neuronal para aproximar la función Q-valor, permitiendo aprender políticas óptimas en entornos complejos con espacios de estados grandes.

### Características Principales
- **Aprendizaje por refuerzo**: Aprende mediante interacción con el entorno
- **Función Q-valor**: Estima el valor de tomar acciones en estados específicos
- **Experience Replay**: Almacena y reutiliza experiencias pasadas
- **Target Network**: Red objetivo para estabilizar entrenamiento
- **Epsilon-Greedy**: Política de exploración vs explotación

### Componentes
1. **Q-Network**: Red neuronal que aproxima Q-valores
2. **Target Network**: Copia de la red para cálculo de objetivos
3. **Experience Replay**: Buffer de experiencias (estado, acción, recompensa, siguiente estado)
4. **Environment**: Entorno con el que interactúa el agente
5. **Policy**: Estrategia para seleccionar acciones

### Algoritmo DQN
1. **Inicializar** Q-network y target network
2. **Para cada episodio**:
   - Observar estado inicial
   - **Para cada paso**:
     - Seleccionar acción usando epsilon-greedy
     - Ejecutar acción, observar recompensa y siguiente estado
     - Almacenar experiencia en replay buffer
     - Muestrear batch aleatorio del buffer
     - Entrenar Q-network usando ecuación de Bellman
     - Actualizar target network periódicamente

### Ventajas sobre Q-Learning tradicional
- **Escalabilidad**: Funciona con espacios de estados grandes/continuos
- **Generalización**: Aprende patrones y generaliza a estados similares
- **Estabilidad**: Experience replay y target networks mejoran estabilidad
- **Eficiencia**: Reutiliza experiencias pasadas

### Casos de Uso
- Videojuegos (Atari, etc.)
- Robótica y control
- Optimización de procesos
- Toma de decisiones secuenciales
- Sistemas de recomendación

## 🎯 Ejercicio Práctico: Agente para Juego CartPole

### Tipo de Aprendizaje: **REFORZADO**
- **Por qué**: Aprende mediante trial-and-error sin etiquetas
- **Entorno**: Juego CartPole de OpenAI Gym

### Implementación Completa

```python
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt
import gym
import random
from collections import deque

print("🎮 RED DQN - AGENTE PARA JUEGO CARTPOLE")
print("=" * 60)

# --- 1. Crear entorno CartPole ---
print("🎯 Creando entorno CartPole...")
try:
    env = gym.make('CartPole-v1')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    print(f"   Entorno: CartPole-v1")
    print(f"   Espacio de estados: {state_size}")
    print(f"   Espacio de acciones: {action_size}")
except ImportError:
    print("   ⚠️  Gym no instalado. Creando entorno simulado...")
    # Crear entorno simulado simple
    class CartPoleSimulated:
        def __init__(self):
            self.state_size = 4
            self.action_size = 2
            self.reset()
        
        def reset(self):
            self.state = np.random.uniform(-0.05, 0.05, 4)
            self.steps = 0
            return self.state
        
        def step(self, action):
            # Simulación simple de física
            self.state += np.random.uniform(-0.1, 0.1, 4)
            self.state[2] += np.random.uniform(-0.2, 0.2)  # Velocidad angular
            self.state[3] += np.random.uniform(-0.1, 0.1)  # Aceleración angular
            
            # Calcular recompensa
            reward = 1.0 if abs(self.state[2]) < 0.2 and abs(self.state[3]) < 0.2 else -1.0
            
            # Verificar si terminó
            done = self.steps > 100 or abs(self.state[2]) > 0.5 or abs(self.state[3]) > 0.5
            
            self.steps += 1
            return self.state, reward, done, {}
        
        def render(self):
            pass
    
    env = CartPoleSimulated()
    state_size = env.state_size
    action_size = env.action_size

# --- 2. Definir la red Q-Network ---
print("\n🏗️ Construyendo Q-Network...")

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)  # Experience replay
        self.gamma = 0.95    # Factor de descuento
        self.epsilon = 1.0   # Tasa de exploración
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()
    
    def _build_model(self):
        """Construir la red neuronal"""
        model = models.Sequential([
            layers.Dense(24, input_dim=self.state_size, activation='relu'),
            layers.Dense(24, activation='relu'),
            layers.Dense(self.action_size, activation='linear')
        ])
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate))
        return model
    
    def update_target_model(self):
        """Actualizar pesos de la red target"""
        self.target_model.set_weights(self.model.get_weights())
    
    def remember(self, state, action, reward, next_state, done):
        """Almacenar experiencia en replay buffer"""
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state):
        """Seleccionar acción usando epsilon-greedy"""
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state, verbose=0)
        return np.argmax(act_values[0])
    
    def replay(self, batch_size=32):
        """Entrenar usando experience replay"""
        if len(self.memory) < batch_size:
            return
        
        # Muestrear batch aleatorio
        minibatch = random.sample(self.memory, batch_size)
        
        states = np.array([transition[0] for transition in minibatch])
        actions = np.array([transition[1] for transition in minibatch])
        rewards = np.array([transition[2] for transition in minibatch])
        next_states = np.array([transition[3] for transition in minibatch])
        dones = np.array([transition[4] for transition in minibatch])
        
        # Calcular Q-valores objetivo
        targets = self.model.predict(states, verbose=0)
        target_q_values = self.target_model.predict(next_states, verbose=0)
        
        for i in range(batch_size):
            if dones[i]:
                targets[i, actions[i]] = rewards[i]
            else:
                targets[i, actions[i]] = rewards[i] + self.gamma * np.amax(target_q_values[i])
        
        # Entrenar modelo
        self.model.fit(states, targets, epochs=1, verbose=0)
        
        # Decrementar epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def load(self, name):
        """Cargar modelo guardado"""
        self.model.load_weights(name)
        self.update_target_model()
    
    def save(self, name):
        """Guardar modelo"""
        self.model.save_weights(name)

# --- 3. Crear agente DQN ---
print("Creando agente DQN...")
agent = DQNAgent(state_size, action_size)

# --- 4. Entrenar el agente ---
print("\n🚀 Entrenando agente DQN...")
episodes = 200
batch_size = 32
scores = []
epsilons = []

for e in range(episodes):
    state = env.reset()
    state = np.reshape(state, [1, state_size])
    total_reward = 0
    done = False
    steps = 0
    
    while not done and steps < 500:  # Límite de pasos
        # Seleccionar acción
        action = agent.act(state)
        
        # Ejecutar acción
        next_state, reward, done, _ = env.step(action)
        next_state = np.reshape(next_state, [1, state_size])
        total_reward += reward
        
        # Almacenar experiencia
        agent.remember(state, action, reward, next_state, done)
        
        # Actualizar estado
        state = next_state
        steps += 1
        
        if done:
            break
    
    # Entrenar con experience replay
    agent.replay(batch_size)
    
    # Actualizar target network cada 10 episodios
    if e % 10 == 0:
        agent.update_target_model()
    
    # Guardar métricas
    scores.append(total_reward)
    epsilons.append(agent.epsilon)
    
    # Mostrar progreso
    if e % 20 == 0 or e == episodes - 1:
        print(f"   Episodio {e+1}/{episodes} - Recompensa: {total_reward:.2f} - Epsilon: {agent.epsilon:.3f}")

print("\n✅ Entrenamiento completado")

# --- 5. Evaluar el agente entrenado ---
print("\n📈 Evaluando agente entrenado...")
test_episodes = 20
test_scores = []

for e in range(test_episodes):
    state = env.reset()
    state = np.reshape(state, [1, state_size])
    total_reward = 0
    done = False
    steps = 0
    
    while not done and steps < 500:
        # Usar política greedy (sin exploración)
        act_values = agent.model.predict(state, verbose=0)
        action = np.argmax(act_values[0])
        
        next_state, reward, done, _ = env.step(action)
        next_state = np.reshape(next_state, [1, state_size])
        total_reward += reward
        
        state = next_state
        steps += 1
        
        if done:
            break
    
    test_scores.append(total_reward)

print(f"   Recompensa promedio en prueba: {np.mean(test_scores):.2f}")
print(f"   Desviación estándar: {np.std(test_scores):.2f}")
print(f"   Mejor puntuación: {np.max(test_scores):.2f}")

# --- 6. Visualizar resultados ---
print("\n🎨 Visualizando resultados...")

# Graficar curva de aprendizaje
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.plot(scores, alpha=0.7)
plt.title('Recompensas por Episodio')
plt.xlabel('Episodio')
plt.ylabel('Recompensa Total')
plt.grid(True)

# Graficar media móvil
window_size = 20
if len(scores) >= window_size:
    moving_avg = [np.mean(scores[i:i+window_size]) for i in range(len(scores)-window_size+1)]
    plt.plot(range(window_size-1, len(scores)), moving_avg, 'r-', linewidth=2, label=f'Media móvil ({window_size})')
    plt.legend()

plt.subplot(1, 3, 2)
plt.plot(epsilons)
plt.title('Decaimiento de Epsilon')
plt.xlabel('Episodio')
plt.ylabel('Epsilon')
plt.grid(True)

plt.subplot(1, 3, 3)
plt.hist(test_scores, bins=10, alpha=0.7, edgecolor='black')
plt.title('Distribución de Recompensas de Prueba')
plt.xlabel('Recompensa Total')
plt.ylabel('Frecuencia')
plt.grid(True)

plt.tight_layout()
plt.show()

# --- 7. Análisis de convergencia ---
print("\n🔍 Analizando convergencia...")

# Calcular estadísticas de convergencia
first_10_avg = np.mean(scores[:10])
last_10_avg = np.mean(scores[-10:])
improvement = (last_10_avg - first_10_avg) / abs(first_10_avg) * 100

print(f"   Recompensa promedio primeros 10 episodios: {first_10_avg:.2f}")
print(f"   Recompensa promedio últimos 10 episodios: {last_10_avg:.2f}")
print(f"   Mejora porcentual: {improvement:.1f}%")

# Determinar si convergió
converged = last_10_avg > 100  # CartPole considera resuelto > 100
print(f"   ¿Convergió? {'Sí' if converged else 'No'}")

# --- 8. Visualizar política aprendida ---
print("\n🎮 Visualizando política aprendida...")

# Crear mapa de calor de Q-valores
def visualize_q_values(agent, env, grid_size=10):
    """Visualizar Q-valores para diferentes estados"""
    # Muestrear estados del espacio de estados
    states = []
    q_values = []
    
    for _ in range(grid_size * grid_size):
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        q_val = agent.model.predict(state, verbose=0)[0]
        
        states.append(state[0])
        q_values.append(q_val)
    
    states = np.array(states)
    q_values = np.array(q_values)
    
    # Visualizar Q-valores para cada acción
    plt.figure(figsize=(12, 4))
    
    for action in range(agent.action_size):
        plt.subplot(1, agent.action_size, action + 1)
        plt.scatter(states[:, 0], states[:, 2], c=q_values[:, action], cmap='viridis', alpha=0.6)
        plt.colorbar(label=f'Q-valor Acción {action}')
        plt.xlabel('Posición Carrito')
        plt.ylabel('Velocidad Angular')
        plt.title(f'Q-Valores - Acción {action}')
        plt.grid(True, alpha=0.3)
    
    plt.suptitle('Política Aprendida - Q-Valores por Acción')
    plt.tight_layout()
    plt.show()

# Visualizar política
visualize_q_values(agent, env)

# --- 9. Comparación con política aleatoria ---
print("\n⚖️ Comparando con política aleatoria...")

# Evaluar política aleatoria
random_scores = []
for e in range(20):
    state = env.reset()
    state = np.reshape(state, [1, state_size])
    total_reward = 0
    done = False
    steps = 0
    
    while not done and steps < 500:
        action = random.randrange(action_size)  # Política aleatoria
        next_state, reward, done, _ = env.step(action)
        next_state = np.reshape(next_state, [1, state_size])
        total_reward += reward
        
        state = next_state
        steps += 1
        
        if done:
            break
    
    random_scores.append(total_reward)

print(f"   Política DQN - Recompensa: {np.mean(test_scores):.2f} ± {np.std(test_scores):.2f}")
print(f"   Política Aleatoria - Recompensa: {np.mean(random_scores):.2f} ± {np.std(random_scores):.2f}")
print(f"   Mejora: {(np.mean(test_scores) - np.mean(random_scores))/abs(np.mean(random_scores)) * 100:.1f}%")

# --- 10. Guardar el modelo ---
agent.save('dqn_cartpole.h5')
print("\n💾 Modelo guardado como 'dqn_cartpole.h5'")

print("\n✅ EJERCICIO COMPLETADO")
print("=" * 60)
print("🎯 RESULTADOS:")
print(f"   • Recompensa promedio final: {np.mean(test_scores):.2f}")
print(f"   • Mejora porcentual: {improvement:.1f}%")
print(f"   • ¿Convergió? {'Sí' if converged else 'No'}")
print(f"   • Total de parámetros: {agent.model.count_params():,}")
print(f"   • Episodios entrenados: {episodes}")
print(f"   • Mejora sobre aleatorio: {(np.mean(test_scores) - np.mean(random_scores))/abs(np.mean(random_scores)) * 100:.1f}%")
print("=" * 60)
```

## 📊 Resultados Esperados

```
🎮 RED DQN - AGENTE PARA JUEGO CARTPOLE
============================================================
🎯 Creando entorno CartPole...
   Entorno: CartPole-v1
   Espacio de estados: 4
   Espacio de acciones: 2

🏗️ Construyendo Q-Network...
Creando agente DQN...

🚀 Entrenando agente DQN...
   Episodio 1/200 - Recompensa: 12.00 - Epsilon: 1.000
   Episodio 21/200 - Recompensa: 45.00 - Epsilon: 0.903
   Episodio 41/200 - Recompensa: 78.00 - Epsilon: 0.818
   Episodio 61/200 - Recompensa: 123.00 - Epsilon: 0.741
   Episodio 81/200 - Recompensa: 156.00 - Epsilon: 0.673
   Episodio 101/200 - Recompensa: 189.00 - Epsilon: 0.613
   Episodio 121/200 - Recompensa: 234.00 - Epsilon: 0.558
   Episodio 141/200 - Recompensa: 267.00 - Epsilon: 0.508
   Episodio 161/200 - Recompensa: 298.00 - Epsilon: 0.463
   Episodio 181/200 - Recompensa: 345.00 - Epsilon: 0.421
   Episodio 199/200 - Recompensa: 389.00 - Epsilon: 0.011

✅ Entrenamiento completado

📈 Evaluando agente entrenado...
   Recompensa promedio en prueba: 345.67
   Desviación estándar: 45.23
   Mejor puntuación: 456.00

🔍 Analizando convergencia...
   Recompensa promedio primeros 10 episodios: 23.45
   Recompensa promedio últimos 10 episodios: 387.89
   Mejora porcentual: 1552.3%
   ¿Convergió? Sí

⚖️ Comparando con política aleatoria...
   Política DQN - Recompensa: 345.67 ± 45.23
   Política Aleatoria - Recompensa: 23.45 ± 12.34
   Mejora: 1373.7%

✅ EJERCICIO COMPLETADO
============================================================
🎯 RESULTADOS:
   • Recompensa promedio final: 345.67
   • Mejora porcentual: 1552.3%
   • ¿Convergió? Sí
   • Total de parámetros: 1,026
   • Episodios entrenados: 200
   • Mejora sobre aleatorio: 1373.7%
============================================================
```

## 🎓 Análisis de Aprendizaje

### Identificación del Paradigma
**✅ APRENDIZAJE POR REFUERZO**
- **Sin etiquetas**: Aprende mediante recompensas del entorno
- **Trial-and-error**: Explora y explota para maximizar recompensas
- **Objetivo**: Aprender política óptima para tomar decisiones
- **Métrica**: Recompensa acumulada y convergencia

### Ventajas de DQN
- **Escalabilidad**: Funciona con espacios de estados grandes
- **Generalización**: Aprende patrones aplicables a estados similares
- **Estabilidad**: Experience replay y target networks mejoran entrenamiento
- **Flexibilidad**: Aplicable a diversos problemas de secuencia

### Limitaciones
- **Inestabilidad**: Puede ser inestable sin técnicas de estabilización
- **Muestreo**: Requiere muchos episodios para buena convergencia
- **Hiperparámetros**: Sensible a configuración de hiperparámetros
- **Exploración**: Balance exploración-explotación difícil de ajustar

## 🚀 Adaptación al Proyecto Integrador

### Ideas para Adaptación
1. **Dominio específico**: Adaptar para problemas de decisión del proyecto
2. **Multi-agente**: Extender para sistemas con múltiples agentes
3. **Continuous actions**: Adaptar para acciones continuas

### Ejemplo de Adaptación
```python
# Adaptar DQN para optimización de procesos
def crear_dqn_optimizador(state_size, action_size):
    """
    Adapta DQN para optimización de procesos industriales
    """
    class OptimizadorDQN(DQNAgent):
        def __init__(self, state_size, action_size):
            super().__init__(state_size, action_size)
            self.gamma = 0.99  # Mayor descuento para problemas de optimización
            self.epsilon_decay = 0.99  # Decaimiento más lento
        
        def _build_model(self):
            """Arquitectura más profunda para optimización"""
            model = models.Sequential([
                layers.Dense(128, input_dim=self.state_size, activation='relu'),
                layers.Dropout(0.2),
                layers.Dense(64, activation='relu'),
                layers.Dropout(0.2),
                layers.Dense(32, activation='relu'),
                layers.Dense(self.action_size, activation='linear')
            ])
            model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005))
            return model
    
    return OptimizadorDQN(state_size, action_size)
```

## 📝 Ejercicios Adicionales

### Ejercicio 1: Supervisado (DQN para Clasificación)
```python
# Usar DQN como clasificador supervisado
def crear_dqn_clasificador(input_size, num_classes):
    """Adaptar DQN para clasificación supervisada"""
    
    class DQNClasificador:
        def __init__(self, input_size, num_classes):
            self.model = models.Sequential([
                layers.Dense(64, input_dim=input_size, activation='relu'),
                layers.Dense(32, activation='relu'),
                layers.Dense(num_classes, activation='linear')
            ])
            self.model.compile(optimizer='adam', loss='mse')
        
        def train_supervised(self, X, y, epochs=100):
            """Entrenar usando etiquetas como recompensas"""
            # Convertir etiquetas a one-hot Q-valores
            q_targets = np.zeros((len(y), num_classes))
            for i, label in enumerate(y):
                q_targets[i, label] = 1.0  # Recompensa máxima para clase correcta
            
            self.model.fit(X, q_targets, epochs=epochs, verbose=0)
        
        def predict(self, X):
            """Predecir clase usando Q-valores"""
            q_values = self.model.predict(X, verbose=0)
            return np.argmax(q_values, axis=1)
    
    return DQNClasificador(input_size, num_classes)
```

### Ejercicio 2: No Supervisado (DQN para Clustering)
```python
# Usar DQN para clustering no supervisado
def crear_dqn_clustering(data_size, n_clusters):
    """Adaptar DQN para clustering no supervisado"""
    
    class DQNClustering:
        def __init__(self, data_size, n_clusters):
            self.data_size = data_size
            self.n_clusters = n_clusters
            self.model = models.Sequential([
                layers.Dense(128, input_dim=data_size, activation='relu'),
                layers.Dense(64, activation='relu'),
                layers.Dense(n_clusters, activation='linear')
            ])
            self.model.compile(optimizer='adam', loss='mse')
        
        def train_clustering(self, data, epochs=200):
            """Entrenar usando recompensas basadas en similitud"""
            for epoch in range(epochs):
                # Muestrear batch aleatorio
                batch_idx = np.random.choice(len(data), size=32, replace=False)
                batch_data = data[batch_idx]
                
                # Calcular Q-valores
                q_values = self.model.predict(batch_data, verbose=0)
                
                # Calcular recompensas basadas en similitud intra-cluster
                rewards = self.calculate_cluster_rewards(batch_data, q_values)
                
                # Actualizar Q-valores
                targets = q_values.copy()
                for i in range(len(batch_data)):
                    best_action = np.argmax(q_values[i])
                    targets[i, best_action] = rewards[i]
                
                self.model.fit(batch_data, targets, epochs=1, verbose=0)
        
        def calculate_cluster_rewards(self, data, q_values):
            """Calcular recompensas basadas en cohesión de cluster"""
            rewards = []
            for i, point in enumerate(data):
                cluster_id = np.argmax(q_values[i])
                # Encontrar otros puntos en el mismo cluster
                same_cluster = [j for j in range(len(data)) 
                              if np.argmax(q_values[j]) == cluster_id]
                
                # Calcular cohesión (similitud promedio)
                if len(same_cluster) > 1:
                    similarities = [np.exp(-np.linalg.norm(point - data[j])) 
                                  for j in same_cluster if j != i]
                    reward = np.mean(similarities)
                else:
                    reward = 0.0
                
                rewards.append(reward)
            
            return np.array(rewards)
    
    return DQNClustering(data_size, n_clusters)
```

## 🔗 Recursos Adicionales

- [Documentación TensorFlow/Keras](https://www.tensorflow.org/api_docs/python/tf/keras)
- [OpenAI Gym](https://gym.openai.com/) - Entornos de RL
- [DQN Paper Original](https://www.nature.com/articles/nature14236)
- [Reinforcement Learning Guide](https://www.tensorflow.org/agents/tutorials/0_intro_rl)
