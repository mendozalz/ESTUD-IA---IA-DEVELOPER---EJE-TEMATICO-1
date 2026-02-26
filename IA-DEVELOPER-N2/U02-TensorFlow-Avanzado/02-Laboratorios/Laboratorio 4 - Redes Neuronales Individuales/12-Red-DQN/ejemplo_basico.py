import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt
import random
from collections import deque

print("RED DQN - AGENTE PARA JUEGO CARTPOLE")
print("=" * 60)

# --- 1. Crear entorno CartPole ---
print("Creando entorno CartPole...")

# Crear entorno simulado simple (sin necesidad de gym)
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

print(f"   Entorno: CartPole Simulado")
print(f"   Espacio de estados: {state_size}")
print(f"   Espacio de acciones: {action_size}")

# --- 2. Definir la red Q-Network ---
print("\nConstruyendo Q-Network...")

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
print("\nEntrenando agente DQN...")
episodes = 100  # Reducido para demostracion
batch_size = 32
scores = []
epsilons = []

for e in range(episodes):
    state = env.reset()
    state = np.reshape(state, [1, state_size])
    total_reward = 0
    done = False
    steps = 0
    
    while not done and steps < 200:  # Límite de pasos reducido
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
print("\nEvaluando agente entrenado...")
test_episodes = 20
test_scores = []

for e in range(test_episodes):
    state = env.reset()
    state = np.reshape(state, [1, state_size])
    total_reward = 0
    done = False
    steps = 0
    
    while not done and steps < 200:
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
print("\nVisualizando resultados...")

# Graficar curva de aprendizaje
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.plot(scores, alpha=0.7)
plt.title('Recompensas por Episodio')
plt.xlabel('Episodio')
plt.ylabel('Recompensa Total')
plt.grid(True)

# Graficar media móvil
window_size = 10
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
print("\nAnalizando convergencia...")

# Calcular estadísticas de convergencia
first_10_avg = np.mean(scores[:10])
last_10_avg = np.mean(scores[-10:])
improvement = (last_10_avg - first_10_avg) / abs(first_10_avg) * 100

print(f"   Recompensa promedio primeros 10 episodios: {first_10_avg:.2f}")
print(f"   Recompensa promedio últimos 10 episodios: {last_10_avg:.2f}")
print(f"   Mejora porcentual: {improvement:.1f}%")

# Determinar si convergió
converged = last_10_avg > 50  # Umbral simplificado
print(f"   ¿Convergió? {'Sí' if converged else 'No'}")

# --- 8. Comparación con política aleatoria ---
print("\nComparando con política aleatoria...")

# Evaluar política aleatoria
random_scores = []
for e in range(20):
    state = env.reset()
    state = np.reshape(state, [1, state_size])
    total_reward = 0
    done = False
    steps = 0
    
    while not done and steps < 200:
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

# --- 9. Guardar el modelo ---
agent.save('dqn_cartpole.h5')
print("\nModelo guardado como 'dqn_cartpole.h5'")

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
