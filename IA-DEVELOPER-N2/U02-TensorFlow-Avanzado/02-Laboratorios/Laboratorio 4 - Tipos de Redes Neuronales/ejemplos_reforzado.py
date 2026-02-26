import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import random
from collections import deque
import warnings
warnings.filterwarnings('ignore')

class EjemplosRefuerzo:
    def __init__(self):
        self.ejemplos = {
            'DQN': self.ejemplo_dqn,
            'Policy_Gradient': self.ejemplo_policy_gradient,
            'Actor_Critic': self.ejemplo_actor_critic
        }
    
    def ejemplo_dqn(self):
        """Ejemplo 1: Deep Q-Network para CartPole"""
        print("=" * 60)
        print("🎮 EJEMPLO 1: DQN - CARTPOLE")
        print("=" * 60)
        
        # Crear entorno CartPole simplificado
        class CartPoleEnv:
            def __init__(self):
                self.gravity = 9.8
                self.masscart = 1.0
                self.masspole = 0.1
                self.total_mass = self.masspole + self.masscart
                self.length = 0.5
                self.polemass_length = self.masspole * self.length
                self.force_mag = 10.0
                self.tau = 0.02
                
                self.theta_threshold_radians = 12 * 2 * np.pi / 360
                self.x_threshold = 2.4
                
                self.reset()
            
            def reset(self):
                self.state = np.random.uniform(low=-0.05, high=0.05, size=(4,))
                self.steps_beyond_done = None
                return np.array(self.state)
            
            def step(self, action):
                x, x_dot, theta, theta_dot = self.state
                force = self.force_mag if action == 1 else -self.force_mag
                
                costheta = np.cos(theta)
                sintheta = np.sin(theta)
                
                temp = (force + self.polemass_length * theta_dot * theta_dot * sintheta) / self.total_mass
                thetaacc = (self.gravity * sintheta - costheta * temp) / (
                    self.length * (4.0 / 3.0 - self.masspole * costheta * costheta / self.total_mass)
                )
                xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass
                
                x = x + self.tau * x_dot
                x_dot = x_dot + self.tau * xacc
                theta = theta + self.tau * theta_dot
                theta_dot = theta_dot + self.tau * thetaacc
                
                self.state = np.array([x, x_dot, theta, theta_dot])
                
                done = bool(
                    x < -self.x_threshold
                    or x > self.x_threshold
                    or theta < -self.theta_threshold_radians
                    or theta > self.theta_threshold_radians
                )
                
                if not done:
                    reward = 1.0
                elif self.steps_beyond_done is None:
                    self.steps_beyond_done = 0
                    reward = 1.0
                else:
                    if self.steps_beyond_done == 0:
                        pass
                    self.steps_beyond_done += 1
                    reward = 0.0
                
                return np.array(self.state), reward, done, {}
        
        # Crear entorno
        env = CartPoleEnv()
        state_size = 4
        action_size = 2
        
        print(f"Entorno: {state_size} estados, {action_size} acciones")
        
        # Construir DQN
        print("🏗️  Construyendo DQN...")
        
        def build_model():
            model = keras.Sequential([
                keras.layers.Dense(24, activation='relu', input_shape=(state_size,)),
                keras.layers.Dense(24, activation='relu'),
                keras.layers.Dense(action_size, activation='linear')
            ])
            model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss='mse')
            return model
        
        # Crear redes
        model = build_model()
        target_model = build_model()
        
        print("✅ DQN construido:")
        model.summary()
        
        # Hiperparámetros
        episodes = 200
        batch_size = 32
        gamma = 0.95
        epsilon = 1.0
        epsilon_min = 0.01
        epsilon_decay = 0.995
        memory = deque(maxlen=2000)
        
        # Entrenamiento
        print("\n🎯 Entrenando DQN...")
        
        rewards = []
        losses = []
        
        for episode in range(episodes):
            state = env.reset()
            total_reward = 0
            done = False
            steps = 0
            
            while not done and steps < 500:
                # Epsilon-greedy
                if np.random.random() <= epsilon:
                    action = random.randrange(action_size)
                else:
                    act_values = model.predict(state.reshape(1, -1), verbose=0)
                    action = np.argmax(act_values[0])
                
                next_state, reward, done, _ = env.step(action)
                total_reward += reward
                
                memory.append((state, action, reward, next_state, done))
                state = next_state
                steps += 1
            
            rewards.append(total_reward)
            
            # Entrenar con experience replay
            if len(memory) > batch_size:
                minibatch = random.sample(memory, batch_size)
                
                states = np.array([transition[0] for transition in minibatch])
                actions = np.array([transition[1] for transition in minibatch])
                rewards_batch = np.array([transition[2] for transition in minibatch])
                next_states = np.array([transition[3] for transition in minibatch])
                dones = np.array([transition[4] for transition in minibatch])
                
                # Q-values actuales
                current_q = model.predict(states, verbose=0)
                
                # Q-values siguientes
                next_q = target_model.predict(next_states, verbose=0)
                max_next_q = np.amax(next_q, axis=1)
                
                # Actualizar Q-values
                for i in range(batch_size):
                    if dones[i]:
                        current_q[i, actions[i]] = rewards_batch[i]
                    else:
                        current_q[i, actions[i]] = rewards_batch[i] + gamma * max_next_q[i]
                
                # Entrenar modelo
                loss = model.train_on_batch(states, current_q)
                losses.append(loss)
            
            # Actualizar epsilon
            if epsilon > epsilon_min:
                epsilon *= epsilon_decay
            
            # Actualizar target model
            if episode % 10 == 0:
                target_model.set_weights(model.get_weights())
            
            if episode % 50 == 0:
                avg_reward = np.mean(rewards[-50:])
                print(f"Episode: {episode}, Avg Reward: {avg_reward:.2f}, Epsilon: {epsilon:.3f}")
        
        # Evaluar modelo entrenado
        print("\n📊 Evaluando modelo entrenado...")
        
        test_rewards = []
        for _ in range(10):
            state = env.reset()
            total_reward = 0
            done = False
            steps = 0
            
            while not done and steps < 500:
                act_values = model.predict(state.reshape(1, -1), verbose=0)
                action = np.argmax(act_values[0])
                state, reward, done, _ = env.step(action)
                total_reward += reward
                steps += 1
            
            test_rewards.append(total_reward)
        
        avg_test_reward = np.mean(test_rewards)
        print(f"Recompensa promedio en prueba: {avg_test_reward:.2f}")
        
        # Visualizar resultados
        plt.figure(figsize=(15, 10))
        
        # Recompensas durante entrenamiento
        plt.subplot(2, 3, 1)
        plt.plot(rewards)
        plt.title('Recompensas por Episodio')
        plt.xlabel('Episodio')
        plt.ylabel('Recompensa')
        plt.grid(True)
        
        # Media móvil de recompensas
        plt.subplot(2, 3, 2)
        window = 50
        if len(rewards) >= window:
            moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
            plt.plot(range(window-1, len(rewards)), moving_avg)
        plt.title(f'Media Móvil (ventana={window})')
        plt.xlabel('Episodio')
        plt.ylabel('Recompensa')
        plt.grid(True)
        
        # Pérdidas durante entrenamiento
        plt.subplot(2, 3, 3)
        if losses:
            plt.plot(losses)
            plt.title('Pérdidas durante Entrenamiento')
            plt.xlabel('Paso de entrenamiento')
            plt.ylabel('Pérdida')
            plt.grid(True)
        
        # Distribución de recompensas
        plt.subplot(2, 3, 4)
        plt.hist(rewards, bins=30, alpha=0.7)
        plt.title('Distribución de Recompensas')
        plt.xlabel('Recompensa')
        plt.ylabel('Frecuencia')
        
        # Progreso del entrenamiento
        plt.subplot(2, 3, 5)
        episodes_x = range(0, len(rewards), 10)
        avg_rewards_10 = [np.mean(rewards[i:i+10]) for i in range(0, len(rewards)-9, 10)]
        plt.plot(episodes_x, avg_rewards_10)
        plt.title('Recompensa Promedio cada 10 episodios')
        plt.xlabel('Episodio')
        plt.ylabel('Recompensa Promedio')
        plt.grid(True)
        
        # Comparación entrenamiento vs prueba
        plt.subplot(2, 3, 6)
        plt.boxplot([rewards[-50:], test_rewards], labels=['Entrenamiento (últimos 50)', 'Prueba'])
        plt.title('Comparación Entrenamiento vs Prueba')
        plt.ylabel('Recompensa')
        
        plt.tight_layout()
        plt.show()
        
        return model, avg_test_reward
    
    def ejemplo_policy_gradient(self):
        """Ejemplo 2: Policy Gradient para Grid World"""
        print("\n" + "=" * 60)
        print("🎯 EJEMPLO 2: POLICY GRADIENT - GRID WORLD")
        print("=" * 60)
        
        # Crear entorno Grid World
        class GridWorld:
            def __init__(self, size=5):
                self.size = size
                self.state = (0, 0)
                self.goal = (size-1, size-1)
                self.obstacles = set()
                
                # Añadir algunos obstáculos
                for _ in range(size):
                    obs = (np.random.randint(0, size), np.random.randint(0, size))
                    if obs != (0, 0) and obs != self.goal:
                        self.obstacles.add(obs)
            
            def reset(self):
                self.state = (0, 0)
                return self.state
            
            def step(self, action):
                x, y = self.state
                
                # Acciones: 0=arriba, 1=derecha, 2=abajo, 3=izquierda
                moves = [(-1, 0), (0, 1), (1, 0), (0, -1)]
                dx, dy = moves[action]
                
                new_x = max(0, min(self.size - 1, x + dx))
                new_y = max(0, min(self.size - 1, y + dy))
                new_state = (new_x, new_y)
                
                # Verificar obstáculos
                if new_state in self.obstacles:
                    new_state = self.state
                
                # Calcular recompensa
                if new_state == self.goal:
                    reward = 10
                    done = True
                else:
                    # Recompensa basada en distancia a la meta
                    dist_before = abs(x - self.goal[0]) + abs(y - self.goal[1])
                    dist_after = abs(new_x - self.goal[0]) + abs(new_y - self.goal[1])
                    reward = -0.1 + (dist_before - dist_after) * 0.1
                    done = False
                
                self.state = new_state
                return new_state, reward, done
        
        # Crear entorno
        env = GridWorld(size=5)
        state_size = 25  # 5x5 grid
        action_size = 4
        
        print(f"Grid World: {env.size}x{env.size}, {len(env.obstacles)} obstáculos")
        
        # Construir Policy Network
        print("🏗️  Construyendo Policy Network...")
        
        def build_policy_model():
            model = keras.Sequential([
                keras.layers.Dense(64, activation='relu', input_shape=(state_size,)),
                keras.layers.Dense(64, activation='relu'),
                keras.layers.Dense(action_size, activation='softmax')
            ])
            return model
        
        policy_model = build_policy_model()
        optimizer = keras.optimizers.Adam(learning_rate=0.01)
        
        print("✅ Policy Network construida:")
        policy_model.summary()
        
        # Hiperparámetros
        episodes = 500
        gamma = 0.99
        
        # Entrenamiento
        print("\n🎯 Entrenando Policy Gradient...")
        
        rewards_history = []
        loss_history = []
        
        for episode in range(episodes):
            state = env.reset()
            state_onehot = np.zeros(state_size)
            state_onehot[state[0] * env.size + state[1]] = 1
            
            states = []
            actions = []
            rewards = []
            
            done = False
            steps = 0
            
            while not done and steps < 100:
                # Obtener probabilidades de acción
                state_tensor = state_onehot.reshape(1, -1)
                action_probs = policy_model(state_tensor, training=False)[0]
                
                # Seleccionar acción
                action = np.random.choice(action_size, p=action_probs)
                
                # Ejecutar acción
                next_state, reward, done = env.step(action)
                
                # Guardar transición
                states.append(state_onehot.copy())
                actions.append(action)
                rewards.append(reward)
                
                # Actualizar estado
                state_onehot = np.zeros(state_size)
                state_onehot[next_state[0] * env.size + next_state[1]] = 1
                
                steps += 1
            
            # Calcular retornos descontados
            returns = []
            G = 0
            for reward in reversed(rewards):
                G = reward + gamma * G
                returns.insert(0, G)
            
            # Normalizar retornos
            returns = np.array(returns)
            returns = (returns - np.mean(returns)) / (np.std(returns) + 1e-8)
            
            # Convertir a tensores
            states_tensor = tf.convert_to_tensor(np.array(states), dtype=tf.float32)
            actions_tensor = tf.convert_to_tensor(np.array(actions), dtype=tf.int32)
            returns_tensor = tf.convert_to_tensor(returns, dtype=tf.float32)
            
            # Calcular gradientes
            with tf.GradientTape() as tape:
                action_probs = policy_model(states_tensor, training=True)
                
                # Calcular log probabilities
                action_masks = tf.one_hot(actions_tensor, action_size)
                selected_action_probs = tf.reduce_sum(action_probs * action_masks, axis=1)
                log_probs = tf.math.log(selected_action_probs + 1e-8)
                
                # Calcular pérdida (negative log likelihood weighted by returns)
                loss = -tf.reduce_mean(log_probs * returns_tensor)
            
            # Aplicar gradientes
            gradients = tape.gradient(loss, policy_model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, policy_model.trainable_variables))
            
            total_reward = sum(rewards)
            rewards_history.append(total_reward)
            loss_history.append(float(loss))
            
            if episode % 50 == 0:
                avg_reward = np.mean(rewards_history[-50:])
                print(f"Episode: {episode}, Avg Reward: {avg_reward:.2f}, Loss: {float(loss):.4f}")
        
        # Evaluar política aprendida
        print("\n📊 Evaluando política aprendida...")
        
        test_rewards = []
        for _ in range(10):
            state = env.reset()
            state_onehot = np.zeros(state_size)
            state_onehot[state[0] * env.size + state[1]] = 1
            
            total_reward = 0
            done = False
            steps = 0
            
            while not done and steps < 100:
                state_tensor = state_onehot.reshape(1, -1)
                action_probs = policy_model(state_tensor, training=False)[0]
                action = np.argmax(action_probs)
                
                next_state, reward, done = env.step(action)
                total_reward += reward
                
                state_onehot = np.zeros(state_size)
                state_onehot[next_state[0] * env.size + next_state[1]] = 1
                
                steps += 1
            
            test_rewards.append(total_reward)
        
        avg_test_reward = np.mean(test_rewards)
        print(f"Recompensa promedio en prueba: {avg_test_reward:.2f}")
        
        # Visualizar resultados
        plt.figure(figsize=(15, 10))
        
        # Recompensas durante entrenamiento
        plt.subplot(2, 3, 1)
        plt.plot(rewards_history)
        plt.title('Recompensas por Episodio')
        plt.xlabel('Episodio')
        plt.ylabel('Recompensa')
        plt.grid(True)
        
        # Media móvil
        plt.subplot(2, 3, 2)
        window = 50
        if len(rewards_history) >= window:
            moving_avg = np.convolve(rewards_history, np.ones(window)/window, mode='valid')
            plt.plot(range(window-1, len(rewards_history)), moving_avg)
        plt.title(f'Media Móvil (ventana={window})')
        plt.xlabel('Episodio')
        plt.ylabel('Recompensa')
        plt.grid(True)
        
        # Pérdidas
        plt.subplot(2, 3, 3)
        plt.plot(loss_history)
        plt.title('Pérdidas durante Entrenamiento')
        plt.xlabel('Episodio')
        plt.ylabel('Pérdida')
        plt.grid(True)
        
        # Distribución de recompensas
        plt.subplot(2, 3, 4)
        plt.hist(rewards_history, bins=30, alpha=0.7)
        plt.title('Distribución de Recompensas')
        plt.xlabel('Recompensa')
        plt.ylabel('Frecuencia')
        
        # Visualizar política aprendida
        plt.subplot(2, 3, 5)
        policy_grid = np.zeros((env.size, env.size))
        for i in range(env.size):
            for j in range(env.size):
                if (i, j) not in env.obstacles and (i, j) != env.goal:
                    state_onehot = np.zeros(state_size)
                    state_onehot[i * env.size + j] = 1
                    state_tensor = state_onehot.reshape(1, -1)
                    action_probs = policy_model(state_tensor, training=False)[0]
                    policy_grid[i, j] = np.max(action_probs)
        
        plt.imshow(policy_grid, cmap='viridis')
        plt.colorbar()
        plt.title('Confianza de la Política')
        plt.xlabel('Columna')
        plt.ylabel('Fila')
        
        # Comparación entrenamiento vs prueba
        plt.subplot(2, 3, 6)
        plt.boxplot([rewards_history[-50:], test_rewards], labels=['Entrenamiento (últimos 50)', 'Prueba'])
        plt.title('Comparación Entrenamiento vs Prueba')
        plt.ylabel('Recompensa')
        
        plt.tight_layout()
        plt.show()
        
        return policy_model, avg_test_reward
    
    def ejemplo_actor_critic(self):
        """Ejemplo 3: Actor-Critic para Navigation"""
        print("\n" + "=" * 60)
        print("🎭 EJEMPLO 3: ACTOR-CRITIC - NAVIGATION")
        print("=" * 60)
        
        # Crear entorno de navegación simple
        class NavigationEnv:
            def __init__(self, size=7):
                self.size = size
                self.agent_pos = (0, 0)
                self.target_pos = (size-1, size-1)
                self.obstacles = set()
                
                # Generar obstáculos
                for _ in range(size * 2):
                    obs = (np.random.randint(0, size), np.random.randint(0, size))
                    if obs != (0, 0) and obs != self.target_pos:
                        self.obstacles.add(obs)
            
            def reset(self):
                self.agent_pos = (0, 0)
                return self._get_state()
            
            def _get_state(self):
                # Estado: posición del agente y objetivo
                state = np.zeros(self.size * self.size * 2)
                state[self.agent_pos[0] * self.size + self.agent_pos[1]] = 1
                state[self.size * self.size + self.target_pos[0] * self.size + self.target_pos[1]] = 1
                return state
            
            def step(self, action):
                x, y = self.agent_pos
                
                # Movimientos
                moves = [(-1, 0), (0, 1), (1, 0), (0, -1)]
                dx, dy = moves[action]
                
                new_x = max(0, min(self.size - 1, x + dx))
                new_y = max(0, min(self.size - 1, y + dy))
                new_pos = (new_x, new_y)
                
                # Verificar obstáculos
                if new_pos in self.obstacles:
                    new_pos = self.agent_pos
                
                # Calcular recompensa
                if new_pos == self.target_pos:
                    reward = 10
                    done = True
                else:
                    # Distancia Manhattan
                    dist_before = abs(x - self.target_pos[0]) + abs(y - self.target_pos[1])
                    dist_after = abs(new_x - self.target_pos[0]) + abs(new_y - self.target_pos[1])
                    reward = -0.1 + (dist_before - dist_after) * 0.5
                    done = False
                
                self.agent_pos = new_pos
                return self._get_state(), reward, done
        
        # Crear entorno
        env = NavigationEnv(size=7)
        state_size = env.size * env.size * 2
        action_size = 4
        
        print(f"Navigation: {env.size}x{env.size}, {len(env.obstacles)} obstáculos")
        
        # Construir Actor-Critic
        print("🏗️  Construyendo Actor-Critic...")
        
        class ActorCritic:
            def __init__(self, state_size, action_size):
                self.state_size = state_size
                self.action_size = action_size
                
                # Actor (policy)
                self.actor = keras.Sequential([
                    keras.layers.Dense(128, activation='relu', input_shape=(state_size,)),
                    keras.layers.Dense(64, activation='relu'),
                    keras.layers.Dense(action_size, activation='softmax')
                ])
                
                # Critic (value)
                self.critic = keras.Sequential([
                    keras.layers.Dense(128, activation='relu', input_shape=(state_size,)),
                    keras.layers.Dense(64, activation='relu'),
                    keras.layers.Dense(1)
                ])
                
                self.optimizer = keras.optimizers.Adam(learning_rate=0.001)
            
            def get_action_and_value(self, state):
                state_tensor = tf.convert_to_tensor(state.reshape(1, -1), dtype=tf.float32)
                
                action_probs = self.actor(state_tensor, training=False)[0]
                value = self.critic(state_tensor, training=False)[0]
                
                action = np.random.choice(self.action_size, p=action_probs.numpy())
                
                return action, action_probs[action].numpy(), value.numpy()
            
            def train_step(self, states, actions, rewards, next_states, dones):
                states_tensor = tf.convert_to_tensor(np.array(states), dtype=tf.float32)
                actions_tensor = tf.convert_to_tensor(np.array(actions), dtype=tf.int32)
                rewards_tensor = tf.convert_to_tensor(np.array(rewards), dtype=tf.float32)
                next_states_tensor = tf.convert_to_tensor(np.array(next_states), dtype=tf.float32)
                dones_tensor = tf.convert_to_tensor(np.array(dones), dtype=tf.float32)
                
                with tf.GradientTape() as tape:
                    # Calcular valores actuales
                    action_probs = self.actor(states_tensor, training=True)
                    values = self.critic(states_tensor, training=True)
                    next_values = self.critic(next_states_tensor, training=True)
                    
                    # Calcular ventajas (advantages)
                    advantages = rewards_tensor + 0.99 * (1 - dones_tensor) * tf.squeeze(next_values) - tf.squeeze(values)
                    
                    # Actor loss (policy gradient)
                    action_masks = tf.one_hot(actions_tensor, self.action_size)
                    selected_action_probs = tf.reduce_sum(action_probs * action_masks, axis=1)
                    actor_loss = -tf.reduce_mean(tf.math.log(selected_action_probs + 1e-8) * tf.stop_gradient(advantages))
                    
                    # Critic loss (value function)
                    critic_loss = tf.reduce_mean(tf.square(advantages))
                    
                    # Loss total
                    total_loss = actor_loss + 0.5 * critic_loss
                
                # Calcular y aplicar gradientes
                gradients = tape.gradient(total_loss, self.actor.trainable_variables + self.critic.trainable_variables)
                self.optimizer.apply_gradients(zip(gradients, self.actor.trainable_variables + self.critic.trainable_variables))
                
                return float(total_loss)
        
        # Crear modelo Actor-Critic
        ac_model = ActorCritic(state_size, action_size)
        
        print("✅ Actor-Critic construido")
        
        # Entrenamiento
        print("\n🎯 Entrenando Actor-Critic...")
        
        episodes = 300
        gamma = 0.99
        
        rewards_history = []
        loss_history = []
        
        for episode in range(episodes):
            state = env.reset()
            
            states = []
            actions = []
            rewards = []
            next_states = []
            dones = []
            
            done = False
            steps = 0
            
            while not done and steps < 100:
                action, prob, value = ac_model.get_action_and_value(state)
                
                next_state, reward, done = env.step(action)
                
                states.append(state)
                actions.append(action)
                rewards.append(reward)
                next_states.append(next_state)
                dones.append(done)
                
                state = next_state
                steps += 1
            
            # Entrenar con la experiencia del episodio
            loss = ac_model.train_step(states, actions, rewards, next_states, dones)
            
            total_reward = sum(rewards)
            rewards_history.append(total_reward)
            loss_history.append(loss)
            
            if episode % 50 == 0:
                avg_reward = np.mean(rewards_history[-50:])
                print(f"Episode: {episode}, Avg Reward: {avg_reward:.2f}, Loss: {loss:.4f}")
        
        # Evaluar modelo entrenado
        print("\n📊 Evaluando Actor-Critic...")
        
        test_rewards = []
        for _ in range(10):
            state = env.reset()
            total_reward = 0
            done = False
            steps = 0
            
            while not done and steps < 100:
                action, prob, value = ac_model.get_action_and_value(state)
                next_state, reward, done = env.step(action)
                total_reward += reward
                state = next_state
                steps += 1
            
            test_rewards.append(total_reward)
        
        avg_test_reward = np.mean(test_rewards)
        print(f"Recompensa promedio en prueba: {avg_test_reward:.2f}")
        
        # Visualizar resultados
        plt.figure(figsize=(15, 10))
        
        # Recompensas
        plt.subplot(2, 3, 1)
        plt.plot(rewards_history)
        plt.title('Recompensas por Episodio')
        plt.xlabel('Episodio')
        plt.ylabel('Recompensa')
        plt.grid(True)
        
        # Media móvil
        plt.subplot(2, 3, 2)
        window = 50
        if len(rewards_history) >= window:
            moving_avg = np.convolve(rewards_history, np.ones(window)/window, mode='valid')
            plt.plot(range(window-1, len(rewards_history)), moving_avg)
        plt.title(f'Media Móvil (ventana={window})')
        plt.xlabel('Episodio')
        plt.ylabel('Recompensa')
        plt.grid(True)
        
        # Pérdidas
        plt.subplot(2, 3, 3)
        plt.plot(loss_history)
        plt.title('Pérdidas durante Entrenamiento')
        plt.xlabel('Episodio')
        plt.ylabel('Pérdida')
        plt.grid(True)
        
        # Visualizar valores aprendidos
        plt.subplot(2, 3, 4)
        value_grid = np.zeros((env.size, env.size))
        for i in range(env.size):
            for j in range(env.size):
                if (i, j) not in env.obstacles:
                    state = np.zeros(state_size)
                    state[i * env.size + j] = 1
                    state[env.size * env.size + env.target_pos[0] * env.size + env.target_pos[1]] = 1
                    state_tensor = tf.convert_to_tensor(state.reshape(1, -1), dtype=tf.float32)
                    value = ac_model.critic(state_tensor, training=False)[0]
                    value_grid[i, j] = value
        
        plt.imshow(value_grid, cmap='viridis')
        plt.colorbar()
        plt.title('Valores Aprendidos por Critic')
        plt.xlabel('Columna')
        plt.ylabel('Fila')
        
        # Distribución de recompensas
        plt.subplot(2, 3, 5)
        plt.hist(rewards_history, bins=30, alpha=0.7)
        plt.title('Distribución de Recompensas')
        plt.xlabel('Recompensa')
        plt.ylabel('Frecuencia')
        
        # Comparación
        plt.subplot(2, 3, 6)
        plt.boxplot([rewards_history[-50:], test_rewards], labels=['Entrenamiento (últimos 50)', 'Prueba'])
        plt.title('Comparación Entrenamiento vs Prueba')
        plt.ylabel('Recompensa')
        
        plt.tight_layout()
        plt.show()
        
        return ac_model, avg_test_reward
    
    def ejecutar_todos(self):
        """Ejecuta todos los ejemplos de refuerzo"""
        print("🎯 EJECUTANDO EJEMPLOS DE APRENDIZAJE POR REFUERZO")
        print("=" * 80)
        
        resultados = {}
        
        for nombre, funcion in self.ejemplos.items():
            print(f"\n🔄 Ejecutando ejemplo: {nombre}")
            modelo, metrica = funcion()
            resultados[nombre] = {'modelo': modelo, 'metrica': metrica}
            print(f"✅ {nombre} completado - Métrica: {metrica:.4f}")
        
        # Resumen
        print("\n" + "=" * 80)
        print("📊 RESUMEN DE RESULTADOS - APRENDIZAJE POR REFUERZO")
        print("=" * 80)
        
        for nombre, resultado in resultados.items():
            print(f"{nombre:15}: {resultado['metrica']:.4f}")
        
        return resultados

def demo_ejemplos_reforzado():
    """Demostración completa de ejemplos de refuerzo"""
    print("=" * 80)
    print("🎯 LABORATORIO 4 - EJEMPLOS DE REFUERZO")
    print("=" * 80)
    
    # Crear y ejecutar ejemplos
    ejemplos = EjemplosRefuerzo()
    resultados = ejemplos.ejecutar_todos()
    
    print("\n" + "=" * 80)
    print("🎉 EJEMPLOS DE REFUERZO COMPLETADOS")
    print("=" * 80)

if __name__ == "__main__":
    demo_ejemplos_reforzado()
