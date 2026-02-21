import numpy as np
import matplotlib.pyplot as plt
import random
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

class Laberinto:
    def __init__(self, tamaño=10, obstaculos=0.2):
        """
        Inicializa el laberinto
        
        Args:
            tamaño: tamaño del laberinto (tamaño x tamaño)
            obstaculos: proporción de obstáculos (0 a 1)
        """
        self.tamaño = tamaño
        self.obstaculos = obstaculos
        self.estado_inicial = (0, 0)
        self.estado_objetivo = (tamaño-1, tamaño-1)
        self.obstaculos_posiciones = set()
        self.generar_laberinto()
        
    def generar_laberinto(self):
        """Genera el laberinto con obstáculos aleatorios"""
        np.random.seed(42)
        
        # Generar obstáculos aleatorios
        n_obstaculos = int(self.tamaño * self.tamaño * self.obstaculos)
        
        for _ in range(n_obstaculos):
            while True:
                x = np.random.randint(0, self.tamaño)
                y = np.random.randint(0, self.tamaño)
                pos = (x, y)
                
                # No colocar obstáculos en inicio o meta
                if pos != self.estado_inicial and pos != self.estado_objetivo:
                    self.obstaculos_posiciones.add(pos)
                    break
    
    def reset(self):
        """Reinicia el laberinto al estado inicial"""
        return self.estado_inicial
    
    def step(self, estado, accion):
        """
        Ejecuta una acción en el laberinto
        
        Args:
            estado: estado actual (x, y)
            accion: acción a ejecutar (0=arriba, 1=derecha, 2=abajo, 3=izquierda)
            
        Returns:
            nuevo_estado, recompensa, terminado
        """
        x, y = estado
        
        # Definir movimientos
        movimientos = {
            0: (-1, 0),  # arriba
            1: (0, 1),   # derecha
            2: (1, 0),   # abajo
            3: (0, -1)   # izquierda
        }
        
        dx, dy = movimientos[accion]
        nuevo_x = x + dx
        nuevo_y = y + dy
        nuevo_estado = (nuevo_x, nuevo_y)
        
        # Verificar límites
        if (nuevo_x < 0 or nuevo_x >= self.tamaño or 
            nuevo_y < 0 or nuevo_y >= self.tamaño):
            return estado, -10, False  # Penalización por salirse
        
        # Verificar obstáculos
        if nuevo_estado in self.obstaculos_posiciones:
            return estado, -10, False  # Penalización por chocar
        
        # Verificar si llegó al objetivo
        if nuevo_estado == self.estado_objetivo:
            return nuevo_estado, 100, True  # Recompensa grande
        
        # Recompensa pequeña por moverse
        return nuevo_estado, -1, False
    
    def visualizar(self, camino=None, politica=None):
        """Visualiza el laberinto"""
        plt.figure(figsize=(8, 8))
        
        # Crear grid
        grid = np.zeros((self.tamaño, self.tamaño, 3))
        
        # Colores base
        grid[:, :] = [0.9, 0.9, 0.9]  # Gris claro para caminos
        
        # Obstáculos
        for x, y in self.obstaculos_posiciones:
            grid[x, y] = [0.2, 0.2, 0.2]  # Negro para obstáculos
        
        # Inicio
        grid[self.estado_inicial] = [0, 1, 0]  # Verde para inicio
        
        # Meta
        grid[self.estado_objetivo] = [1, 0, 0]  # Rojo para meta
        
        # Camino (si se proporciona)
        if camino:
            for x, y in camino[1:-1]:  # Excluir inicio y meta
                grid[x, y] = [0, 0.5, 1]  # Azul para camino
        
        # Política (si se proporciona)
        if politica:
            flechas = ['↑', '→', '↓', '←']
            for x in range(self.tamaño):
                for y in range(self.tamaño):
                    if (x, y) not in self.obstaculos_posiciones and (x, y) != self.estado_objetivo:
                        accion = np.argmax(politica[x, y])
                        plt.text(y, x, flechas[accion], ha='center', va='center', fontsize=8)
        
        plt.imshow(grid)
        plt.title('Laberinto')
        plt.xticks(range(self.tamaño))
        plt.yticks(range(self.tamaño))
        plt.grid(True, alpha=0.3)
        plt.show()

class QLearningAgente:
    def __init__(self, laberinto, learning_rate=0.1, discount_factor=0.95, epsilon=0.1):
        """
        Inicializa el agente Q-Learning
        
        Args:
            laberinto: instancia del laberinto
            learning_rate: tasa de aprendizaje
            discount_factor: factor de descuento
            epsilon: probabilidad de exploración
        """
        self.laberinto = laberinto
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.q_table = defaultdict(lambda: defaultdict(float))
        self.entrenamiento_info = []
        
    def get_q_value(self, estado, accion):
        """Obtiene el valor Q para un estado-acción"""
        return self.q_table[estado][accion]
    
    def set_q_value(self, estado, accion, valor):
        """Establece el valor Q para un estado-acción"""
        self.q_table[estado][accion] = valor
    
    def elegir_accion(self, estado, entrenando=True):
        """
        Elige una acción usando política epsilon-greedy
        
        Args:
            estado: estado actual
            entrenando: si está en modo entrenamiento (usa exploración)
            
        Returns:
            acción seleccionada
        """
        if entrenando and random.random() < self.epsilon:
            # Exploración: acción aleatoria
            return random.randint(0, 3)
        else:
            # Explotación: mejor acción conocida
            q_valores = [self.get_q_value(estado, a) for a in range(4)]
            max_q = max(q_valores)
            # En caso de empate, elegir aleatoriamente
            mejores_acciones = [a for a, q in enumerate(q_valores) if q == max_q]
            return random.choice(mejores_acciones)
    
    def actualizar_q_value(self, estado, accion, recompensa, proximo_estado):
        """
        Actualiza el valor Q usando la ecuación de Bellman
        
        Q(s,a) = Q(s,a) + α[r + γ * max(Q(s',a')) - Q(s,a)]
        """
        # Valor Q actual
        q_actual = self.get_q_value(estado, accion)
        
        # Mejor valor Q para el próximo estado
        max_q_proximo = max([self.get_q_value(proximo_estado, a) for a in range(4)])
        
        # Ecuación de Bellman
        nuevo_q = q_actual + self.learning_rate * (
            recompensa + self.discount_factor * max_q_proximo - q_actual
        )
        
        self.set_q_value(estado, accion, nuevo_q)
    
    def entrenar(self, n_episodios=1000):
        """Entrena al agente usando Q-Learning"""
        print(f"🧠 Entrenando agente Q-Learning por {n_episodios} episodios...")
        
        recompensas_episodio = []
        pasos_episodio = []
        
        for episodio in range(n_episodios):
            estado = self.laberinto.reset()
            recompensa_total = 0
            pasos = 0
            terminado = False
            
            while not terminado and pasos < 1000:  # Límite para evitar bucles infinitos
                # Elegir acción
                accion = self.elegir_accion(estado, entrenando=True)
                
                # Ejecutar acción
                proximo_estado, recompensa, terminado = self.laberinto.step(estado, accion)
                
                # Actualizar Q-value
                self.actualizar_q_value(estado, accion, recompensa, proximo_estado)
                
                # Actualizar estado
                estado = proximo_estado
                recompensa_total += recompensa
                pasos += 1
            
            recompensas_episodio.append(recompensa_total)
            pasos_episodio.append(pasos)
            
            # Reducir epsilon gradualmente
            self.epsilon = max(0.01, self.epsilon * 0.995)
            
            # Mostrar progreso
            if (episodio + 1) % 100 == 0:
                recompensa_promedio = np.mean(recompensas_episodio[-100:])
                pasos_promedio = np.mean(pasos_episodio[-100:])
                print(f"   Episodio {episodio + 1}: Recompensa promedio = {recompensa_promedio:.2f}, "
                      f"Pasos promedio = {pasos_promedio:.1f}, Epsilon = {self.epsilon:.3f}")
        
        self.entrenamiento_info = {
            'recompensas': recompensas_episodio,
            'pasos': pasos_episodio
        }
        
        print("✅ Entrenamiento completado")
        
    def visualizar_entrenamiento(self):
        """Visualiza el progreso del entrenamiento"""
        if not self.entrenamiento_info:
            print("❌ No hay información de entrenamiento")
            return
        
        plt.figure(figsize=(12, 4))
        
        # Recompensas
        plt.subplot(1, 2, 1)
        recompensas = self.entrenamiento_info['recompensas']
        plt.plot(recompensas, alpha=0.7)
        # Media móvil
        ventana = 50
        media_movil = np.convolve(recompensas, np.ones(ventana)/ventana, mode='valid')
        plt.plot(range(ventana-1, len(recompensas)), media_movil, 'r-', linewidth=2)
        plt.title('Recompensas por Episodio')
        plt.xlabel('Episodio')
        plt.ylabel('Recompensa Total')
        plt.legend(['Recompensas', f'Media móvil ({ventana})'])
        plt.grid(True)
        
        # Pasos
        plt.subplot(1, 2, 2)
        pasos = self.entrenamiento_info['pasos']
        plt.plot(pasos, alpha=0.7)
        media_movil_pasos = np.convolve(pasos, np.ones(ventana)/ventana, mode='valid')
        plt.plot(range(ventana-1, len(pasos)), media_movil_pasos, 'r-', linewidth=2)
        plt.title('Pasos por Episodio')
        plt.xlabel('Episodio')
        plt.ylabel('Número de Pasos')
        plt.legend(['Pasos', f'Media móvil ({ventana})'])
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()
        
    def obtener_politica(self):
        """Obtiene la política óptima aprendida"""
        politica = np.zeros((self.laberinto.tamaño, self.laberinto.tamaño, 4))
        
        for x in range(self.laberinto.tamaño):
            for y in range(self.laberinto.tamaño):
                estado = (x, y)
                if estado not in self.laberinto.obstaculos_posiciones and estado != self.laberinto.estado_objetivo:
                    q_valores = [self.get_q_value(estado, a) for a in range(4)]
                    politica[x, y] = q_valores
        
        return politica
    
    def probar_agente(self, max_pasos=100):
        """Prueba al agente entrenado"""
        print("🧪 Probando agente entrenado...")
        
        estado = self.laberinto.reset()
        camino = [estado]
        recompensa_total = 0
        pasos = 0
        terminado = False
        
        while not terminado and pasos < max_pasos:
            # Elegir mejor acción (sin exploración)
            accion = self.elegir_accion(estado, entrenando=False)
            
            # Ejecutar acción
            proximo_estado, recompensa, terminado = self.laberinto.step(estado, accion)
            
            # Actualizar estado
            estado = proximo_estado
            camino.append(estado)
            recompensa_total += recompensa
            pasos += 1
        
        print(f"📊 Resultados de la prueba:")
        print(f"   Pasos: {pasos}")
        print(f"   Recompensa total: {recompensa_total}")
        print(f"   Éxito: {'Sí' if terminado else 'No'}")
        
        # Visualizar laberinto con camino
        self.laberinto.visualizar(camino=camino)
        
        return camino, recompensa_total, terminado

def demo_qlearning_laberinto():
    """Demostración completa de Q-Learning en laberinto"""
    print("=" * 60)
    print("🧠 Q-LEARNING EN LABERINTO (APRENDIZAJE POR REFUERZO)")
    print("=" * 60)
    
    # Crear laberinto
    laberinto = Laberinto(tamaño=10, obstaculos=0.2)
    
    # Visualizar laberinto inicial
    print("🗺️  Laberinto generado:")
    laberinto.visualizar()
    
    # Crear agente
    agente = QLearningAgente(laberinto, learning_rate=0.1, discount_factor=0.95, epsilon=0.1)
    
    # Entrenar agente
    agente.entrenar(n_episodios=1000)
    
    # Visualizar progreso del entrenamiento
    agente.visualizar_entrenamiento()
    
    # Probar agente entrenado
    camino, recompensa, exito = agente.probar_agente()
    
    # Mostrar política aprendida
    politica = agente.obtener_politica()
    print("🗺️  Política aprendida:")
    laberinto.visualizar(politica=politica)
    
    print("\n" + "=" * 60)
    print("🎉 DEMOSTRACIÓN COMPLETADA")
    print("=" * 60)

if __name__ == "__main__":
    demo_qlearning_laberinto()
