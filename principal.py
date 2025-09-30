import random
import torch
import torch.nn as nn
from agentes import Agent
import numpy as np
import torch.nn.functional as F

class Connect4State:
    def __init__(self, rows=6, cols=7): 
        """
        Inicializa el estado del juego Connect4.
        
        Args:
            Definir qué hace a un estado de Connect4.
        """
        self.board = np.zeros((rows, cols), dtype=int)  # Tablero de 6 filas y 7 columnas
        self.current_player = 1  # Jugador 1 comienza

    def copy(self):  
        """
        Crea una copia profunda del estado actual.
        
        Returns:
            Una nueva instancia de Connect4State con los mismos valores.
        """
        new_state = Connect4State()
        new_state.board = self.board.copy()
        new_state.current_player = self.current_player
        return new_state

    """def update_state(self):
        """"""
        Modifica las variables internas del estado luego de una jugada.

        Args:
            ... (_type_): _description_
            ... (_type_): _description_
        """"""
        self.current_player = 2 if self.current_player == 1 else 1"""

    def update_state(self, col): #chat me dice que falta agregarle col para poder marcar donde se jugo
        for row in range(5, -1, -1):
            if self.board[row, col] == 0:
                self.board[row, col] = self.current_player
                self.current_player *= -1
                return
        raise ValueError("Columna llena")

    def __eq__(self, other):
        """
        Compara si dos estados son iguales.
        
        Args:
            other: Otro estado para comparar.
            
        Returns:
            True si los estados son iguales, False en caso contrario.
        """
        return np.array_equal(self.board, other.board) and self.current_player == other.current_player

    def __hash__(self): 
        """
        Genera un hash único para el estado.
        
        Returns:
            Hash del estado basado en el tablero y jugador actual.
        """
        return hash((self.board.tobytes(), self.current_player))

    def __repr__(self):
        """
        Representación en string del estado.
        
        """
        return f"Player: {self.current_player}\nBoard:\n{self.board}"

class Connect4Environment:
    def __init__(self, rows=6, cols=7):
        """
        Inicializa el ambiente del juego Connect4.
        
        Args:
            Definir las variables de instancia de un ambiente de Connect4

        """
        self.state = Connect4State()
        self.done = False
        self.winner = None
        self.rows = rows
        self.cols = cols

    def reset(self):
        """
        Reinicia el ambiente a su estado inicial para volver a realizar un episodio.
        
        """
        self.state = Connect4State()
        self.done = False
        self.winner = None
        return self.state

    def available_actions(self):
        """
        Obtiene las acciones válidas (columnas disponibles) en el estado actual.
        
        Returns:
            Lista de índices de columnas donde se puede colocar una ficha.
        """
        indices_col = []
        for col in range(self.state.board.shape[1]):
            if self.state.board[0, col] == 0:
                indices_col.append(col)
        return indices_col

    def step(self, action):
        """
        Ejecuta una acción.
        El estado es modificado acorde a la acción y su interacción con el ambiente.
        Devuelve la tupla: nuevo_estado, reward, terminó_el_juego?, ganador
        Si terminó_el_juego==false, entonces ganador es None.
        
        Args:
            action: Acción elegida por un agente.
            
        """
        if self.done:
            raise Exception("El juego ya terminó. Llama a reset() para reiniciar.")
        # Realiza la jugada
        self.state.update_state(action)
        # Verifica si hay ganador
        self.winner = self.check_winner()
        self.done = self.winner is not None or len(self.available_actions()) == 0
        reward = 0
        if self.winner is not None:
            reward = 1
        elif self.done:
            reward = 0.5  # Empate
        return self.state.copy(), reward, self.done, self.winner
    
    def check_winner(self):
        """
        Chequea si hay un ganador en el tablero actual.
        """
        board = self.state.board
        for r in range(6):
            for c in range(7):
                player = board[r, c]
                if player == 0:
                    continue
                # Horizontal
                if c <= 3 and all(board[r, c+i] == player for i in range(4)):
                    return player
                # Vertical
                if r <= 2 and all(board[r+i, c] == player for i in range(4)):
                    return player
                # Diagonal \
                if r <= 2 and c <= 3 and all(board[r+i, c+i] == player for i in range(4)):
                    return player
                # Diagonal /
                if r <= 2 and c >= 3 and all(board[r+i, c-i] == player for i in range(4)):
                    return player
        return None

    def render(self):
        """
        Muestra visualmente el estado actual del tablero en la consola.

        """
        print(np.flip(self.state.board, 0))

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim): 
        """
        Inicializa la red neuronal DQN para el aprendizaje por refuerzo.
        
        Args:
            input_dim: Dimensión de entrada (número de features del estado).
            output_dim: Dimensión de salida (número de acciones posibles).
        """
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, x):
        """
        Pasa la entrada a través de la red neuronal.
        
        Args:
            x: Tensor de entrada.
            
        Returns:
            Tensor de salida con los valores Q para cada acción.
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class DeepQLearningAgent:
    def __init__(self, state_shape, n_actions, device, gamma, epsilon, epsilon_min, epsilon_decay, lr, batch_size, memory_size, target_update_every): 
        """
        Inicializa el agente de aprendizaje por refuerzo DQN.
        
        Args:
            state_shape: Forma del estado (filas, columnas).
            n_actions: Número de acciones posibles.
            device: Dispositivo para computación ('cpu' o 'cuda').
            gamma: Factor de descuento para recompensas futuras.
            epsilon: Probabilidad inicial de exploración.
            epsilon_min: Valor mínimo de epsilon.
            epsilon_decay: Factor de decaimiento de epsilon.
            lr: Tasa de aprendizaje.
            batch_size: Tamaño del batch para entrenamiento.
            memory_size: Tamaño máximo de la memoria de experiencias.
            target_update_every: Frecuencia de actualización de la red objetivo.
        """
        self.state_shape = state_shape
        self.n_actions = n_actions
        self.device = device
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.lr = lr
        self.batch_size = batch_size
        self.memory_size = memory_size
        self.target_update_every = target_update_every

        self.q_network = DQN(np.prod(state_shape), n_actions).to(device)
        self.target_net = DQN(np.prod(state_shape), n_actions).to(device)
        self.target_net.load_state_dict(self.q_network.state_dict())
        self.target_net.eval()

        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=lr)
        self.memory = []
        self.steps_done = 0

    def preprocess(self, state):
        """
        Convierte el estado del juego a un tensor de PyTorch.
        
        Args:
            state: Estado del juego.
            
        Returns:
            Tensor de PyTorch con el estado aplanado.
        """
        board = state.board.astype(np.float32).flatten()
        return torch.tensor(board, dtype=torch.float32, device=self.device).unsqueeze(0)

    def select_action(self, state, valid_actions): 
        """
        Selecciona una acción usando la política epsilon-greedy.
        
        Args:
            state: Estado actual del juego.
            valid_actions: Lista de acciones válidas.
            
        Returns:
            Índice de la acción seleccionada.
        """
        # Política epsilon-greedy
        if np.random.rand() < self.epsilon:
            return random.choice(valid_actions)
        with torch.no_grad():
            state_tensor = self.preprocess(state)
            q_values = self.q_network(state_tensor).cpu().numpy().flatten()
            # Selecciona la acción válida con mayor valor Q
            q_valid = [(a, q_values[a]) for a in valid_actions]
            best_action = max(q_valid, key=lambda x: x[1])[0]
            return best_action


    def store_transition(self, s, a, r, s_next, done):
        """
        Almacena una transición (estado, acción, recompensa, siguiente estado, terminado) en la memoria.
        
        Args:
            s: Estado actual.
            a: Acción tomada.
            r: Recompensa obtenida.
            s_next: Siguiente estado.
            done: Si el episodio terminó.
        """
        if len(self.memory) >= self.memory_size:
            self.memory.pop(0)
        self.memory.append((s, a, r, s_next, done))

    def train_step(self): 
        """
        Ejecuta un paso de entrenamiento usando experiencias de la memoria.
        
        Returns:
            Valor de la función de pérdida si se pudo entrenar, None en caso contrario.
        """
        if len(self.memory) < self.batch_size:
            return None
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.cat([self.preprocess(s) for s in states])
        actions = torch.tensor(actions, dtype=torch.long, device=self.device).unsqueeze(1)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_states = torch.cat([self.preprocess(s) for s in next_states])
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1)

        # Q(s,a)
        q_values = self.q_network(states).gather(1, actions)
        # max_a' Q_target(s',a')
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0].unsqueeze(1)
            target = rewards + self.gamma * next_q_values * (1 - dones)

        loss = torch.nn.functional.mse_loss(q_values, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Actualiza la red objetivo cada cierto número de pasos
        self.steps_done += 1
        if self.steps_done % self.target_update_every == 0:
            self.target_net.load_state_dict(self.q_network.state_dict())

        return loss.item()


    def update_epsilon(self):
        """
        Actualiza el valor de epsilon para reducir la exploración gradualmente.
        """
        # Decaimiento de epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            self.epsilon = max(self.epsilon, self.epsilon_min)

class TrainedAgent(Agent):
    def __init__(self, model_path: str, state_shape: tuple, n_actions: int, device='cpu'):
        """
        Inicializa un agente DQN pre-entrenado.
        
        Args:
            model_path: Ruta al archivo del modelo entrenado.
            state_shape: Forma del estado del juego.
            n_actions: Número de acciones posibles.
            device: Dispositivo para computación.
        """
        self.device = device
        self.model = DQN(np.prod(state_shape), n_actions).to(device)
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.eval()
        self.state_shape = state_shape
        self.n_actions = n_actions


    def play(self, state, valid_actions): 
        """
        Selecciona la mejor acción según el modelo entrenado.
        
        Args:
            state: Estado actual del juego.
            valid_actions: Lista de acciones válidas.
            
        Returns:
            Índice de la mejor acción según el modelo.
        """
        board = state.board.astype(np.float32).flatten()
        state_tensor = torch.tensor(board, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.model(state_tensor).cpu().numpy().flatten()
        # Solo considerar acciones válidas
        q_valid = [(a, q_values[a]) for a in valid_actions]
        best_action = max(q_valid, key=lambda x: x[1])[0]
        return best_action
