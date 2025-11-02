import torch
import torch.nn as nn
import numpy as np
from agentes import Agent
from collections import deque
import random
import utils
import torch.nn.functional as F

class Connect4State:
    def __init__(self, board=None, current_player=1, rows=6, cols=7):
        """
        Inicializa el estado del juego Connect4.

        Args:
            board: Tablero del juego (matriz numpy). Si es None, se crea uno vacío.
            current_player: Jugador actual (1 o 2).
            rows: Número de filas del tablero.
            cols: Número de columnas del tablero.
        """
        self.rows = rows
        self.cols = cols
        if board is None:
            self.board = utils.create_board(rows, cols)
        else:
            self.board = board
        self.current_player = current_player

    def copy(self):
        """
        Crea una copia profunda del estado actual.

        Returns:
            Una nueva instancia de Connect4State con los mismos valores.
        """
        return Connect4State(
            board=np.copy(self.board),
            current_player=self.current_player,
            rows=self.rows,
            cols=self.cols
        )

    def update_state(self, col, player):
        """
        Modifica las variables internas del estado luego de una jugada.

        Args:
            col (int): Columna donde se coloca la ficha.
            player (int): Jugador que realiza la jugada (1 o 2).
        """
        utils.insert_token(self.board, col, player)
        # Cambiar al siguiente jugador
        self.current_player = 3 - self.current_player

    def __eq__(self, other):
        """
        Compara si dos estados son iguales.

        Args:
            other: Otro estado para comparar.

        Returns:
            True si los estados son iguales, False en caso contrario.
        """
        if not isinstance(other, Connect4State):
            return False
        return (np.array_equal(self.board, other.board) and
                self.current_player == other.current_player)

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
        return f"Connect4State(current_player={self.current_player})\n{self.board}"

class Connect4Environment:
    def __init__(self, rows=6, cols=7):
        """
        Inicializa el ambiente del juego Connect4.

        Args:
            rows: Número de filas del tablero (default: 6).
            cols: Número de columnas del tablero (default: 7).
        """
        self.rows = rows
        self.cols = cols
        self.state = None
        self.reset()

    def reset(self):
        """
        Reinicia el ambiente a su estado inicial para volver a realizar un episodio.

        Returns:
            El estado inicial del juego.
        """
        self.state = Connect4State(rows=self.rows, cols=self.cols)
        return self.state

    def available_actions(self):
        """
        Obtiene las acciones válidas (columnas disponibles) en el estado actual.

        Returns:
            Lista de índices de columnas donde se puede colocar una ficha.
        """
        valid_actions = []
        for col in range(self.cols):
            # Una columna es válida si la fila superior está vacía
            if self.state.board[0, col] == 0:
                valid_actions.append(col)
        return valid_actions

    def step(self, action):
        """
        Ejecuta una acción.
        El estado es modificado acorde a la acción y su interacción con el ambiente.
        Devuelve la tupla: nuevo_estado, reward, terminó_el_juego?, info
        Si terminó_el_juego==false, entonces ganador es None.

        Args:
            action: Acción elegida por un agente (columna donde colocar la ficha).

        Returns:
            tuple: (nuevo_estado, reward, done, info)
        """
        current_player = self.state.current_player

        # Actualizar el estado con la acción
        self.state.update_state(action, current_player)

        # Verificar si el juego terminó
        done, winner = utils.check_game_over(self.state.board)

        # Calcular recompensa
        reward = 0
        if done:
            if winner == current_player:
                reward = 1  # Ganó el jugador actual
            elif winner is None:
                reward = 0  # Empate
            else:
                reward = -1  # Perdió el jugador actual

        info = {"winner": winner if done else None}

        return self.state, reward, done, info

    def render(self):
        """
        Muestra visualmente el estado actual del tablero en la consola.
        """
        print("\n" + "  ".join([str(i) for i in range(self.cols)]))
        for row in self.state.board:
            row_str = "  ".join(["." if cell == 0 else str(cell) for cell in row])
            print(row_str)
        print()

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        """
        Inicializa la red neuronal DQN para el aprendizaje por refuerzo.

        Args:
            input_dim: Dimensión de entrada (número de features del estado).
            output_dim: Dimensión de salida (número de acciones posibles).
        """
        super(DQN, self).__init__()

        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, output_dim)

        self.relu = nn.ReLU()

    def forward(self, x):
        """
        Pasa la entrada a través de la red neuronal.

        Args:
            x: Tensor de entrada.

        Returns:
            Tensor de salida con los valores Q para cada acción.
        """
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        x = self.fc5(x)
        return x

class DeepQLearningAgent:
    def __init__(self, state_shape, n_actions, device,
                 gamma, epsilon=1.0, epsilon_min=0.1, epsilon_decay=0.995,
                 lr=0.001, batch_size=64, memory_size=10000, target_update_every=100):
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
        self.batch_size = batch_size
        self.target_update_every = target_update_every

        # Dimensión de entrada es el tamaño del tablero aplanado
        input_dim = state_shape[0] * state_shape[1]

        # Red Q principal y red objetivo
        self.q_network = DQN(input_dim, n_actions).to(device)
        self.target_network = DQN(input_dim, n_actions).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        # Optimizador y función de pérdida
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

        # Memoria de experiencias (replay buffer)
        self.memory = deque(maxlen=memory_size)

        # Contador de pasos para actualizar la red objetivo
        self.steps = 0

    def preprocess(self, state):
        """
        Convierte el estado del juego a un tensor de PyTorch.

        Args:
            state: Estado del juego.

        Returns:
            Tensor de PyTorch con el estado aplanado.
        """
        # Aplanar el tablero y convertir a tensor
        board_flat = state.board.flatten()
        return torch.FloatTensor(board_flat).to(self.device)

    def select_action(self, state, valid_actions):
        """
        Selecciona una acción usando la política epsilon-greedy.

        Args:
            state: Estado actual del juego.
            valid_actions: Lista de acciones válidas.

        Returns:
            Índice de la acción seleccionada.
        """
        if random.random() < self.epsilon:
            # Exploración: elegir acción aleatoria
            return random.choice(valid_actions)
        else:
            # Explotación: elegir mejor acción según Q-network
            with torch.no_grad():
                state_tensor = self.preprocess(state).unsqueeze(0)
                q_values = self.q_network(state_tensor).squeeze(0)

                # Enmascarar acciones inválidas con un valor muy bajo
                mask = torch.full((self.n_actions,), float('-inf'), device=self.device)
                mask[valid_actions] = 0
                q_values = q_values + mask

                return q_values.argmax().item()

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
        self.memory.append((s, a, r, s_next, done))

    def train_step(self):
        """
        Ejecuta un paso de entrenamiento usando experiencias de la memoria.

        Returns:
            Valor de la función de pérdida si se pudo entrenar, None en caso contrario.
        """
        if len(self.memory) < self.batch_size:
            return None

        # Muestrear un batch de experiencias
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # Convertir a tensores
        states_tensor = torch.stack([self.preprocess(s) for s in states])
        actions_tensor = torch.LongTensor(actions).to(self.device)
        rewards_tensor = torch.FloatTensor(rewards).to(self.device)
        next_states_tensor = torch.stack([self.preprocess(s) for s in next_states])
        dones_tensor = torch.FloatTensor(dones).to(self.device)

        # Calcular valores Q actuales
        current_q_values = self.q_network(states_tensor).gather(1, actions_tensor.unsqueeze(1)).squeeze(1)

        # Calcular valores Q objetivo usando la red objetivo
        with torch.no_grad():
            next_q_values = self.target_network(next_states_tensor).max(1)[0]
            target_q_values = rewards_tensor + (1 - dones_tensor) * self.gamma * next_q_values

        # Calcular pérdida y realizar backpropagation
        loss = self.loss_fn(current_q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Actualizar la red objetivo periódicamente
        self.steps += 1
        if self.steps % self.target_update_every == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        return loss.item()

    def update_epsilon(self):
        """
        Actualiza el valor de epsilon para reducir la exploración gradualmente.
        """
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

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
        self.state_shape = state_shape
        self.n_actions = n_actions
        self.device = torch.device(device)

        # Cargar el modelo entrenado
        input_dim = state_shape[0] * state_shape[1]
        self.model = DQN(input_dim, n_actions).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
        self.model.eval()

        self.name = "TrainedAgent"

    def preprocess(self, state):
        """
        Convierte el estado del juego a un tensor de PyTorch.

        Args:
            state: Estado del juego.

        Returns:
            Tensor de PyTorch con el estado aplanado.
        """
        board_flat = state.board.flatten()
        return torch.FloatTensor(board_flat).to(self.device)

    def play(self, state, valid_actions):
        """
        Selecciona la mejor acción según el modelo entrenado.

        Args:
            state: Estado actual del juego.
            valid_actions: Lista de acciones válidas.

        Returns:
            Índice de la mejor acción según el modelo.
        """
        with torch.no_grad():
            state_tensor = self.preprocess(state).unsqueeze(0)
            q_values = self.model(state_tensor).squeeze(0)

            # Enmascarar acciones inválidas con un valor muy bajo
            mask = torch.full((self.n_actions,), float('-inf'), device=self.device)
            mask[valid_actions] = 0
            q_values = q_values + mask

            return q_values.argmax().item()
