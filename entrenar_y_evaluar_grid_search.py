import os
import torch
import csv
from principal import Connect4Environment, Connect4State, DeepQLearningAgent, TrainedAgent
from agentes import Agent, RandomAgent, DefenderAgent
from connect4 import Connect4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def entrenar(episodes:int=500,
             gamma:float=0.99,
             epsilon_start:float=1.0,
             epsilon_min:float=0.1,
             epsilon_decay:float=0.995,
             alpha:float=0.001,
             batch_size:int=64,
             memory_size:int=500,
             target_update_every:int=100,
             opponent:Agent=None,
             verbose:bool=True):
    '''
    Entrena un Agente DQN y devuelve el agente entrenado (sin guardar en disco).
    '''
    nombre_oponente:str = 'None' if opponent==None else opponent.name
    model_name:str = f"trained_model_vs_{nombre_oponente}_{episodes}_{gamma}_" + \
                     f"{epsilon_start}_{epsilon_min}_{epsilon_decay}" + \
                     f"{alpha}_{batch_size}_{memory_size}_{target_update_every}"

    if verbose:
        print(f"Entrenando: {model_name}")

    # Inicialización del ambiente
    env:Connect4Environment = Connect4Environment()

    # Inicialización del agente
    agent:DeepQLearningAgent = DeepQLearningAgent(
        state_shape=(env.rows, env.cols),
        n_actions=env.cols,
        device=device,
        gamma=gamma,
        epsilon=epsilon_start,
        epsilon_min=epsilon_min,
        epsilon_decay=epsilon_decay,
        lr=alpha,
        batch_size=batch_size,
        memory_size=memory_size,
        target_update_every=target_update_every
    )

    # Entrenamiento
    for episode in range(episodes):
        state:Connect4State = env.reset()
        done:bool = False
        episode_losses = []
        dqn_player = 1  # DQN siempre es jugador 1

        while not done:
            valid_actions = env.available_actions()
            if env.state.current_player==dqn_player or opponent==None:
                # Turno del DQN (o no hay oponente)
                action = agent.select_action(state, valid_actions)
                next_state, reward, done, _ = env.step(action)
                # Solo almacenar experiencias cuando DQN juega
                agent.store_transition(state, action, reward, next_state, done)
                loss = agent.train_step()
                if loss is not None:
                    episode_losses.append(loss)
            else:
                # Turno del oponente
                action = opponent.play(state, valid_actions)
                next_state, reward, done, _ = env.step(action)

            state = next_state

        agent.update_epsilon()

        if verbose and (episode + 1) % 100 == 0:
            avg_loss = sum(episode_losses) / len(episode_losses) if episode_losses else 0
            print(f"  Episodio {episode + 1}/{episodes} | Epsilon: {agent.epsilon:.4f} | Loss: {avg_loss:.6f}")

    if verbose:
        print("  Entrenamiento completado!\n")

    return agent, model_name


def evaluar_agente(agent_dqn, episodes=1000, verbose=False, trained_first=True, opponent="random"):
    '''
    Evalúa un agente DQN entrenado contra Random o Defender.
    Devuelve las estadísticas de victorias.
    '''
    agente_random: Agent = RandomAgent("Random")
    agente_defensor: Agent = DefenderAgent("Defender")

    # Crear un TrainedAgent temporal usando la red del agente entrenado
    class TempTrainedAgent(Agent):
        def __init__(self, q_network, device, n_actions):
            self.model = q_network
            self.device = device
            self.n_actions = n_actions
            self.name = "TrainedAgent"

        def preprocess(self, state):
            board_flat = state.board.flatten()
            return torch.FloatTensor(board_flat).to(self.device)

        def play(self, state, valid_actions):
            with torch.no_grad():
                state_tensor = self.preprocess(state).unsqueeze(0)
                q_values = self.model(state_tensor).squeeze(0)
                mask = torch.full((self.n_actions,), float('-inf'), device=self.device)
                mask[valid_actions] = 0
                q_values = q_values + mask
                return q_values.argmax().item()

    agente_entrenado = TempTrainedAgent(agent_dqn.q_network, device, 7)

    if opponent == "defender":
        if trained_first:
            agent1, agent2 = agente_entrenado, agente_defensor
        else:
            agent1, agent2 = agente_defensor, agente_entrenado
    else:  # opponent == "random"
        if trained_first:
            agent1, agent2 = agente_entrenado, agente_random
        else:
            agent1, agent2 = agente_random, agente_entrenado

    wins1, wins2, draws = 0, 0, 0
    for _ in range(episodes):
        juego = Connect4(agent1=agent1, agent2=agent2)
        ganador = juego.play(render=verbose)
        if ganador == 0:
            draws += 1
        elif ganador == 1:
            wins1 += 1
        else:
            wins2 += 1

    # Ajustar las victorias según quién es el agente entrenado
    if trained_first:
        wins_trained = wins1
        losses_trained = wins2
    else:
        wins_trained = wins2
        losses_trained = wins1

    return {
        "ganadas": wins_trained,
        "perdidas": losses_trained,
        "empates": draws,
        "porcentaje_victorias": wins_trained / episodes
    }


if __name__ == '__main__':
    # Configuración del Grid Search
    alphas = [0.005, 0.001, 0.0005, 0.0001]
    gammas = [0.9, 0.95, 0.98, 0.99]
    epsilons = [1.0, 0.9, 0.8, 0.7]
    episodes_train = 1000
    episodes_eval = 1000
    batch_size = 128
    memory_size = 1000
    target_update_every = 100
    verbose = True

    # Oponentes para entrenar
    opponents = [
        None,
        RandomAgent("Random"),
        DefenderAgent("Defensor")
    ]

    # Contra quién evaluar (puedes cambiar a "defender" si prefieres)
    eval_opponent = "random"

    # Variables para trackear el mejor modelo
    best_model_agent = None
    best_model_name = None
    best_win_rate = 0.0

    # Lista para guardar todos los resultados
    all_results = []

    # Contador de modelos
    total_models = len(alphas) * len(gammas) * len(epsilons) * len(opponents)
    current_model = 0

    print(f"=== GRID SEARCH INICIADO ===")
    print(f"Total de modelos a entrenar: {total_models}")
    print(f"Evaluación contra: {eval_opponent.upper()}")
    print(f"Device: {device}\n")

    # Grid Search
    for alpha in alphas:
        for gamma in gammas:
            for epsilon in epsilons:
                for opponent in opponents:
                    current_model += 1
                    nombre_oponente = 'None' if opponent == None else opponent.name

                    print(f"\n[{current_model}/{total_models}] " + "="*50)
                    print(f"Hiperparámetros: alpha={alpha}, gamma={gamma}, epsilon={epsilon}")
                    print(f"Oponente entrenamiento: {nombre_oponente}")

                    # Entrenar agente
                    agent, model_name = entrenar(
                        episodes=episodes_train,
                        gamma=gamma,
                        epsilon_start=epsilon,
                        epsilon_min=0.1,
                        epsilon_decay=0.995,
                        alpha=alpha,
                        batch_size=batch_size,
                        memory_size=memory_size,
                        target_update_every=target_update_every,
                        opponent=opponent,
                        verbose=verbose
                    )

                    # Evaluar agente
                    print(f"Evaluando contra {eval_opponent}...")
                    results = evaluar_agente(
                        agent,
                        episodes=episodes_eval,
                        verbose=False,
                        trained_first=True,
                        opponent=eval_opponent
                    )

                    win_rate = results["porcentaje_victorias"]
                    print(f"Resultado: {win_rate:.2%} victorias ({results['ganadas']}/{episodes_eval})")

                    # Guardar resultado en la lista
                    all_results.append({
                        "modelo": model_name,
                        "alpha": alpha,
                        "gamma": gamma,
                        "epsilon_start": epsilon,
                        "oponente_entrenamiento": nombre_oponente,
                        "oponente_evaluacion": eval_opponent,
                        "episodios_eval": episodes_eval,
                        "ganadas": results["ganadas"],
                        "perdidas": results["perdidas"],
                        "empates": results["empates"],
                        "porcentaje_victorias": win_rate
                    })

                    # Verificar si es el mejor modelo hasta ahora
                    if win_rate > best_win_rate:
                        best_win_rate = win_rate
                        best_model_agent = agent
                        best_model_name = model_name
                        print(f">>> NUEVO MEJOR MODELO! Win rate: {win_rate:.2%}")

    # Guardar solo el mejor modelo
    print("\n" + "="*60)
    print("=== GRID SEARCH COMPLETADO ===")
    print(f"\nMejor modelo: {best_model_name}")
    print(f"Win rate: {best_win_rate:.2%}")

    # Crear carpeta para el mejor modelo
    os.makedirs("best_model", exist_ok=True)
    model_path = os.path.join("best_model", f"{best_model_name}.pth")
    torch.save(best_model_agent.q_network.state_dict(), model_path)
    print(f"\nMejor modelo guardado en: {model_path}")

    # Guardar todos los resultados en CSV
    csv_path = "resultados_grid_search.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = all_results[0].keys()
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)

    print(f"Todos los resultados guardados en: {csv_path}")

    # Mostrar top 5 modelos
    print("\n=== TOP 5 MODELOS ===")
    sorted_results = sorted(all_results, key=lambda x: x["porcentaje_victorias"], reverse=True)
    for i, result in enumerate(sorted_results[:5], 1):
        print(f"{i}. {result['porcentaje_victorias']:.2%} - alpha={result['alpha']}, "
              f"gamma={result['gamma']}, epsilon={result['epsilon_start']}, "
              f"vs_{result['oponente_entrenamiento']}")

