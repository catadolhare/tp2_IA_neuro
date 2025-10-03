import torch
from principal import Connect4Environment, Connect4State, DeepQLearningAgent, TrainedAgent
from agentes import Agent, RandomAgent, DefenderAgent
import argparse
import random

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

    ''' Entrenar un Agente DQN en la cantidad de episodios y con los 
        parámetros indicados. 
        Entrena jugando contra el agente opponent, si está definido.
        Si opponent==None, entrena jugando contra sí mismo. '''

    nombre_oponente:str = 'None' if opponent==None else opponent.name
    model_name:str = f"trained_model_vs_{"None_Random_Defender"}_{episodes}_{gamma}_" + \
                     f"{epsilon_start}_{epsilon_min}_{epsilon_decay}" + \
                     f"{alpha}_{batch_size}_{memory_size}_{target_update_every}"
    if verbose: print(model_name, flush=True)
    
    # Inicialización del ambiente
    env: Connect4Environment = Connect4Environment()

    # Inicialización del agente DQN
    agent: Agent = DeepQLearningAgent(
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

    # Oponentes
    random_opponent = RandomAgent("Random")
    defender_opponent = DefenderAgent("Defensor")

    # Definir fases: 60% / 30% / 10%
    phase1_end = int(episodes * 0.6)
    phase2_end = int(episodes * 0.9)

    for episode in range(episodes):
        # Selección de rival por fase
        if episode < phase1_end:
            opponent = None                 # 1ª fase: self-play
        elif episode < phase2_end:
            opponent = random_opponent      # 2ª fase: random
        else:
            opponent = defender_opponent    # 3ª fase: defender

        dqn_player = 1  # tu agente siempre jugador 1

        state: Connect4State = env.reset()
        done = False
        episode_losses = []

        while not done:
            valid_actions = env.available_actions()

            if env.state.current_player == dqn_player or opponent is None:
                # Turno del DQN (o self-play)
                action = agent.select_action(state, valid_actions)
                next_state, reward, done, _ = env.step(action)
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

        if (episode + 1) % 100 == 0:
            avg_loss = sum(episode_losses) / len(episode_losses) if episode_losses else 0
            print(f"Episodio {episode+1}/{episodes} | "
                  f"Fase={'Self' if episode<phase1_end else 'Random' if episode<phase2_end else 'Defender'} | "
                  f"Epsilon={agent.epsilon:.3f} | Loss={avg_loss:.4f}")

    if verbose: print(flush=True)
    
    torch.save(agent.q_network.state_dict(), f"{model_name}.pth")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Entrenar un agente usando DQL en el ambiente de 'Connect4'.")

    # Agregar argumentos
    parser.add_argument('-n', '--episodes', type=int, default=1000, help='Número de episodios para entrenar al agente (default: 1000)')
    parser.add_argument('-g', '--gamma', type=float, default=0.99, help='Factor de descuento (default: 0.99)')
    parser.add_argument('-es', '--epsilon_start', type=float, default=1.0, help='Valor inicial de la tasa de exploración (default: 1.0)')
    parser.add_argument('-em', '--epsilon_min', type=float, default=0.1, help='Valor mínimo de la tasa de exploración (default: 0.1)')
    parser.add_argument('-ed', '--epsilon_decay', type=float, default=0.995, help='Decay de la tasa de exploración (default: 0.995)')
    parser.add_argument('-a', '--alpha', type=float, default=0.001, help='Tasa de aprendizaje (default: 0.001)')
    parser.add_argument('-bs', '--batch_size', type=int, default=128, help='Tamaño del batch usado para aprendizaje (default: 128)')
    parser.add_argument('-ms', '--memory_size', type=int, default=1000, help='Tamaño de la memoria de experiencias del agente (default: 1000)')
    parser.add_argument('-ue', '--target_update_every', type=int, default=100, help='Cada cuánto actualizar red objetivo (default: 100)')
    parser.add_argument('-of', '--opponent_model_path', type=str, help='Archivo pth del agente DQN para usar como oponente.')
    parser.add_argument('-v', '--verbose', action='store_true', help='Activar modo verbose para ver más detalles durante el entrenamiento')

    # Parsear los argumentos
    args = parser.parse_args()
    
    if args.opponent_model_path != None:
        agente_entrenado:Agent = TrainedAgent(
            model_path=args.opponent_model_path,
            state_shape=(6, 7),
            n_actions=7,
            device=device
        )
    else: 
        agente_entrenado = None


    # Oponente al azar
    opponent = RandomAgent("Random")

    # Oponente defensor
    #opponent = DefenderAgent("Defensor")

    # Oponente entrenado previamente (otro .pth)
    #opponent = agente_entrenado

    entrenar(
        episodes=1000,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_min=0.1,
        epsilon_decay=0.995,
        alpha=0.001,
        batch_size=128,
        memory_size=1000,
        target_update_every=100,
        opponent=opponent,
        verbose=True
    )

