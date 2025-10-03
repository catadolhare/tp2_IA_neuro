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
    model_name:str = f"trained_model_vs_{nombre_oponente}_{episodes}_{gamma}_" + \
                     f"{epsilon_start}_{epsilon_min}_{epsilon_decay}" + \
                     f"{alpha}_{batch_size}_{memory_size}_{target_update_every}"
    if verbose: print(model_name, flush=True)
    
    # Inicialización del ambiente
    env:Connect4Environment = Connect4Environment()
    
    # Inicialización del agente
    agent:Agent = DeepQLearningAgent(
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
    
        if (episode + 1) % 100 == 0:
            avg_loss = sum(episode_losses) / len(episode_losses) if episode_losses else 0
            if verbose: print(f"Episodio {episode + 1} finalizado. Epsilon: {agent.epsilon:.4f} | Loss promedio: {avg_loss:.6f}", flush=True)

    if verbose: print(flush=True)
    
    torch.save(agent.q_network.state_dict(), f"{model_name}.pth")


if __name__ == '__main__':

    # Definir oponentes
    opponents = [
        None, 
        RandomAgent("Random"), 
        DefenderAgent("Defensor")
    ]

    # Llamar al grid search
    
    alphas=[0.001, 0.0005, 0.0001]
    gammas=[0.90, 0.95, 0.99]
    epsilons=[1.0, 0.8, 0.5]
    episodes=500             # podés ajustar para test rápido
    batch_size=128
    memory_size=1000
    target_update_every=100
    verbose=True

    for alpha in alphas:
        for gamma in gammas:
            for epsilon in epsilons:
                for opponent in opponents:
                    nombre_oponente:str = 'None' if opponent==None else opponent.name
                    print(f"\nEntrenando con alpha={alpha}, gamma={gamma}, epsilon_start={epsilon}, oponente={nombre_oponente} \n")

                    entrenar(
                            episodes=episodes,
                            gamma=gamma,
                            epsilon_start=epsilon,
                            epsilon_min=0.1,
                            epsilon_decay=0.995,
                            alpha=alpha,
                            batch_size=batch_size,
                            memory_size=memory_size,
                            target_update_every=target_update_every,
                            opponent=opponent,
                            verbose=True
                            )

