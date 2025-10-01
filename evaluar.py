from connect4 import Connect4
from agentes import RandomAgent, DefenderAgent
from principal import TrainedAgent

def jugar(model_path, opponent_name="random", verbose=False, episodes=10):
    """
    Juega partidas entre un TrainedAgent y un oponente dado.
    
    Args:
        model_path: ruta al archivo .pth entrenado.
        opponent_name: 'random' o 'defender'
        verbose: mostrar tablero durante el juego
        episodes: número de partidas a jugar
    """
    # Crear agente entrenado
    agente_entrenado = TrainedAgent(model_path, state_shape=(6,7), n_actions=7)

    # Seleccionar oponente
    if opponent_name.lower() == "random":
        opponent = RandomAgent("Random")
    elif opponent_name.lower() == "defender":
        opponent = DefenderAgent("Defensor")
    else:
        raise ValueError("Oponente desconocido. Usa 'random' o 'defender'.")

    # Estadísticas
    wins, losses, draws = 0, 0, 0

    for i in range(episodes):
        game = Connect4(agent1=agente_entrenado, agent2=opponent)
        winner = game.play(render=verbose)

        if winner == 1:  # agente entrenado es jugador 1
            wins += 1
        elif winner == 2:
            losses += 1
        else:
            draws += 1

    print("\nResultados contra", opponent.name)
    print(f"Partidas: {episodes}")
    print(f"Ganadas: {wins} ({wins/episodes:.2%})")
    print(f"Perdidas: {losses} ({losses/episodes:.2%})")
    print(f"Empates: {draws} ({draws/episodes:.2%})")

if __name__ == '__main__':
    # ---------------- CONFIGURACIÓN ----------------
    model_path = "trained_model_vs_None_1000_0.99_1.0_0.1_0.9950.001_128_1000_100.pth"  # cambia por tu archivo .pth
    opponent = "random"   # "random" o "defender"
    episodes = 100         # número de partidas
    verbose = True        # True para mostrar el tablero en cada turno
    # ------------------------------------------------

    jugar(model_path, opponent_name=opponent, verbose=verbose, episodes=episodes)
