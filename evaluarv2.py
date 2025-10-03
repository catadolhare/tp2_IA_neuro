from connect4 import Connect4
from agentes import Agent, RandomAgent, DefenderAgent
from principal import TrainedAgent

def main(episodes=1000, verbose=False, trained_first=True, agent="random"):
    agente_random: Agent = RandomAgent("Random")
    agente_defensor: Agent = DefenderAgent("Defender")
    agente_entrenado = TrainedAgent(
        "trained_model_vs_None_1000_0.99_1.0_0.1_0.9950.001_128_1000_100.pth",
        state_shape=(6,7), n_actions=7
    )
    if agent == "defender":
        if trained_first:
            agent1 = agente_entrenado
            agent2 = agente_defensor
        else:
            agent1 = agente_defensor
            agent2 = agente_entrenado
    else:
        if trained_first:
            agent1 = agente_entrenado
            agent2 = agente_random
        else:
            agent1 = agente_random
            agent2 = agente_entrenado

    print(f"Juego: {agent1.name} (Jugador 1) vs. {agent2.name} (Jugador 2)")

    wins1, wins2, draws = 0, 0, 0

    for i in range(episodes):
        juego = Connect4(agent1=agent1, agent2=agent2)
        ganador = juego.play(render=verbose)

        if ganador == 0:
            draws += 1
        elif ganador == 1:
            wins1 += 1
        else:
            wins2 += 1

    print("\nResultados finales:")
    print(f"Total partidas: {episodes}")
    print(f"{agent1.name} ganó {wins1} veces ({wins1/episodes:.2%})")
    print(f"{agent2.name} ganó {wins2} veces ({wins2/episodes:.2%})")
    print(f"Empates: {draws} ({draws/episodes:.2%})")

if __name__ == '__main__':
    # Configuración
    main(episodes=10000, verbose=False, trained_first=True, agent="random")
