import argparse
from connect4 import Connect4
from agentes import Agent, DefenderAgent, HumanAgent, RandomAgent
from principal import TrainedAgent

# adaptamos el codigo de jugar_humano_contra_defensor.py para que juegue el agente entrenado contra el agente random
def main(verbose, trained_first=True):
    agente_random:Agent = RandomAgent("Random")
    agente_entrenado = TrainedAgent("trained_model_vs_None_1000_0.99_1.0_0.1_0.9950.001_128_1000_100.pth", state_shape=(6,7), n_actions=7)

    if trained_first:
        agent1 = agente_entrenado
        agent2 = agente_random
    else:
        agent1 = agente_random
        agent2 = agente_entrenado

    print(f"Juego: {agent1.name} (Jugador 1) vs. {agent2.name} (Jugador 2)")

    juego = Connect4(agent1=agent1, agent2=agent2)
    ganador = juego.play(render=verbose)

    print("\nResultado del juego:")
    if ganador == 0:
        print("Empate")
    elif ganador == 1:
        print(f"Ganó el jugador {agent1.name}")
    else:
        print(f"Ganó el jugador {agent2.name}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Jugar una partida de Conecta 4: Humano vs Agente Defensor.")
    parser.add_argument('--human_first', action='store_true', help='El humano primero (Jugador 1)')
    parser.add_argument('-v', '--verbose', action='store_true', help='Mostrar el tablero en cada turno')
    
    args = parser.parse_args()
    main(args.verbose, args.human_first) 
