import os
import glob
import csv
from connect4 import Connect4
from agentes import Agent, RandomAgent, DefenderAgent
from principal import TrainedAgent

def evaluar_modelo(model_path, episodes=1000, verbose=False, trained_first=True, opponent="random"):
    agente_random: Agent = RandomAgent("Random")
    agente_defensor: Agent = DefenderAgent("Defender")
    agente_entrenado = TrainedAgent(model_path, state_shape=(6,7), n_actions=7)

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

    return {
        "modelo": os.path.basename(model_path),
        "oponente": opponent,
        "episodios": episodes,
        "ganadas_entrenado": wins1,
        "perdidas_entrenado": wins2,
        "empates": draws,
        "porcentaje_victorias": wins1 / episodes
    }

if __name__ == '__main__':
    # Carpeta donde están guardados los modelos
    carpeta_modelos = "grid_search_trained"  
    modelos = glob.glob(os.path.join(carpeta_modelos, "*.pth"))

    opponent = "random"   # o "defender"
    episodes = 1000
    verbose = False
    trained_first = True

    resultados = []
    for modelo in modelos:
        print(f"Evaluando {modelo} ...")
        res = evaluar_modelo(modelo, episodes=episodes, verbose=verbose, trained_first=trained_first, opponent=opponent)
        resultados.append(res)
        print(f" → {res['porcentaje_victorias']:.2%} victorias")

    # Guardar resultados en CSV
    with open("resultados_evaluacion.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=resultados[0].keys())
        writer.writeheader()
        writer.writerows(resultados)

    print("\n✅ Resultados guardados en resultados_evaluacion.csv")
