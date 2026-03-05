# Agente de IA para Connect4

## Descripción

Este proyecto implementa un agente basado en aprendizaje automático para jugar Connect4.  
El objetivo fue entrenar un modelo capaz de aprender estrategias competitivas mediante interacción con el entorno del juego y evaluación frente a distintos oponentes.

---

## Componentes del proyecto

- Implementación del entorno del juego Connect4.
- Desarrollo de distintos agentes de juego.
- Entrenamiento de un modelo basado en redes neuronales.
- Evaluación del desempeño contra distintos oponentes.
- Búsqueda de hiperparámetros mediante grid search.

---

## Estructura del repositorio

- `connect4.py` – Implementación del entorno del juego Connect4.
- `agentes.py` – Definición de los distintos agentes que interactúan con el entorno.
- `entrenar.py` – Script principal para entrenar el modelo.
- `evaluar.py` – Evaluación del desempeño del agente entrenado.
- `entrenar_grid_search.py` – Ejecución de múltiples entrenamientos variando hiperparámetros.
- `evaluar_grid_search.py` – Comparación de resultados obtenidos en el grid search.
- `jugar_humano_contra_defensor.py` – Permite jugar contra un agente defensor.
- `jugar_trained_contra_random.py` – Partidas entre el agente entrenado y un agente aleatorio.
- `grid_search_trained/` – Modelos entrenados con distintas configuraciones.
- `informe/` – Informe técnico del proyecto.

---

## Resultados

El entrenamiento permitió desarrollar agentes capaces de competir eficazmente contra oponentes básicos, identificando configuraciones de hiperparámetros que mejoran el desempeño mediante experimentación sistemática.

---

## Tecnologías

- Python
- PyTorch
- Redes neuronales
- Aprendizaje por refuerzo

---

## Autoría

- Catalina Dolhare  
- Joaquín Schanz  
- Camila Cauzzo
