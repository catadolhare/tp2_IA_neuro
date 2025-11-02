# Informe Técnico - TP2: Agente de Connect4 con Deep Q-Learning

**Autores:** Cauzzo, Dolhare, Schanz

## 1. Decisiones de Arquitectura

### 1.1 Red Neuronal (DQN)
Se implementó una red neuronal profunda con la siguiente arquitectura:
- **Capa de entrada:** 42 neuronas (tablero 6×7 aplanado)
- **Capas ocultas:** 256 → 256 → 128 → 64 neuronas
- **Capa de salida:** 7 neuronas (una por columna)
- **Función de activación:** ReLU en capas ocultas
- **Función de pérdida:** MSE (Mean Squared Error)

**Justificación:** La arquitectura profunda con disminución gradual de neuronas permite capturar patrones complejos del juego mientras reduce la dimensionalidad progresivamente. Se optó por una red más profunda que ancha para capturar jerarquías de características.

### 1.2 Algoritmo de Aprendizaje
Se implementó **Deep Q-Learning** con las siguientes características clave:

- **Replay Buffer:** Memoria de experiencias (deque) para romper correlaciones temporales
- **Target Network:** Red objetivo actualizada periódicamente para estabilizar el entrenamiento
- **Epsilon-Greedy:** Exploración decreciente para balancear exploración vs. explotación
- **Enmascaramiento de acciones:** Penalización de movimientos inválidos con -∞

### 1.3 Preprocesamiento del Estado
El tablero se representa como una matriz numpy donde:
- 0 = casilla vacía
- 1 = jugador 1 (agente DQN)
- 2 = jugador 2 (oponente)

El estado se aplana a un vector de 42 elementos antes de pasarlo a la red.

## 2. Grid Search de Hiperparámetros

Se realizó una búsqueda exhaustiva sobre 192 configuraciones diferentes:

### 2.1 Espacio de Búsqueda
- **Learning rate (α):** [0.0001, 0.0005, 0.001, 0.005]
- **Factor de descuento (γ):** [0.9, 0.95, 0.98, 0.99]
- **Epsilon inicial:** [0.7, 0.8, 0.9, 1.0]
- **Oponentes de entrenamiento:** [None (auto-juego), RandomAgent, DefenderAgent]

### 2.2 Parámetros Fijos
- Episodios de entrenamiento: 1000
- Epsilon mínimo: 0.1
- Epsilon decay: 0.995
- Batch size: 128
- Tamaño de memoria: 1000
- Actualización de target network: cada 100 pasos

## 3. Resultados Obtenidos

### 3.1 Mejor Modelo
El modelo con **mejor desempeño** logró **92.7% de victorias** contra RandomAgent:

- **α = 0.005**
- **γ = 0.95**
- **ε₀ = 0.8**
- **Oponente de entrenamiento:** DefenderAgent
- **Resultado:** 927 victorias / 73 derrotas / 0 empates (en 1000 partidas)

### 3.2 Top 5 Configuraciones
1. **92.7%** - α=0.005, γ=0.95, ε=0.8, vs Defensor
2. **92.7%** - α=0.005, γ=0.98, ε=0.7, vs Defensor
3. **91.1%** - α=0.005, γ=0.95, ε=0.7, vs Defensor
4. **88.1%** - α=0.005, γ=0.9, ε=0.9, vs Defensor
5. **87.9%** - α=0.005, γ=0.98, ε=1.0, vs Random

### 3.3 Análisis de Resultados

**Hallazgos principales:**

1. **Learning rate alto es crucial:** Los mejores resultados se obtuvieron consistentemente con α=0.005. Valores menores (0.0001-0.001) produjeron convergencia más lenta y peor desempeño.

2. **Gamma óptimo en valores medios-altos:** γ=0.95 y γ=0.98 demostraron el mejor balance. Valores muy bajos (0.9) priorizan demasiado recompensas inmediatas, mientras que valores muy altos (0.99) pueden introducir inestabilidad.

3. **Entrenar contra DefenderAgent es superior:** Los 5 mejores modelos fueron entrenados contra DefenderAgent. Esto sugiere que enfrentar un oponente con estrategia (aunque simple) es más efectivo que auto-juego o jugar contra un agente aleatorio.

4. **Epsilon inicial moderado funciona mejor:** ε₀=0.8 apareció en el mejor modelo, sugiriendo que iniciar con exploración moderada (no máxima) ayuda al aprendizaje.

5. **Variabilidad en resultados:** Se observa variabilidad significativa entre configuraciones similares, indicando que el proceso de aprendizaje tiene componentes estocásticos importantes.

## 4. Grado de Éxito

**Éxito alcanzado:** ✓ **ALTO**

- El agente supera ampliamente al RandomAgent (92.7% vs 8% esperado de ventaja natural del primer jugador)
- La arquitectura DQN converge consistentemente en 1000 episodios
- El grid search identificó configuraciones robustas que generalizan bien

**Limitaciones identificadas:**
- No se evaluó contra oponentes más sofisticados (minimax, MCTS)
- La variabilidad en resultados sugiere sensibilidad a inicialización aleatoria
- No se exploró el desempeño cuando el agente juega segundo

## 5. Conclusiones

El proyecto implementó exitosamente un agente de Connect4 basado en Deep Q-Learning que alcanza un nivel de juego competente. La búsqueda sistemática de hiperparámetros reveló que:

1. Un learning rate relativamente alto (0.005) es esencial para convergencia efectiva
2. Entrenar contra oponentes con estrategia básica produce mejores políticas que auto-juego
3. La arquitectura profunda propuesta es suficiente para capturar la complejidad del juego

El código desarrollado es modular, extensible y bien documentado, permitiendo futuras mejoras como la implementación de variantes del algoritmo (Double DQN, Dueling DQN) o exploración de otras arquitecturas de red.

---

**Código completo disponible en:**
- [principal.py](principal.py) - Implementación del ambiente y DQN
- [entrenar_y_evaluar_grid_search.py](entrenar_y_evaluar_grid_search.py) - Grid search completo
- [resultados_grid_search.csv](resultados_grid_search.csv) - Resultados detallados de 192 experimentos
