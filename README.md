# PCNN Chess - AlphaZero Lite

PCNN Chess es una implementación de motor de ajedrez que combina algoritmos de búsqueda clásicos con una Red Neuronal para la evaluación de posiciones. Está diseñado con un enfoque en Inteligencia Artificial Explicable (XAI), proporcionando retroalimentación narrativa en tiempo real sobre el proceso de toma de decisiones del motor.

## Descripción del Proyecto

Este proyecto implementa un motor de ajedrez que aprende a evaluar posiciones utilizando un regresor Perceptrón Multicapa (MLP). A diferencia de los motores tradicionales que dependen únicamente de funciones de evaluación hechas a mano, PCNN Chess utiliza una red neuronal para predecir la probabilidad de ganar de un estado del tablero dado. El motor está envuelto en una GUI personalizada basada en Pygame que muestra el tablero, el proceso de pensamiento del motor y comentarios en lenguaje natural.

## Características Principales

### IA y Motor de Búsqueda
- **Evaluación con Red Neuronal**: Utiliza `sklearn.neural_network.MLPRegressor` para evaluar posiciones del tablero.
- **Algoritmo de Búsqueda**: Implementa Minimax con poda Alpha-Beta para un recorrido eficiente del árbol de juego.
- **Profundización Iterativa (Iterative Deepening)**: Busca a profundidades crecientes dentro de un límite de tiempo para asegurar que se encuentre la mejor jugada dadas las restricciones.
- **Búsqueda en Reposo (Quiescence Search)**: Extiende la búsqueda en los nodos hoja para secuencias tácticas (capturas) para mitigar el efecto horizonte.
- **Tabla de Transposición**: Almacena en caché los resultados de evaluación de posiciones visitadas anteriormente para acelerar la búsqueda.

### Interfaz Gráfica de Usuario (GUI)
- **Interfaz Pygame**: Una visualización responsiva del tablero de ajedrez.
- **Multihilo**: La IA se ejecuta en un hilo separado, asegurando que la interfaz de usuario permanezca responsiva durante los cálculos.
- **Características Interactivas**: Incluye validación de movimientos, un botón de "Rendirse" y una superposición transparente de Fin del Juego.
- **Registro en Tiempo Real**: Muestra los registros internos del motor (Intención, Análisis, Razonamiento) en una barra lateral.

### Inteligencia Artificial Explicable (XAI)
- **Comentarios en Lenguaje Natural**: Genera explicaciones legibles por humanos para las jugadas del motor (por ejemplo, "Desarrollando el caballo para controlar el centro").
- **Visualización del Proceso de Pensamiento**: Muestra las jugadas candidatas consideradas y sus evaluaciones.

## Estructura del Proyecto

- **main.py**: El punto de entrada de la aplicación. Inicializa la IA, el Motor y la GUI.
- **ai_core.py**: La lógica central. Contiene la clase `NeuralBrain` (gestión de la Red Neuronal) y la clase `ChessEngine` (algoritmos de búsqueda).
- **gui.py**: Maneja la interfaz gráfica de usuario, la entrada del usuario y el renderizado usando Pygame.
- **features.py**: Responsable de convertir el estado del tablero de ajedrez en un vector de características numéricas (usando bitboards) para la red neuronal.
- **commentary_narrator.py**: Lógica para generar comentarios en lenguaje natural basados en el estado del tablero y el tipo de jugada.
- **replay_buffer.py**: Gestiona el almacenamiento del historial de partidas para un posible entrenamiento offline.
- **engine.py**: Implementación monolítica heredada (mantenida como referencia).

## Instalación

1. Asegúrate de tener Python 3.10 o superior instalado.
2. Crea un entorno virtual (opcional pero recomendado):
   ```bash
   python -m venv venv
   source venv/bin/activate  # En Windows: venv\Scripts\activate
   ```
3. Instala las dependencias requeridas:
   ```bash
   pip install python-chess pygame numpy scikit-learn joblib pandas kagglehub
   ```

## Uso

Para iniciar el juego, ejecuta el script `main.py`:

```bash
python main.py
```

1. Ingresa tu nombre y elige tu color (Blancas o Negras).
2. Haz clic en "Iniciar" para comenzar el juego.
3. Realiza movimientos haciendo clic en una pieza y luego en la casilla de destino.
4. Observa los registros de "RAZONAMIENTO" en la barra lateral para entender el pensamiento de la IA.
5. Usa el botón "Rendirse" para abandonar si la posición está perdida.

## Detalles Técnicos

- **Características de Entrada**: El tablero se representa como un conjunto de bitboards, extraídos en `features.py`, capturando posiciones de piezas y equilibrio material.
- **Arquitectura del Modelo**: Una red neuronal feed-forward (MLP) con capas ocultas (128, 64) y activación ReLU.
- **Aprendizaje**: El modelo puede ser entrenado offline usando el historial de partidas o potencialmente online (implementación TD-Lambda presente en el código).

## Licencia

Este proyecto es de código abierto.
