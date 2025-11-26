# Guía de Migración y Arquitectura - PCNN Chess Refactorizado

## 1. Cambios de Arquitectura

El proyecto ha pasado de una estructura monolítica (`engine.py`) a una arquitectura modular por capas:

*   **`ai_core.py`**: Contiene la lógica de negocio y la IA.
    *   `NeuralBrain`: Ahora usa `MLPRegressor` (Red Neuronal) en lugar de `SGDRegressor`.
    *   `ChessEngine`: Implementa *Iterative Deepening* y *Quiescence Search*.
*   **`gui.py`**: Maneja la presentación (Pygame) y la concurrencia (Threading).
*   **`main.py`**: Punto de entrada limpio.
*   **`features.py`**: Optimizado con operaciones de bits (Bitboards).

## 2. Instrucciones de Ejecución

Para ejecutar la nueva versión:

1.  Asegúrate de estar en el entorno virtual (`venv`).
2.  Ejecuta el nuevo punto de entrada:
    ```bash
    python main.py
    ```

## 3. Migración de Modelos

El nuevo modelo usa `MLPRegressor` (Red Neuronal), que **NO es compatible** con el antiguo modelo `SGDRegressor` guardado en `models/pcnn_model.pkl`.

**Pasos para migrar:**
1.  Borra el archivo `models/pcnn_model.pkl` antiguo.
2.  Al iniciar `main.py`, el sistema detectará que no hay modelo y descargará/entrenará uno nuevo automáticamente desde Kaggle o memoria sintética.
3.  El `ReplayBuffer` (`replay_buffer.pkl`) puede mantenerse, pero se recomienda borrarlo para empezar con experiencias limpias adaptadas a la nueva red neuronal.

## 4. Mejoras de Rendimiento

*   **Multithreading:** La interfaz ya no se congela mientras la IA piensa.
*   **Vectorización:** La extracción de características es ~50x más rápida gracias a NumPy y Bitboards.
*   **Estabilidad Táctica:** *Quiescence Search* reduce drásticamente los errores por "efecto horizonte" (dejarse piezas en intercambios largos).

## 5. Diseño de Software

*   **SRP (Single Responsibility Principle):** Cada clase tiene una única razón para cambiar.
*   **Concurrencia:** Uso de `threading` y `queue` para comunicación segura entre UI y IA.
*   **Inyección de Dependencias:** `ChessGUI` recibe `brain` y `engine` en su constructor, facilitando el testing.
