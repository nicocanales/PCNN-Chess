import sys
import os

# Asegurar que el directorio actual está en el path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ai_core import NeuralBrain, ChessEngine
from gui import ChessGUI

def main():
    """
    Punto de entrada principal de PCNN Chess.
    Inicializa los componentes y lanza la GUI.
    """
    print("Inicializando PCNN Chess...")
    
    # 1. Inicializar Cerebro (Red Neuronal)
    brain = NeuralBrain()
    
    # 2. Inicializar Motor de Ajedrez (Búsqueda)
    engine = ChessEngine(brain)
    
    # 3. Inicializar Interfaz Gráfica
    gui = ChessGUI(brain, engine)
    
    # 4. Ejecutar Loop Principal
    try:
        gui.run()
    except KeyboardInterrupt:
        print("\nCerrando aplicación...")
        sys.exit()
    except Exception as e:
        print(f"Error crítico: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
