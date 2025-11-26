import random
import joblib
import numpy as np
from collections import deque
import os

class ReplayBuffer:
    def __init__(self, capacity=20000):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, target):
        """
        Agrega una experiencia (estado, valor) al buffer.
        """
        self.buffer.append((state, target))

    def sample(self, batch_size):
        """
        Retorna un batch aleatorio de experiencias.
        """
        batch_size = min(batch_size, len(self.buffer))
        batch = random.sample(self.buffer, batch_size)
        
        states = np.array([b[0] for b in batch])
        targets = np.array([b[1] for b in batch])
        
        return states, targets

    def __len__(self):
        return len(self.buffer)

    def save(self, path="replay_buffer.pkl"):
        """Guarda el buffer en disco."""
        try:
            joblib.dump(self.buffer, path)
            print(f"Replay Buffer guardado en {path} ({len(self.buffer)} muestras)")
        except Exception as e:
            print(f"Error guardando buffer: {e}")

    def load(self, path="replay_buffer.pkl"):
        """Carga el buffer desde disco."""
        if os.path.exists(path):
            try:
                self.buffer = joblib.load(path)
                print(f"Replay Buffer cargado: {len(self.buffer)} muestras")
            except Exception as e:
                print(f"Error cargando buffer: {e}")
