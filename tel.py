import numpy as np
from typing import List, Tuple
import cmath
import matplotlib.pyplot as plt
from scipy.linalg import expm
import time

class QuantumCellularAutomaton:
    
    def __init__(self, grid_size: int, init_state: np.ndarray = None):  
        self.grid_size = grid_size
        self.grid = np.zeros((grid_size, grid_size), dtype=complex)
        if init_state is not None:
            if init_state.shape == (grid_size, grid_size):
                self.grid = init_state
            else:
                raise ValueError("Dimensione dello stato iniziale non valida.")
        else:
           
            self.grid = self._random_quantum_state()
        self.hamiltonian = self._build_hamiltonian()

    def _random_quantum_state(self) -> np.ndarray:
        state = np.random.random((self.grid_size, self.grid_size)) + \
                1j * np.random.random((self.grid_size, self.grid_size))
        norm = np.sqrt(np.sum(np.abs(state)**2))
        return state / norm

    def _build_hamiltonian(self) -> np.ndarray:
        size = self.grid_size ** 2
        H = np.zeros((size, size), dtype=complex)
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                idx = i * self.grid_size + j
               
                neighbors = self._get_neighbors(i, j)
                for ni, nj in neighbors:
                    n_idx = ni * self.grid_size + nj
                    H[idx, n_idx] = 0.1 * (1 + 1j) 
                H[idx, idx] = 1.0  
        return H + H.conj().T  

    def _get_neighbors(self, i: int, j: int) -> List[Tuple[int, int]]:
        neighbors = []
        for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ni, nj = (i + di) % self.grid_size, (j + dj) % self.grid_size
            neighbors.append((ni, nj))
        return neighbors

    def evolve(self, dt: float, steps: int) -> None:
        U = expm(-1j * self.hamiltonian * dt)
        state_vector = self.grid.flatten()
        for _ in range(steps):
            state_vector = U @ state_vector
            
            norm = np.sqrt(np.sum(np.abs(state_vector)**2))
            state_vector /= norm
        self.grid = state_vector.reshape((self.grid_size, self.grid_size))

    def measure_observable(self, observable: np.ndarray = None) -> float:
        if observable is None:
            
            return np.sum(np.abs(self.grid)**2)
        state_vector = self.grid.flatten()
        return np.real(state_vector.conj().T @ observable @ state_vector)

    def visualize(self, step: int = 0) -> None:
        plt.imshow(np.abs(self.grid)**2, cmap='viridis', interpolation='nearest')
        plt.colorbar(label='Densità di probabilità')
        plt.title(f"Stato quantistico al passo {step}")
        plt.show()

def main():
    grid_size = 20
    dt = 0.1
    steps = 10
    
    
    qca = QuantumCellularAutomaton(grid_size)
    
    
    print("Visualizzazione stato iniziale...")
    qca.visualize(step=0)

    
    start_time = time.time()
    for t in range(steps):
        qca.evolve(dt, 1)
        print(f"Passo {t+1}/{steps}, Valore atteso densità: {qca.measure_observable():.4f}")
        if (t + 1) % 5 == 0:
            qca.visualize(step=t+1)
    print(f"Tempo di simulazione: {time.time() - start_time:.2f} secondi")

if __name__ == "__main__":  
    main()