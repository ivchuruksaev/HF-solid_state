import numpy as np

def hartree_fock_iteration(S, H_core, max_iterations=100, convergence_threshold=1e-6):
    """
    Простой алгоритм метода Хартри-Фока.
    S: матрица перекрытия
    H_core: основная матрица Хартри-Фока
    """
    print("Starting Hartree-Fock iteration...")
    F = H_core.copy()  # Начальная Фоковская матрица
    energy = 0.0

    for iteration in range(max_iterations):
        # Решаем уравнение Hψ = εψ
        eigenvalues, eigenvectors = np.linalg.eigh(F)

        # Обновляем энергию и матрицу F (упрощенно)
        energy_new = np.sum(eigenvalues)
        if abs(energy_new - energy) < convergence_threshold:
            print(f"Converged at iteration {iteration+1}")
            break
        energy = energy_new

        print(f"Iteration {iteration+1}: Energy = {energy:.6f}")

    return energy, eigenvalues, eigenvectors
