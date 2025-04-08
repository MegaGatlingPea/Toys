import numpy as np


def trap_energy(X):  # 1000, 6, 2
    energy = 0.5 * (X[:, :, 0]**2 + X[:, :, 1]**2)  
    total_energy = energy.sum(axis=1) 
    # print(total_energy[0])
    return total_energy   # J

def coul_energy(X, epsilon=1e-10):  # Coulomb
    n_samples, n_particles, _ = X.shape


    diff = X[:, :, np.newaxis, :] - X[:, np.newaxis, :, :]  
    # diff = np.minimum(diff, 2 * L - diff)  # 周期效应

    distances = np.linalg.norm(diff, axis=-1)  
    distances = np.maximum(distances, epsilon)  # 确保所有距离不小于 epsilon

    # 防止自己与自己计算，设置对角线为无穷大
    distances[:, np.eye(distances.shape[1], dtype=bool)] = np.inf

    # 计算库伦势能矩阵
    energy_matrix = 1 / distances  # 每对粒子的库伦势能  1000, 6, 6
    total_energies = np.sum(energy_matrix, axis=(1, 2)) / 2  

    return total_energies  

def X_Hamiltonian(X, LJ_alpha=1.0):
    return 1.0 * trap_energy(X) + LJ_alpha * coul_energy(X)  # LJ_alpha取0.2

