import numpy as np
import matplotlib.pyplot as plt

def rozkladCholesky(M):
    n = len(M)
    L = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1):
            s = sum(L[i][k] * L[j][k] for k in range(j))
            if i == j:
                L[i][j] = np.sqrt(M[i][i] - s)
            else:
                L[i][j] = (1.0 / L[j][j] * (M[i][j] - s))
    return L


def rozwiazanieUkladuRownan(M, b):
    L = rozkladCholesky(M)
    y = np.zeros(len(b))
    x = np.zeros(len(b))
    for i in range(len(b)):
        y[i] = (b[i] - sum(L[i][j] * y[j] for j in range(i))) / L[i][i]
    for i in range(len(b) - 1, -1, -1):
        x[i] = (y[i] - sum(L[j][i] * x[j] for j in range(i + 1, len(b)))) / L[i][i]

    return x


def Welkier_Patryk_Aproks_MNK(x, y, n):
    M = np.zeros((len(x), n + 1))
    for i in range(n + 1):
        for j in range(len(x)):
            M[j][i] = x[j] ** i

    M_t = np.transpose(M)
    M_t_M = np.dot(M_t, M)
    M_t_y = np.dot(M_t, y)

    wspolczynniki = rozwiazanieUkladuRownan(M_t_M, M_t_y)

    plt.scatter(x, y, label='Punkty')
    xp = np.linspace(min(x), max(x), 100)
    yp = np.zeros_like(xp)
    for i in range(n + 1):
        yp += wspolczynniki[i] * xp ** i
    plt.plot(xp, yp, label='Wielomian aproksymacyjny', color='black')
    plt.legend()
    plt.show()

    return wspolczynniki


x = np.array([1, 2, 3, 4, 5])
y = np.array([1, 3, 5, 7, 9])
n = 3
wspolczynniki = Welkier_Patryk_Aproks_MNK(x, y, n)
print("Współczynniki wielomianu aproksymacyjnego:", wspolczynniki)
