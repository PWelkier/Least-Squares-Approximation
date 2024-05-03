import numpy as np
import matplotlib.pyplot as plt


def matrix_multiply(A, B):
    result = np.zeros((A.shape[0], B.shape[1]))
    for i in range(A.shape[0]):
        for j in range(B.shape[1]):
            for k in range(A.shape[1]):
                result[i][j] += A[i][k] * B[k][j]
    return result


def matrix_transpose(A):
    result = np.zeros((A.shape[1], A.shape[0]))
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            result[j][i] = A[i][j]
    return result


def cholesky_decomposition(A):
    n = len(A)
    L = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1):
            s = sum(L[i][k] * L[j][k] for k in range(j))
            if i == j:
                L[i][j] = np.sqrt(A[i][i] - s)
            else:
                L[i][j] = (1.0 / L[j][j] * (A[i][j] - s))
    return L


def solve_linear_equations(A, b):
    L = cholesky_decomposition(A)
    y = np.zeros(len(b))
    x = np.zeros(len(b))
    for i in range(len(b)):
        y[i] = (b[i] - sum(L[i][j] * y[j] for j in range(i))) / L[i][i]
    for i in range(len(b) - 1, -1, -1):
        x[i] = (y[i] - sum(L[j][i] * x[j] for j in range(i + 1, len(b)))) / L[i][i]

    return x


def Welkier_Patryk_Aproks_MNK(x, y, n):

    A = np.zeros((len(x), n + 1))
    for i in range(n + 1):
        for j in range(len(x)):
            A[j][i] = x[j] ** i

    A_transpose = matrix_transpose(A)
    A_transpose_A = matrix_multiply(A_transpose, A)
    A_transpose_y = np.dot(A_transpose, y)

    coefficients = solve_linear_equations(A_transpose_A, A_transpose_y)

    plt.scatter(x, y, label='Punkty')
    xp = np.linspace(min(x), max(x), 100)
    yp = np.zeros_like(xp)
    for i in range(n + 1):
        yp += coefficients[i] * xp ** i
    plt.plot(xp, yp, label='Wielomian aproksymacyjny', color='red')
    plt.legend()
    plt.show()

    return coefficients


x = np.array([1, 2, 3, 4, 5])
y = np.array([1, 3, 5, 7, 9])
n = 3
coefficients = Welkier_Patryk_Aproks_MNK(x, y, n)
print("Współczynniki wielomianu aproksymacyjnego:", coefficients)