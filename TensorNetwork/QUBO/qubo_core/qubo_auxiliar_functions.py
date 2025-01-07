import numpy as np


def matrix_QUBO_to_dict(matrix: np.array) -> dict:
    """
    Function that returns the QUBO problem representation in a dictionary

    Args:
        matrix (np.array): matrix of the QUBO problem

    Raises:
        ValueError: _description_

    Returns:
        dict: dictionary of the QUBO problem
    """
    # Checking that matrix is square
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Tha matrix is not square")

    qubo_dict = {}
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[0]):
            if matrix[row, column] != 0:
                qubo_dict[(row, column)] = matrix[row, column]
    return qubo_dict


def evaluar_qubo(Q_matrix, x):
    """
    Function that evaluates the cost of a given solution of a QUBO problem.
    Args:
        Q_matrix (_type_): QUBO marix representation of the problem.
        x (_type_): solution to check the value.

    Returns:
        _type_: cost of the solution.
    """
    x = np.array(x)
    return np.dot(x, np.dot(Q_matrix, x))


def generar_matriz_qubo(n: int, n_diagonales: int) -> np.array:
    """
    Function that generates a matrix for a QUBO problem given a number of variables and a number of diagonals.

    Args:
        n (int): Number of variables.
        n_diagonales (int): Number of diagonals.

    Returns:
        np.array: Returns the QUBO matrix.
    """

    Q_matrix = np.random.rand(n, n) * 2 - 1
    #Q_matrix = np.random.randint(1, 10, size=(n, n))

    for i in range(n):
        for j in range(i + 1, n):
            Q_matrix[i, j] = 0
    # number of diagonals og the QUBO problem
    for i in range(n):
        for j in range(0, i - (n_diagonales) + 1):
            Q_matrix[i, j] = 0
    # Checking that all the element are positive
    for i in range(n):
        Q_matrix[i, i] = abs(Q_matrix[i, i])

    return Q_matrix