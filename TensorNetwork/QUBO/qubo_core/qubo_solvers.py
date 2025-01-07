import greedy
import numpy as np
import dimod
import neal
from dwave.system import DWaveSampler, EmbeddingComposite
from dwave.system import LeapHybridSampler

from quantum_sim.TensorNetwork.QUBO.qubo_core.qubo_auxiliar_functions import evaluar_qubo
import random


# dwave solver
def qubo_dimod_solver(
    Diccionario: dict,
    qubo_sampler: str,
    dwave_num_reads: int = 10,
    dwave_annealing_time: int = 20,
    fuerza: float = 0,
):
    """
    Function that allows solving problems or codes using the D-Wave library. It is possible to use the following modes:
        - "dwave": Requires an updated D-Wave key.
        - "neal": A simulated annealing simulator.
        - "hybrid": A solver that uses classical techniques and D-Wave's quantum computers.

    Args:
        Diccionario (dict): Dictionary representing the QUBO problem.
        qubo_sampler (str): Simulation method.
        dwave_num_reads (int, optional): Number of reads. Defaults to 10.
        dwave_annealing_time (int, optional): Annealing time. Defaults to 20.
        fuerza (float, optional): Strength parameter. Defaults to 0.

    Raises:
        ValueError: If an invalid mode is provided.
        ValueError: If the input parameters are incorrect.

    Returns:
        _type_: Solution to the QUBO problem.
    """
    J = Diccionario
    h = {}
    problem = dimod.BinaryQuadraticModel(h, J, 0.0, dimod.BINARY)
    api_key = "DEV-caa8e3d0f6dbfb0175cf148a15109b11979d3329"
    if qubo_sampler == "dwave":
        sampler = EmbeddingComposite(DWaveSampler(token=api_key))
        if fuerza == 0:
            result = sampler.sample(
                problem, num_reads=dwave_num_reads, annealing_time=dwave_annealing_time
            )
        else:
            result = sampler.sample(
                problem,
                num_reads=dwave_num_reads,
                annealing_time=dwave_annealing_time,
                chain_strength=fuerza,
            )
    elif qubo_sampler == "neal":
        solver = neal.SimulatedAnnealingSampler()
        result = solver.sample(problem)
    elif qubo_sampler == "hybrid":
        solver = LeapHybridSampler(token=api_key)
        result = solver.sample(problem)
    elif qubo_sampler == "brute_force":
        raise ValueError(f"Sampler '{qubo_sampler}' no implementado.")
    else:
        raise ValueError(f"Sampler '{qubo_sampler}' no reconocido.")
    return result.record[0][0]

# recocido simulado
def generar_vecino(x):
    """
    Generate a neighboring configuration

    Args:
        x (_type_): the configuration to evaluate

    Returns:
        _type_: a neighboring configuration
    """
    vecino = x.copy()
    idx = np.random.randint(len(x))
    vecino[idx] = 1 - vecino[idx]  
    return vecino


def recocido_simulado(Q_matrix, x_inicial, temperatura_inicial, tasa_enfriamiento, iteraciones):
    """
    Algorithm of simulated annealing

    Args:
        Q_matrix (_type_): QUBO matrix
        x_inicial (_type_): initial configuration
        temperatura_inicial (_type_): 
        tasa_enfriamiento (_type_): _description_
        iteraciones (_type_): 

    Returns:
        _type_: the best configuration obtained
    """
    x_actual = x_inicial.copy()
    costo_actual = evaluar_qubo(Q_matrix,x_actual)
    mejor_x = x_actual.copy()
    mejor_costo = costo_actual
    temperatura = temperatura_inicial

    for i in range(iteraciones):
        # Generar un vecino y calcular su coste
        x_vecino = generar_vecino(x_actual)
        costo_vecino = evaluar_qubo(Q_matrix,x_vecino)
        
        # Calcular la diferencia de costo
        delta_costo = costo_vecino - costo_actual

        # Decidir si aceptamos el nuevo estado
        if delta_costo < 0 or np.random.rand() < np.exp(-delta_costo / temperatura):
            x_actual, costo_actual = x_vecino, costo_vecino

        # Actualizar el mejor estado encontrado
        if costo_actual < mejor_costo:
            mejor_x, mejor_costo = x_actual.copy(), costo_actual

        # Reducir la temperatura
        temperatura *= tasa_enfriamiento

    return mejor_x


# recocid iterativo
def qubo_solver_rs(Q_matrix:np.array, number_layers:int)->np.array:
    """
    Function that applies annealing simulation iteratively
    Args:
    - Q_matrix: QUBO matrix
    
    Return:
    - solution: the best configuration obtained
    """
    # Determinamos el tamaño del problema
    n_variables = Q_matrix.shape[0]
    solution = np.zeros(n_variables, dtype=int)

    # Matrix QUBO auxiliar para las iteraciones
    Q_matrix_aux = Q_matrix.copy()

    # Creamos el bucle iterativo
    for variable in range(n_variables-1):
        x_inicial = np.random.randint(2, size=n_variables-variable)
        result_vector = recocido_simulado(Q_matrix_aux, x_inicial, 10.0, 0.99, 1000)
        # Obtenemos la solucion de la variable
        solution[variable] = result_vector[0]

        # Cambiamos la matriz auxiliar segun el resultado obtenido
        # Sumamos los valores de la variable en caso de haber salido 1
        if solution[variable] == 1:
            for column in range(Q_matrix_aux.shape[1]):
                Q_matrix_aux[column][column] += Q_matrix_aux[column][0]
        # Borramos la primera fila y columna
        Q_matrix_aux = Q_matrix_aux[1:,1:]

    # Ultima variable
    if Q_matrix_aux[0][0] < 0:
        solution[-1] = 1
    
    return solution

def random_qubo_solver(Q_matrix:np.array)-> np.array:
    """
    Function that generates 400 random qubo solutions and selects the best one.

    Args:
        Q_matrix (np.array): QUBO matrix

    Returns:
        np.array: best qubo solution of the algorithm
    """
    n_variables = len(Q_matrix[-1])
    best = np.inf
    best_c = np.zeros(n_variables)
    for i in range(400):
        random_solution = np.random.randint(2,size=n_variables)
        if evaluar_qubo(Q_matrix, random_solution) < best:
            best_c = random_solution.copy()
            best = evaluar_qubo(Q_matrix, random_solution)
    return best_c


def objetivo(solucion, Q):
    """
    Calcula el valor de la función objetivo para una solución dada.
    :param solucion: Vector de variables discretas.
    :param Q: Matriz Q que define el problema QUDO.
    :return: Valor de la función objetivo.
    """
    return np.dot(solucion, np.dot(Q, solucion))

def recocido_simulado_qudo(Q, valores_discretos, temperatura_inicial, tasa_enfriamiento, iteraciones):
    """
    Implementa el algoritmo de recocido simulado para resolver un problema QUDO.
    :param Q: Matriz Q del problema QUDO.
    :param valores_discretos: Lista de valores que las variables pueden tomar.
    :param temperatura_inicial: Temperatura inicial del algoritmo.
    :param tasa_enfriamiento: Factor por el cual se reduce la temperatura.
    :param iteraciones: Número máximo de iteraciones.
    :return: Mejor solución encontrada y su valor objetivo.
    """
    n = Q.shape[0]  # Número de variables
    solucion_actual = np.random.choice(valores_discretos, size=n)  # Solución inicial aleatoria
    valor_actual = objetivo(solucion_actual, Q)
    mejor_solucion = solucion_actual.copy()
    mejor_valor = valor_actual

    temperatura = temperatura_inicial

    for _ in range(iteraciones):
        # Generar una nueva solución vecina
        nueva_solucion = solucion_actual.copy()
        indice = random.randint(0, n - 1)  # Elegir una variable al azar
        nuevo_valor = random.choice(valores_discretos)  # Asignar un nuevo valor a la variable
        nueva_solucion[indice] = nuevo_valor

        # Evaluar la nueva solución
        nuevo_valor_objetivo = objetivo(nueva_solucion, Q)

        # Aceptar la nueva solución con una probabilidad basada en la temperatura
        delta = nuevo_valor_objetivo - valor_actual
        if delta < 0 or random.random() < np.exp(-delta / temperatura):
            solucion_actual = nueva_solucion
            valor_actual = nuevo_valor_objetivo

            # Actualizar el mejor encontrado
            if valor_actual < mejor_valor:
                mejor_solucion = solucion_actual.copy()
                mejor_valor = valor_actual

        # Enfriar la temperatura
        temperatura *= tasa_enfriamiento

    return mejor_solucion, mejor_valor