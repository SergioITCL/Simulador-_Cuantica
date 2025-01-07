import numpy as np
import matplotlib.pyplot as plt

class QuantumCircuit:

    def __init__(self,number_qubits):
                self.number_qubits = number_qubits

    def _state_normalization(self,quantum_state):
        ''' 
        method that normalizes a quantum state
        

        Not finished
        '''
         

    def initial_state(self, coefficients=[]):
        if coefficients==[]:
            state = np.zeros(2**self.number_qubits)
            state[0] = 1
            return state
        else:
            state=np.array(coefficients)
            if state.shape[0]==2**self.number_qubits:
                return state  
            else:
                return print('the number of coefficients is not correct')
    

    def measures_histogram(self,quantum_state):

        '''
        method that plots the asociated probabilities of the quantum state


        Not finisehd
        '''
        plt.hist(quantum_state, bins=10, color='blue', edgecolor='black')
        plt.xlabel('Coeficientes')
        plt.ylabel('Frecuencia')
        plt.title('Histograma de Coeficientes')
        plt.savefig('mi_grafica.png')
        # Mostrar el histograma

                
        
            
            