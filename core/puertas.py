from math import sqrt
import numpy as np


'''
This module contains basic quantum gates used in several calculations
'''


X = np.array([[0, 1], [1, 0]])

Z = np.array([[1, 0], [0, -1]])

H = np.array([[1/sqrt(2), 1/sqrt(2)], [1/sqrt(2), -1/sqrt(2)]])

K = np.array([[2, 3], [4, 5]])

Y = np.array([[0, -1j],
              [1j, 0]])

I = np.array([[1, 0], [0, 1]]) # noqa

def cnot_adyacent(control_qubit):
    if control_qubit==0:
        CNOT = np.array([[1, 0, 0, 0],
                 [0, 1, 0, 0],
                 [0, 0, 0, 1],
                 [0, 0, 1, 0]])
    else:
        CNOT = np.array([[1, 0, 0, 0],
                 [0, 0, 0, 1],
                 [0, 0, 1, 0],
                 [0, 1, 0, 0]])  
    return CNOT      


CNOT = np.array([[1, 0, 0, 0],
                 [0, 1, 0, 0],
                 [0, 0, 0, 1],
                 [0, 0, 1, 0]])

CNOT3=np.array([[1, 0, 0, 0,0,0,0,0],
                 [0, 1, 0, 0,0,0,0,0],
                 [0, 0, 1, 0,0,0,0,0],
                 [0, 0, 0, 1,0,0,0,0],
                 [0, 0, 0, 0,0,1,0,0],
                 [0, 0, 0, 0,1,0,0,0],
                 [0, 0, 0, 0,0,0,0,1],
                 [0, 0, 0, 0,0,0,1,0]])

SWAP = np.array([[1, 0, 0, 0],
                 [0, 0, 1, 0],
                 [0, 1, 0, 0],
                 [0, 0, 0, 1]])
