import numpy as np
from time import time
time_total=time()
Q=153
elementos = 10
encontrado=False
cantidad = 12
np.random.seed(1)
pesos = np.random.randint(1, 11, size=elementos)
print(len(pesos))
pesos.sort()
print(pesos)
Co= np.zeros((cantidad+1, 2))

for j in range(0,elementos):
    inicio=time()
    for i in range(0,Co.shape[0]):
        Co[i][0]=pesos[j]*(i%(cantidad+1))+Co[i][0]
        
        if Co[i][0]==Q:
            Co[i][j+1]=(i%(cantidad+1))
            L=Co[i]
            Co=0
            print(f"La fila con el valor más alto en la columna {0} es: {L}")
            encontrado=True
            break

        
        if Co[i][0]<Q:
            Co[i][j+1]=(i%(cantidad+1))
        else:
            Co[i][0]=-1
       
        if Co[i][0] != 0:
            for n in range(Co.shape[0]):
                if n != i and Co[i][0] == Co[n][0] and i>n:
                    Co[i][0]=-1

    if encontrado==True:
        break  
    Co = Co[Co[:, 0] != -1]
    
    print("cambio de objeto",j)
    fin=time()
    print(f"Tiempo de ejecución: {fin-inicio} segundos")
    #for i in range(0,Co.shape[0]):
        #print(np.round(Co[i]).astype(int))
    print(Co.shape[0])
    Co = np.repeat(Co, cantidad+1, axis=0)
    column_of_zeros = np.zeros((Co.shape[0], 1))  # Columna de ceros
    Co = np.hstack((Co, column_of_zeros))
    
indice_fila_max = np.argmax(Co[:, 0])

# Paso 2: Seleccionar la fila correspondiente al índice encontrado
fila_maxima = Co[indice_fila_max]

print(f"La fila con el valor más alto en la columna {0} es: {fila_maxima}")
fin_t=time()
print(f"Tiempo total de ejecución: {fin_t-time_total} segundos")