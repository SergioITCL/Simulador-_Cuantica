# Notebook que permite resolver el problema del Knapsack con valores asociados

"""
En este Notebook se implementa el algoritmo que permite resolver el problema de la mochila, la idea es encontrar la configuración de objetos, cada uno asociado con un valor y un peso, que maximiza el valor total de los objetos seleccionados pero sin superar la capacidad C de la mochila.


Versiones:
- v1. Aprovechamiento de la sparsity del problema e implementación en GPU, sirve para resolver problemas donde cada objeto puede ser seleccionado i veces.
- v2. Aprovechamiento de la sparsity de los tensores generados teniendo en cuenta que solamente se forman dos diagonales no nulas, solo permite resolver el 0-1 knapsack
- v3. Limpieza del notebook
   -v3.2 Esta versión utiliza la librería decimal para poder trabajar con valores de tau mucho mas grandes, es más lento pero obtiene ligeramente mejores resultados
   
- v4. Generalización del método de sparsity en el caso de que cada objeto pueda ser seleccionado varias veces. <br>
   -v4.2. Esta versión es un desarrollo análogo a a la v4. pero sin aprovechar los cálculos intermedios.

"""
