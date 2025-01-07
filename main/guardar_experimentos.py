import matplotlib.pyplot as plt
import math
import numpy as np
import random
# Crear un diccionario con las configuraciones deseadas
custom_rcparams = {
    'figure.figsize': (12, 8),         # Tamaño predeterminado de las figuras
    'axes.titlesize': 50,             # Tamaño de los títulos de los ejes
    'axes.labelsize': 50,             # Tamaño de las etiquetas de los ejes
    'xtick.labelsize': 30,            # Tamaño de las etiquetas del eje x
    'ytick.labelsize': 30,            # Tamaño de las etiquetas del eje y
    'legend.fontsize': 25,            # Tamaño de las fuentes de la leyenda
    'lines.linewidth': 2,             # Grosor de las líneas
    'lines.markersize': 8,            # Tamaño de los marcadores
    'savefig.dpi': 300,               # Resolución al guardar la figura
}
plt.rcParams.update(custom_rcparams)


def guardar_experimento(nombre_archivo: str, descripcion_experimento: str ,parametros: dict, resultados:dict):
    """
    Funcion que permite guardar los datos introducidos en un experimento y los obtenidos tras el experimento
    Args:
        nombre_archivo (str): _description_
        descripcion_experimento (_type_): _description_
        parametros (_type_): _description_
        resultados (_type_): _description_
    """
    with open(nombre_archivo, 'a') as file:
        file.write(f"Descripción del experimento: {descripcion_experimento}\n")
        file.write("Los parámetros utilizados en el experimento son:\n")
        for parametro in parametros:
            file.write(f"parametro: {parametro}={parametros[parametro]}\n")
        file.write("Los resultados obtenidos en el experimento son:\n")
        for resultado in resultados:
            file.write(f"parametro: {resultado}={resultados[resultado]}\n")
        file.write("="*40 + "\n")


def plot_function(datos_1, datos_2, nombre_eje_x, nombre_eje_y, color = 'blue',marker ='o', titulo = None, etiqueta=None, format='pdf'):
    plt.figure("figura")
    plt.plot(datos_1, datos_2, marker = marker, linestyle = 'None', color = color, label=etiqueta)
    plt.axhline(y=0, color='gray', linestyle='--', linewidth=3)
    plt.xlabel(nombre_eje_x)
    plt.ylabel(nombre_eje_y)
    #plt.yscale("exp")
    plt.tight_layout()
    plt.legend()
    plt.savefig(titulo, format=format, dpi=300)


# Función para calcular la media por posición
def media_por_posicion(lista_de_listas):
    # Usamos zip para agrupar los elementos por posición
    agrupados = zip(*lista_de_listas)
    # Calculamos la media de cada grupo
    medias = [sum(grupo) / len(grupo) for grupo in agrupados]
    return medias

def guardar_datos(filename: str, **datos):
    """
    Guarda en un archivo datos etiquetados.

    :param filename: Nombre del archivo donde se guardarán los datos.
    :param datos: Pares etiqueta=valores a guardar.
    """
    with open(filename, 'w') as file:
        for etiqueta, valores in datos.items():
            # Escribimos la etiqueta
            file.write(f"{etiqueta}:\n")
            # Convertimos los valores en una cadena delimitada por comas y escribimos
            file.write(", ".join(map(str, valores)) + "\n")
    print(f"Datos guardados en {filename}")


def random_color_rgb():
    r = np.random.rand()
    g = np.random.rand()
    b = np.random.rand()
    return (r, g, b)