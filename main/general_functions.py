from math import log
def convertir_a_base_n(numero, base,digits):
    if numero == 0:
        return '0'*digits
    
    caracteres = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    resultado = ""
    
    while numero > 0:
        resultado = caracteres[numero % base] + resultado
        numero = numero // base
    
    return resultado.zfill(digits)


def base_a_decimal(numero: list, base: int) -> int:
    """Function that converts a dinary numer into decimal

    Args:
        numero (list): dinary number to convert
        base (int): base of the dinary number

    Returns:
        int: decimal representation of the number
    """
    numero = str(numero)
    numero = numero[::-1]
    decimal = 0
    for index, digit in enumerate(numero):
        decimal += base**index * int(digit)
    return decimal

def dinariy_list(number: int, base: int)->list:
    din_list = []
    digits = int(log(number) / log(base))
    for number in range(number):
        din_number = convertir_a_base_n(number, base, digits)
        din_list.append(din_number)

    return din_list

b = base_a_decimal('0'*4, 2)

b = '0' * 4      # Esto crea la cadena '0000'
b = b[1:] + '5'  # Reemplaza el último carácter
print(b)
c = b[1:] +'4'
print(c)