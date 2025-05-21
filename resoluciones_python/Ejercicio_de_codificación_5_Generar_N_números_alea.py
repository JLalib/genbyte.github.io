import numpy as np

def generar_numeros_enteros_aleatorios(N, minimo, maximo):
    # Genera N números enteros aleatorios entre minimo y maximo (inclusive)
    return list(np.random.randint(minimo, maximo + 1, size=N))

# Ejemplo de uso
N = 5
minimo = 1
maximo = 10
resultado = generar_numeros_enteros_aleatorios(N, minimo, maximo)
print(resultado)  # Ejemplo: [4, 10, 1, 8, 2]
  