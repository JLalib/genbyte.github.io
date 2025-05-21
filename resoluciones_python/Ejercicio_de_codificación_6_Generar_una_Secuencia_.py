import numpy as np

def generar_secuencia_numerica(minimo, maximo, paso):
    # Genera una secuencia desde minimo hasta maximo (sin incluirlo) con el paso indicado
    return np.arange(minimo, maximo, paso)

# Ejemplo de uso
minimo = 0
maximo = 10
paso = 2
resultado = generar_secuencia_numerica(minimo, maximo, paso)
print(resultado)  # [0 2 4 6 8]