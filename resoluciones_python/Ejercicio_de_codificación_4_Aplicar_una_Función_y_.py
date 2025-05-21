def aplicar_funcion_y_filtrar(lista, valor_umbral):
    # Elevar al cuadrado cada número y filtrar los que sean mayores al umbral
    resultado = [x**2 for x in lista if x**2 > valor_umbral]
    return resultado

# Ejemplo de uso
numeros = [1, 2, 3, 4, 5]
valor_umbral = 3
resultado = aplicar_funcion_y_filtrar(numeros, valor_umbral)
print(resultado)  # [4, 9, 16, 25]

    