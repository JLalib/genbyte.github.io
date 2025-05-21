def contar_frecuencia(lista):
    frecuencia = {}
    for elemento in lista:
        if elemento in frecuencia:
            frecuencia[elemento] += 1
        else:
            frecuencia[elemento] = 1
    return frecuencia

# Ejemplo de uso
elementos = [1, 2, 2, 3, 1, 2, 4, 5, 4]
resultado = contar_frecuencia(elementos)
print(resultado)  # {1: 2, 2: 3, 3: 1, 4: 2, 5: 1}

    