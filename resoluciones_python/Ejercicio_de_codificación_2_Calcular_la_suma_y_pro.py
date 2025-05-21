def calcular_suma_y_promedio(lista_numeros):
    if not lista_numeros:  # Verifica si la lista está vacía
        return {"suma": 0, "promedio": 0}

    suma = sum(lista_numeros)
    promedio = suma / len(lista_numeros)
    return {"suma": suma, "promedio": promedio}

# Pruebas
numeros = [1, 2, 3, 4, 5]
resultado = calcular_suma_y_promedio(numeros)
print("Suma:", resultado["suma"])         # 15
print("Promedio:", resultado["promedio"]) # 3.0

    