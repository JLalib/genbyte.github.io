import pandas as pd

def calcular_promedio(dataframe):
    # Calcula el promedio de cada columna
    return dataframe.mean()

# Ejemplo de uso
data = {
    'A': [1, 2, 3, 4],
    'B': [5, 6, 7, 8],
    'C': [9, 10, 11, 12]
}

df = pd.DataFrame(data)
resultado = calcular_promedio(df)
print(resultado)