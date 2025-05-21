import pandas as pd

def seleccionar_datos(dataframe, criterio):
    # Usa el método query para filtrar el DataFrame según el criterio dado
    try:
        resultado = dataframe.query(criterio)
        return resultado
    except Exception as e:
        print(f"Error al aplicar el criterio: {e}")
        return pd.DataFrame()  # Devuelve un DataFrame vacío en caso de error

# Ejemplo de uso
data = {
    'nombre': ['Alice', 'Bob', 'Charlie', 'David'],
    'edad': [20, 22, 18, 25],
    'calificaciones': [90, 88, 75, 95]
}

df = pd.DataFrame(data)
criterio = 'edad > 18'
resultado = seleccionar_datos(df, criterio)
print(resultado)