import pandas as pd

def rellenar_con_media(dataframe, columna):
    # Calcular la media ignorando los valores nulos
    media = dataframe[columna].mean()
    
    # Rellenar los valores nulos con la media calculada
    dataframe[columna] = dataframe[columna].fillna(media)
    
    return dataframe

# Ejemplo de uso
data = {
    'nombre': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
    'edad': [20, None, 18, 25, None],
    'calificaciones': [90, 88, None, None, 95]
}
 
df = pd.DataFrame(data)
columna = 'calificaciones'
 
resultado = rellenar_con_media(df, columna)
print(resultado)