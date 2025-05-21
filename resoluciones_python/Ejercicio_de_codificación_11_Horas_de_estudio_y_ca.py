import matplotlib.pyplot as plt

def graficar_linea(x, y):
    # Crear el gráfico de línea conectando los puntos (x, y)
    plt.plot(x, y, marker='o', linestyle='-', color='blue')

    # Título y etiquetas
    plt.title("Relación entre horas de estudio y calificaciones")
    plt.xlabel("Horas de Estudio")
    plt.ylabel("Calificación")

    # Mostrar la cuadrícula
    plt.grid(True)

    # Mostrar la gráfica
    plt.show()

# Datos de entrada
horas_estudio = [1, 2, 3, 4, 5, 6, 7, 8]
calificaciones = [55, 60, 65, 70, 75, 80, 85, 90]

# Llamada a la función
graficar_linea(horas_estudio, calificaciones)

    