import matplotlib.pyplot as plt

def graficar_temperaturas(dias, temperaturas):
    """
    Genera un gráfico de líneas que muestra la variación de temperaturas a lo largo de la semana.

    Parámetros:
        dias (list): Lista con los días de la semana (str).
        temperaturas (list): Lista con las temperaturas correspondientes (float/int).
    """

    # Tamaño del gráfico
    plt.figure(figsize=(10, 6))

    # Crear gráfico de líneas
    plt.plot(dias, temperaturas, color='blue', linestyle='--', marker='o', label='Temperatura')

    # Añadir título y etiquetas
    plt.title('Temperaturas Semanales')
    plt.xlabel('Días')
    plt.ylabel('Temperatura (°C)')

    # Añadir leyenda
    plt.legend()

    # Mostrar el gráfico
    plt.grid(True)
    plt.tight_layout()
    plt.show()