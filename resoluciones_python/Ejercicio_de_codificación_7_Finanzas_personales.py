import numpy as np

def analizar_finanzas(ingresos, gastos):
    """
    Analiza las finanzas personales de un estudiante durante un año. 
    Calcula el balance mensual, el total de ingresos, el total de gastos 
    y el saldo final.
    
    Args:
    ingresos (np.array): Array de ingresos mensuales (12 elementos).
    gastos (np.array): Array de gastos mensuales (12 elementos).
    
    Returns:
    np.array: Array con el balance mensual, el total de ingresos, 
              el total de gastos y el saldo final.
    """
    import numpy as np

def analizar_finanzas(ingresos, gastos):
    # Convertimos a arrays de NumPy por si vienen como listas
    ingresos = np.array(ingresos)
    gastos = np.array(gastos)

    # Validación de que todos los datos sean positivos
    if np.any(ingresos < 0) or np.any(gastos < 0):
        raise ValueError("Todos los valores de ingresos y gastos deben ser positivos.")

    # Balance mensual: ingresos - gastos
    balance_mensual = ingresos - gastos

    # Total de ingresos y total de gastos
    total_ingresos = ingresos.sum()
    total_gastos = gastos.sum()

    # Saldo final del año
    saldo_final = total_ingresos - total_gastos

    # Retornar los resultados en un array de objetos
    return [balance_mensual, total_ingresos, total_gastos, saldo_final]

# Ejemplo de uso
ingresos_mensuales = [1500, 1600, 1700, 1650, 1800, 1900, 2000, 2100, 2200, 2300, 2400, 2500]
gastos_mensuales   = [1000, 1100, 1200, 1150, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000]

resultado = analizar_finanzas(ingresos_mensuales, gastos_mensuales)

# Mostrar resultados
print("Balance mensual:", resultado[0])
print("Total ingresos:", resultado[1])
print("Total gastos:", resultado[2])
print("Saldo final:", resultado[3])