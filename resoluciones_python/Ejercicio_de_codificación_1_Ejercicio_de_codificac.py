def operaciones_matematicas(a, b):
    suma = a + b
    resta = a - b
    multiplicacion = a * b

    if b != 0:
        division = a / b
        residuo = a % b
    else:
        division = "División por cero no permitida"
        residuo = "División por cero no permitida"

    return (suma, resta, multiplicacion, division, residuo)

# Prueba de la función
a = 10
b = 3
resultado = operaciones_matematicas(a, b)
print("Suma:", resultado[0])           # 13
print("Resta:", resultado[1])          # 7
print("Multiplicación:", resultado[2]) # 30
print("División:", resultado[3])       # 3.333...
print("Residuo:", resultado[4])        # 1