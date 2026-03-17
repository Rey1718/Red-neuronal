import random
import math

# pesos iniciales
w = [random.random(), random.random()]
b = random.random()

lr = 0.01  # learning rate

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def neuron(x):
    suma = x[0]*w[0] + x[1]*w[1] + b
    return sigmoid(suma)

# datos (AND)
datos = [
    ([0, 0], 0),
    ([0, 1], 0),
    ([1, 0], 0),
    ([1, 1], 1),
]


# entrenamiento
for epoch in range(1000000):
    for x, y_real in datos:
        y_pred = neuron(x)
        
        error = y_pred - y_real
        #derivada
        d = y_pred * (1 - y_pred)
        # ajustar pesos
        w[0] -= lr * error * x[0]
        w[1] -= lr * error * x[1]
        b -= lr * error
resultado = 1 if y_pred > 0.5 else 0
# prueba final
for x, y_real in datos:
    print(x, "->", neuron(x))