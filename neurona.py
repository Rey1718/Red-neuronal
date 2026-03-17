import random

#pesos iniciales
#w=peso b=sesgo
w = [random.random(), random.random()]
b = random.random()

def relu(x):
    return max(0, x)

def neurona(x):
    #suma = x[0]*w[0] + x[1]*w[1] + b
    suma = x[0]*w[0] + x[1]*w[1] + b
    return relu(suma)

# prueba
entrada = [2, 2]
print(neurona(entrada))
