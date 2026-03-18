import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# datos XOR
X = torch.tensor([
    [0,0],
    [0,1],
    [1,0],
    [1,1]
], dtype=torch.float32)

y = torch.tensor([[0], [1], [1], [0]]).float()

# modelo
model = nn.Sequential(
    nn.Linear(2, 4),  # capa oculta
    nn.Sigmoid(),
    nn.Linear(4, 1),  # salida
    nn.Sigmoid()
)

# función de pérdida
loss_fn = nn.BCELoss()

# optimizador
optimizer = optim.SGD(model.parameters(), lr=0.01)

# entrenamiento
for epoch in range(100000):
    y_pred = model(X)
    
    loss = loss_fn(y_pred, y)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
     
    if epoch % 1000 == 0:
      print(f"Epoch {epoch}, Loss: {loss.item()}")
# prueba
with torch.no_grad():
    print(model(X))
   



   # Crear una malla de puntos
x = np.linspace(0, 1, 100)
y = np.linspace(0, 1, 100)
xx, yy = np.meshgrid(x, y)

grid = torch.tensor(np.c_[xx.ravel(), yy.ravel()]).float()

# Predicciones
with torch.no_grad():
    preds = model(grid)
    preds = preds.reshape(xx.shape)

# Dibujar
plt.contourf(xx, yy, preds, levels=50)
plt.colorbar()

# Datos originales
plt.scatter(X[:,0], X[:,1], c=y, edgecolors='k')
plt.title("Frontera de decisión")
plt.show()


test = torch.tensor([
    [0.1, 0.1],
    [0.1, 0.9],
    [0.9, 0.1],
    [0.9, 0.9]
]).float()

with torch.no_grad():
    hidden = model[0](X)  # primera capa
    print(hidden)