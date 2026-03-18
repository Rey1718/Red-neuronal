import torch
import torch.nn as nn
import torch.optim as optim

# datos XOR
X = torch.tensor([
    [0,0],
    [0,1],
    [1,0],
    [1,1]
], dtype=torch.float32)

y = torch.tensor([
    [0],
    [1],
    [1],
    [0]
], dtype=torch.float32)

# modelo
model = nn.Sequential(
    nn.Linear(2, 2),  # capa oculta
    nn.Sigmoid(),
    nn.Linear(2, 1),  # salida
    nn.Sigmoid()
)

# función de pérdida
loss_fn = nn.MSELoss()

# optimizador
optimizer = optim.SGD(model.parameters(), lr=0.1)

# entrenamiento
for epoch in range(10000):
    y_pred = model(X)
    
    loss = loss_fn(y_pred, y)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# prueba
with torch.no_grad():
    print(model(X))