import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
import socket
from collections import deque

# ===== Red neuronal DQN =====
class DQN(nn.Module):
    def __init__(self, input_size, num_actions=3):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, num_actions)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# ===== Replay Buffer =====
class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    def push(self, state, action, reward, next_state):
        self.buffer.append((state, action, reward, next_state))
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states = zip(*batch)
        return (np.array(states), np.array(actions), np.array(rewards), np.array(next_states))
    def __len__(self):
        return len(self.buffer)

# ===== Configuración =====
input_size = 6  # distance, dir.x, dir.z, cooldown, hit, out_of_bounds
num_actions = 3
gamma = 0.99
epsilon = 0.2
batch_size = 32
lr = 1e-3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = DQN(input_size, num_actions).to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)
criterion = nn.MSELoss()
memory = ReplayBuffer()

# ===== Socket server =====
HOST = '127.0.0.1'
PORT = 5005
server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind((HOST, PORT))
server.listen(1)
print("Esperando conexión Unity...")
conn, addr = server.accept()
print("Conectado a Unity:", addr)

# ===== Función para elegir acción =====
def select_action(state):
    global epsilon
    if random.random() < epsilon:
        return random.randint(0, num_actions-1)
    state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        q_values = model(state_tensor)
    return torch.argmax(q_values).item()

# ===== Entrenamiento DQN =====
def train():
    if len(memory) < batch_size:
        return
    states, actions, rewards, next_states = memory.sample(batch_size)
    states = torch.tensor(states, dtype=torch.float32).to(device)
    actions = torch.tensor(actions, dtype=torch.long).to(device)
    rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
    next_states = torch.tensor(next_states, dtype=torch.float32).to(device)

    q_values = model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
    with torch.no_grad():
        target = rewards + gamma * model(next_states).max(1)[0]
    loss = criterion(q_values, target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# ===== Loop principal =====
prev_action = 0
last_state = None

while True:
    try:
        data = conn.recv(1024)
        if not data:
            break
        obs = np.array([float(x) for x in data.decode().split(',')], dtype=np.float32)
        if len(obs) != input_size:
            print(f"Error: Unity envió {len(obs)} observables, se esperan {input_size}")
            continue

        state = obs
        action = 1
        conn.send(bytes([action]))

        # Reward
        reward = 1.0 if state[4] == 1 else -0.1
        if state[5] == 1:
            reward -= 1.0

        if last_state is not None:
            memory.push(last_state, prev_action, reward, state)
            train()
            print(f"Reward: {reward:.2f} | Acción anterior: {prev_action} | Buffer: {len(memory)}")

        last_state = state
        prev_action = action

    except Exception as e:
        print("Error loop principal:", e)
        break