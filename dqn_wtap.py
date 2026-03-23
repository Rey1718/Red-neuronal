import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import os
# ========================
# CONFIG
# ========================
STATE_SIZE = 3
ACTION_SIZE = 5

GAMMA = 0.99
LR = 0.001
BATCH_SIZE = 256
MEMORY_SIZE = 50000

EPSILON = 1.0
EPSILON_MIN = 0.02
EPSILON_DECAY = 0.995

EPISODES = 2000
MAX_STEPS = 200

# ========================
# ENTORNO (SIN UNITY)
# ========================
class ChaseEnv:

    def __init__(self):
        self.reset()

    def reset(self):
        self.agent = np.array([0.0, 0.0])
        self.target = np.array([
            random.uniform(-10, 10),
            random.uniform(-10, 10)
        ])
        self.prev_distance = self.get_distance()
        return self.get_state()

    def get_distance(self):
        return np.linalg.norm(self.agent - self.target)

    def get_state(self):
        dx = self.target[0] - self.agent[0]
        dz = self.target[1] - self.agent[1]
        dist = self.get_distance()
        return np.array([dx/10, dz/10, dist/20], dtype=np.float32)

    def step(self, action):
        speed = 0.5

        # acciones
        if action == 0: self.agent[0] += speed
        elif action == 1: self.agent[0] -= speed
        elif action == 2: self.agent[1] += speed
        elif action == 3: self.agent[1] -= speed
        elif action == 4: pass  # idle

        distance = self.get_distance()
        reward = -0.01

        delta = self.prev_distance - distance

        # acercarse
        reward += delta * 2

        # alejarse castigo
        if delta < 0:
            reward += delta * 2

        # quedarse quieto castigo
        if abs(delta) < 0.001:
            reward -= 0.05

        # bonus cercanía
        if distance < 2:
            reward += 0.2

        done = False

        # alcanzó objetivo
        if distance < 0.5:
            reward += 0.9
            done = True

        self.prev_distance = distance

        return self.get_state(), reward, done, {}

# ========================
# RED NEURONAL (MEJORADA)
# ========================
class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(STATE_SIZE, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, ACTION_SIZE)
        )

    def forward(self, x):
        return self.net(x)

# ========================
# SETUP
# ========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Usando:", device)

model = DQN().to(device)
target_model = DQN().to(device)
target_model.load_state_dict(model.state_dict())

optimizer = optim.Adam(model.parameters(), lr=LR)
loss_fn = nn.MSELoss()

memory = deque(maxlen=MEMORY_SIZE)
epsilon = EPSILON

# ========================
# FUNCIONES
# ========================
def choose_action(state):
    global epsilon

    if random.random() < epsilon:
        return random.randint(0, ACTION_SIZE - 1)

    state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        q_values = model(state_t)

    return torch.argmax(q_values[0]).item()

def store_experience(s, a, r, s_next, done):
    memory.append((s, a, r, s_next, done))

def train():
    if len(memory) < BATCH_SIZE:
        return

    batch = random.sample(memory, BATCH_SIZE)
    states, actions, rewards, next_states, dones = zip(*batch)

    states = torch.tensor(np.array(states), dtype=torch.float32).to(device)
    next_states = torch.tensor(np.array(next_states), dtype=torch.float32).to(device)
    actions = torch.tensor(actions).to(device)
    rewards = torch.tensor(rewards).to(device)
    dones = torch.tensor(dones).float().to(device)

    q_values = model(states)
    next_q_values = target_model(next_states)

    target = q_values.clone()

    for i in range(BATCH_SIZE):
        target_q = rewards[i]
        if not dones[i]:
            target_q += GAMMA * torch.max(next_q_values[i])
        target[i][actions[i]] = target_q

    loss = loss_fn(q_values, target)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


if os.path.exists("checkpoint.pth"):
    checkpoint = torch.load("checkpoint.pth")

    model.load_state_dict(checkpoint["model"])
    target_model.load_state_dict(checkpoint["model"])
    epsilon = checkpoint["epsilon"]

    print("✅ Checkpoint cargado")
else:
    print("⚠️ Entrenamiento desde cero")
# ========================
# ENTRENAMIENTO
# ========================
env = ChaseEnv()

try:
    for ep in range(EPISODES):

        state = env.reset()

        total_reward = 0
        direction_changes = 0
        idle_count = 0
        prev_action = None

        for step in range(MAX_STEPS):

            action = choose_action(state)

            # métricas w-tap
            if prev_action is not None and action != prev_action:
                direction_changes += 1

            if action == 4:  # idle
                idle_count += 1

            next_state, reward, done, _ = env.step(action)

            # BONUS por W-tap correcto
            if prev_action is not None:
                if prev_action == 0 and action == 1:
                    reward += 1
                if prev_action == 1 and action == 0:
                    reward += 1

            # CASTIGO por W-tap deficiente
            if prev_action is not None:
                # castigo si no cambia de dirección y no está idle
                if action != 4 and action == prev_action:
                    reward -= 0.5  # penaliza correr en la misma dirección
                # castigo extra si idle demasiado
                if idle_count > 3:
                    reward -= 0.5

            store_experience(state, action, reward, next_state, done)
            train()

            state = next_state
            prev_action = action
            total_reward += reward

            if done:
                break

        # epsilon decay
        if epsilon > EPSILON_MIN:
            epsilon *= EPSILON_DECAY

        # actualizar target
        if ep % 20 == 0:
            target_model.load_state_dict(model.state_dict())

        # métricas
        wtap_score = direction_changes / (step + 1)

        print(f"Ep:{ep} | Reward:{total_reward:.2f} | Steps:{step} | Wtap:{wtap_score:.2f} | Eps:{epsilon:.3f}")

        # ========================
        # GUARDAR MODELO CADA 20 EPISODIOS
        # ========================
        if ep % 20 == 0:
            torch.save({
                "model": model.state_dict(),
                "epsilon": epsilon
            }, "checkpoint.pth")

except KeyboardInterrupt:
    print("🛑 Detenido")

    torch.save({
        "model": model.state_dict(),
        "epsilon": epsilon
    }, "checkpoint.pth")

    print("💾 Guardado de emergencia")