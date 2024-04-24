import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
from PIL import Image

# Assuming the Elevator environment from previous context
from pyRDDLGym.Elevator import Elevator

# Neural Network for Q-value Approximation
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, action_dim)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        return self.fc2(x)


class DQNAgent:
    def __init__(self, state_dim, action_dim, gamma=0.99, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995, learning_rate=0.001):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.memory = deque(maxlen=2000)
        self.gamma = gamma  # discount factor
        self.epsilon = epsilon  # exploration rate
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        self.model = QNetwork(state_dim, action_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.action_dim)
        state = torch.from_numpy(state).float().unsqueeze(0)
        act_values = self.model(state)
        return torch.argmax(act_values[0]).item()

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward if done else reward + self.gamma * torch.max(self.model(torch.from_numpy(next_state).float().unsqueeze(0))[0]).item()
            current_q = self.model(torch.from_numpy(state).float().unsqueeze(0))[0][action]
            loss = (current_q - target) ** 2

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay



def train_dqn(episodes):
    env = Elevator(is_render=False)
    agent = DQNAgent(env.observation_space.n, env.action_space.n)
    batch_size = 32

    for e in range(episodes):
        state = env.reset()
        state = np.reshape(state, [1, agent.state_dim])
        total_reward = 0
        while True:
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            next_state = np.reshape(next_state, [1, agent.state_dim])
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            if done:
                print("episode: {}/{}, score: {}, e: {:.2}".format(e, episodes, total_reward, agent.epsilon))
                break
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)

    return agent

if __name__ == "__main__":
    agent = train_dqn(1000)  # Train with 1000 episodes












# import torch
# import torch.nn as nn
# import torch.optim as optim
# import numpy as np
# from collections import deque
# import random

# class QNetwork(nn.Module):
#     def __init__(self, state_dim, action_dim):
#         super(QNetwork, self).__init__()
#         # Example network
#         self.fc1 = nn.Linear(state_dim, 64)
#         self.relu = nn.ReLU()
#         self.fc2 = nn.Linear(64, action_dim)

#     def forward(self, x):
#         x = self.relu(self.fc1(x))
#         return self.fc2(x)


# class DQNAgent:
#     def __init__(self, state_dim, action_dim, gamma=0.99, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995, learning_rate=0.001):
#         self.state_dim = state_dim
#         self.action_dim = action_dim
#         self.memory = deque(maxlen=2000)
#         self.gamma = gamma  # discount factor
#         self.epsilon = epsilon  # exploration rate
#         self.epsilon_min = epsilon_min
#         self.epsilon_decay = epsilon_decay
#         self.learning_rate = learning_rate
#         self.model = QNetwork(state_dim, action_dim)
#         self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

#     def remember(self, state, action, reward, next_state, done):
#         self.memory.append((state, action, reward, next_state, done))

#     def act(self, state):
#         if np.random.rand() <= self.epsilon:
#             return np.random.choice(self.action_dim)
#         state = torch.from_numpy(state).float().unsqueeze(0)
#         act_values = self.model(state)
#         return torch.argmax(act_values[0]).item()

#     def replay(self, batch_size):
#         minibatch = random.sample(self.memory, batch_size)
#         for state, action, reward, next_state, done in minibatch:
#             target = reward
#             if not done:
#                 next_state = torch.from_numpy(next_state).float().unsqueeze(0)
#                 target = (reward + self.gamma * torch.max(self.model(next_state)[0]).item())
#             state = torch.from_numpy(state).float().unsqueeze(0)
#             target_f = self.model(state)
#             target_f[0][action] = target
#             self.optimizer.zero_grad()
#             loss = nn.MSELoss()(self.model(state), target_f)
#             loss.backward()
#             self.optimizer.step()
#         if self.epsilon > self.epsilon_min:
#             self.epsilon *= self.epsilon_decay


# def train_dqn(episode):
#     loss = 0
#     agent = DQNAgent(state_dim=env.observation_space.shape[0], action_dim=env.action_space.n)
#     for e in range(episode):
#         state = env.reset()
#         state = np.reshape(state, [1, agent.state_dim])
#         total_reward = 0
#         for time in range(500):
#             action = agent.act(state)
#             next_state, reward, done, _ = env.step(action)
#             next_state = np.reshape(next_state, [1, agent.state_dim])
#             agent.remember(state, action, reward, next_state, done)
#             state = next_state
#             total_reward += reward
#             if done:
#                 print("episode: {}/{}, score: {}, e: {:.2}".format(e, episode, total_reward, agent.epsilon))
#                 break
#             if len(agent.memory) > batch_size:
#                 agent.replay(batch_size)
#     return agent
