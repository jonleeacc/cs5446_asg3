import numpy as np
import torch
from pyRDDLGym.Elevator import Elevator
from aivle_gym.agent_env import AgentEnv
from aivle_gym.env_serializer import SampleSerializer
from PIL import Image

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
from collections import deque

from collections import namedtuple



## DO NOT CHANGE THIS CODE
def convert_state_to_list(state, env_features):
    out = []
    for i in env_features:
        out.append(state[i])
    return out
    



## TODO: Define your model here:
class YourModel(nn.Module):
    def __init__(self, input_size):
        super(YourModel, self).__init__()
        # Your model layers and initializations here
        # Model layers
        self.fc1 = nn.Linear(input_size, 20)  # First layer, input size: state configuration. output 20
        self.fc2 = nn.Linear(20, 20)  # Second layer, input 20, output 20
        # self.fc3 = nn.Linear(20, 20)  # Second layer, input 20, output 20
        # self.fc4 = nn.Linear(20, 20)  # Second layer, input 20, output 20
        self.fc5 = nn.Linear(20, 6)   # Third layer, input 20, output 6
 
    def forward(self, x):
        # x will be a tensor with shape [batch_size, 11]
        # Your forward pass logic here
        # Ensure the output has shape [batch_size, 6]

        x = F.relu(self.fc1(x))  # Apply ReLU to the output of the first layer
        x = F.relu(self.fc2(x))  # Apply ReLU to the output of the second layer
        # x = F.relu(self.fc3(x))  # Apply ReLU to the output of the third layer
        # x = F.relu(self.fc4(x))  # Apply ReLU to the output of the fourth layer
        x = self.fc5(x)     # Output layer, no activation function here
        output = torch.sigmoid(x)  # Apply sigmoid to the output of the third layer

        return output


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

def train_model():
    env = Elevator(is_render=False)
    state = env.reset()
    env_features = list(env.observation_space.keys())

    input_size = len(env_features)
    model = YourModel(input_size)
    target_model = YourModel(input_size)
    optimizer = optim.Adam(model.parameters(), lr=0.1)
    criterion = nn.MSELoss()
    target_model.load_state_dict(model.state_dict())
    target_model.eval()

    discount_factor = 0.9
    epsilon = 0.9
    batch_size = 64
    memory = ReplayBuffer(10000)

    num_epochs = 50
    update_target_every = 5
    Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))
    for epoch in range(num_epochs):
        total_loss = 0
        state = env.reset()
        for _ in range(env.horizon):
            state_desc = env.disc2state(state)
            print(f'state {state_desc, env_features}')
            state_list = convert_state_to_list(state_desc, env_features)
            state_tensor = torch.FloatTensor([state_list])

            model.train()
            action_values = model(state_tensor)
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = torch.argmax(action_values).item()

            next_state, reward, terminated, info = env.step(action)
            next_state_desc = env.disc2state(next_state)
            next_state_list = convert_state_to_list(next_state_desc, env_features)
            memory.push(state_list, action, reward, next_state_list, terminated)

            if len(memory) > batch_size:
                transitions = memory.sample(batch_size)
                batch = Transition(*zip(*transitions))

                non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                        batch.next_state)), dtype=torch.bool)
                non_final_next_states = torch.FloatTensor([s for s in batch.next_state
                                                           if s is not None])
                state_batch = torch.FloatTensor(batch.state)
                action_batch = torch.LongTensor(batch.action)
                reward_batch = torch.FloatTensor(batch.reward)

                state_action_values = model(state_batch).gather(1, action_batch.unsqueeze(1))

                next_state_values = torch.zeros(batch_size)
                next_state_values[non_final_mask] = target_model(non_final_next_states).max(1)[0].detach()

                expected_state_action_values = (next_state_values * discount_factor) + reward_batch

                loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            if terminated:
                state = env.reset()
            else:
                state = next_state

        epsilon = max(0.01, epsilon * 0.99)

        if epoch % update_target_every == 0:
            target_model.load_state_dict(model.state_dict())

        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Total Loss: {total_loss}')

    torch.save(model.state_dict(), 'model.pt')
    print("Model saved to model.pt")

class DeepRLAgent:
    def __init__(self, model_path, input_size):

        self.brain = YourModel(input_size=input_size)  # Instantiate model

        # Load the model state dictionary
        self.brain.load_state_dict(torch.load(model_path))

        # Load the model
        # self.brain = torch.load(model_path)

        self.brain.eval()  # Set the network to evaluation mode
        if torch.cuda.is_available():
            self.brain.cuda()

    def select_action(self, state):
        state_tensor = torch.FloatTensor(state)
        if torch.cuda.is_available():
            state_tensor = state_tensor.cuda()

        #state_tensor = torch.FloatTensor(state).cuda()
        with torch.no_grad():
            action_values = self.brain(state_tensor)
        return np.argmax(action_values.cpu().numpy())

    def step(self, state):
        return self.select_action(state)

class ElevatorDeepRLAgentEnv(AgentEnv):
    def __init__(self, port: int, model_path: str):
        self.base_env = Elevator()
        self.model_path = model_path

        super().__init__(
            SampleSerializer(),
            self.base_env.action_space,
            self.base_env.observation_space,
            self.base_env.reward_range,
            uid=0,
            port=port,
            env=self.base_env,
        )

    def create_agent(self, **kwargs):
        env_features = list(self.base_env.observation_space.keys())
        input_size = len(env_features)

        agent = DeepRLAgent(self.model_path, input_size)
        return agent



def main():
    # rendering matters: we save each step as png and convert to png under the hook. set is_render=True to do so
    is_render = True
    render_path = 'temp_vis'
    env = Elevator(is_render=is_render, render_path=render_path)

    model_path = "model.pt"
    agent_env = ElevatorDeepRLAgentEnv(0, model_path)
    agent = agent_env.create_agent()
    state = env.reset()
    env_features = list(env.observation_space.keys())
    
    total_reward = 0
    for t in range(env.horizon):
        
        state_desc = env.disc2state(state)
        state_list = convert_state_to_list(state_desc, env_features)
        action = agent.step(state_list)
        
        next_state, reward, terminated, info = env.step(action)
        
        if is_render:
            env.render()
            
        total_reward += reward
        print()
        print(f'state      = {state}')
        print(f'action     = {action}')
        print(f'next state = {next_state}')
        print(f'reward     = {reward}')
        print(f'total_reward     = {total_reward}')
        state = next_state

    env.close()
    
    if is_render:
        env.save_render()
        img = Image.open(f'{render_path}/elevator.gif').convert('RGB')
        img.show()


if __name__ == "__main__":
    #train_model()
    main()
