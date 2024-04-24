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

from collections import namedtuple, deque

## DO NOT CHANGE THIS CODE
def convert_state_to_list(state, env_features):
    out = []
    for i in env_features:
        out.append(state[i])
    return out
    
'''For Reference
State:
['num-person-waiting___f0', 'num-person-waiting___f1', \
    'num-person-waiting___f2', 'num-person-waiting___f3', \
        'num-person-in-elevator___e0', 'elevator-dir-up___e0', \
            'elevator-closed___e0', 'elevator-at-floor___e0__f0', \
                'elevator-at-floor___e0__f1', 'elevator-at-floor___e0__f2', \
                    'elevator-at-floor___e0__f3']
'''


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


## TODO: Define your model here:
class YourModel(nn.Module):
    def __init__(self, input_size):
        super(YourModel, self).__init__()
        # Your model layers and initializations here
        # Model layers
        self.fc1 = nn.Linear(input_size, 126)  # First layer, input size: state configuration. output 20
        self.fc2 = nn.Linear(126, 126)  # Second layer, input 20, output 20
        # self.fc3 = nn.Linear(20, 20)  # Second layer, input 20, output 20
        # self.fc4 = nn.Linear(20, 20)  # Second layer, input 20, output 20
        self.fc5 = nn.Linear(126, 6)   # Third layer, input 20, output 6
 
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

print('ok')
def test_env():
    env = Elevator(is_render=False)  # Disable rendering for training
    state = env.reset()
    env_features = list(env.observation_space.keys())
    #print(env_features)
    def parse_state(state):
        state_desc = env.disc2state(state)
        state_list = convert_state_to_list(state_desc, env_features)
        print(f'state_list {state_list}')
        # print(f'actions {env.action_space}')
        # print(f'act {env.action_space.sample()}')
    parse_state(state)
    #help(env.action_space)
    for i in range(6):
        action = i
        new_state, reward, term, trun = env.step(action)
        parse_state(new_state)
        print(reward)



def train_model():

    def parse_state(state):
        state_desc = env.disc2state(state)
        state_list = convert_state_to_list(state_desc, env_features)
        print(f'state_list {state_list}')

    # Create an environment instance
    env = Elevator(is_render=False)  # Disable rendering for training
    #help(env.step)
    state = env.reset()
    print(f'env reset {env.disc2state(state)}')
    env_features = list(env.observation_space.keys())

    # Model and training setup
    input_size = len(env_features)
    model = YourModel(input_size)
    optimizer = optim.Adam(model.parameters(), lr=0.1)
    criterion = nn.MSELoss()

    discount_factor = 0.9
    epsilon = 0.9

    memory = ReplayMemory(10000)

    # Training loop
    num_epochs = 50  # Total number of epochs to train
    for epoch in range(num_epochs):
        total_loss = 0
        print(f'Epoch {epoch}')
        # print('horizon',env.horizon)
        env.reset()
        for i in range(env.horizon):
            #print(f'horizon {env.horizon}')
            state_desc = env.disc2state(state)
            #print(f'state {state_desc, env_features}')
            state_list = convert_state_to_list(state_desc, env_features)
            state_tensor = torch.FloatTensor([state_list])  # Convert state list to tensor and add batch dimension
            
            # Predict action values from the model
            model.train()
            action_values = model(state_tensor)
            # ε-greedy action selection
            if random.random() < epsilon:
                
                action = env.action_space.sample()  # Random action
                # if i ==10:
                #     print('action',action)
                #     print('avail actions',env.action_space)
            else:
                action = torch.argmax(action_values).item()  # Select action with the highest value
                # print('action',action)
                # print('avail actions',env.action_space)
            
            # Take action in the environment
            #print(f'env.step{env.step(action)}')
            next_state, reward, terminated, info = env.step(action)
            next_state_desc = env.disc2state(next_state)
            next_state_list = convert_state_to_list(next_state_desc, env_features)
            next_state_tensor = torch.FloatTensor([next_state_list])
            
            memory.push(state_tensor, action, next_state_tensor, reward)
            # # Calculate the target value for training
            # next_action_values = model(next_state_tensor)
            # target_value = reward + discount_factor * torch.max(next_action_values).item()  # Using a discount factor of 0.90
            # target_values = action_values.clone()
            # target_values[0, action] = target_value
            
            # # Calculate loss and perform backpropagation
            # loss = criterion(action_values, target_values)
            # optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()
            
            # total_loss += loss.item()
            # if terminated:
            #     break
            # if i>0 and prev_state_list!= state_list:
            #     print('diff')
            #     print(prev_state_list)
            #     print(state_list)
            # if i%100 == 0:
            #     print(i)
            #     print(f'action {action}')
            #     print(f'epsilon {epsilon}')
            #     print(state_list)
            #     # print(model)
            # if i%101 == 0:
            #     print(i)
            #     print(f'action {action}')
            #     print(f'epsilon {epsilon}')
            #     print(state_list)
            # if i%100 == 0 and i>0:
            #     print(i)
            #     print(state_list)
            #     print(prev_state_list)
            #     for i in range(10):
            #         action = env.action_space.sample()
            #         next_state, reward, terminated, info = env.step(action)
            #         print('current', state_list)
            #         print('action',action)
            #         parse_state(next_state)

            #     print('____________________')

            prev_state_list = state_list


            state = next_state

        epsilon = max(0.01, epsilon * 0.99)
        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Average Loss: {total_loss / env.horizon}')

    # Save the trained model
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
    #test_env()
    train_model()
    main()
