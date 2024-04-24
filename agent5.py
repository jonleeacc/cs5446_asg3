import numpy as np
import torch
from pyRDDLGym.Elevator import Elevator
from aivle_gym.agent_env import AgentEnv
from aivle_gym.env_serializer import SampleSerializer
from PIL import Image

import math
import random

from collections import namedtuple, deque
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

## DO NOT CHANGE THIS CODE
def convert_state_to_list(state, env_features):
    out = []
    for i in env_features:
        out.append(state[i])
    return out
    
## TODO: Define your model here:
# class YourModel(nn.Module):
#     def __init__(self):
#         super(YourModel, self).__init__()
#         # Your model layers and initializations here
 
#     def forward(self, x):
#         # x will be a tensor with shape [batch_size, 11]
#         # Your forward pass logic here
#         # Ensure the output has shape [batch_size, 6]
#         return output

class DQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        # self.layer1 = nn.Linear(n_observations, 128)
        # self.layer2 = nn.Linear(128, 128)
        # self.layer3 = nn.Linear(128, n_actions)
        self.layer1 = nn.Linear(n_observations, 30)
        self.layer2 = nn.Linear(30, 30)
        self.layer3 = nn.Linear(30, 30)
        self.layer4 = nn.Linear(30, 30)
        self.layer5 = nn.Linear(30, n_actions)


    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        x = F.relu(self.layer4(x))
        output = self.layer5(x)
        return output

# if GPU is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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



env = Elevator()

# BATCH_SIZE is the number of transitions sampled from the replay buffer
# GAMMA is the discount factor as mentioned in the previous section
# EPS_START is the starting value of epsilon
# EPS_END is the final value of epsilon
# EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
# TAU is the update rate of the target network
# LR is the learning rate of the ``AdamW`` optimizer
BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-4

# Get number of actions from gym action space
# print(f'act space: {dir(env.action_space)}')
# print(f'act space: {env.action_space.sample}')
n_actions = env.action_space.n
# Get the number of state observations
state = env.reset()
env_features = list(env.observation_space.keys())

def parse_state(state):
    state_desc = env.disc2state(state)
    state_list = convert_state_to_list(state_desc, env_features)
    return state_list

state_list = parse_state(state)
n_observations = len(state_list)

policy_net = DQN(n_observations, n_actions).to(device)
target_net = DQN(n_observations, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(10000)


steps_done = 0


def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return the largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state).max(1).indices.view(1, 1)
    else:
        return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)


def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1).values
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()


num_episodes = 50


for i_episode in range(num_episodes):
    print(i_episode)
    # Initialize the environment and get its state
    TEST_FLOAT = True
    if TEST_FLOAT:
        state = env.reset()
        #state = parse_state(state)
        state_list = parse_state(state)
        #state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        state_tensor = torch.FloatTensor([state_list])
        for t in range(env.horizon):
            #action = select_action(state) #state_tensor
            action = select_action(state_tensor) 
            observation, reward, terminated, truncated = env.step(action.item())
            reward = torch.tensor([reward], device=device)
            done = terminated or truncated

            if terminated:
                next_state = None
            else:
                #next_state = torch.tensor(parse_state(observation), dtype=torch.float32, device=device).unsqueeze(0)
                next_state = torch.FloatTensor([parse_state(observation)])
            # Store the transition in memory
            #memory.push(state, action, next_state, reward) #state_tensor
            memory.push(state_tensor, action, next_state, reward) #state_tensor

            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the policy network)
            optimize_model()

            # Soft update of the target network's weights
            # θ′ ← τ θ + (1 −τ )θ′
            target_net_state_dict = target_net.state_dict()
            policy_net_state_dict = policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
            target_net.load_state_dict(target_net_state_dict)

            if done:
                break
    else:
        state = env.reset()
        state = parse_state(state)
        # state_list = parse_state(state)
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        # state_tensor = torch.FloatTensor([state_list])
        for t in range(env.horizon):
            action = select_action(state) #state_tensor
            # action = select_action(state_tensor) 
            observation, reward, terminated, truncated = env.step(action.item())
            reward = torch.tensor([reward], device=device)
            done = terminated or truncated

            if terminated:
                next_state = None
            else:
                next_state = torch.tensor(parse_state(observation), dtype=torch.float32, device=device).unsqueeze(0)
                # next_state = torch.FloatTensor([parse_state(observation)])
            # Store the transition in memory
            memory.push(state, action, next_state, reward) #state_tensor
            # memory.push(state_tensor, action, next_state, reward) #state_tensor

            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the policy network)
            optimize_model()

            # Soft update of the target network's weights
            # θ′ ← τ θ + (1 −τ )θ′
            target_net_state_dict = target_net.state_dict()
            policy_net_state_dict = policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
            target_net.load_state_dict(target_net_state_dict)

            if done:
                break

torch.save(policy_net_state_dict, 'model.pt')





########################################
class DeepRLAgent:
    def __init__(self, model_path, input_size, n_actions):
        # # Load the model
        # self.brain = torch.load(model_path)
        self.brain = DQN(input_size,n_actions)

        # Load the model state dictionary
        self.brain.load_state_dict(torch.load(model_path))
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
        n_actions = self.base_env.action_space.n
        agent = DeepRLAgent(self.model_path, input_size, n_actions)
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
    main()
