import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from scipy.stats import entropy
import torch.nn.functional as F
import random
from collections import deque
import warnings

# Ignore all warnings
# warnings.filterwarnings("ignore")

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        experience = (state, action, reward, next_state, done)
        self.buffer.append(experience)

    def sample(self, batch_size):
        batch = random.sample(self.buffer, min(batch_size,len(self.buffer)))
        state, action, reward, next_state, done = zip(*batch)
        return np.array(state), np.array(action), np.array(reward), np.array(next_state), np.array(done)

    def train_sequence(self, batch_size):
        batch = list(self.buffer)[-batch_size:]
        state, action, reward, next_state, done = zip(*batch)
        return np.array(state), np.array(action), np.array(reward), np.array(next_state), np.array(done)   

    
    def __len__(self):
        return len(self.buffer)    




class Net(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=[128,256]):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size[0])
        self.fc2 = nn.Linear(hidden_size[0], hidden_size[1])
        self.fc3 = nn.Linear(hidden_size[1], output_size)

    def forward(self, x):
        x=self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x
    
    
class Agent:
    def __init__(self, state_size, action_size, hidden_size=128, lr=1e-3,sigma=0.1):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = Net(state_size, action_size, hidden_size).to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr)
        self.action_desk = [1-sigma,1,1+sigma,1-sigma,1,1+sigma] 
        self.entropy_array = []

    def current_entropy(self,p1,p2):
        a = entropy(p1)
        b = entropy(p2)
        self.entropy_array.append((a+b)/2)

    def get_action(self, state):
        state = torch.FloatTensor(state) 
        with torch.no_grad():
            policy = self.net(state)
            probs1 = F.softmax(policy[:len(policy)//2], dim=0)
            probs2 = F.softmax(policy[len(policy)//2:], dim=0)
            probs1 = np.clip(probs1.numpy(), 1e-5, 1-1e-5)  # Clip values to avoid edge cases
            probs2 = np.clip(probs2.numpy(), 1e-5, 1-1e-5)  # Clip values to avoid edge cases
            probs1 /= np.sum(probs1)  # Normalize to ensure sum is 1
            probs2 /= np.sum(probs2)  # Normalize to ensure sum is 1
            delta_idx = np.random.choice(range(len(policy)//2), p=probs1)
            self.current_entropy(probs1,probs2)
            weight_idx = np.random.choice(range(len(policy)//2), p=probs2)

        return [self.action_desk[delta_idx],self.action_desk[weight_idx]],policy.detach()
    

    def train(self, state, action, reward, next_state, done, alpha=0.7,gamma=0.97):
        state = torch.FloatTensor(state).to(self.device)
        action = torch.FloatTensor(action).to(self.device)
        reward = torch.FloatTensor(np.array([reward])).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        done = torch.FloatTensor(np.array([done])).to(self.device)
        # done = done
        if len(done.shape) > 1:
            reward = reward.view(-1, 1)  # Reshape to [128, 1] if necessary
            done = done.view(-1, 1)  # Reshape to [128, 1] if necessary


        # predicted values with current state
        state_value = self.net(state)
        next_state_value = self.net(next_state)

        state_value_prime = state_value + alpha * (reward + (1-done)*(gamma * next_state_value - state_value))

        self.optimizer.zero_grad()
        loss = self.criterion(state_value_prime, state_value)
        loss.backward()
        self.optimizer.step()

