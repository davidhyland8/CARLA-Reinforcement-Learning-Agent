from collections import deque, namedtuple
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch.utils.tensorboard import SummaryWriter
import time
import numpy as np
import random
from ImageDQN import ImageDQN
from DQN import DQN

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))


class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def memorise(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
    
class DQNAgent:
    def __init__(self, memory, min_memory, gamma, epsilon_decay, epsilon_min, learning_rate, i, o, tb_dir, models_dir, image):
        self.memory = ReplayMemory(memory)
        self.min_memory = min_memory
        self.gamma = gamma
        self.epsilon = 1
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.learning_rate = learning_rate
        self.clip_delta = 1.0
        self.action_space = o
        self.tb_dir = tb_dir
        self.models_dir = models_dir
        self.image = image
        if self.tb_dir is not None:
            self.tensorboard = SummaryWriter(self.tb_dir + f'/carla-agent-{int(time.time())}')
        self.model = self.create_model(i, o, learning_rate)
        self.target_model = self.create_model(i, o, learning_rate)
        self.update_target_model()
    
    def create_model(self, inputs, outputs, lr):
        if self.image:
            return ImageDQN(inputs, outputs, lr)
        else:
            return DQN(inputs, outputs, lr)
        
    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())
        
    def memorise(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def best_action(self, state):
        obs_ = T.tensor([state]).to(self.model.device)
        obs = obs_.float()
        actions = self.model.forward(obs)
        action_ = T.argmax(actions).item()
        
        return action_
        
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            action = np.random.randint(0, self.action_space)
        else:
            action = self.best_action(state)
            
        self.decay()
        return action
    
    def decay(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon, self.epsilon_min)
        
    def load_model(self, path):
        self.model.load_state_dict(T.load(path))
        self.target_model.load_state_dict(T.load(path))

    def learn(self, batch_size):
        if len(self.memory) < self.min_memory:
            return
        
        self.model.optimiser.zero_grad()
        
        minibatch = self.memory.sample(batch_size)
        batch = Transition(*zip(*minibatch))
        
        state_batch_ = T.tensor(batch.state).to(self.model.device)
        state_batch = state_batch_.float()
        
        action_batch_ = T.tensor(batch.action).to(self.model.device)
        action_batch = T.reshape(action_batch_, (batch_size, 1, 1))
        
        reward_batch_ = T.tensor(batch.reward).to(self.model.device)
        reward_batch = T.reshape(reward_batch_, (batch_size, 1)).float()
        
        next_state_batch_ = T.tensor(batch.next_state).to(self.model.device)
        next_state_batch = next_state_batch_.float()
        
        done_batch_ = T.tensor(batch.done).to(self.model.device)
        done_batch = T.reshape(done_batch_, (batch_size, 1))
        
        state_action_values = self.model.forward(state_batch).to(self.model.device).gather(2, action_batch)
        y_pred_ = state_action_values.max(1)[0]
        y_pred = y_pred_.float()

        next_state_values = self.target_model(next_state_batch).to(self.model.device)
        max_qs = next_state_values.max(2)[0]
        y_true_ = (max_qs * self.gamma) + reward_batch
        
        y_true = y_true_.float()
        y_true[done_batch] = reward_batch[done_batch]
        
        loss = self.model.loss(y_pred, y_true).to(self.model.device)
        loss.backward()
        self.model.optimiser.step()

        return loss
    
    def save_model(self, r, t, ep):
        if self.models_dir is not None:
            T.save(self.model.state_dict(), self.models_dir + f"/carla-agent->r:{r}-t:{t}-ep:{ep}-{int(time.time())}")