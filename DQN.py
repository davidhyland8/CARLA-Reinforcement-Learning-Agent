import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch.utils.tensorboard import SummaryWriter

class DQN(nn.Module):
    def __init__(self, inputs, outputs, learning_rate):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(inputs, 100)
        nn.init.kaiming_normal_(self.fc1.weight)
        self.fc2 = nn.Linear(100, 50)
        nn.init.kaiming_normal_(self.fc2.weight)
        self.fc3 = nn.Linear(50, 25)
        nn.init.kaiming_normal_(self.fc3.weight)
        self.fc4 = nn.Linear(25, 15)
        nn.init.kaiming_normal_(self.fc4.weight)
        self.fc5 = nn.Linear(15, 8)
        nn.init.kaiming_normal_(self.fc5.weight)
        self.fc6 = nn.Linear(8, outputs)
        nn.init.kaiming_normal_(self.fc6.weight)
        
        self.loss = nn.MSELoss()
        self.learning_rate = learning_rate
        self.optimiser = optim.Adam(self.parameters(), self.learning_rate)
        self.device = T.device("cuda" if T.cuda.is_available() else "cpu")
        self.to(self.device)
        
        
    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        
        x = self.fc2(x)
        x = F.relu(x)
        
        x = self.fc3(x)
        x = F.relu(x)
        
        x = self.fc4(x)
        x = F.relu(x)
        
        x = self.fc5(x)
        x = F.relu(x)
        
        x = self.fc6(x)
        
        return x
    
