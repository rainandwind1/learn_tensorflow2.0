import torch
import tensorflow as tf
import numpy as np
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim

class Net(nn.Module):
    
    def __init__(self):
        super(Net,self).__init__()
        self.conv1 = nn.Conv2d(1,6,3)
        self.conv2 = nn.Conv2d(6,16,3)
        self.fc1 = nn.Linear(16*6*6,120)
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84,10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)),(2,2))
        x = F.max_pool2d(F.relu(self.conv2(x)),2)

        x = x.view(-1,self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self,x):
        #print(x.size())
        size = x.size()[1:]
        #print(size)
        num_features = 1
        for s in size:
            num_features *=  s
        #print(num_features)
        return num_features
    
net = Net()
params = list(net.parameters())


input = torch.randn(1,1,32,32)
out = net(input)
#print(out)

target = torch.randn(1,10)
criterion = nn.MSELoss()
loss = criterion(out,target)

optimizer = optim.SGD(net.parameters(),lr = 0.01)
optimizer.zero_grad()
loss.backward()
optimizer.step()
