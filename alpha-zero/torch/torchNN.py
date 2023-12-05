import torch
print(torch.__version__)

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Network(nn.Module):
    def __init__(self):
        super().__init__()
        # super(Network, self).__init__()

        self.dim_of_policy = 21
        self.dim_of_value = 1
        self.shared_layers = nn.Sequential(
            nn.Conv2d(4, 64, kernel_size=1),   #self.input_shape[0]
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            # nn.MaxPool2d(kernel_size=2),
            # nn.Conv2d(in_channels=32, out_channels=64, kernel_size=1),
            # nn.ReLU(),
            # nn.Dropout(0.1),
            # nn.MaxPool2d(kernel_size=(2, 2)),
            # nn.Flatten(),
            # nn.Linear(64 , 128),
        )
        
        self.policy_head = nn.Sequential(
            nn.Conv2d(64, 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128, self.dim_of_policy),
            nn.Softmax(dim=1)
        )
        
        self.value_head = nn.Sequential(
            nn.Conv2d(64, 3, kernel_size=3, padding=1),
            nn.BatchNorm2d(3),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(48, self.dim_of_value),
            nn.Tanh()
        )
        
        
    def forward(self, x):
        # Forward pass through shared layers
        x_shared = self.shared_layers(x)
        
        # Forward pass through policy and value heads
        policy_output = self.policy_head(x_shared)
        value_output = self.value_head(x_shared)
        
        return policy_output, value_output
    
# input_shape = (4, 4, 4)
# dim_of_policy = 21
# dim_of_value = 1
# model = Network()
# print(model)

# import numpy as np
# import torch.optim as optim

# data = np.zeros((1, 4, 4, 4), dtype=np.float32)
# data[0, 0, 1, 1] = 1.0
# data[0, 0, 3, 3] = 1.0
# data[0, 1, 1, 1] = 1.0
# data[0, 1, 3, 3] = 3.0

# Convert the numpy array to a PyTorch tensor
# data_tensor = torch.tensor(data)
# model = Network()
# # print(model)
# # print(model(data_tensor))

# criterion_policy = nn.CrossEntropyLoss()
# criterion_value = nn.MSELoss()

# optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
# optimizer.zero_grad()
# outputs_policy, outputs_value = model(data_tensor)
# print(outputs_value.item())
# print(outputs_policy[0])
# loss_policy = criterion_policy(outputs_policy, torch.ones(1,21))
# loss_value = criterion_value(outputs_value, torch.ones([1,1]))
# loss = loss_policy + loss_value
# loss.backward()
# optimizer.step()

# print(loss.item())
