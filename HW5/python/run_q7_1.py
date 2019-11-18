import numpy as np
import matplotlib.pyplot as plt
import sys
import scipy.io

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import torchvision
import torchvision.transforms as transforms

class TwoLayerFCNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(TwoLayerFCNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.sigmoid(x)
        x = self.fc2(x)
        x = F.softmax(x, dim=1)
        return x

class NISTDataset(torch.utils.data.Dataset):
    def __init__(self, path='../data/nist36_train.mat', name="train"):
        # data laoding and pre-processing
        data = scipy.io.loadmat(path)
        x, y = data[name+'_data'], data[name+'_labels']
        self.x = torch.tensor(x, dtype=torch.float)
        self.y = torch.tensor(y, dtype=torch.long).argmax(1)
        print("x:", x.shape, x.dtype, x.max(), x.min())
        print("y:", y.shape, y.dtype)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return (self.x[idx], self.y[idx])

def pytorchNIST():
    input_size = 1024
    hidden_size = 64
    output_size = 36
    num_epochs = 50
    learning_rate = 1.0
    batch_size = 12

    # load the data
    trainset = NISTDataset()

    # Define the network, loss function and optimizer
    net = TwoLayerFCNet(input_size, hidden_size, output_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                        shuffle=True, num_workers=4)

    for epoch in range(num_epochs):
        total_loss = 0.0
        total_acc = 0.0
        for i, data in enumerate(trainloader, 0):
            x, y = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            # print statistics
            total_acc += (outputs.argmax(1) == y).sum().float() / len(y)
            total_loss += loss.item() * batch_size

        total_acc /= len(trainloader)
        if epoch % 2 == 0:
            print("epoch: {:02d} \t loss: {:.2f} \t acc : {:.2f}".format(epoch,total_loss,total_acc))

def main():
    if sys.argv[1] == "7.1.1":
        pytorchNIST()

if __name__ == '__main__':
    main()
