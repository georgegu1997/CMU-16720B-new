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
        # Note that cross entropy loss is actually log_softmax + nll_loss
        x = F.log_softmax(x, dim=1)
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

def trainNetwork(num_epochs, trainloader, device, optimizer, net, criterion):
    net.to(device)
    epoch_list = []
    train_acc_list = []
    train_loss_list = []

    for epoch in range(num_epochs):
        total_loss = 0.0
        total_acc = 0.0
        for i, data in enumerate(trainloader, 0):
            x, y = data[0].to(device), data[1].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            # print statistics
            total_acc += (outputs.argmax(1) == y).sum().float() / len(y)
            total_loss += loss.item() * len(x)

        total_acc /= len(trainloader)
        if epoch % 2 == 0:
            print("epoch: {:02d} \t loss: {:.2f} \t acc : {:.2f}".format(epoch,total_loss,total_acc))
            epoch_list.append(epoch+1)
            train_acc_list.append(total_acc)
            train_loss_list.append(total_loss)

    return net, epoch_list, train_loss_list, train_acc_list

def visualizeTrainProgress(epoch_list, train_loss_list, train_acc_list):
    plt.figure(figsize=(8, 12))
    plt.subplot(211)
    plt.plot(epoch_list, train_acc_list, label="Training")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")

    plt.subplot(212)
    plt.plot(epoch_list, train_loss_list, label="Training")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.subplot(211)

def NISTFC(device):
    input_size = 1024
    hidden_size = 64
    output_size = 36
    num_epochs = 50
    batch_size = 50
    learning_rate = 2e-3 * batch_size

    # load the data
    trainset = NISTDataset()

    # Define the network, loss function and optimizer
    net = TwoLayerFCNet(input_size, hidden_size, output_size)
    # Note that cross entropy loss is actually log_softmax + nll_loss
    criterion = F.nll_loss
    optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                        shuffle=True, num_workers=4)

    # Train the network
    net, epoch_list, train_loss_list, train_acc_list = trainNetwork(num_epochs, trainloader, device, optimizer, net, criterion)

    # Visualize the training progress and save plot
    visualizeTrainProgress(epoch_list, train_loss_list, train_acc_list)
    plt.title("Q 7.1.1")
    plt.savefig('../results/q7_1_1.png')

# Borrowed from the Pytorch official tutorial on training a CNN on MNIST
# https://github.com/pytorch/examples/blob/master/mnist/main.py
class ConvNet(nn.Module):
    def __init__(self, fc_in=9216):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(fc_in, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

def MNISTConv(device):
    batch_size = 64
    # Data pre-processing
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    # dataset and dataLoader
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    trainloader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True, transform=transform),
        batch_size=batch_size, shuffle=True, **kwargs
        )

    # Define the network, loss function and optimizer
    net = TwoLayerFCNet()
    optimizer = optim.Adadelta(net.parameters(), lr=args.lr)

    pass

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    if sys.argv[1] == "7.1.1":
        print("Q7.1.1: Train a 2-layer MLP on NIST36 training set")
        NISTFC(device)
    if sys.argv[1] == "7.1.2":
        print("Q7.1.2: Train a CNN on MNIST")
        MNISTConv(device)


if __name__ == '__main__':
    main()
