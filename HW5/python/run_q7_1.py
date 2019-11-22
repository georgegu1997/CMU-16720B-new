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

'''
To check the overall accuracy over a dataset
Refer to: https://gist.github.com/jcjohnson/6e41e8512c17eae5da50aebef3378a4c
'''
def checkAccLoss(net, loader, criterion, device):
    total_loss = 0.0
    num_correct, num_samples = 0, 0
    net.eval()
    for data in loader:
        x, y = data[0].to(device), data[1].to(device)
        outputs = net(x)
        num_correct += (outputs.argmax(1) == y).sum()
        num_samples += x.shape[0]
        loss = criterion(outputs, y)
        total_loss += loss.item()
    acc = float(num_correct) / num_samples
    return acc, total_loss

'''
Run a single training epoch
Refer to: https://gist.github.com/jcjohnson/6e41e8512c17eae5da50aebef3378a4c
'''
def runEpoch(net, criterion, loader, optimizer, device):
    net.train()
    for data in loader:
        x, y = data[0].to(device), data[1].to(device)
        outputs = net(x)
        loss = criterion(outputs, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

'''
Utilities for training a network and Visualize its training progress
'''
def trainNetwork(num_epochs, trainloader, device, optimizer, net, criterion, valloader = None, acc_freq=1):
    net.to(device)
    epoch_list = []
    train_acc_list = []
    train_loss_list = []
    valid_acc_list = []
    valid_loss_list = []

    # Before training, the initial loss and accuracy
    train_acc, train_loss = checkAccLoss(net, trainloader, criterion, device)
    epoch_list.append(0)
    train_acc_list.append(train_acc)
    train_loss_list.append(train_loss)
    if not valloader is None:
        valid_acc, valid_loss = checkAccLoss(net, valloader, criterion, device)
        valid_acc_list.append(valid_acc)
        valid_loss_list.append(valid_loss)
        print("epoch: {:02d} \t loss: {:.2f} \t acc : {:.2f} \t valid loss: {:.2f} \t valid acc: {:.2f}".format(
            0,train_loss,train_acc,valid_loss,valid_acc
        ))
    else:
        print("epoch: {:02d} \t loss: {:.2f} \t acc : {:.2f}".format(0,train_loss,train_acc))

    # Start the training process
    for epoch in range(num_epochs):
        # forward, backward and update
        runEpoch(net, criterion, trainloader, optimizer, device)
        # compute loss and accuracy, record them
        if epoch % acc_freq == 0:
            train_acc, train_loss = checkAccLoss(net, trainloader, criterion, device)
            if valloader is None:
                print("epoch: {:02d} \t loss: {:.2f} \t acc : {:.2f}".format(epoch,train_loss,train_acc))
            # If valloader passed in, then run through valloader as validation
            else:
                valid_acc, valid_loss = checkAccLoss(net, valloader, criterion, device)
                print("epoch: {:02d} \t loss: {:.2f} \t acc : {:.2f} \t valid loss: {:.2f} \t valid acc: {:.2f}".format(
                    epoch,train_loss,train_acc,valid_loss,valid_acc
                ))
                valid_acc_list.append(valid_acc)
                valid_loss_list.append(valid_loss)
            epoch_list.append(epoch+1)
            train_acc_list.append(train_acc)
            train_loss_list.append(train_loss)

    if valloader is None:
        return net, epoch_list, train_loss_list, train_acc_list
    else:
        return net, epoch_list, train_loss_list, train_acc_list, valid_loss_list, valid_acc_list

def visualizeTrainProgress(epoch_list, train_loss_list, train_acc_list, valid_loss_list=None, valid_acc_list=None):
    plt.figure(figsize=(8, 12))
    plt.subplot(211)
    plt.plot(epoch_list, train_acc_list, label="Training")
    if not valid_acc_list is None:
        plt.plot(epoch_list, valid_acc_list, label="Validation")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")

    plt.subplot(212)
    plt.plot(epoch_list, train_loss_list, label="Training")
    if not valid_loss_list is None:
        plt.plot(epoch_list, valid_loss_list, label="Validation")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.subplot(211)

'''
For Q7.1.1
'''
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

    return net

'''
For Q7.1.2
'''
# Borrowed from the Pytorch official tutorial on training a CNN on MNIST
# https://github.com/pytorch/examples/blob/master/mnist/main.py
class ConvNet(nn.Module):
    def __init__(self, fc_in=9216, num_class=10):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(fc_in, 128)
        self.fc2 = nn.Linear(128, num_class)

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
    num_epochs = 15
    learning_rate = 0.1

    # Data pre-processing
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    # dataset and dataLoader
    kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}
    trainloader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('../data', train=True, download=True, transform=transform),
        batch_size=batch_size, shuffle=True, **kwargs
        )

    # Define the network, loss function and optimizer
    net = ConvNet().to(device)
    optimizer = torch.optim.Adadelta(net.parameters(), lr=learning_rate)
    criterion = F.nll_loss

    # Train the network
    net, epoch_list, train_loss_list, train_acc_list = trainNetwork(num_epochs, trainloader, device, optimizer, net, criterion)

    # Visualize the training progress and save plot
    visualizeTrainProgress(epoch_list, train_loss_list, train_acc_list)
    plt.title("Q 7.1.2")
    plt.savefig('../results/q7_1_2.png')

    return net

'''
Dataset and hyperparameters for Q7.1.3
'''
class NISTImageDataset(torch.utils.data.Dataset):
    def __init__(self, path='../data/nist36_train.mat', name="train", transform=None):
        # data laoding and pre-processing
        data = scipy.io.loadmat(path)
        x, y = data[name+'_data'], data[name+'_labels']
        # Reshape the data to image
        x = x.reshape((x.shape[0], 1, 32, 32))
        self.x = torch.tensor(x, dtype=torch.float)
        self.y = torch.tensor(y, dtype=torch.long).argmax(1)
        self.transform = transform
        print("x:", x.shape, x.dtype, x.max(), x.min())
        print("mean:", x.mean(), "\tstd:", x.std())
        print("y:", y.shape, y.dtype)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        x, y = self.x[idx], self.y[idx]
        if self.transform:
            x = self.transform(x)
        return (x, y)

def NISTConv(device):
    batch_size = 64
    num_epochs = 50
    learning_rate = 0.1

    # dataset and dataLoader
    kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}
    transform = transforms.Normalize((0.7916,), (0.3058,))
    trainloader = torch.utils.data.DataLoader(
        NISTImageDataset(transform=transform), batch_size=batch_size, shuffle=True, **kwargs
        )

    # Define the network, loss function and optimizer
    net = ConvNet(fc_in=12544, num_class=36).to(device)
    optimizer = torch.optim.Adadelta(net.parameters(), lr=learning_rate)
    criterion = F.nll_loss

    # Train the network
    net, epoch_list, train_loss_list, train_acc_list = trainNetwork(num_epochs, trainloader, device, optimizer, net, criterion)

    # Visualize the training progress and save plot
    visualizeTrainProgress(epoch_list, train_loss_list, train_acc_list)
    plt.title("Q 7.1.3")
    plt.savefig('../results/q7_1_3.png')
    return net

'''
For Q7.1.4
'''
def EMNISTConv(device):
    batch_size = 64
    num_epochs = 20
    learning_rate = 0.1

    # Data pre-processing
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    # dataset and dataLoader
    kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}
    torchvision.datasets.EMNIST.url = 'http://www.itl.nist.gov/iaui/vip/cs_links/EMNIST/gzip.zip'

    # Use the testing dataset as the validation set during training to see whether overfit
    trainset = torchvision.datasets.EMNIST('../data', split="balanced", train=True, download=True, transform=transform)
    testset = torchvision.datasets.EMNIST('../data', split="balanced", train=False, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, **kwargs)
    valloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True, **kwargs)

    # Define the network, loss function and optimizer
    net = ConvNet(num_class=47).to(device)
    optimizer = torch.optim.Adadelta(net.parameters(), lr=learning_rate)
    criterion = F.nll_loss

    # Train the network
    net, epoch_list, train_loss_list, train_acc_list, valid_loss_list, valid_acc_list = trainNetwork(
        num_epochs, trainloader, device, optimizer, net, criterion, valloader=valloader
    )

    # Visualize the training progress and save plot
    visualizeTrainProgress(epoch_list, train_loss_list, train_acc_list, valid_loss_list, valid_acc_list)
    plt.title("Q 7.1.4")
    plt.savefig('../results/q7_1_4.png')

    torch.save(net.state_dict(), "../results/EMNISTConv.pk")
    return net

def testEMNISTConv(device, model_path="../results/EMNISTConv.pk"):
    from q4 import findLetters
    from run_q4 import cluster
    import os, skimage, string
    # Load the pre-trained network
    net = ConvNet(num_class=47).to(device)
    net.load_state_dict(torch.load(model_path, map_location=device))
    net.eval()

    for img in os.listdir('../images'):
        im1 = skimage.img_as_float(skimage.io.imread(os.path.join('../images',img)))
        bboxes, bw = findLetters(im1)

        # Cluster
        classes = cluster(bboxes)
        # Group the bboxes by class
        line_labels = np.unique(classes)
        line_idx = []
        for label in line_labels:
            this_line_idx = np.where(classes == label)[0]
            # Sort characters by x coordinates
            this_line_idx = this_line_idx[bboxes[this_line_idx, 1].argsort()]
            line_idx.append(this_line_idx)
        # Sort lines by the first y index
        first_ys = np.array([bboxes[line[0], 0] for line in line_idx])
        sorted_line_idx = []
        for i in first_ys.argsort():
            sorted_line_idx.append(line_idx[i])
        line_idx = sorted_line_idx

        # crop the bounding boxes
        X = []
        pad_width = 2
        img_width = 28
        for box in bboxes:
            y1, x1, y2, x2 = box
            cx, cy = (x1+x2)/2.0, (y1+y2)/2.0
            l = max(y2-y1, x2-x1)
            nx1, nx2 = int(cx-l/2), int(cx+l/2)
            ny1, ny2 = int(cy-l/2), int(cy+l/2)
            crop = bw[ny1:ny2, nx1:nx2].copy()
            crop = skimage.transform.resize(crop.astype(float), (img_width-pad_width*2, img_width-pad_width*2))
            crop = (crop < 0.8).T
            crop = np.pad(crop, pad_width=pad_width, mode='constant', constant_values=0)
            X.append(crop)
            # plt.imshow(crop, cmap='gray')
            # plt.show()
        X = np.array(X)

        # Convert input to torch format and forward
        X = torch.Tensor(X, device=device).unsqueeze(1)
        probs = net(X)
        pred_label = probs.argmax(axis=1)

        # The mapping of the EMNIST balanced dataset
        chars = ''.join([str(_) for _ in range(10)]) + string.ascii_uppercase[:26] + "abdefghnqrt"
        text_by_line = []
        for r in line_idx:
            line = ""
            for idx in r:
                line += chars[pred_label[int(idx)]]
            text_by_line.append(line)

        print()
        for line in text_by_line:
            print(line)

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    if sys.argv[1] == "7.1.1":
        print("Q7.1.1: Train a 2-layer MLP on NIST36 training set")
        NISTFC(device)
    if sys.argv[1] == "7.1.2":
        print("Q7.1.2: Train a CNN on MNIST")
        MNISTConv(device)
    if sys.argv[1] == "7.1.3":
        print("Q7.1.3: Train a CNN on NIST36")
        NISTConv(device)
    if sys.argv[1] == "7.1.4":
        if sys.argv[2] == "train":
            print("Q7.1.4: Train a CNN on EMNIST Balanced")
            EMNISTConv(device)
        if sys.argv[2] == "test":
            print("Q7.1.4: Test the trained Conv on findLetters in Q4")
            testEMNISTConv(device)

if __name__ == '__main__':
    main()
