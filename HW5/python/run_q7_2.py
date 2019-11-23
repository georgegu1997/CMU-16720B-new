import numpy as np
import matplotlib.pyplot as plt
import sys, os
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
import torch.nn.functional as F
from run_q7_1 import checkAccLoss, runEpoch, trainNetwork, visualizeTrainProgress

# Borrowed from the Pytorch official tutorial on training a CNN on MNIST
# https://github.com/pytorch/examples/blob/master/mnist/main.py
# Adapted to make it deeper and work on RGB images
class ConvNet4Conv(nn.Module):
    def __init__(self, channel_in=3, fc_in=9216, num_class=10):
        super(ConvNet4Conv, self).__init__()
        self.conv1 = nn.Conv2d(channel_in, 128, 3, 3)
        self.conv2 = nn.Conv2d(128, 256, 3, 1)
        self.conv3 = nn.Conv2d(256, 512, 3, 1)
        self.conv4 = nn.Conv2d(512, 512, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.25)
        self.dropout3 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(fc_in, 256)
        self.fc2 = nn.Linear(256, num_class)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.conv2(x)
        x = F.max_pool2d(x, 3)
        x = self.dropout1(x)

        x = F.relu(self.conv3(x))
        x = self.conv4(x)
        x = F.max_pool2d(x, 3)
        x = self.dropout2(x)

        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout3(x)
        x = self.fc2(x)
        x = F.log_softmax(x, dim=1)
        return x

def main():
    if len(sys.argv) < 3:
        raise ValueError("Usage: python run_q7_2.py [image folder] [network name]")

    num_epochs = 100
    batch_size = 64

    if sys.argv[1] == '102':
        num_classes = 102
        dataset_folder = "../data/oxford-flowers102"
    elif sys.argv[1] == "17":
        num_classes = 17
        dataset_folder = "../data/oxford-flowers17"
    else:
        raise ValueError("Uknown input argument:", sys.argv[1])

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    '''
    Transformation function for training and testing dataset
    Copied from: https://gist.github.com/jcjohnson/6e41e8512c17eae5da50aebef3378a4c
    '''
    train_transform = T.Compose([
        T.Resize(256),
        T.RandomResizedCrop(224),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    val_transform = T.Compose([
        T.Resize(224),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    # define the train and validation dataset
    trainset = torchvision.datasets.ImageFolder(
        root=os.path.join(dataset_folder, "train"),
        transform = train_transform
    )
    valset = torchvision.datasets.ImageFolder(
        root=os.path.join(dataset_folder, "val"),
        transform = val_transform
    )
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, num_workers=4)
    valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size, num_workers=4)

    # Define the network
    if sys.argv[2] == "squeeze":
        net = torchvision.models.squeezenet1_1(pretrained=True)

        # Reinitialize the classifier layer
        final_conv = nn.Conv2d(512, num_classes, kernel_size=1)
        net.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            final_conv,
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        # optimize only the classifier layer
        for param in net.parameters():
            param.requires_grad = False
        for param in net.classifier.parameters():
            param.requires_grad = True

        # the optimizer only on the part to be fine tuned
        optimizer = torch.optim.Adam(net.classifier.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()

    elif sys.argv[2] == "small":
        net = ConvNet4Conv(channel_in=3, fc_in=18432, num_class=num_classes)
        optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)
        criterion = F.nll_loss

    else:
        raise ValueError("Uknown input argument:", sys.argv[1])

    net, epoch_list, train_loss_list, train_acc_list, valid_loss_list, valid_acc_list = \
        trainNetwork(num_epochs, trainloader, device, optimizer, net, criterion, valloader = valloader, acc_freq=1)

    # Visualize the training progress and save plot
    visualizeTrainProgress(epoch_list, train_loss_list, train_acc_list, valid_loss_list, valid_acc_list)
    plt.title("Q 7.2.1")
    plt.savefig('../results/q7_2_1_%s_%s.png' % (sys.argv[1], sys.argv[2]))

if __name__ == '__main__':
    main()
