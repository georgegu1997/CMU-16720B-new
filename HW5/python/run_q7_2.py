import numpy as np
import sys, os
import torch
import torch.nn as nn
import torchvision
from run_q7_1 import checkAccLoss, runEpoch, trainNetwork, visualizeTrainProgress

def main():
    if len(sys.argv) < 3:
        raise ValueError("Usage: python run_q7_2.py [image folder] [network name]")

    num_epochs = 20


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
        T.Scale(256),
        T.RandomSizedCrop(224),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    val_transform = T.Compose([
        T.Scale(224),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    # define the train and validation dataset
    trainset = torchvision.datasets.ImageFolder(
        root=os.path.join(dataset_folder, "train"),
    )
    valset = torchvision.datasets.ImageFolder(
        root=os.path.join(dataset_folder, "val"),
    )
    trainloader = DataLoader(trainset, batch_size=batch_size, num_workers=4)
    valloader = DataLoader(valset, batch_size=batch_size, num_workers=4)

    # Define the network
    if sys.argv[2] == "squeeze":
        net = torchvision.models.squeezenet1_1(pretrained=True)

        # Reinitialize the classifier layer
        final_conv = nn.Conv2d(512, num_classes, kernel_size=1)
        self.classifier = nn.Sequential(
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
        optimizer = torch.optim.Adam(model.classifier.parameters(), lr=1e-3)



        net, epoch_list, train_loss_list, train_acc_list, valid_loss_list, valid_acc_list = \
            trainNetwork(num_epochs, trainloader, device, optimizer, net, criterion, valloader = None, acc_freq=1)

    elif sys.argv[2] == "small":
        pass
    else:
        raise ValueError("Uknown input argument:", sys.argv[1])




    pass

if __name__ == '__main__':
    main()
