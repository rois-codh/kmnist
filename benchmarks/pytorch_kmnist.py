# pytorch_kmnist.py
#----------------
# Train a small CNN in Pytorch to identify 10 Japanese characters in classical script
# Based on MNIST CNN from wandb pytorch examples
# https://github.com/wandb/examples/blob/master/pytorch-cnn-fashion/train.py

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets
from torch.autograd import Variable
import torch.nn.functional as F
from utils import load_train_data, load_test_data, load
from PIL import Image

import wandb
import os

wandb.init()
config = wandb.config

config.dropout = 0.5
config.channels_one = 16
config.channels_two = 32
config.batch_size = 100
config.epochs = 50

class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()

        # Convolution 1
        self.cnn1 = nn.Conv2d(in_channels=1, out_channels=config.channels_one, kernel_size=5, stride=1, padding=0)
        self.relu1 = nn.ReLU()
        # Max pool 1
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        # Convolution 2
        self.cnn2 = nn.Conv2d(in_channels=config.channels_one, out_channels=config.channels_two, kernel_size=5, stride=1, padding=0)
        self.relu2 = nn.ReLU()

        # Max pool 2
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        self.dropout = nn.Dropout(p=config.dropout)

        # Fully connected 1 (readout)
        self.fc1 = nn.Linear(config.channels_two*4*4, 10)

    def forward(self, x):
        # Convolution 1
        out = self.cnn1(x)
        out = self.relu1(out)

        # Max pool 1
        out = self.maxpool1(out)

        # Convolution 2
        out = self.cnn2(out)
        out = self.relu2(out)

        # Max pool 2
        out = self.maxpool2(out)

        # Resize
        # Original size: (100, 32, 7, 7)
        # out.size(0): 100
        # New out size: (100, 32*7*7)
        out = out.view(out.size(0), -1)
        out = self.dropout(out)
        # Linear function (readout)
        out = self.fc1(out)

        return out

class KMnistDataset(torch.utils.data.Dataset):
    """`MNIST <http://yann.lecun.com/exdb/mnist/>`_ Dataset.
    Args:
        root (string): Root directory of dataset where ``processed/training.pt``
            and  ``processed/test.pt`` exist.
        train (bool, optional): If True, creates dataset from ``training.pt``,
            otherwise from ``test.pt``.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """
    

    def __init__(self, root='.', train=True, transform=None, target_transform=None, download=False):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set

        if self.train:
            self.train_data, self.train_labels = load_train_data(self.root)
            self.train_labels = torch.LongTensor(self.train_labels)
        else:
            self.test_data, self.test_labels = load_test_data(self.root)
            self.test_labels =  torch.LongTensor(self.test_labels)

        
            
    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.train:
            img, target = self.train_data[index], self.train_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img, mode='L')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)
    
    
def main():
    normalize = transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
                                         std=[x/255.0 for x in [63.0, 62.1, 66.7]])

    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.1307,), (0.3081,))])

    train_dataset = KMnistDataset(root='./dataset',
                                  train=True,
                                  transform=transform
                               )

    test_dataset = KMnistDataset(root='./dataset',
                                 train=False,
                                 transform=transform
                               )


    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=config.batch_size,
                                               shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=config.batch_size,
                                              shuffle=False)


    model = CNNModel()
    wandb.watch(model)

    criterion = nn.CrossEntropyLoss()
    config.learning_rate = 0.001

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    iter = 0
    for epoch in range(config.epochs):
        for i, (images, labels) in enumerate(train_loader):

            images = Variable(images)
            labels = Variable(labels)

            # Clear gradients w.r.t. parameters
            optimizer.zero_grad()

            # Forward pass to get output/logits
            outputs = model(images)

            # Calculate Loss: softmax --> cross entropy loss
            loss = criterion(outputs, labels)

            # Getting gradients w.r.t. parameters
            loss.backward()

            # Updating parameters
            optimizer.step()

            iter += 1

            if iter % 100 == 0:
                # Calculate Accuracy
                correct = 0
                correct_arr = [0.0] * 10
                total = 0
                total_arr = [0.0] * 10

                # Iterate through test dataset
                for images, labels in test_loader:
                    images = Variable(images)

                    # Forward pass only to get logits/output
                    outputs = model(images)

                    # Get predictions from the maximum value
                    _, predicted = torch.max(outputs.data, 1)

                    # Total number of labels
                    total += labels.size(0)


                    correct += (predicted == labels).sum()
                    for label in range(10):
                        correct_arr[label] += (((predicted == labels) + (labels == label)) == 2).sum() * 100
                        total_arr[label] += (labels == label).sum()


                accuracy = float(correct) / total

                metrics = {'kmnist_val_acc': accuracy, 'val_loss': loss}
                wandb.log(metrics)


if __name__ == '__main__':
   main()
