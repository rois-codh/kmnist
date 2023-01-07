import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import numpy as np

from config import *

# set dataset path
PATH_TO_TRAIN_FILE = config['TRAIN_FILE']
PATH_TO_TEST_FILE = config['TEST_FILE']
PATH_TO_TRAIN_LABEL_FILE = config['TRAIN_LABEL']
PATH_TO_TEST_LABEL_FILE = config['TEST_LABEL']
TRAIN_IMAGE_NUM = config['TRAIN_NUM']


# load data
class MyDataset(Dataset):
    def __init__(self, data, target, transform=None):
        self.data = torch.from_numpy(data).float()
        self.data = self.data.unsqueeze(1)
        self.data = torch.cat((self.data, self.data, self.data), 1)
        self.target = torch.from_numpy(target).long()
        self.transform = transform

    def __getitem__(self, index):
        x = self.data[index]
        y = self.target[index]

        if self.transform:
            x = self.transform(x)

        return x, y

    def __len__(self):
        return len(self.data)


# preprocess images
transform = transforms.Compose([transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                transforms.Resize((config['TRAN_SIZE'], config['TRAN_SIZE']))])


# configure dataloader for train data
def get_train_dataloader(batch_size, shuffle, num_workers):
    train_images = np.load(PATH_TO_TRAIN_FILE)['arr_0'][:TRAIN_IMAGE_NUM]
    train_labels = np.load(PATH_TO_TRAIN_LABEL_FILE)['arr_0'][:TRAIN_IMAGE_NUM]
    train_dataset = MyDataset(train_images, train_labels, transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    return train_loader


# configure dataloader for validation data
def get_validation_dataloader(batch_size, shuffle, num_workers):
    val_images = np.load(PATH_TO_TRAIN_FILE)['arr_0'][TRAIN_IMAGE_NUM:]
    val_labels = np.load(PATH_TO_TRAIN_LABEL_FILE)['arr_0'][TRAIN_IMAGE_NUM:]
    val_dataset = MyDataset(val_images, val_labels, transform)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    return val_loader


# configure dataloader for test data
def get_test_dataloader(batch_size, shuffle, num_workers):
    test_images = np.load(PATH_TO_TEST_FILE)['arr_0']
    test_labels = np.load(PATH_TO_TEST_LABEL_FILE)['arr_0']
    test_dataset = MyDataset(test_images, test_labels, transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    return test_loader
