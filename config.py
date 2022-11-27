import os

PROJECT_PATH = '/userhome/cs2/mingzeng/codes/kmnist/'

config = {
    'LOG_PATH': PROJECT_PATH + 'log/',
    'TRAIN_FILE': PROJECT_PATH + 'kmnist-train-imgs.npz',
    'TEST_FILE': PROJECT_PATH + 'kmnist-test-imgs.npz',
    'TRAIN_LABEL': PROJECT_PATH + 'kmnist-train-labels.npz',
    'TEST_LABEL': PROJECT_PATH + 'kmnist-test-labels.npz',
    'TRAIN_NUM': 54000,
    'CUDA_VISIBLE_DEVICES': "0",
    'TRAN_SIZE': 224,
    'TRAN_CROP': 224,
    'N_CLASSES': 10
}
