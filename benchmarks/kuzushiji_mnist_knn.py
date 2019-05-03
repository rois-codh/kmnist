# kNN with neighbors=4 benchmark for Kuzushiji-MNIST
# Acheives 97.4% test accuracy

from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import wandb
import time

def load(f):
    return np.load(f)['arr_0']

# Load the data
x_train = load('../dataset/kmnist-train-imgs.npz')
x_test = load('../dataset/kmnist-test-imgs.npz')
y_train = load('../dataset/kmnist-train-labels.npz')
y_test = load('../dataset/kmnist-test-labels.npz')

# Flatten images
x_train = x_train.reshape(-1, 784)
x_test = x_test.reshape(-1, 784)

# Set hyperparameters
N_TRAIN = len(y_train)
N_TEST = len(y_test)
N_NEIGHBORS = 4
WEIGHTS = "distance"

wandb.init(project="kmnist")
config = {
  "n_train" : N_TRAIN,
  "n_test" : N_TEST,
  "n_neighbors" : N_NEIGHBORS,
  "weights" : WEIGHTS
}
print(config)
wandb.config.update(config)

clf = KNeighborsClassifier(n_neighbors=4, weights='distance', n_jobs=-1)
print('Fitting', clf)
clf.fit(x_train, y_train)
print('Evaluating', clf)
test_score = clf.score(x_test, y_test)
print('Test accuracy:', test_score)
wandb.log({"accuracy" : test_score})
