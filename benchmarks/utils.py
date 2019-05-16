# utility functions for benchmarks
import os

# load data file into array
def load(f):
    return np.load(f)['arr_0']

def load_train_data(datadir):
  x_train = load(os.path.join(datadir, 'kmnist-train-imgs.npz'))
  y_train = load(os.path.join(datadir, 'kmnist-train-labels.npz'))
  return x_train, y_train

def load_test_data(datadir):
  x_test = load(os.path.join(datadir, 'kmnist-test-imgs.npz'))
  y_test = load(os.path.join(datadir, 'kmnist-test-labels.npz'))
  return x_test, y_test
