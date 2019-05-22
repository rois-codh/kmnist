# utility functions for benchmarks
import numpy as np
import os

from tensorflow.keras.callbacks import Callback
import wandb

# extend Keras callback to log benchmark-specific key, "kmnist_val_acc"
class KmnistCallback(Callback):
  def on_epoch_end(self, epoch, logs={}):
    if "val_accuracy" in logs:
        # latest version of tensorflow
        wandb.log({"kmnist_val_acc" : logs["val_accuracy"]})
    elif "val_acc" in logs:
        # older version of tensorflow
        wandb.log({"kmnist_val_acc" : logs["val_acc"]})
    else:
        raise Exception("Keras logs object missing validation accuracy")

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
