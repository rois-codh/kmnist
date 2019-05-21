# kNN with neighbors=4 benchmark for Kuzushiji-MNIST

from sklearn.neighbors import KNeighborsClassifier
import argparse
import numpy as np
import os
import time
from utils import load_train_data, load_test_data, load
import wandb

# default configuration / hyperparameter values
# you can modify these below or via command line
MODEL_NAME = ""
DATA_HOME = "./dataset"
K_NEIGHBORS = 4
WEIGHTS = "distance"

def train_knn(args):
  # Load the data
  x_train, y_train = load_train_data(args.data_home)
  x_test, y_test = load_test_data(args.data_home)

  # Flatten images
  x_train = x_train.reshape(-1, 784)
  x_test = x_test.reshape(-1, 784)

  # Set hyperparameters
  N_TRAIN = len(y_train)
  N_TEST = len(y_test)

  wandb.init()
  config = {
    "model_type" : "knn",
    "n_train" : N_TRAIN,
    "n_test" : N_TEST,
    "k_neighbors" : args.k_neighbors,
    "weights" : args.weights
  }
  wandb.config.update(config)

  clf = KNeighborsClassifier(n_neighbors=args.k_neighbors, weights=args.weights, n_jobs=-1)
  print('Fitting', clf)
  clf.fit(x_train, y_train)
  print('Evaluating', clf)
  test_score = clf.score(x_test, y_test)
  print('Test accuracy:', test_score)
  # store train accuracy as validation accuracy as well to simplify
  # comparison to CNN/other scripts
  wandb.log({"accuracy" : test_score})
  wandb.log({"kmnist_val_acc" : test_score})

if __name__ == "__main__":
  parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument(
    "-m",
    "--model_name",
    type=str,
    default=MODEL_NAME,
    help="Name of this model/run (model will be saved to this file)")
  parser.add_argument(
    "--data_home",
    type=str,
    default=DATA_HOME,
    help="Relative path to training/test data")
  parser.add_argument(
    "-k",
    "--k_neighbors",
    type=int,
    default=K_NEIGHBORS,
    help="k for k-nearest-neighbors, or number of nearest neighbors")
  parser.add_argument(
    "--weights",
    type=str,
    default=WEIGHTS,
    help="weight function used in prediction to sklearn.neighbors")
  parser.add_argument(
    "-q",
    "--dry_run",
    action="store_true",
    help="Dry run (do not log to wandb)")

  args = parser.parse_args()

  # easier testing--don't log to wandb if dry run is set
  if args.dry_run:
    os.environ['WANDB_MODE'] = 'dryrun'

  # create run name
  if args.model_name:
    os.environ['WANDB_DESCRIPTION'] = args.model_name

  train_knn(args)
 
