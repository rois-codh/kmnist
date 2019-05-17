# Based on MNIST CNN from Keras' examples: https://github.com/keras-team/keras/blob/master/examples/mnist_cnn.py (MIT License)

from __future__ import print_function
import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

import argparse
import numpy as np
import os
from utils import load_train_data, load_test_data, load
import wandb
from wandb.keras import WandbCallback

# default configuration / hyperparameter values
# you can modify these below or via command line
MODEL_NAME = ""
DATA_HOME = "./dataset" 
BATCH_SIZE = 128
EPOCHS = 10
L1_SIZE = 32
L2_SIZE = 64
DROPOUT_1 = 0.25
DROPOUT_2 = 0.5
FC1_SIZE = 128
NUM_CLASSES = 10

# input image dimensions
img_rows, img_cols = 28, 28

def train_cnn(args):
  # initialize wandb logging to your project
  wandb.init()
  config = {
    "model_type" : "cnn",
    "batch_size" : args.batch_size,
    "num_classes" : args.num_classes,
    "epochs" : args.epochs,
    "l1_size": args.l1_size,
    "l2_size" : args.l2_size,
    "dropout_1" : args.dropout_1,
    "dropout_2" : args.dropout_2,
    "fc1_size" : args.fc1_size
  }
  wandb.config.update(config)

  # Load the data form the relative path provided
  x_train, y_train = load_train_data(args.data_home)
  x_test, y_test = load_test_data(args.data_home)

  if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
  else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

  x_train = x_train.astype('float32')
  x_test = x_test.astype('float32')
  x_train /= 255
  x_test /= 255
  N_TRAIN = len(x_train)
  N_TEST = len(x_test)
  wandb.config.update({"n_train" : N_TRAIN, "n_test" : N_TEST})
  print('{} train samples, {} test samples'.format(N_TRAIN, N_TEST))

  # Convert class vectors to binary class matrices
  y_train = keras.utils.to_categorical(y_train, NUM_CLASSES)
  y_test = keras.utils.to_categorical(y_test, NUM_CLASSES)

  # Build model
  model = Sequential()
  model.add(Conv2D(args.l1_size, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
  model.add(Conv2D(args.l2_size, (3, 3), activation='relu'))
  model.add(MaxPooling2D(pool_size=(2, 2)))
  model.add(Dropout(args.dropout_1))
  model.add(Flatten())
  model.add(Dense(args.fc1_size, activation='relu'))
  model.add(Dropout(args.dropout_2))
  model.add(Dense(args.num_classes, activation='softmax'))

  model.compile(loss="categorical_crossentropy",
              optimizer="adadelta",
              metrics=['accuracy'])

  model.fit(x_train, y_train,
            batch_size=args.batch_size,
            epochs=args.epochs,
            verbose=1,
            validation_data=(x_test, y_test),
            callbacks=[WandbCallback()])

  train_score = model.evaluate(x_train, y_train, verbose=0)
  test_score = model.evaluate(x_test, y_test, verbose=0)
  print('Train loss:', train_score[0])
  print('Train accuracy:', train_score[1])
  print('Test loss:', test_score[0])
  print('Test accuracy:', test_score[1])

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
    "-b",
    "--batch_size",
    type=int,
    default=BATCH_SIZE,
    help="batch size")
  parser.add_argument(
    "--dropout_1",
    type=float,
    default=DROPOUT_1,
    help="dropout  layers")
  parser.add_argument(
    "--dropout_2",
    type=float,
    default=DROPOUT_2,
    help="dropout before dense layers")
  parser.add_argument(
    "-e",
    "--epochs",
    type=int,
    default=EPOCHS,
    help="number of training epochs (passes through full training data)")
  parser.add_argument(
    "--fc1_size",
    type=int,
    default=FC1_SIZE,
    help="size of fully-connected layer")
  parser.add_argument(
    "--l1_size",
    type=int,
    default=L1_SIZE,
    help="size of first conv layer")
  parser.add_argument(
    "--l2_size",
    type=int,
    default=L2_SIZE,
    help="size of second conv layer")
  parser.add_argument(
    "--num_classes",
    type=int,
    default=NUM_CLASSES,
    help="number of classes (default: 10)")
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
  if not args.model_name:
    print("warning: no run name provided")
  else:
    os.environ['WANDB_DESCRIPTION'] = args.model_name

  train_cnn(args)

