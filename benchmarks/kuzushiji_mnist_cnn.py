# Based on MNIST CNN from Keras' examples: https://github.com/keras-team/keras/blob/master/examples/mnist_cnn.py (MIT License)

from __future__ import print_function
import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
#from tensorflow.keras.optimizers import Adadelta
import numpy as np
import wandb
from wandb.keras import WandbCallback

# Set hyperparameters
BATCH_SIZE = 128
NUM_CLASSES = 10
EPOCHS = 25
L1_SIZE = 32
L2_SIZE = 64
DROPOUT_1 = 0.25
DROPOUT_2 = 0.5
FC1_SIZE = 128

# input image dimensions
img_rows, img_cols = 28, 28

wandb.init(project="kmnist")
config = {
  "batch_size" : BATCH_SIZE,
  "num_classes" : NUM_CLASSES,
  "epochs" : EPOCHS,
  "l1_size": L1_SIZE,
  "l2_size" : L2_SIZE,
  "dropout_1" : DROPOUT_1,
  "dropout_2" : DROPOUT_2,
  "fc1_size" : FC1_SIZE 
}
wandb.config.update(config)

def load(f):
    return np.load(f)['arr_0']

# Load the data
x_train = load('../dataset/kmnist-train-imgs.npz')
x_test = load('../dataset/kmnist-test-imgs.npz')
y_train = load('../dataset/kmnist-train-labels.npz')
y_test = load('../dataset/kmnist-test-labels.npz')

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
model.add(Conv2D(L1_SIZE, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(L2_SIZE, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(DROPOUT_1))
model.add(Flatten())
model.add(Dense(FC1_SIZE, activation='relu'))
model.add(Dropout(DROPOUT_2))
model.add(Dense(NUM_CLASSES, activation='softmax'))

model.compile(loss="categorical_crossentropy",
              optimizer="adadelta",
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=BATCH_SIZE,
          epochs=EPOCHS,
          verbose=1,
          validation_data=(x_test, y_test),
          callbacks=[WandbCallback()])

train_score = model.evaluate(x_train, y_train, verbose=0)
test_score = model.evaluate(x_test, y_test, verbose=0)
print('Train loss:', train_score[0])
print('Train accuracy:', train_score[1])
print('Test loss:', test_score[0])
print('Test accuracy:', test_score[1])
