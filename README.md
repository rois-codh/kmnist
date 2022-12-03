# Kuzushiji-MNIST Classification

This repository impletments the classification for [Kuzushiji-MNIST](https://github.com/rois-codh/kmnist) in Pytorch with model ResNet & ResMLP.



## Download the dataset

(1) Get in the project folder in the terminal.

(2) Run

```shell
wget http://codh.rois.ac.jp/kmnist/dataset/kmnist/kmnist-train-imgs.npz
wget http://codh.rois.ac.jp/kmnist/dataset/kmnist/kmnist-train-labels.npz
wget http://codh.rois.ac.jp/kmnist/dataset/kmnist/kmnist-test-imgs.npz
wget http://codh.rois.ac.jp/kmnist/dataset/kmnist/kmnist-test-labels.npz
```



## MLP/CNN Model

**You will need GPU to run the following code.**

**Results will be stored in `./log`**

**Model parameters will be stored in `./models`**



Available model names:

```
ResMLP-12, ResMLP-24, ResNet-18, ResNet-34
```

To train and test the model:

```shell
python classification.py --model [model name] --gpu [GPU No.] --train 1 --test 1 --train_batch [train batch size] --test_batch [test batch size] --epoch [number of train epoch]
```

Only to test the model

```shell
python classification.py --model [model name] --gpu [GPU No.] --test_batch [test batch size] 
```



For ResMLP-12

```shell
python classification.py --model ResMLP-12 --gpu 0 --train 1 --test 1 --train_batch 64 --test_batch 500 --epoch 30
```

For ResMLP-24

```shell
python classification.py --model ResMLP-24 --gpu 0 --train 1 --test 1 --train_batch 64 --test_batch 500 --epoch 30
```

For ResNet-18

```shell
python classification.py --model ResNet-18 --gpu 0 --train 1 --test 1 --train_batch 64 --test_batch 500 --epoch 30
```

For ResNet-34

```shell
python classification.py --model ResNet-34 --gpu 0 --train 1 --test 1 --train_batch 64 --test_batch 500 --epoch 30
```

