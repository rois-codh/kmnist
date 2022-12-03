# Kuzushiji-MNIST Classification

This repository impletments the classification for [Kuzushiji-MNIST](https://github.com/rois-codh/kmnist) in Pytorch with model ResNet & ResMLP.



## Download the dataset

```shell
wget http://codh.rois.ac.jp/kmnist/dataset/kmnist/kmnist-train-imgs.npz
wget http://codh.rois.ac.jp/kmnist/dataset/kmnist/kmnist-train-labels.npz
wget http://codh.rois.ac.jp/kmnist/dataset/kmnist/kmnist-test-imgs.npz
wget http://codh.rois.ac.jp/kmnist/dataset/kmnist/kmnist-test-labels.npz
```



## MLP/CNN Model

**You will need GPU to run the following code.**

Available model names:

```
ResMLP-12, ResMLP-24, ResNet-18, ResNet-34
```

To test the model:

```shell
python classification.py --model [model name] --gpu [GPU No.]
```

To train the model:

```shell
python classification.py --model [model name] --gpu [GPU No.] --train 1
```

Results are stored in `./log`
