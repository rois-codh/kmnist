# Kuzushiji-MNIST Benchmark

This W&B Benchmark instruments the Kuzushiji-MNIST (kmnist) dataset. [Read the paper](https://arxiv.org/abs/1812.01718) and see
the [original repository](https://github.com/rois-codh/kmnist) to learn more about this dataset, approaches so far, and Kuzushiji, the cursive style of Japanese writing used up until 150 years ago.

## Objective

This benchmark is a fresh reimagining of the well-known deep learning baseline of handwritten digits (mnist). It preserves the technical simplicity of the canonical computer vision problem of classifying ten different characters from black and white images. It offers more headroom for creative exploration, since the solution space is far less saturated than mnist and visual intuition is unreliable (only a small number of experts can read this classical Kuzushiji script, regardless of fluency in contemporary Japanese). This is an exciting challenge with a tangible potential outcome of making classical Japanese literature more accessible.

## Dataset

The kmnist benchmark uses a version of Kuzushiji-MNIST that perfectly substitutes for MNIST. The data consists of 10 classes of characters, where each image is 28x28, for a total of 70K examples (60K train, 10K test).

## Usage

The ``benchmarks`` directory contains two starter scripts: a K-nearest-neighbors and simple convolutional net approach to kmnist. From ``benchmarks``, you can run:
* ``python knn_kmnist.py`` to train the KNN
* ``python cnn_kmnist.py`` to train the CNN

Run either script with ``-h`` to see all the options&mdash;hyperparameters and config can be set at the top of each script or overriden via the command line for convenience.

## Initial benchmark

Please refer to the benchmarks listed in the [original repository](https://github.com/rois-codh/kmnist). We will add benchmark numbers for this repository with the current settings as they become available.

## References

Deep Learning for Classical Japanese Literature. Tarin Clanuwat et al. [arXiv:1812.01718](https://arxiv.org/abs/1812.01718)
