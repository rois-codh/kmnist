# Weights & Biases Benchmark for Kuzushiji-MNIST

This community benchmark is a fork of [Kuzushiji-MNIST](https://github.com/rois-codh/kmnist) instrumented with [Weights & Biases](https://www.wandb.com) to track and visualize model training and facilitate collaborative deep learning solutions.

## How to participate
You can learn more and [join the benchmark here](https://app.wandb.ai/wandb/kmnist/benchmark).

## Objective

Given an image, correctly classify it as showing one of ten classical Japanese cursive characters. This benchmark is a fresh reimagining of the well-known deep learning baseline of handwritten digits (mnist). It preserves the technical simplicity of mnist and offers more headroom for creativity, since the solution space is less explored and visual intuition is unreliable (only a small number of experts can read Kuzushiji script, regardless of fluency in contemporary Japanese). This is an exciting challenge with a tangible potential outcome of making classical Japanese literature more accessible.

## Dataset

Kuzushiji-MNIST perfectly replaces the well-known MNIST. This balanced dataset contains images of 10 classes of classical Japanese characters in cursive script, where each image is 28x28 black and white pixels, for a total of 70K examples (60K train, 10K test).

## Usage

Please refer to the [benchmark instructions](https://app.wandb.ai/wandb/kmnist/benchmark) to get started.
The ``benchmarks`` directory contains two starter scripts: a K-nearest-neighbors and simple convolutional net approach to kmnist.
Run either script with ``-h`` to see all the options&mdash;hyperparameters and config can be set at the top of each script or overriden via the command line for convenience.

## References
For more information on the dataset and existing approaches, please see [the README](https://github.com/rois-codh/kmnist) and the [models benchmarked](https://github.com/rois-codh/kmnist#benchmarks--results-) in the original repository, as well as the original paper Deep Learning for Classical Japanese Literature by Tarin Clanuwat et al. [arXiv:1812.01718](https://arxiv.org/abs/1812.01718).
