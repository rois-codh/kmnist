# Kuzushiji-MNIST

[![License: CC BY-SA 4.0](https://img.shields.io/badge/License-CC%20BY--SA%204.0-blue.svg)](https://creativecommons.org/licenses/by-sa/4.0/)  
📚 [Read the paper](https://arxiv.org/abs/1812.01718) to learn more about Kuzushiji, the datasets and our motivations for making them!

**Kuzushiji-MNIST** is a drop-in replacement for the MNIST dataset (28x28 grayscale, 70,000 images), provided in the original MNIST format as well as a NumPy format. Since MNIST restricts us to 10 classes, we chose one character to represent each of the 10 rows of Hiragana when creating Kuzushiji-MNIST.

**Kuzushiji-49**, as the name suggests, has 49 classes (28x28 grayscale, 266,407 images), is a much larger, but imbalanced dataset containing 48 Hiragana characters and one Hiragana iteration mark.

**Kuzushiji-Kanji** is an imbalanced dataset of total 3832 Kanji characters (64x64 grayscale, 140,426 images), ranging from 1,766 examples to only a single example per class.

<p align="center">
  <img src="images/kmnist_examples.png">
  The 10 classes of Kuzushiji-MNIST, with the first column showing each character's modern hiragana counterpart.
</p>

## Get the data 💾

🌟 You can run `python download_data.py` to interactively select and download any of these datasets!

### Kuzushiji-MNIST

Kuzushiji-MNIST contains 70,000 28x28 grayscale images spanning 10 classes (one from each column of [hiragana](https://upload.wikimedia.org/wikipedia/commons/thumb/2/28/Table_hiragana.svg/768px-Table_hiragana.svg.png)), and is perfectly balanced like the original MNIST dataset (6k/1k train/test for each class).

| File            | Examples | Download (MNIST format)    | Download (NumPy format)      |
|-----------------|--------------------|----------------------------|------------------------------|
| Training images | 60,000             | [train-images-idx3-ubyte.gz](https://storage.googleapis.com/kuzushiji-mnist/train-images-idx3-ubyte.gz) (17MB) | [kuzushiji10-train-imgs.npz](https://storage.googleapis.com/kuzushiji-mnist/kuzushiji10-train-imgs.npz) (16MB)   |
| Training labels | 60,000             | [train-labels-idx1-ubyte.gz](https://storage.googleapis.com/kuzushiji-mnist/train-labels-idx1-ubyte.gz) (30KB) | [kuzushiji10-train-labels.npz](https://storage.googleapis.com/kuzushiji-mnist/kuzushiji10-train-labels.npz) (50KB)  |
| Testing images  | 10,000             | [t10k-images-idx3-ubyte.gz](https://storage.googleapis.com/kuzushiji-mnist/t10k-images-idx3-ubyte.gz) (3MB) | [kuzushiji10-test-imgs.npz](https://storage.googleapis.com/kuzushiji-mnist/kuzushiji10-test-imgs.npz) (3MB)   |
| Testing labels  | 10,000             | [t10k-labels-idx1-ubyte.gz](https://storage.googleapis.com/kuzushiji-mnist/t10k-labels-idx1-ubyte.gz) (5KB)  | [kuzushiji10-test-labels.npz](https://storage.googleapis.com/kuzushiji-mnist/kuzushiji10-test-labels.npz) (8KB) |

We recommend using standard top-1 accuracy on the test set for evaluating on Kuzushiji-MNIST.

##### Which format do I download?
If you're looking for a drop-in replacement for the MNIST or Fashion-MNIST dataset (for tools that currently work with these datasets), download the data in MNIST format.

Otherwise, it's recommended to download in NumPy format, which can be loaded into an array as easy as:  
`arr = np.load(filename)['arr_0']`.

### Kuzushiji-49

Kuzushiji-49 contains 266,407 images spanning 49 classes.

| File            | Examples |  Download (NumPy format)      |
|-----------------|--------------------|----------------------------|
| Training images | 184,628            | [kuzushiji49-train-imgs.npz](https://storage.googleapis.com/kuzushiji-mnist/kuzushiji49-train-imgs.npz) (50MB)   |
| Training labels | 184,628            | [kuzushiji49-train-labels.npz](https://storage.googleapis.com/kuzushiji-mnist/kuzushiji49-train-labels.npz) (200KB)  |
| Testing images  | 46,185             | [kuzushiji49-test-imgs.npz](https://storage.googleapis.com/kuzushiji-mnist/kuzushiji49-test-imgs.npz) (13MB)   |
| Testing labels  | 46,185             | [kuzushiji49-test-labels.npz](https://storage.googleapis.com/kuzushiji-mnist/kuzushiji49-test-labels.npz) (50KB) |

### Kuzushiji-Kanji

Kuzushiji-Kanji is a large 64x64 dataset of 3832 Kanji characters, both common and rare.  
We hope to have the full Kuzushiji-Kanji dataset available for download within the next few days!

![Examples of Kuzushiji-Kanji classes](images/kkanji_examples.png)

## Benchmarks & Results 📈

Have more results to add to the table? Feel free to submit an [issue](https://github.com/rois-codh/kmnist/issues/new) or [pull request](https://github.com/rois-codh/kmnist/compare)!

|Model                            | MNIST | KMNIST | K49 |
|---------------------------------|-------|--------|-----|
|[4-Nearest Neighbour Baseline](benchmarks/kuzushiji_mnist_knn.py)     |97.14% | 91.56% |86.01%|
|[Keras Simple CNN Benchmark](benchmarks/kuzushiji_mnist_cnn.py)       |99.06% | 95.12% |89.25%|
|PreActResNet-18                  |**99.56%** | 97.82% |96.64%|
|PreActResNet-18 + Input Mixup    |99.54% | 98.41% |97.04%|
|PreActResNet-18 + Manifold Mixup |99.54% | **98.83%** | **97.33%** |

For MNIST and KMNIST we use a standard accuracy metric, while Kuzushiji-49 is evaluated using balanced accuracy (so that all classes have equal weight).

## License

Both the dataset itself and the contents of this repo are licensed under a permissive [CC-BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) license, except where specified within some benchmark scripts.
