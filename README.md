# Kuzushiji-MNIST
Repository for Kuzushiji_MNIST, Kuzushiji49, and Kuzushiji_Kanji

Kuzushiji-MNIST, a drop-in replacement for the MNIST dataset.
Kuzushiji-49, a much larger, but imbalanced dataset containing 48 Hiragana characters and one Hiragana iteration mark.
Kuzushiji-Kanji, an imbalanced dataset of 3832 characters, including rare characters with very few samples.

The 10 classes of Kuzushiji-MNIST, with the first column showing each character's modern hiragana counterpart.

![Image showing examples of each class of Kuzushiji MNIST](https://storage.googleapis.com/kuzushiji-mnist/kuzushiji-mnist-30example-labelled.png)

## Get the data

You can run `python download_data.py` to interactively select and download any of these datasets!

### Kuzushiji-MNIST

Kuzushiji-MNIST contains 70,000 28x28 grayscale images spanning 10 classes (one from each column of [hiragana](https://upload.wikimedia.org/wikipedia/commons/thumb/2/28/Table_hiragana.svg/768px-Table_hiragana.svg.png) except ん), and is perfectly balanced like the original MNIST dataset (6k/1k train/test for each class).

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

Kuzushiji-49 contains 230,000 images spanning 49 classes.

| File            | Examples |  Download (NumPy format)      |
|-----------------|--------------------|----------------------------|
| Training images | 184,628            | [kuzushiji49-train-imgs.npz](https://storage.googleapis.com/kuzushiji-mnist/kuzushiji49-train-imgs.npz) (50MB)   |
| Training labels | 184,628            | [kuzushiji49-train-labels.npz](https://storage.googleapis.com/kuzushiji-mnist/kuzushiji49-train-labels.npz) (200KB)  |
| Testing images  | 46,185             | [kuzushiji49-test-imgs.npz](https://storage.googleapis.com/kuzushiji-mnist/kuzushiji49-test-imgs.npz) (13MB)   |
| Testing labels  | 46,185             | [kuzushiji49-test-labels.npz](https://storage.googleapis.com/kuzushiji-mnist/kuzushiji49-test-labels.npz) (50KB) |

## Benchmarks & Results

### Kuzushiji-MNIST Classification

| Model                                                    | Train accuracy | Test accuracy |
|----------------------------------------------------------|----------------|---------------|
| [Keras Simple CNN Benchmark](benchmarks/kuzushiji_mnist_cnn.py) | 99.31%         | 98.24%        |
| [4-Nearest-Neighbour benchmark](benchmarks/kuzushiji_mnist_knn.py) | N/A            | 97.4%         |
