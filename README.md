# Kuzushiji-MNIST
Repository for Kuzushiji_MNIST, Kuzushiji49, and Kuzushiji_Kanji

The 10 classes of Kuzushiji-MNIST, with the first column showing each character's modern hiragana counterpart.

![Image showing examples of each class of Kuzushiji MNIST](https://storage.googleapis.com/kuzushiji-mnist/kuzushiji-mnist-30example-labelled.png)

## Get the data
### Kuzushiji-MNIST

Kuzushiji-MNIST contains 70,000 28x28 grayscale images spanning 10 classes (one from each column of [hiragana](https://upload.wikimedia.org/wikipedia/commons/thumb/2/28/Table_hiragana.svg/768px-Table_hiragana.svg.png) except ん), and is perfectly balanced like the original MNIST dataset (6k/1k train/test for each class).

| File            | Examples | Download (MNIST format)    | Download (NumPy format)      |
|-----------------|--------------------|----------------------------|------------------------------|
| Training images | 60,000             | train-images-idx3-ubyte.gz | [kuzushiji10-train-imgs.npz](https://storage.googleapis.com/kuzushiji-mnist/kuzushiji10-train-imgs.npz) (16MB)   |
| Training labels | 60,000             | train-labels-idx1-ubyte.gz | [kuzushiji10-train-labels.npz](https://storage.googleapis.com/kuzushiji-mnist/kuzushiji10-train-imgs.npz)   |
| Testing images  | 10,000             | t10k-images-idx3-ubyte.gz  | [kuzushiji10-test-labels.npz](https://storage.googleapis.com/kuzushiji-mnist/kuzushiji10-test-imgs.npz) (3MB)   |
| Testing labels  | 10,000             | t10k-labels-idx1-ubyte.gz  | [kuzushiji10-test-labels.npz](https://storage.googleapis.com/kuzushiji-mnist/kuzushiji10-test-imgs.npz)  |

##### Which format do I download?
If you're looking for a drop-in replacement for the MNIST or Fashion-MNIST dataset (for tools that currently work with these datasets), download the data in MNIST format.

Otherwise, it's recommended to download in NumPy format, which can be loaded into an array as easy as:  
`arr = np.load(filename)['arr_0']`.
