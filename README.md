# Machine Learning Models

Six PyTorch notebooks comparing a multi-layer perceptron (MLP) with a LeNet-5 convolutional network on three image
datasets: MNIST (handwritten digits), Fashion-MNIST (clothing) and CIFAR-10 (small colour photos). They were written for
a machine learning class, together with a [final report](Machine%20Learning%20Final%20Report.docx.pdf) and
[slides](Survey%20Analysis%20of%20Multi-Layer%20Perceptrons%20and%20Common%20Convolutional.pptx) that survey how these
networks developed and how they behave.

Every notebook has an **Open in Colab** link, so you can run it in the browser with no setup. In Colab, choose
Runtime, then Change runtime type, then a GPU to train faster.

## Results

All six runs use the same recipe: SGD at a learning rate of 0.01, batch size 128, 30 epochs, cross-entropy loss and no
data augmentation. The table shows accuracy on the test set after the last epoch.

| Model | Dataset | Layers | Parameters | Test accuracy (epoch 30) | Notebook |
| --- | --- | --- | ---: | ---: | --- |
| MLP | MNIST | 784-128-64-10 | 109,386 | **94.96%** | [Open in Colab](https://colab.research.google.com/github/jrdpaulino/Machine-Learning-Models/blob/main/MNIST.ipynb) |
| MLP | Fashion-MNIST | 784-256-128-10 | 235,146 | **85.05%** | [Open in Colab](https://colab.research.google.com/github/jrdpaulino/Machine-Learning-Models/blob/main/MLP%20Fashion-MNIST.ipynb) |
| MLP | CIFAR-10 | 3072-256-128-10 | 820,874 | **46.41%** | [Open in Colab](https://colab.research.google.com/github/jrdpaulino/Machine-Learning-Models/blob/main/MLP%20CIFAR10.ipynb) |
| LeNet-5 | MNIST | 2 conv + 3 fc | 61,706 | **98.37%** | [Open in Colab](https://colab.research.google.com/github/jrdpaulino/Machine-Learning-Models/blob/main/LeNet%20MNIST.ipynb) |
| LeNet-5 | Fashion-MNIST | 2 conv + 3 fc | 61,706 | **86.58%** | [Open in Colab](https://colab.research.google.com/github/jrdpaulino/Machine-Learning-Models/blob/main/LeNet%20Fashion-MNIST.ipynb) |
| LeNet-5 | CIFAR-10 | 2 conv + 3 fc | 62,006 | **51.66%** | [Open in Colab](https://colab.research.google.com/github/jrdpaulino/Machine-Learning-Models/blob/main/LeNet%20CIFAR10.ipynb) |

**What it shows**

- The convolutional network beats the MLP on every dataset, by about 3.4 points on MNIST, 1.5 on Fashion-MNIST and
  5.3 on CIFAR-10. It does this with fewer parameters than any of the MLPs, because convolutions reuse the same small
  filters across the whole image.
- The gap widens as the images get harder. A flattened MLP loses the 2-D structure that matters most in colour
  photos.
- CIFAR-10 stays near 50% for both models. That is expected for networks this small trained without augmentation
  or normalisation, and well below what modern architectures reach. These notebooks are a baseline, not a
  benchmark attempt.

**Caveats**

- Accuracy is measured on the test set after every epoch, and there is no separate validation split. I report the
  last epoch rather than the best one to avoid picking a flattering number, but the test set was still watched
  during training.
- Each result is a single run with no fixed random seed, so differences of a point or less may be noise.
- `MNIST.ipynb` is the MLP on MNIST. It keeps its original name.

## Running locally

```
pip install torch torchvision matplotlib numpy tensorboard
jupyter notebook
```

Each notebook downloads its dataset automatically into `./data` and logs training and validation loss to TensorBoard.

## What I would try next

Normalising the inputs, adding data augmentation, using a held-out validation set, and comparing against a deeper
network. I would also still like to train a model on audio frequencies for choir.
