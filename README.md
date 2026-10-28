# Classification of handwritten digits from MNIST dataset

This repository contains the implementation of a neural network which classifies handwritten digits from the MNIST dataset. 
The network is implemented without any deep learning frameworks such as PyTorch, but relies only on standard numerical packages as NumPy.

The development is inspired by Nielsen's [*Neural Networks and Deep Learning*](http://neuralnetworksanddeeplearning.com) book.

## Goal and motivation

The goal is to develop a working neural network capable of classifying MNIST digits with good accuracy (97% accuracy would be a first goal). The broad motivation is to learn the fundamentals of neural networks, together with common problems and solutions in this type of computational architecture.

## Structure

- `src`: contains the Jupyter notebook where the network is tested and the Python scripts used to develop the neural network class
- `data`: contains the data from MNIST dataset

## The network

- Fully connected feedforward architecture (784 → ... → 10)
- Sigmoid activations, cross-entropy (and quadratic) loss
- Trained with mini-batch stochastic gradient descent and backpropagation

## Requirements

- Python 3.x
- NumPy
- Matplotlib
- Jupyter
- Scikit-learn

## Usage

```bash
git clone https://github.com/nfrancini/NN_MNIST_from_scratch.git
cd NN_MNIST_from_scratch
pip install -r requirements.txt
jupyter notebook src/learn_MNIST.ipynb```