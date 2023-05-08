# Multi-Layer Perceptron from Scratch and PyTorch

This code implements a multilayer perceptron (MLP) for binary classification in PyTorch. The model is trained on the Berthier dataset, which consists of two classes of two-dimensional points. The first implementation of the MLP is built from scratch, while the second implementation uses PyTorch's built-in `nn.Module`.

The `berthier_data()` function generates the Berthier dataset. The `visualize_samples()` function visualizes the generated dataset with matplotlib.

### MLP from Scratch

The `MultiLayerPerceptronFromScratch` class implements the MLP from scratch. The class constructor takes as input the learning rate, number of epochs, the x1 and x2 coordinates of the dataset, the x1 and x2 labels, the number of inputs and outputs, and the batch size. The class has methods for forward and backward propagation, training, plotting, and testing. 

The `forward()` method calculates the forward propagation of the network, and the `backward()` method updates the weights using the backpropagation algorithm. The `train()` method trains the model on the dataset for the given number of epochs and returns the loss of each epoch. The `plot()` method plots the loss curve. The `test()` method returns the output of the model given a new input.

The `visualize_classification()` function takes an instance of the `MultiLayerPerceptronFromScratch` class, the dataset, and the labels, and visualizes the model's classification with matplotlib.

### MLP with PyTorch

The `MLP` class implements the MLP using PyTorch's built-in `nn.Module`. The class constructor takes as input the number of inputs and outputs. The class has a `forward()` method that calculates the forward propagation of the network.

The `train_model()` function trains the model on the dataset using the PyTorch optimizer and the `nn.MSELoss` loss function. The `train_model2()` function trains the model on the dataset using the PyTorch optimizer and the `nn.BCELoss` loss function. Both functions return the trained model.

The `visualize_classification()` function takes an instance of the `MLP` class, the dataset, and the labels, and visualizes the model's classification with matplotlib.
