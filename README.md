# Simple Neural Network from Scratch

This is a simple implementation of a neural network from scratch.

## Description

This project is a basic, feedforward neural network built using only **NumPy** and **Pandas**.

The network is designed for classification tasks, such as recognizing handwritten digits from the MNIST dataset (input size 784 for 28x28 images, output size 10 for digits 0-9).


## Quick Start 🚀

Follow these steps to set up the environment and run the model training.

### 1. Setup Environment

Install the project dependencies from `pyproject.toml` and activate the virtual environment.

```bash
uv sync
source .venv/bin/activate
```

### 2. Run Script


```bash
python NN.py
```

## How to Use 🛠️

1.  **Initialize the Network**:
    ```python
    # nn.py
    nn = NeuralNetwork(input_layer_size=784, hidden_layer_sizes=(16, 16), output_layer_size=10)
    ```

2.  **Load Data**:
    ```python
    # Make sure 'train.csv' is in the same directory
    nn.get_training_data('train.csv')
    ```

3.  **Perform a Forward Pass**:
    ```python
    # Get the network's output for the 20th sample in the dataset
    predicted_output, actual_output = nn.forward_propagation(20)
    print("Predicted Output:", predicted_output)
    ```

## TODOs

* Implement the backpropagation algorithm to calculate gradients for weights and biases.
* Complete the `training` method to iterate over the dataset and update weights.
* Implement the `prediction` method for making predictions on new data.