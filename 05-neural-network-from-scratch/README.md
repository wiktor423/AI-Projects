# Neural Network from Scratch

A pure NumPy implementation of a Multilayer Perceptron (MLP) trained and evaluated on the Fashion-MNIST dataset. This project does not use PyTorch, TensorFlow, or any other deep learning framework, implementing all layer connections, forward/backward propagation, and activation functions from scratch.

## Core Features

- **Layer Abstraction & Implementations**:
  - **Fully Connected (Dense) Layer**: Custom forward (`X * W + b`) and backward pass (gradient calculation and SGD updates).
  - **Activation Functions**: Explicit gradient computations for `Sigmoid` (with clipping for numerical stability), `Tanh`, `ReLU`, and `LeakyReLU`.
- **Modular Training Pipeline**:
  - **Network Class**: Handles compilation, forward propagation, and fitting (using stochastic gradient descent).
  - **Loss Abstraction**: MSE (Mean Squared Error) and its analytical derivative.
- **Experimental Analysis**:
  - **Activation Function Performance**: Evaluates convergence and test accuracy of Sigmoid, Tanh, ReLU, and LeakyReLU across 3 random seeds. Saves learning curve plots as `activation_losses.png`.
  - **Architecture Comparison**: Examines performance differences between a shallow model (`784 -> 128 -> 10`) and a deeper model (`784 -> 64 -> 32 -> 10`).

## Directory Structure

- [main.py](file:///home/wiktor-klepacz/projects/EITI/EARIN/AI-Labs/05-neural-network-from-scratch/main.py): Complete MLP codebase, experiments setup, and visualization code.
- [activation_losses.png](file:///home/wiktor-klepacz/projects/EITI/EARIN/AI-Labs/05-neural-network-from-scratch/activation_losses.png): Plot showing training loss progression for different activation functions.
- [assignment_5.md](file:///home/wiktor-klepacz/projects/EITI/EARIN/AI-Labs/05-neural-network-from-scratch/assignment_5.md): Lab guidelines and requirements.
- [lab5_cg105_g27_v2_Klepacz_Savy.pdf](file:///home/wiktor-klepacz/projects/EITI/EARIN/AI-Labs/05-neural-network-from-scratch/lab5_cg105_g27_v2_Klepacz_Savy.pdf): Final report documenting experimental methodology and findings.

## Setup and Usage

### Requirements
Ensure you have the required dependencies:
```bash
pip install numpy scikit-learn matplotlib
```
*Note: Scikit-learn is only used to fetch the Fashion-MNIST dataset (`fetch_openml`) and split the data (`train_test_split`).*

### How to Run
Run the MLP training and experiments:
```bash
python main.py
```
This script will automatically:
1. Download the Fashion-MNIST dataset from OpenML.
2. Train multiple MLP networks comparing `Tanh`, `Sigmoid`, `ReLU`, and `LeakyReLU` activations (3 seeds each).
3. Generate and save `activation_losses.png`.
4. Train and report accuracies for Shallow vs. Deep architectures.

## Authors
- **Savy Timothée**
- **Klepacz Wiktor**
- **Group Number**: 27
