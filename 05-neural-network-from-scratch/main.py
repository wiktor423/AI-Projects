from abc import abstractmethod, ABC
from typing import List
from sklearn.datasets import fetch_openml
import numpy as np


dataset = fetch_openml('Fashion-MNIST', version=1, as_frame=False)

class Layer(ABC):
    """Basic building block of the Neural Network"""

    def __init__(self) -> None:
        self._learning_rate = 0.01

    @abstractmethod
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward propagation of x through layer"""
        pass

    @abstractmethod
    def backward(self, output_error_derivative) -> np.ndarray:
        """Backward propagation of output_error_derivative through layer"""
        pass

    @property
    def learning_rate(self):
        return self._learning_rate

    @learning_rate.setter
    def learning_rate(self, learning_rate):
        assert learning_rate < 1, f"Given learning_rate={learning_rate} is larger than 1"
        assert learning_rate > 0, f"Given learning_rate={learning_rate} is smaller than 0"
        self._learning_rate = learning_rate


class FullyConnected(Layer):
    def __init__(self, input_size: int, output_size: int) -> None:
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        
        # Initialize weights and bias
        self.weights = np.random.randn(input_size, output_size) * 0.1
        self.bias = np.zeros((1, output_size))
        self.input = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.input = x
        return np.dot(x, self.weights) + self.bias

    def backward(self, output_error_derivative) -> np.ndarray:
        input_error_derivative = np.dot(output_error_derivative, self.weights.T)
        weights_error_derivative = np.dot(self.input.T, output_error_derivative)
        bias_error_derivative = np.sum(output_error_derivative, axis=0, keepdims=True)

        self.weights -= self.learning_rate * weights_error_derivative
        self.bias -= self.learning_rate * bias_error_derivative
        
        return input_error_derivative


class Tanh(Layer):
    def forward(self, x: np.ndarray) -> np.ndarray:
        self.output = np.tanh(x)
        return self.output

    def backward(self, output_error_derivative) -> np.ndarray:
        return output_error_derivative * (1 - self.output ** 2)

class Sigmoid(Layer):
    def forward(self, x: np.ndarray) -> np.ndarray:
        #np.clip to prevent overflow
        x = np.clip(x, -500, 500)
        self.output = 1 / (1 + np.exp(-x))
        return self.output

    def backward(self, output_error_derivative) -> np.ndarray:
        return output_error_derivative * (self.output * (1 - self.output))

class ReLU(Layer):
    def forward(self, x: np.ndarray) -> np.ndarray:
        self.input = x
        return np.maximum(0, x)

    def backward(self, output_error_derivative) -> np.ndarray:
        return output_error_derivative * (self.input > 0)

class LeakyReLU(Layer):
    def __init__(self, alpha: float = 0.01) -> None:
        super().__init__()
        self.alpha = alpha

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.input = x
        return np.where(x > 0, x, self.alpha * x)

    def backward(self, output_error_derivative) -> np.ndarray:
        return output_error_derivative * np.where(self.input > 0, 1, self.alpha)



class Loss:
    def __init__(self, loss_function: callable, loss_function_derivative: callable) -> None:
        self.loss_function = loss_function
        self.loss_function_derivative = loss_function_derivative

    def loss(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        return self.loss_function(y_pred, y_true)

    def loss_derivative(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        return self.loss_function_derivative(y_pred, y_true)


class Network:
    def __init__(self, layers: List[Layer], learning_rate: float) -> None:
        self.layers = layers
        self.learning_rate = learning_rate

    def compile(self, loss: Loss) -> None:
        self.loss = loss
        for layer in self.layers:
            layer.learning_rate = self.learning_rate

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Forward propagation of x through all layers"""
        output = x 
        for layer in self.layers: 
            output = layer.forward(output)
        return output

    def fit(self,
            x_train: np.ndarray,
            y_train: np.ndarray,
            epochs: int,
            learning_rate: float,
            verbose: int = 0) -> List[float]:
            
        self.learning_rate = learning_rate
        for layer in self.layers:
            layer.learning_rate = learning_rate

        history = []
        n_samples = len(x_train)

        for epoch in range(epochs):
            err = 0
            for i in range(n_samples):
                x = x_train[i:i+1]
                y = y_train[i:i+1]
                
                # Forward
                y_pred = self(x)
                
                # Loss computation
                err += self.loss.loss(y_pred, y)
                
                # Backward
                error = self.loss.loss_derivative(y_pred, y)
                for layer in reversed(self.layers):
                    error = layer.backward(error)
                    
            err /= n_samples
            history.append(float(err))
            if verbose:
                print(f"Epoch {epoch + 1}/{epochs} - loss: {err:.6f}")
                
        return history


import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


#Loss function
def mse(y_pred, y_true):
    return np.mean((y_pred - y_true) ** 2)

def mse_derivative(y_pred, y_true):
    return 2 * (y_pred - y_true) / y_true.size


#Evaluation helper
def get_accuracy(network, X, y_onehot):
    correct = 0
    for i in range(len(X)):
        pred = network(X[i:i+1])
        if np.argmax(pred) == np.argmax(y_onehot[i:i+1]):
            correct += 1
    return correct / len(X)

#Data loading
X = dataset.data / 255.0
y = dataset.target.astype(int)
y_onehot = np.eye(10)[y]
X_train, X_test, y_train, y_test = train_test_split(X, y_onehot, test_size=0.2, random_state=42)


def run_experiments():
    seeds = [42, 123, 999]
    activations = {
        'Tanh': Tanh,
        'Sigmoid': Sigmoid,
        'ReLU': ReLU,
        'LeakyReLU': LeakyReLU
    }
    
    epochs = 10
    learning_rate = 0.05
    results_acc = {}
    loss_curves = {}

    print("--- Running Experiment 1 & 2: Activation Function Comparison ---")
    mse_loss = Loss(mse, mse_derivative)

    for name, ActClass in activations.items():
        print(f"\nTraining with {name} activation...")
        seed_accuracies = []
        avg_losses = np.zeros(epochs)
        
        for seed in seeds:
            np.random.seed(seed)
            net = Network([
                FullyConnected(784, 64),
                ActClass(),
                FullyConnected(64, 10)
            ], learning_rate=learning_rate)
            
            net.compile(mse_loss)
            
            history = net.fit(X_train, y_train, epochs=epochs, learning_rate=learning_rate, verbose=0)
            
            acc = get_accuracy(net, X_test, y_test)
            seed_accuracies.append(acc)
            avg_losses += np.array(history)
            
        avg_losses /= len(seeds)
        mean_acc = np.mean(seed_accuracies)
        std_acc = np.std(seed_accuracies)
        
        results_acc[name] = (mean_acc, std_acc)
        loss_curves[name] = avg_losses
        print(f"{name} - Test Accuracy (over 3 seeds): {mean_acc:.4f} ± {std_acc:.4f}")

    # Plotting Loss Curves
    plt.figure(figsize=(10, 6))
    for name, losses in loss_curves.items():
        plt.plot(range(1, epochs + 1), losses, label=name)
    plt.title("Training Loss Curver per Activation (Fashion-MNIST)")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.grid()
    plt.savefig("activation_losses.png")
    plt.show()

    print("\n--- Running Experiment 3: Architecture Comparison")
    archs = [
        {"name": "Shallow (128)", "layers": [FullyConnected(784, 128), ReLU(), FullyConnected(128, 10)]},
        {"name": "Deep (64->32)", "layers": [FullyConnected(784, 64), ReLU(), FullyConnected(64, 32), ReLU(), FullyConnected(32, 10)]}
    ]

    for arch in archs:
        np.random.seed(42)
        net = Network(arch["layers"], learning_rate=learning_rate)
        net.compile(mse_loss)
        print(f"Training {arch['name']}...")
        net.fit(X_train[:5000], y_train[:5000], epochs=epochs, learning_rate=learning_rate, verbose=1)
        acc = get_accuracy(net, X_test[:1000], y_test[:1000])
        print(f"{arch['name']} - Test Accuracy: {acc:.4f}")

if __name__ == "__main__":
    run_experiments()
