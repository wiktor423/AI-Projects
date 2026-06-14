# Assignment 5 – Multilayer Perceptron with Backpropagation (5 points)

## Common Instructions

The goal of this assignment is to implement a **Multilayer Perceptron (MLP)** from scratch using NumPy,
train it on an image classification dataset, and analyse the effect of the component assigned to your group.

Use the provided `template.py` and implement all required methods. The implementation must work
for arbitrary layer configurations — do not hardcode the number of layers or neurons.

---

## Skeleton Overview (all groups)

### `FullyConnected` class

| Method | Description |
|--------|-------------|
| `__init__(input_size, output_size)` | Initialise weights `(input_size × output_size)` and bias `(1 × output_size)` randomly. |
| `forward(x)` | Compute `x @ W + b`. Cache the input for `backward`. |
| `backward(output_error_derivative)` | Compute gradients for `W` and `b`, update parameters, return gradient for the previous layer. |

### `Loss` class

| Method | Description |
|--------|-------------|
| `loss(y_pred, y_true)` | Return the scalar loss value by calling `self.loss_function`. |
| `loss_derivative(y_pred, y_true)` | Return `∂L/∂y_pred` by calling `self.loss_function_derivative`. |

### `Network` class

| Method | Description |
|--------|-------------|
| `compile(loss)` | Store the `Loss` object. Set `learning_rate` on all layers. |
| `__call__(x)` | Run forward propagation through all layers in sequence. |
| `fit(...)` | Train: forward → loss → backward through all layers in reverse. Return per-epoch loss list. |

---

## Scoring (5 points)

| Component | Points  |
|-----------|---------|
| **Code** — correct implementation of all required methods | 1.0     |
| **Report** — plots, analysis, and written conclusions | 2.0     |
| **Discussion** — oral discussion with the instructor | 2.0     |
| **Total** | **5.0** |

> **Note:** If during the oral discussion it becomes apparent that the student does not understand
> their own solution, the instructor reserves the right to award **0 points** for the entire assignment.

---

# Group A — Activation Function Comparison on MNIST

## Task

Implement **all four** activation functions listed below and compare their effect on MNIST classification.
Use the same architecture, optimiser (SGD), loss (MSE), and hyperparameters for every activation.

## Activation functions to implement

| Class | Formula | Derivative |
|-------|---------|-----------|
| `Tanh` | $\tanh(x)$ | $1 - \tanh^2(x)$ |
| `Sigmoid` | $\frac{1}{1+e^{-x}}$ | $\sigma(x)(1-\sigma(x))$ |
| `ReLU` | $\max(0, x)$ | $\mathbf{1}[x>0]$ |
| `LeakyReLU(α=0.01)` | $x$ if $x>0$ else $\alpha x$ | $1$ if $x>0$ else $\alpha$ |

## Experiments

1. Train each activation with the same architecture and report mean ± std test accuracy over **3 seeds**.
2. Plot training loss curves for all 5 activations on a single figure.
3. **Architecture comparison**: for the best-performing activation, also test a second architecture
   (e.g. different depth or width).
4. **Conclusions**: discuss vanishing gradient risk, convergence speed.

---

# Group B — Activation Function Comparison on FashionMNIST

## Task

Repeat the same activation function comparison as Group A, but on **FashionMNIST** instead of MNIST.

Load with:
```python
from sklearn.datasets import fetch_openml
dataset = fetch_openml('Fashion-MNIST', version=1, as_frame=False)
```

FashionMNIST has the same shape as MNIST (28×28 greyscale, 10 classes, 70 000 samples)
but is significantly harder.

## Activation functions to implement

Same as Group A — implement all four: Tanh, Sigmoid, ReLU, LeakyReLU (α=0.01).

## Experiments

1. Same structure as Group A (per-activation mean ± std over 3 seeds, combined loss curve).
2. **Architecture comparison**: test at least two architectures with the best-performing activation.
3. **Conclusions**: relate performance differences to the increased complexity of FashionMNIST.

---

# Group C — Optimiser Comparison

## Task

Implement **four gradient-based optimisers** as classes and compare their convergence on MNIST.
Use a fixed architecture, activation (ReLU), and loss (MSE).

## Optimisers to implement

| Class | Update rule |
|-------|------------|
| `SGD` | $W \leftarrow W - \eta \nabla W$ |
| `Momentum(β=0.9)` | $v \leftarrow \beta v + (1-\beta)\nabla W$; $W \leftarrow W - \eta v$ |
| `RMSProp(β=0.9)` | $s \leftarrow \beta s + (1-\beta)\nabla W^2$; $W \leftarrow W - \eta \nabla W / \sqrt{s+\epsilon}$ |
| `Adam(β₁=0.9, β₂=0.999)` | Combined first and second moment estimates with bias correction |

## Design

Each optimiser is a class with a `step(layer)` method that reads `layer.weights_grad` and
`layer.bias_grad` and updates `layer.weights` and `layer.bias`.
`FullyConnected.backward` should **store** the gradients as attributes instead of applying them directly —
the optimiser handles the update.

```python
# In Network.fit, after the backward pass through all layers:
for layer in self.layers:
    if isinstance(layer, FullyConnected):
        self.optimizer.step(layer)
```

## Experiments

1. Compare all 4 optimisers with the **same learning rate** and report convergence curves.
2. For each optimiser, find a good learning rate via a small sweep and report best results over 3 seeds.
3. **Architecture comparison**: test at least two architectures for the best setups.
4. **Conclusions**: discuss convergence speed, stability, sensitivity to learning rate.

---

# Group D — Loss Function Comparison

## Task

Implement **three loss functions** and compare their effect on MNIST classification.
Use a fixed architecture, activation (ReLU), and optimiser (SGD).

## Loss functions to implement

| Function | Formula | Derivative w.r.t. `y_pred` |
|----------|---------|--------------------------|
| MSE | $\frac{1}{n}\sum(y_p - y_t)^2$ | $\frac{2}{n}(y_p - y_t)$ |
| MAE | $\frac{1}{n}\sum\|y_p - y_t\|$ | $\frac{1}{n}\text{sign}(y_p - y_t)$ |
| Cross-Entropy | $-\frac{1}{n}\sum y_t \log(\text{softmax}(y_p))$ | $\frac{1}{n}(\text{softmax}(y_p) - y_t)$ |

For Cross-Entropy, the softmax+CE derivative is combined into a single expression for numerical
stability. The output layer is **linear** (no activation) when using CE — softmax is applied inside the loss.

## Experiments

1. Train with each loss and compare accuracy and training curves over **3 seeds**.
2. **Architecture comparison**: test at least two architectures per loss.
3. Investigate training stability: does MAE converge slower?
4. **Conclusions**: discuss suitability of each loss for multi-class classification.

---

# Group E — Regularisation Study

## Task

Implement **two regularisation techniques** — L2 weight decay and Dropout — and study their
effect on overfitting on MNIST.

## Components to implement

### `FullyConnected` with L2 weight decay

Add a `lambda_` parameter. During `backward`, add `lambda_ * W` to the weight gradient:

```
dL/dW_regularised = dL/dW + λ * W
```

### `Dropout` layer

```
forward  (training): mask = Bernoulli(1-rate) / (1-rate)  [inverted dropout]
                     return x * mask
forward  (inference): return x  [no scaling needed]
backward:            return delta * mask
```

The `Network` must propagate a `training` flag to all `Dropout` layers (set `training=False`
during evaluation).

## Experiments

1. Train **three configurations** with the same architecture and hyperparameters:
   - No regularisation (baseline)
   - L2 weight decay (test λ ∈ {1e-4, 1e-3, 1e-2})
   - Dropout (test rate ∈ {0.2, 0.4, 0.5})
2. For each configuration, report train **and** test accuracy per epoch (plot both on the same figure
   to visualise the generalisation gap).
3. **Architecture comparison**: test at least two architectures per regularisation method.
4. **Conclusions**: which technique reduces overfitting most? What are the trade-offs?

---

## Technical Requirements

- Submit a `.py` file and a report in `.pdf` format.
- Code must follow **PEP 8** and include comments on non-obvious logic.
- All experiments must be reproducible from the submitted code (set seeds explicitly).
- Do **not** use ML frameworks for the MLP (PyTorch, TensorFlow, scikit-learn models, etc.).
  Only `numpy` and `matplotlib` are allowed for the network implementation.

## Submission Guidelines

Submit all files to your private GitHub repository no later than:

- **02.05.2026 until 23:59** for Monday group
- **04.05.2026 until 23:59** for Wednesday group
- **06.05.2026 until 23:59** for Friday group

File naming convention:
```
lab5_cg{class_group_number}_g{group_number}_v{variant_letter}_{surname1}_{surname2}.(py|pdf|zip)
```
E.g. `lab5_cg101_g1_vA_Smith_Jankowski.py`