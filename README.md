# Project: Artificial Neural Network from Scratch with NumPy

## Overview
This project implements a fully functional **Artificial Neural Network (ANN)** from scratch using only **Python and NumPy** — no machine learning frameworks (e.g., TensorFlow, PyTorch, Scikit-learn) are used. The aim is to deeply understand the internals of neural networks, including forward and backward propagation, optimization, and evaluation metrics.

---

## Features

Modular `deep_learning` class supports:

1. Multiple hidden layers  
2. Custom weight/bias initialization strategies  
3. Activation functions: ReLU, Leaky ReLU, Softmax  
4. Loss functions:  
   - Cross-Entropy (for classification)  
   - Mean Squared Error (for regression)  
5. L2 Regularization  
6. Adam Optimizer  
7. Evaluation Metrics: Accuracy, R² Score  
8. Dropout Regularization  

---

## Architecture

### Custom Class: `deep_learning`

| Method | Description |
|--------|-------------|
| `base_model(lmbda, lr)` | Initializes the model with regularization and learning rate |
| `add_layers(input_size, output_size, act, weight_int, base_int)` | Adds a layer with activation and initialization |
| `train(X, y, model_type, optimizer, epochs)` | Trains the model using chosen optimizer |
| `forward()` and `backward_*()` | Forward and backward propagation |
| `predict()` | Generates predictions |
| `accuracy()`, `r2_score()` | Evaluates model performance |

_Example usage is shown in: `try_dl_numpy.py`_

---

## Results

### 1. Breast Cancer Dataset (Classification)

- `train_loss`: **0.1178**
- `Train Accuracy`: **98.24%**
- `Test Accuracy`: **96.49%**

No significant overfitting observed.

---

### 2. California Housing Dataset (Regression)

- `train_loss`: **0.9883**
- `Train R²`: **0.7452**
- `Test R²`: **0.7251**

No significant overfitting.  
R² score can improve further with additional layers and dropout (in progress).

---

### 3. MNIST Dataset (Classification)

- `train_loss`: **0.3017**
- `Train Accuracy`: **95.29%**
- `Test Accuracy`: **95.01%**

Stable training and generalization across epochs.

---

## Future Improvements

- Implement custom adaptive dropout strategy  
- Add learning rate decay functionality  
- Add mini-batch training (batch/mini-batch SGD)

---

This project is an excellent educational tool for mastering neural networks at a fundamental level without abstracting away any logic.
