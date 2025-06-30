# Project: Artificial Neural Network from Scratch with NumPy


# Overview:

This project implements a fully functional Artificial Neural Network (ANN) from scratch using only Python and NumPy. No machine learning frameworks (e.g., TensorFlow, PyTorch, Scikit-learn) are used. The goal is to provide a transparent understanding of forward propagation, backpropagation, optimization, and evaluation techniques in neural networks.

# Features

Modular ANN class with support for:

1)Multiple layers

2)Custom weight/bias initialization strategies

3)ReLU, Leaky ReLU, and Softmax activation functions
  
4)Cross-entropy loss (classification)
  
5)Mean Squared Error loss (regression)
  
6)L2 Regularization
  
7)Adam optimizer
  
8)Accuracy and R2 score evaluation

9)Dropout

# Architecture

Custom class: deep_learning

Methods include:

base_model(lmbda, lr) – initialize model and adding layers into model 

add_layers(input_size, output_size, act, weight_int, base_int) – add layers

train(X, y, model_type, optimizer, epochs) – train model

forward() and backward_*() – propagation

predict() – make predictions

accuracy(), r2_score() – evaluation metrics

# example usages in try_dl_numpy

# Results

# 1) testing model with sklearn.datasets.load_digits

train Loss = 0.1521, Accuracy = 0.9708

Test Accuracy: 0.9444

No significant overfitting observed. 

# 2) testing model with sklearn.datasets.fetch_california_housing

train Loss = 0.9710, r2 score = 0.7524

Test r2: 0.7331692

 No significant overfitting observed. r2 score can be improve by more layers, Dropouts(coming soon)

 # 3) testing model with keras.datasets.mnist

 train Loss = 0.1248, Accuracy = 0.9641
 
 Test Accuracy: 0.9587

 # Future Improvements

working on adaptive dropout(custom dropout method currently working)

Add learning rate decay

Add batch training (mini-batch SGD)



 



