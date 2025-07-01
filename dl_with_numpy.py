import numpy as np
import math

class deep_learning:
    def __init__(self):
        pass
    
    # to create a model 
    def base_model(self, lmbda=0.0, lr=0.001):
        self.weights = []
        self.biases = []
        self.shapes = []
        self.activation = []
        self.lmbda = lmbda
        self.lr = lr
        self.drop_outs=[]
        self.grads_W = []
        self.grads_b = []

    # helper function for weight type finding (local function)
    def __weight_find(self, weight_int, input_size, output_size):
        if weight_int == "uniform":
            return np.random.uniform(low=-0.5, high=0.5, size=(input_size, output_size))
        elif weight_int == "heuniform":
            return np.random.uniform(low=-math.sqrt(6 / input_size), high=math.sqrt(6 / input_size), size=(input_size, output_size))
        elif weight_int == "Glorotuniform":
            return np.random.uniform(low=-math.sqrt(6 / (input_size + output_size)), high=math.sqrt(6 / (input_size + output_size)), size=(input_size, output_size))
        elif weight_int == "henormal":
            return np.random.normal(loc=0.0, scale=math.sqrt(2 / input_size), size=(input_size, output_size))
        elif weight_int == "normal":
            return np.random.normal(loc=0.0, scale=0.05, size=(input_size, output_size))
        elif weight_int == "Glorotnormal":
            return np.random.normal(loc=0.0, scale=math.sqrt(2 / (input_size + output_size)), size=(input_size, output_size))
        else:
            raise ValueError("Weight initializer not found. Choose from: ['uniform', 'heuniform', 'Glorotuniform', 'henormal', 'normal', 'Glorotnormal']")
            # if not found weights name it raises error, used for debugging 

    # helper function for bias type finding (local function)
    def __bias_finder(self, base_int, output_size):
        if base_int == "ones":
            return np.ones((1, output_size))
        elif base_int == "zeros":
            return np.zeros((1, output_size))
        else:
            raise ValueError("Bias initializer not found. Choose from: ['zeros', 'ones']") # if not found bias argument, used for debugging

    # helper function for activation checking 
    def __activation_finder(self, activation, x):
        if activation == "relu":
            return self.relu(x)
        if activation == "leaky_relu":
            return self.leaky_relu(x)
        if activation == "softmax":
            return self.softmax(x)
        return x

    # helper function for adding layers 
    def add_layers(self, input_size, output_size, act="relu", weight_int="heuniform", base_int="zeros",dropout=0.0):
        W = self.__weight_find(weight_int, input_size, output_size)
        b = self.__bias_finder(base_int, output_size)
        self.weights.append(W)
        self.biases.append(b)
        self.shapes.append((input_size, output_size))
        self.activation.append(act)
        self.drop_outs.append(dropout)

    # activation functions and derivative function used in back propagation
    def relu(self, z):
        return np.maximum(0, z)

    def leaky_relu(self, z):
        return np.maximum(0.1 * z, z)

    def relu_derivative(self, z):
        return (z > 0).astype(float)

    def leaky_relu_derivative(self, z):
        dz = np.ones_like(z)
        dz[z < 0] = 0.1
        return dz

    def softmax(self, Z):
        expZ = np.exp(Z - np.max(Z, axis=1, keepdims=True))
        return expZ / np.sum(expZ, axis=1, keepdims=True)

    # loss functions for classification
    def cross_entropy_loss(self, y_pred, y_true):
        m = y_true.shape[0]
        probs = y_pred[range(m), y_true]
        return -np.mean(np.log(probs + 1e-9))

    # loss function for regression
    def meansqrerror(self, y_pred, y_true):
        mse = np.mean((y_pred - y_true) ** 2)
        reg_term = 0
        for W in self.weights:
            reg_term += np.sum(np.square(W))
        reg_term = (self.lmbda / 2) * reg_term
        return mse + reg_term

    # adam optimizer
    def optimizer_adam(self, beta_1=0.9, beta_2=0.99, epsilon=1e-07):
        if not hasattr(self, 'm_w'):
            self.m_w = [np.zeros_like(w) for w in self.weights]
            self.v_w = [np.zeros_like(w) for w in self.weights]
            self.m_b = [np.zeros_like(b) for b in self.biases]
            self.v_b = [np.zeros_like(b) for b in self.biases]
            self.t = 0

        self.t += 1

        for i in range(len(self.weights)):
            grad_w = self.grads_W[i]    
            grad_b = self.grads_b[i]    

            self.m_w[i] = beta_1 * self.m_w[i] + (1 - beta_1) * grad_w
            self.v_w[i] = beta_2 * self.v_w[i] + (1 - beta_2) * (grad_w ** 2)

            self.m_b[i] = beta_1 * self.m_b[i] + (1 - beta_1) * grad_b
            self.v_b[i] = beta_2 * self.v_b[i] + (1 - beta_2) * (grad_b ** 2)

            m_hat_w = self.m_w[i] / (1 - beta_1 ** self.t)
            v_hat_w = self.v_w[i] / (1 - beta_2 ** self.t)

            m_hat_b = self.m_b[i] / (1 - beta_1 ** self.t)
            v_hat_b = self.v_b[i] / (1 - beta_2 ** self.t)

            self.weights[i] -= self.lr * m_hat_w / (np.sqrt(v_hat_w) + epsilon)
            self.biases[i] -= self.lr * m_hat_b / (np.sqrt(v_hat_b) + epsilon)   # updating weights as per forumla 

    # forward propagation 
    def forward(self, x, training=True):
        n_layers = len(self.weights)
        A = x
        self.Zs = []
        self.As = [x]
        self.masks = []

        for i in range(n_layers):
            Z = np.dot(A, self.weights[i]) + self.biases[i]
            A = self.__activation_finder(self.activation[i], Z)
            
            if training and self.drop_outs[i]>0:
                dropout_mask = (np.random.rand(*A.shape) > self.drop_outs[i]).astype(float)
                A *= dropout_mask
                A /= (1.0 - self.drop_outs[i])
                self.masks.append(dropout_mask)
            else:
                self.masks.append(None)

            self.Zs.append(Z)
            self.As.append(A)
        return A
    
    # back propagation for classification models 
    def backward_classification(self, y_true):
        m = y_true.shape[0]
        grads_W = [0] * len(self.weights)
        grads_b = [0] * len(self.biases)

        y_true_onehot = np.zeros_like(self.As[-1]) # used one hot encodeding for classification
        y_true_onehot[np.arange(m), y_true] = 1

        dZ = (self.As[-1] - y_true_onehot) / m # softmax derivative 

        for i in reversed(range(len(self.weights))):
            A_prev = self.As[i]
            grads_W[i] = A_prev.T @ dZ + self.lmbda * self.weights[i] # l2 regularization
            grads_b[i] = np.sum(dZ, axis=0, keepdims=True)

            if i != 0:
                dA = dZ @ self.weights[i].T  # @ used for matrix multiplication and .T is used for transpose 
                dZ = dA * self.relu_derivative(self.Zs[i - 1]) 

        self.grads_W = grads_W
        self.grads_b = grads_b

    # back propagation for regression models
    def backward_regression(self, X, Y):
        m = X.shape[0]
        grads_W = [0] * len(self.weights)
        grads_b = [0] * len(self.biases)

        dZ = (2 / m) * (self.As[-1] - Y) # used meansquare loss function and its derivative

        for i in reversed(range(len(self.weights))):
            A_prev = self.As[i]
            grads_W[i] = A_prev.T @ dZ + self.lmbda * self.weights[i] # l2 regularization 
            grads_b[i] = np.sum(dZ, axis=0, keepdims=True)

            if i != 0:
                dA = dZ @ self.weights[i].T   # @ used for matrix multiplication and .T is used for transpose 
                dZ = dA * self.relu_derivative(self.Zs[i - 1])

        self.grads_W = grads_W
        self.grads_b = grads_b

    # start train 
    def train(self, X, y, model_type="classification",optimizer=True, epochs=100): # used classification as default model
        for epoch in range(epochs):
            y_pred = self.forward(X)
            if model_type == "classification":
                loss = self.cross_entropy_loss(y_pred, y)
                self.backward_classification(y)
                if optimizer==True:
                    self.optimizer_adam()  # adam optimizer for better converging model
            if model_type == "regression":
                loss = self.meansqrerror(y_pred, y)
                self.backward_regression(X, y)
                if optimizer==True:
                    self.optimizer_adam()   # set adam optimizer as option 

            # for understanding and report for every epochs
            if epoch % 10 == 0 or epoch == epochs - 1:
                if model_type == "classification":
                    acc = self.accuracy(X, y)
                    print(f"Epoch {epoch+1}: Loss = {loss:.4f}, Accuracy = {acc:.4f}")
                if model_type == "regression":
                    r2 = self.r2_score(X, y)
                    print(f"Epoch {epoch+1}: Loss = {loss:.4f}, r2 score = {r2:.4f}")

    # for prediction using this model
    def predict(self, X, model_type="classification"):
        y_pred = self.forward(X,training=False)
        if model_type == "classification":
            return np.argmax(y_pred, axis=1) # onehot encoding 
        return y_pred

    # used for classification models to check accuracy
    def accuracy(self, X, y):
        y_pred = self.predict(X)
        if y.ndim > 1 and y.shape[1] > 1:
            y = np.argmax(y, axis=1)
        else:
            y = y.flatten() # to avoid error making y as 1D shape
        return np.mean(y_pred == y)

    # used in regression models to chack r2 score
    def r2_score(self, X, y_true):
        y_pred = self.predict(X, model_type="regression")
        y_true = np.array(y_true).flatten() # making 1D shape
        y_pred = np.array(y_pred).flatten() # making 1D shape
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        return r2

