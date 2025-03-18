import h5py
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from framework import (

    InputLayer,

    FullyConnectedLayer,

    TanhLayer,

    LogisticSigmoidLayer, 

    LogLoss

)

def RMSE(Y, Yhat):
    """
    returns float
    """
    mse = np.mean((Y - Yhat) ** 2)  
    rmse = np.sqrt((mse / len(Y))+1e-8)  
    return rmse

def SMAPE(Y, Yhat):
    """
    Returns float
    """
    epsilon = 1e-8
    smape = np.mean(np.abs(Y - Yhat) / (np.abs(Y) + np.abs(Yhat) + epsilon))
    return smape

def accuracy(Y, Yhat):
    """
    Compute classification accuracy
    """
    predictions = (Yhat >= 0.5).astype(int)  # Convert probabilities to binary predictions
    return np.mean(predictions == Y)


def train_optimized(X_train, Y_train, X_val, Y_val, learning_rate=0.001, max_epochs=100000, tol=1e-10):
    input_dim = X_train.shape[1]
    output_dim =  1 #Y_train.shape[1]
    #print("input_dim:, ", X_train.shape[1])
    #print("output_dim: ", output_dim)
    L1 = InputLayer(X_train)
    L2 = FullyConnectedLayer(input_dim, input_dim)
    L3 = TanhLayer()
    L4 = FullyConnectedLayer(input_dim, output_dim) # since the tanh layer produces an output of x by 1
    L5 = LogisticSigmoidLayer()
    L6 = LogLoss()
    layers = [L1, L2, L3, L4, L5, L6]
    train_mse, val_mse = [], []
    Y_hat = []
    tic = time.perf_counter()  # Start timer
    prev_mse = float('inf')
    
    for epoch in range(max_epochs):
        # Forward pass for training data
        X = X_train
        for i in range(len(layers) - 1):  # Do not include loss layer in forward pass

            X = layers[i].forward(X)
        
        # Compute training loss
        train_loss = layers[-1].eval(Y_train, X)
        train_mse.append(train_loss)

        # Compute gradient from loss
        grad = layers[-1].gradient(Y_train, X)  

        # Backpropagation
        for i in range(len(layers) - 2, 0, -1):  
            newgrad = layers[i].backward2(grad)
            #print("grad shape: ", grad.shape)
            #print("newgrad shape: ", newgrad.shape)
            if isinstance(layers[i], FullyConnectedLayer):
                # if newgrad.shape[1] != 1:  
                #     newgrad = newgrad.reshape(-1, 1)
                layers[i].updateWeights(grad, learning_rate)  
            grad = newgrad
        
        #Now validation 

        X_val_temp = X_val
        for i in range(len(layers) - 1):  # Do not include loss layer in forward pass
            X_val_temp = layers[i].forward(X_val_temp)
        
        # Compute validation loss
        val_loss = layers[-1].eval(X_val_temp, Y_val)
        val_mse.append(val_loss)

        #print(f"Epoch {epoch}: Train Loss = {train_loss}, Val Loss ={val_loss}")
        if epoch % 10000 == 0:
            train_acc = accuracy(Y_train, X)
            val_acc = accuracy(Y_val, X_val_temp)
            print(f"Epoch {epoch}: Train Loss = {train_loss:.10f}, Val Loss = {val_loss:.10f}, Train Acc = {train_acc:.10f}, Val Acc = {val_acc:.10f}")

        
        # Check for convergence
        if abs(prev_mse - val_loss) < tol:
            print(f"Converged at epoch {epoch}")
            break
        prev_mse = val_loss
    toc = time.perf_counter()  # End timer
    print(f"Training Time: {toc - tic:.2f} seconds")
    Y_hat = grad
    #print("RMSE: ", RMSE(Y_train, Y_hat))
    #print("SMAPE: ", SMAPE(Y_train, Y_hat))
    return train_mse, val_mse


def train(X_train, Y_train, X_val, Y_val, learning_rate=0.001, max_epochs=100000, tol=1e-10):
    input_dim = X_train.shape[1]
    output_dim =  1 #Y_train.shape[1]
    #print("input_dim:, ", X_train.shape[1])
    #print("output_dim: ", output_dim)
    L1 = InputLayer(X_train)
    L2 = FullyConnectedLayer(input_dim, input_dim)
    L3 = TanhLayer()
    L4 = FullyConnectedLayer(input_dim, output_dim) # since the tanh layer produces an output of x by 1
    L5 = LogisticSigmoidLayer()
    L6 = LogLoss()
    layers = [L1, L2, L3, L4, L5, L6]
    train_mse, val_mse = [], []
    Y_hat = []
    tic = time.perf_counter()  # Start timer
    prev_mse = float('inf')
    
    for epoch in range(max_epochs):
        # Forward pass for training data
        X = X_train
        for i in range(len(layers) - 1):  # Do not include loss layer in forward pass

            X = layers[i].forward(X)
        
        # Compute training loss
        train_loss = layers[-1].eval(Y_train, X)
        train_mse.append(train_loss)

        # Compute gradient from loss
        grad = layers[-1].gradient(Y_train, X)  # Use training predictions and labels

        # Backpropagation
        for i in range(len(layers) - 2, 0, -1):  # Go backward through hidden layers
            newgrad = layers[i].backward(grad)
            #print("grad shape: ", grad.shape)
            #print("newgrad shape: ", newgrad.shape)
            if isinstance(layers[i], FullyConnectedLayer):
                # if newgrad.shape[1] != 1:  
                #     newgrad = newgrad.reshape(-1, 1)
                layers[i].updateWeights(grad, learning_rate)  # Update weights with training gradient
            grad = newgrad
        
        
        #Now validation 
        X_val_temp = X_val
        for i in range(len(layers) - 1):  # Do not include loss layer in forward pass
            X_val_temp = layers[i].forward(X_val_temp)
        
        # Compute validation loss
        val_loss = layers[-1].eval(X_val_temp, Y_val)
        val_mse.append(val_loss)

        #print(f"Epoch {epoch}: Train Loss = {train_loss}, Val Loss ={val_loss}")
        if epoch % 10000 == 0:
            train_acc = accuracy(Y_train, X)
            val_acc = accuracy(Y_val, X_val_temp)
            print(f"Epoch {epoch}: Train Loss = {train_loss:.10f}, Val Loss = {val_loss:.10f}, Train Acc = {train_acc:.10f}, Val Acc = {val_acc:.10f}")

        
        # Check for convergence
        if abs(prev_mse - val_loss) < tol:
            print(f"Converged at epoch {epoch}")
            break
        prev_mse = val_loss
    toc = time.perf_counter()  # End timer
    print(f"Training Time: {toc - tic:.2f} seconds")
    Y_hat = grad
    #print("RMSE: ", RMSE(Y_train, Y_hat))
    #print("SMAPE: ", SMAPE(Y_train, Y_hat))
    return train_mse, val_mse


def plot_mse(train_mse, val_mse):
    plt.plot(train_mse, label="Training Log Loss")
    plt.plot(val_mse, label="Validation Log Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Log Loss")
    plt.title("Log Loss vs Epoch")
    plt.legend()
    plt.show()
    
def plot_mse_optimized(train_mse, val_mse):
    plt.plot(train_mse, label="Training Log Loss")
    plt.plot(val_mse, label="Validation Log Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Log Loss")
    plt.title("Log Loss vs Epoch optimized")
    plt.legend()
    plt.show()
    


if __name__ == "__main__":

    with h5py.File('flattened_data.h5', 'r') as f:
        loaded_features = f['features'][:]
        loaded_labels = f['labels'][:]

    # Shuffle the dataset
    indices = np.random.permutation(len(loaded_features))
    loaded_features = loaded_features[indices]
    loaded_labels = loaded_labels[indices]

    # Split the dataset into 1/3 for training and 2/3 for validation
    split_index = len(loaded_features) // 3
    X_train, X_val = loaded_features[split_index:], loaded_features[:split_index]
    Y_train, Y_val = loaded_labels[split_index:], loaded_labels[:split_index]

    print("Training data shape:", X_train.shape)
    print("Validation data shape:", X_val.shape)

    # Training using the regular and optimized methods
    print("\n\nRegular Training\n_____________________________________")
    train_mse, val_mse = train(X_train, Y_train, X_val, Y_val)

    print("\n\nOptimized Training for Speed\n_____________________________________")
    train_mse2, val_mse2 = train_optimized(X_train, Y_train, X_val, Y_val)

    plot_mse(train_mse, val_mse)
    plot_mse_optimized(train_mse2, val_mse2)