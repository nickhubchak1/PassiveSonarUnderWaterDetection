#-----------------------------------------------------
# Deep Learning Final Project 2025
# Under Water Passive Acoustic Source Localization
# Author: Nick Hubchak, Priontu Chowdhury, Colin Woods
# All Rights Reserved 2025-2030
#----------------------------------------------------
import h5py
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from prio_framework import (

    InputLayer,

    FullyConnectedLayer,

    TanhLayer,

    LinearLayer, 

    SquaredError,
    
    ReLULayer

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


# def batch_generator(X, Y, batch_size=64):
#     """
#     Generator function to yield batches of data.
#     """
#     num_samples = X.shape[0]
#     while True:
#         for start in range(0, num_samples, batch_size):
#             end = min(start + batch_size, num_samples)
#             yield X[start:end], Y[start:end]

def train_with_batching(X_train, Y_train, X_val, Y_val, learning_rate=0.01, max_epochs=100000, tol=1e-10, batch_size=64, patience = 20):
    input_dim = X_train.shape[1]
    output_dim =  1
    Y_train = Y_train.reshape(-1, 1) #force to be (3000, 1)

    best_val_loss = float("inf")
    patience_counter = 0

    L1 = InputLayer(X_train)
    L2 = FullyConnectedLayer(input_dim, 128)
    L3 = ReLULayer()
    L4 = FullyConnectedLayer(128, 128)
    L5 = ReLULayer()
    L6 = FullyConnectedLayer(128, 128)
    L7 = ReLULayer()
    L8 = FullyConnectedLayer(128, 128)
    L9 = ReLULayer()
    L10 = FullyConnectedLayer(128, 128)
    L11 = ReLULayer()
    L12 = FullyConnectedLayer(128, 128)
    L13 = ReLULayer()
    L14 = FullyConnectedLayer(128, 128)
    L15 = ReLULayer()
    L16 = FullyConnectedLayer(128, 1)
    L17 = LinearLayer()
    L18 = SquaredError()
    
    layers = [ 
              L1 , 
              L2 , L3 , 
              L4 , L5 , 
              L6 , L7 , 
              L8 , L9 , 
              L10, L11, 
              L12, L13, 
              L14, L15, 
              L16, L17, 
              L18 ]
    
    train_mse, val_mse = [], []
    Y_hat = []
    tic = time.perf_counter()
    prev_mse = float('inf')
    
    # Initialize the batch generator
    #train_gen = batch_generator(X_train, Y_train, batch_size=batch_size)
    
    for epoch in range(max_epochs):
        # Training loop with mini-batches
        #X_batch, Y_batch = next(train_gen)
        
        # Forward pass for training data
        X = X_train
        for layer in layers[:-1]:
            X = layer.forward(X)
        #print("X shape going into squared error: ", X.shape)
        #print("Y_train shape: ", Y_train.shape)
        train_loss = layers[-1].eval(Y_train, X)
        train_mse.append(train_loss)

        grad = layers[-1].gradient(Y_train, X)
        #print("grad shape from Squared Error forward training: ", grad.shape)
        # Backpropagation
        for i in range(len(layers) - 2, 0, -1):
            newgrad = layers[i].backward2(grad)
            if isinstance(layers[i], FullyConnectedLayer):
                layers[i].updateWeights(grad, learning_rate)
            grad = newgrad
        
        # Validation
        X_val_temp = X_val
        for layer in layers[:-1]:
            X_val_temp = layer.forward(X_val_temp)
        
        val_loss = layers[-1].eval(X_val_temp, Y_val)
        val_mse.append(val_loss)

        if epoch % 100 == 0:
            train_smape = SMAPE(Y_train, X)
            train_rmse = RMSE(Y_train, X)
            val_smape = SMAPE(Y_val, X_val_temp)
            val_rmse = RMSE(Y_val, X_val_temp)
            print(f"Epoch {epoch}: Train Loss = {train_loss:.10f}, Val Loss = {val_loss:.10f}, Train SMAPE = {train_smape:.10f}, Train RMSE = {train_rmse:.10f}, Val SMAPE = {val_smape:.10f}, Val RMSE = {val_rmse:.10f}")

        if abs(prev_mse - val_loss) < tol:
            print(f"Converged at epoch {epoch}")
            break
        prev_mse = val_loss
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print("Convergence detected. Early stopping initiated.")
            break
        
        # learning_rate = learning_rate * 0.99999

    toc = time.perf_counter()
    print(f"Training Time: {toc - tic:.2f} seconds")
    return train_mse, val_mse

def plot_mse(train_mse, val_mse):
    plt.plot(train_mse, label="Train Squared Error")
    plt.plot(val_mse, label="Validation Squared Error")
    plt.xlabel("Epoch")
    plt.ylabel("Squared Error")
    plt.title("Squared Error vs Epoch")
    plt.legend()
    plt.savefig("Graphs/Deep_MLP.png")
    plt.show()
    
if __name__ == "__main__":

    with h5py.File('Training_and_Validation.h5', 'r') as f:
        X_train_reduced = f['xTrainReduced'][:]
        Y_train = f['yTrain'][:]
        X_val_reduced = f['xValidationReduced'][:]
        Y_val = f['yValidation'][:]

    print("Training data shape after reduction:", X_train_reduced.shape)
    print("Validation data shape after reduction:", X_val_reduced.shape)

    print("\n\nRunning Deep MLP with forward and back prop....\n_____________________________________")
    train_mse, val_mse = train_with_batching(X_train_reduced, Y_train, X_val_reduced, Y_val)
    plot_mse(train_mse, val_mse)
