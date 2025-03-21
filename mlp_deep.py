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
import copy
import os

from deep_framework import (

    InputLayer,

    FullyConnectedLayer,

    TanhLayer,

    LinearLayer, 

    SquaredError,
    
    ReLULayer

)

def RMSE(Y, Yhat):
    mse = np.mean((Y - Yhat) ** 2)  
    rmse = np.sqrt((mse / len(Y))+1e-8)  
    return rmse

def SMAPE(Y, Yhat):
    epsilon = 1e-8
    smape = np.mean(np.abs(Y - Yhat) / (np.abs(Y) + np.abs(Yhat) + epsilon))
    return smape

def accuracy(Y, Yhat):
    predictions = ((Y - Yhat) <= 2).astype(int)  
    return np.mean(predictions)


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
    # X_mean = 4.8561368037430336e-11
    # X_var = 2.2572550317402957e-10
    
    # X_train = (X_train - X_mean)/X_var
    # X_val = (X_val - X_mean) / X_var
    
    input_dim = X_train.shape[1]
    output_dim =  1
    Y_train = Y_train.reshape(-1, 1) #force to be (3000, 1)
    
    # print("Y_train: ", Y_train)

    best_val_loss = float("inf")
    patience_counter = 0

    L1 = InputLayer(X_train)
    L2 = FullyConnectedLayer(input_dim, input_dim)
    L3 = TanhLayer()
    L4 = FullyConnectedLayer(input_dim, 256)
    L5 = TanhLayer()
    L6 = FullyConnectedLayer(256, 256)
    L7 = TanhLayer()
    L8 = FullyConnectedLayer(256, 256)
    L9 = TanhLayer()
    L10 = FullyConnectedLayer(256, 256)
    L11 = TanhLayer()
    L12 = FullyConnectedLayer(256, 256)
    L13 = TanhLayer()
    L14 = FullyConnectedLayer(256, 256)
    L15 = TanhLayer()
    L16 = FullyConnectedLayer(256, 1)
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
    train_acc, val_acc = [], []
    Y_hat = []
    tic = time.perf_counter()
    prev_mse = float('inf')
    
    # Initialize the batch generator
    #train_gen = batch_generator(X_train, Y_train, batch_size=batch_size)
    
    for epoch in range(max_epochs):
        # Training loop with mini-batches
        #X_batch, Y_batch = next(train_gen)
        
        # Forward pass for training data
        X = copy.deepcopy(X_train)
        for layer in layers[:-1]:
            X = layer.forward(X)
        #print("X shape going into squared error: ", X.shape)
        #print("Y_train shape: ", Y_train.shape)
        train_predictions = X
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
        X_val_temp = copy.deepcopy(X_val)
        for layer in layers[:-1]:
            X_val_temp = layer.forward(X_val_temp)
        
        val_predictions = X_val_temp
        val_loss = layers[-1].eval(X_val_temp, Y_val)
        val_mse.append(val_loss)

        if epoch % 10 == 0:
            train_smape = SMAPE(Y_train, X)
            train_rmse = RMSE(Y_train, X)
            val_smape = SMAPE(Y_val, X_val_temp)
            val_rmse = RMSE(Y_val, X_val_temp)
            train_accuracy = accuracy(Y_train, X)
            val_accuracy = accuracy(Y_val, X_val_temp)
            train_acc.append(train_accuracy)
            val_acc.append(val_accuracy)
            print(f"Epoch {epoch}: Train Loss = {train_loss:.10f}, Val Loss = {val_loss:.10f}, Train Accuracy: {train_accuracy:.4f}, Val Accuracy: {val_accuracy:.4f}, Train SMAPE = {train_smape:.10f}, Train RMSE = {train_rmse:.10f}, Val SMAPE = {val_smape:.10f}, Val RMSE = {val_rmse:.10f}")

        # if abs(prev_mse - val_loss) < tol:
        #     print(f"Converged at epoch {epoch}")
        #     break
        # prev_mse = val_loss
        
        if val_loss <= best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print("Convergence detected. Early stopping initiated.")
            break
        
        # learning_rate = learning_rate * 0.99999

    X_pred = copy.deepcopy(X_val)
    for layer in layers[:-1]:  # Exclude SquaredError
        X_pred = layer.forward(X_pred)
    # print(X_pred)
    # Y_pred = X_pred.reshape(-1, 1)
    
    # print("X_val_temp: \n", X_val_temp)

    toc = time.perf_counter()
    print(f"Training Time: {toc - tic:.2f} seconds")
    return train_mse, val_mse, train_acc, val_acc, train_predictions, val_predictions

def plot_mse(train_mse, val_mse):
    
    plt.plot(train_mse, label="Train Squared Error")
    plt.plot(val_mse, label="Validation Squared Error")
    plt.xlabel("Epoch")
    plt.ylabel("Squared Error")
    plt.title("Squared Error vs Epoch")
    plt.legend()
    plt.savefig("Graphs/DeepNet_MSE.png")
    # plt.show()


# def plot_predictions(Y_true, Y_pred, title="Predictions vs Ground Truth"):
#     plt.figure(figsize=(8,6))
#     plt.scatter(Y_true, Y_pred, alpha=0.5, label="Predictions")
#     plt.plot([Y_true.min(), Y_true.max()], [Y_true.min(), Y_true.max()], 'r--', label="Ideal")
#     plt.xlabel("Ground Truth")
#     plt.ylabel("Predictions")
#     plt.title(title)
#     plt.legend()
#     plt.grid()
#     plt.savefig("Graphs/DeepNet_Predictions_vs_GroundTruth.png")
#     # plt.show()

def plot_accuracy_curve(train_acc, val_acc):
    plt.figure(figsize=(8,6))
    plt.plot(train_acc, label="Train Accuracy")
    plt.plot(val_acc, label="Val Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs Epochs")
    plt.legend()
    plt.grid()
    plt.savefig("Graphs/DeepNet_Accuracy_Curve.png")
    # plt.show()
    
def plot_ground_truth_vs_prediction(Y_validation, prediction, title="Ground Truth vs Prediction for Validation"):
    plt.figure(figsize=(8, 6))
    Y_validation = Y_validation.reshape(-1, 1) #force to be (3000, 1)

    m = 1  
    x = np.arange(len(prediction))  
    #pred_adjusted = m * prediction.T + x
    #pred_adjusted = pred_adjusted.T
    print("Yvalidation shape: ", Y_validation.shape)
    print("prediction: ", prediction.shape)
    plt.scatter(Y_validation, prediction, alpha=0.5, label="Predictions", color='blue')
    plt.plot([min(Y_validation), max(Y_validation)], [min(Y_validation), max(Y_validation)], 
             linestyle='dashed', color='red', label="Ideal Prediction (y = x)")
    
    plt.xlabel("Ground Truth (y_validation)")
    plt.ylabel("Predicted (X_fused)")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig("Graphs/DeepNet_groundtruth_and_prediction.png")
    #plt.show()

if __name__ == "__main__":

    with h5py.File('Training_and_Validation.h5', 'r') as f:
        X_train_reduced = f['xTrainReduced'][:]
        Y_train = f['yTrain'][:]
        X_val_reduced = f['xValidationReduced'][:]
        Y_val = f['yValidation'][:]

    print("Training data shape after reduction:", X_train_reduced.shape)
    print("Validation data shape after reduction:", X_val_reduced.shape)

    print("\n\nRunning Deep MLP with forward and back prop....\n_____________________________________")
    train_mse, val_mse, train_acc, val_acc, train_predictions, val_predictions = train_with_batching(X_train_reduced, Y_train, X_val_reduced, Y_val)
    plot_mse(train_mse, val_mse)
    plot_accuracy_curve(train_acc, val_acc)
    plot_ground_truth_vs_prediction(Y_val, val_predictions )

#  # Generate Predictions for the final model
#     X_pred = X_val_reduced
#     for layer in layers[:-1]:  # Exclude SquaredError
#         X_pred = layer.forward(X_pred)

    # Plot results
    # plot_mse(train_mse, val_mse)
    # print("Y_val: \n", Y_val)
    # plot_predictions(Y_val, X_pred)
    
    
    # results_df = pd.DataFrame({"Ground Truth": Y_val, "Predictions": np.squeeze(X_pred.reshape(-1, 1))})
    # csv_filename = os.path.join("Graphs", "Ground_Truth_vs_Predictions.csv")
    # results_df.to_csv(csv_filename, index=False)
    
    # plot_ground_truth_vs_prediction(Y_val, val_predictions)
    # plt.figure()
    # plt.plot(val_predictions)
    
    plt.show()