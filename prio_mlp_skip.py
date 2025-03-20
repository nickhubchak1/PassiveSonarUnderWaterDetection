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
    ReLULayer,
    AddLayer  
)

def RMSE(Y, Yhat):
    mse = np.mean((Y - Yhat) ** 2)  
    rmse = np.sqrt((mse / len(Y)) + 1e-8)  
    return rmse

def SMAPE(Y, Yhat):
    epsilon = 1e-8
    smape = np.mean(np.abs(Y - Yhat) / (np.abs(Y) + np.abs(Yhat) + epsilon))
    return smape

def accuracy(Y, Yhat):
    predictions = ((Y - Yhat) <= 2).astype(int)  
    return np.mean(predictions)

def train_with_batching(X_train, Y_train, X_val, Y_val, learning_rate=0.01, max_epochs=100000, tol=1e-10, batch_size=64, patience=20):
    input_dim = X_train.shape[1]
    output_dim = 1
    Y_train = Y_train.reshape(-1, 1)

    best_val_loss = float("inf")
    patience_counter = 0

    L1 = InputLayer(X_train)
    L2 = FullyConnectedLayer(input_dim, 128)
    L3 = TanhLayer()
    L4 = FullyConnectedLayer(128, 128)
    L5 = TanhLayer()
    Add1 = AddLayer()

    L6 = FullyConnectedLayer(128, 128)
    L7 = TanhLayer()
    L8 = FullyConnectedLayer(128, 128)
    L9 = TanhLayer()
    Add2 = AddLayer()

    L10 = FullyConnectedLayer(128, 128)
    L11 = TanhLayer()
    L12 = FullyConnectedLayer(128, 128)
    L13 = TanhLayer()
    Add3 = AddLayer()

    L14 = FullyConnectedLayer(128, 128)
    L15 = TanhLayer()
    L16 = FullyConnectedLayer(128, 1)
    L17 = LinearLayer()
    L18 = SquaredError()

    layers = [
        L1, L2, L3, L4, L5, Add1,  
        L6, L7, L8, L9, Add2,  
        L10, L11, L12, L13, Add3,  
        L14, L15, L16, L17,  
        L18
    ]

    train_mse, val_mse = [], []
    train_acc, val_acc = [], []
    
    tic = time.perf_counter()
    prev_mse = float('inf')

    for epoch in range(max_epochs):
        X = X_train
        skip1, skip2, skip3 = None, None, None

        for layer in layers[:-1]:  
            if isinstance(layer, AddLayer):
                X = layer.forward(X, skip1 if layer == Add1 else skip2 if layer == Add2 else skip3)
            else:
                X = layer.forward(X)

            if layer == Add1:
                skip1 = X
            elif layer == Add2:
                skip2 = X
            elif layer == Add3:
                skip3 = X

        train_loss = layers[-1].eval(Y_train, X)
        train_mse.append(train_loss)

        grad = layers[-1].gradient(Y_train, X)

        for i in range(len(layers) - 2, 0, -1):
            if isinstance(layers[i], AddLayer):
                grad, skip_grad = layers[i].backward2(grad)
                if layers[i] == Add1 and skip1 is not None:
                    grad += skip_grad
                if layers[i] == Add2 and skip2 is not None:
                    grad += skip_grad
                if layers[i] == Add3 and skip3 is not None:
                    grad += skip_grad
            else:
                newgrad = layers[i].backward2(grad)
                if isinstance(layers[i], FullyConnectedLayer):
                    layers[i].updateWeights(grad, learning_rate)
                grad = newgrad

        X_val_temp = X_val
        skip1, skip2, skip3 = None, None, None

        for layer in layers[:-1]:
            if isinstance(layer, AddLayer):
                X_val_temp = layer.forward(X_val_temp, skip1 if layer == Add1 else skip2 if layer == Add2 else skip3)
            else:
                X_val_temp = layer.forward(X_val_temp)

            if layer == Add1:
                skip1 = X_val_temp
            elif layer == Add2:
                skip2 = X_val_temp
            elif layer == Add3:
                skip3 = X_val_temp

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

        if val_loss <= best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print("Convergence detected. Early stopping initiated.")
            break

    toc = time.perf_counter()
    print(f"Training Time: {toc - tic:.2f} seconds")
    return train_mse, val_mse, train_acc, val_acc, layers

def plot_mse(train_mse, val_mse):
    plt.plot(train_mse, label="Train Squared Error")
    plt.plot(val_mse, label="Validation Squared Error")
    plt.xlabel("Epoch")
    plt.ylabel("Squared Error")
    plt.title("Squared Error vs Epoch")
    plt.legend()
    plt.savefig("Graphs/Resnet_MSE.png")
    # plt.show()
    
def plot_predictions(Y_true, Y_pred, title="Predictions vs Ground Truth"):
    plt.figure(figsize=(8,6))
    plt.scatter(Y_true, Y_pred, alpha=0.5, label="Predictions")
    plt.plot([Y_true.min(), Y_true.max()], [Y_true.min(), Y_true.max()], 'r--', label="Ideal")
    plt.xlabel("Ground Truth")
    plt.ylabel("Predictions")
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.savefig("Graphs/Resnet_Predictions_vs_GroundTruth.png")
    # plt.show()

def plot_accuracy_curve(train_acc, val_acc):
    plt.figure(figsize=(8,6))
    plt.plot(train_acc, label="Train Accuracy")
    plt.plot(val_acc, label="Val Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs Epochs")
    plt.legend()
    plt.grid()
    plt.savefig("Graphs/Resnet_Accuracy_Curve.png")
    # plt.show()

if __name__ == "__main__":
    with h5py.File('Training_and_Validation.h5', 'r') as f:
        X_train_reduced = f['xTrainReduced'][:]
        Y_train = f['yTrain'][:]
        X_val_reduced = f['xValidationReduced'][:]
        Y_val = f['yValidation'][:]

    print("Training data shape after reduction:", X_train_reduced.shape)
    print("Validation data shape after reduction:", X_val_reduced.shape)

    print("\n\nRunning Residual Network MLP with forward and back prop....\n_____________________________________")
    train_mse, val_mse, train_acc, val_acc, layers = train_with_batching(X_train_reduced, Y_train, X_val_reduced, Y_val)
    plot_mse(train_mse, val_mse)
    plot_accuracy_curve(train_acc, val_acc)



 # Generate Predictions for the final model
    X_pred = X_val_reduced
    for layer in layers[:-1]:  # Exclude SquaredError
        X_pred = layer.forward(X_pred)

    # Plot results
    # plot_mse(train_mse, val_mse)
    # plot_predictions(Y_val, X_pred)
    

    
    plt.show()