#-----------------------------------------------------
# Deep Learning Final Project 2025
# Under Water Passive Acoustic Source Localization
# Author: Nick Hubchak, Priontu Chowdhury, Colin Woods
# All Rights Reserved 2025-2030
#----------------------------------------------------
import h5py
import time
import numpy as np
import matplotlib.pyplot as plt

from framework import (

    InputLayer,

    FullyConnectedLayer,

    TanhLayer,

    ReLULayer,

    LinearLayer, 

    SquaredError

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

def train_validate(X_train, Y_train, X_val, Y_val, learning_rate=0.0001, max_epochs=100000, tol=1e-6, patience=64):
    input_dim = X_train.shape[1]
    output_dim =  1
    Y_train = Y_train.reshape(-1, 1) #force to be (3000, 1)

    L1 = InputLayer(X_train)
    #print("Setting up Sparse Fully conected layer")
    L2 = FullyConnectedLayer(input_dim, input_dim)
    L3 = TanhLayer()
    L4 = FullyConnectedLayer(input_dim, 750)  
    L5 = TanhLayer()
    L8 = FullyConnectedLayer(750, output_dim)  
    L9 = LinearLayer()
    L10 = SquaredError() 
    #L3 = ReLULayer() #Used for testing results were mixed val loss was lower by 0.05 and converged at epoch 1400, relu hits the vanishing gradients easier

    
    layers = [L1, L2, L3, L4, L5,L8, L9, L10]
    
    train_mse, val_mse = [], []
    Y_hat = []
    tic = time.perf_counter()
    prev_mse = float('inf')
    patience_counter = 0

    
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
        train_prediction = X
        train_loss = layers[-1].eval(Y_train, X)
        train_mse.append(train_loss)

        grad = layers[-1].gradient(Y_train, X)
        #print("grad shape from Squared Error forward training: ", grad.shape)
        # Backpropagation
        for i in range(len(layers) - 2, 0, -1):
            newgrad = layers[i].backward2(grad)
            if isinstance(layers[i], FullyConnectedLayer):
                #print(f"Layer {i} weight gradient norm: {np.linalg.norm(grad)}") 
                layers[i].updateWeights(grad, learning_rate)
            grad = newgrad
        
        # Validation
        X_val_temp = X_val
        for layer in layers[:-1]:
            X_val_temp = layer.forward(X_val_temp)
        val_predictions = X_val_temp
        val_loss = layers[-1].eval(X_val_temp, Y_val)
        val_mse.append(val_loss)

        if epoch % 100 == 0:
            train_smape = SMAPE(Y_train, X)
            train_rmse = RMSE(Y_train, X)
            train_accuracy = accuracy(Y_train, X)
            val_smape = SMAPE(Y_val, X_val_temp)
            val_rmse = RMSE(Y_val, X_val_temp)
            val_accuracy = accuracy(Y_val, X_val_temp)
            print(f"Epoch {epoch}: Train Loss = {train_loss:.10f}, Val Loss = {val_loss:.10f}, Train Acc = {train_accuracy:.10f}, Val Acc = {val_accuracy:.10f}, Train SMAPE = {train_smape:.10f}, Train RMSE = {train_rmse:.10f}, Val SMAPE = {val_smape:.10f}, Val RMSE = {val_rmse:.10f}")
        
        if abs(prev_mse - val_loss) < tol:
            print(f"Converged at epoch {epoch}")
            break

        if(prev_mse >= val_loss):
            patience_counter = 0
            prev_mse = val_loss   
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print("Convergence detected. Early stopping initiated.")
            break
        prev_mse = val_loss

    toc = time.perf_counter()
    print(f"Training Time: {toc - tic:.2f} seconds")
    return train_mse, val_mse, train_accuracy, val_accuracy, train_prediction, val_predictions

def plot_accuracy_curve(train_acc, val_acc):
    plt.figure(figsize=(8,6))
    plt.plot(train_acc, label="Train Accuracy")
    plt.plot(val_acc, label="Val Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs Epochs")
    plt.legend()
    plt.grid()
    plt.savefig("Graphs/mlp_Accuracy_Curve.png")
    # plt.show()

def plot_mse(train_mse, val_mse):
    plt.plot(train_mse, label="Train Squared Error")
    plt.plot(val_mse, label="Validation Squared Error")
    plt.xlabel("Epoch")
    plt.ylabel("Squared Error")
    plt.title("Squared Error vs Epoch")
    plt.legend()
    plt.savefig("Graphs/mlp_squared_error.png")
    #plt.show()

def plot_ground_truth_vs_prediction(Y_validation, prediction, title="Ground Truth vs Prediction for Validation"):
    plt.figure(figsize=(8, 6))
    Y_validation = Y_validation.reshape(-1, 1) #force to be (3000, 1)

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
    plt.savefig("Graphs/mlp_ground_truth_prediction.png")
    #plt.show()
    
if __name__ == "__main__":

    with h5py.File('Training_and_Validation.h5', 'r') as f:
        X_train_reduced = f['xTrainReduced'][:]
        Y_train = f['yTrain'][:]
        X_val_reduced = f['xValidationReduced'][:]
        Y_val = f['yValidation'][:]

    
    print("Xtrain reduced: ", X_train_reduced[0:50])
    print("Ytrain reduced: ", Y_train[0:50])

    print("Training data shape after reduction:", X_train_reduced.shape)
    print("Validation data shape after reduction:", X_val_reduced.shape)

    print("\n\nRunning shallow multiclass MLP with forward and back prop....\n_____________________________________")
    train_mse, val_mse, train_accuracy, val_accuracy, train_predictions, val_predictions = train_validate(X_train_reduced, Y_train, X_val_reduced, Y_val)
    plot_mse(train_mse, val_mse)
    plot_ground_truth_vs_prediction(Y_val, val_predictions)
    plot_accuracy_curve(train_accuracy, val_accuracy)
    plt.show()