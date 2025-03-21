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
    
    AddLayer, 

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

def train_validate_with_skip(X_train, Y_train, X_val, Y_val, learning_rate=0.00005, max_epochs=100000, tol=1e-10, patience=20):
    input_dim = X_train.shape[1]
    output_dim =  1
    Y_train = Y_train.reshape(-1, 1) #force to be (3000, 1)

    # Main network branch
    L1_main = InputLayer(X_train)
    L2_main = FullyConnectedLayer(input_dim, 128)
    L3_main = TanhLayer()
    L4_main = FullyConnectedLayer(128, 128)
    L5_main = TanhLayer()
    L6_main = FullyConnectedLayer(128, output_dim)
    
    # Residual connection branch
    L1_res = InputLayer(X_train)
    L2_res = FullyConnectedLayer(input_dim, 128)
    L3_res = LinearLayer()
    L4_res = FullyConnectedLayer(128, output_dim)
    L5_res = LinearLayer()

    # Fuse the two branches
    L4 = AddLayer(L4_main, L3_res)

    L7 = AddLayer(L6_main, L5_res)

    # Loss function
    L8 = SquaredError()

    layers = [L1_main, L2_main, L3_main, L4_main, L5_main, L6_main, L1_res, L2_res, L3_res, L4, L4_res, L5_res, L7, L8]
    
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
        X_main = X_train
        X_res = X_train
        for layer in [L1_main, L2_main, L3_main, L4_main]:
            X_main = layer.forward(X_main)
        for layer in [L1_res, L2_res, L3_res]:
            X_res = layer.forward(X_res)
        X_fused = L4.forward(X_main, X_res)

        X_main = X_fused
        X_res = X_fused
        for layer in [L5_main, L6_main]:
            X_main = layer.forward(X_main)
        for layer in [L4_res, L5_res]:
            X_res = layer.forward(X_res)
        X_fused = L7.forward(X_main, X_res)
        #print("X_fused shape going into squared error: ", X_fused.shape)
        #print("Y_train shape: ", Y_train.shape)
        train_loss = L8.eval(Y_train, X_fused)
        train_mse.append(train_loss)

        grad = L8.gradient(Y_train, X_fused)
        #print("grad shape from Squared Error forward training: ", grad.shape)
        # Backpropagation, doing backwards order from forward prop


        grad_main, grad_res = L7.backward2(grad)
        for i in range(len([L5_main, L6_main]) - 1, 0, -1):
            layer = [L5_main, L6_main][i]
            newgrad_main = layer.backward2(grad_main)
            if isinstance(layer, FullyConnectedLayer):
                layer.updateWeights(grad_main, learning_rate)
            grad_main = newgrad_main
        for i in range(len([L4_res, L5_res]) - 1, 0, -1):
            layer = [L4_res, L5_res][i]
            newgrad_res = layer.backward2(grad_res)
            if isinstance(layer, FullyConnectedLayer):
                layer.updateWeights(grad_res, learning_rate)
            grad_res = newgrad_res
        
        #print("shape of grad_main after back: ", grad_main.shape)
        #print("shape of grad_res after back: ", grad_res.shape)
        grad_main, grad_res = L4.backward2(grad_main)
        #print("grad_main shape: ", grad_main.shape)
        for i in range(len([L1_main, L2_main, L3_main, L4_main]) - 1, 0, -1):
            layer = [L1_main, L2_main, L3_main, L4_main][i]
            newgrad_main = layer.backward2(grad_main)
            if isinstance(layer, FullyConnectedLayer):
                layer.updateWeights(grad_main, learning_rate)
            grad_main = newgrad_main
        for i in range(len([L1_res, L2_res, L3_res]) - 1, 0, -1):
            layer = [L1_res, L2_res, L3_res][i]
            newgrad_res = layer.backward2(grad_res)
            if isinstance(layer, FullyConnectedLayer):
                layer.updateWeights(grad_res, learning_rate)
            grad_res = newgrad_res


        
        # Validation


        # for layer in [L1_main, L2_main, L3_main, L4_main]:
        #     X_main = layer.forward(X_main)
        # for layer in [L1_res, L2_res, L3_res]:
        #     X_res = layer.forward(X_res)
        # X_fused = L4.forward(X_main, X_res)

        # X_main = X_fused
        # X_res = X_fused
        # for layer in [L5_main, L6_main]:
        #     X_main = layer.forward(X_main)
        # for layer in [L4_res, L5_res]:
        #     X_res = layer.forward(X_res)
        # X_fused = L7.forward(X_main, X_res)



        X_val_temp_main = X_val
        X_val_temp_res = X_val
        for layer in [L1_main, L2_main, L3_main, L4_main]:
            X_val_temp_main = layer.forward(X_val_temp_main)
        for layer in [L1_res, L2_res, L3_res]:
            X_val_temp_res = layer.forward(X_val_temp_res)
        X_val_temp_fused = L4.forward(X_val_temp_main, X_val_temp_res) # addition portion

        X_val_temp_main = X_val_temp_fused
        X_val_temp_res = X_val_temp_fused
        for layer in [L5_main, L6_main]:
            #print("Xval_temp_main shape: ", X_val_temp_main.shape)
            X_val_temp_main = layer.forward(X_val_temp_main)
        for layer in [L4_res, L5_res]:
            X_val_temp_res = layer.forward(X_val_temp_res)
        X_val_temp_fused = L7.forward(X_val_temp_main, X_val_temp_res)


        
        val_loss = L8.eval(X_val_temp_fused, Y_val)
        val_mse.append(val_loss)

        if epoch % 100 == 0:
            train_smape = SMAPE(Y_train, X_fused)
            train_rmse = RMSE(Y_train, X_fused)
            train_accuracy = accuracy(Y_train, X_fused)
            val_smape = SMAPE(Y_val, X_val_temp_fused)
            val_rmse = RMSE(Y_val, X_val_temp_fused)
            val_accuracy = accuracy(Y_val, X_val_temp_fused)
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
    return train_mse, val_mse, train_accuracy, val_accuracy, X_fused, X_val_temp_fused

def plot_accuracy_curve(train_acc, val_acc):
    plt.figure(figsize=(8,6))
    plt.plot(train_acc, label="Train Accuracy")
    plt.plot(val_acc, label="Val Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs Epochs")
    plt.legend()
    plt.grid()
    plt.savefig("Graphs/mlp_shallow_skip_Accuracy_Curve.png")
    #plt.show()

def plot_ground_truth_vs_prediction(Y_validation, prediction, title="Ground Truth vs Prediction for Validation"):
    plt.figure(figsize=(8, 6))
    plt.scatter(Y_validation, prediction, alpha=0.5, label="Predictions", color='blue')
    plt.plot([min(Y_validation), max(Y_validation)], [min(Y_validation), max(Y_validation)], 
             linestyle='dashed', color='red', label="Ideal Prediction (y = x)")
    
    plt.xlabel("Ground Truth (y_validation)")
    plt.ylabel("Predicted (X_fused)")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig("Graphs/mlp_shallow_skip_ground_Truth_vs_pred.png")
    #plt.show()

def plot_mse(train_mse, val_mse):
    plt.plot(train_mse, label="Train Squared Error")
    plt.plot(val_mse, label="Validation Squared Error")
    plt.xlabel("Epoch")
    plt.ylabel("Squared Error")
    plt.title("Squared Error vs Epoch")
    plt.savefig("Graphs/mlp_shallow_skip_squared_error.png")
    plt.legend()
    #plt.show()
    
if __name__ == "__main__":

    with h5py.File('Training_and_Validation.h5', 'r') as f:
        X_train_reduced = f['xTrainReduced'][:]
        Y_train = f['yTrain'][:]
        X_val_reduced = f['xValidationReduced'][:]
        Y_val = f['yValidation'][:]

    print("Training data shape after reduction:", X_train_reduced.shape)
    print("Validation data shape after reduction:", X_val_reduced.shape)

    print("\n\nRunning shallow multiclass MLP with forward and back prop and SKIP RESIDUAL....\n_____________________________________")
    train_mse, val_mse, train_accuracy, val_accuracy, X_train_fused, X_val_fused = train_validate_with_skip(X_train_reduced, Y_train, X_val_reduced, Y_val)
    plot_mse(train_mse, val_mse)
    plot_ground_truth_vs_prediction(Y_train, X_train_fused)
    plot_ground_truth_vs_prediction(Y_val, X_val_fused)
    plot_accuracy_curve(train_accuracy, val_accuracy)
    plt.show()

