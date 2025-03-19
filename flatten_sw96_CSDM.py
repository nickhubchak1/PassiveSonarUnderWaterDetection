#-----------------------------------------------------
# Deep Learning Final Project 2025
# Under Water Passive Acoustic Source Localization
# Author: Nick Hubchak, Priontu Chowdhury, Colin Woods
# All Rights Reserved 2025-2030
#----------------------------------------------------
import numpy as np
import json
import h5py

def pca_numpy(X, num_components):
    """
    Perform PCA on dataset X using NumPy.
    
    Parameters:
    - X: Input data of shape (num_samples, num_features)
    - num_components: Number of principal components to retain

    Returns:
    - X_reduced: Transformed data with reduced dimensions
    - explained_variance: Percentage of variance retained
    """
    X_mean = np.mean(X, axis=0)
    X_centered = X - X_mean

    covariance_matrix = np.cov(X_centered, rowvar=False)

    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)  # eigh is optimized for symmetric matrices
    sorted_indices = np.argsort(eigenvalues)[::-1]  # Descending order
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]

    selected_eigenvectors = eigenvectors[:, :num_components]

    X_reduced = np.dot(X_centered, selected_eigenvectors)


    explained_variance = np.sum(eigenvalues[:num_components]) / np.sum(eigenvalues)
    print(f"Variance retained: {explained_variance * 100:.2f}%")
    return X_reduced, explained_variance

def pca_svd(X, num_components):
    """
    Perform PCA using Singular Value Decomposition (SVD), avoiding large covariance matrices.

    Parameters:
    - X: Input data of shape (num_samples, num_features)
    - num_components: Number of principal components to retain

    Returns:
    - X_reduced: Transformed data with reduced dimensions
    - explained_variance: Percentage of variance retained
    """
    X_mean = np.mean(X, axis=0)
    X_centered = X - X_mean

    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)

    V_selected = Vt[:num_components, :]  # (num_components, num_features)

    X_reduced = np.dot(X_centered, V_selected.T)  # (num_samples, num_components)

    print("S: ", S)
    explained_variance = np.sum(S[:num_components]**2) / np.sum(S**2)

    print(f"Variance retained: {explained_variance * 100:.2f}%")
    
    return X_reduced, explained_variance

def read_from_sw96_CSDM(path):
    print("*** Make sure the sw96_CSDM.zip is properly downloaded from github and uncompressed ****")
    #Reading in from sw_96_CSDM dataset downloaded
    with open(path+'/feature_metadata.json', 'r') as f:
        metadata = json.load(f)

    feature_shape = metadata["shape"]
    feature_dtype = np.complex128  


    n_elements = np.prod(feature_shape)

    real_part = np.fromfile(path + '/features.dat', dtype=np.float64, count=n_elements)
    imag_part = np.fromfile(path  + '/features.dat', dtype=np.float64, count=n_elements)

    features = real_part + 1j * imag_part 

    features = features.reshape(*feature_shape)
    feature_shape = metadata["shape"]
    feature_dtype = metadata["dtype"]

    frequencies = np.load(path + '/frequencies.npy') 
    labels = np.load(path + '/labels.npy') 

    print("Features shape:", features.shape)
    print("Frequencies shape:", frequencies.shape)
    print("Labels shape:", labels.shape)

    n_samples = features.shape[0]
    n_features = np.prod(features.shape[1:])  

    features_flattened = features.reshape(n_samples, n_features)

    features_normalized = np.abs(features_flattened) 
    features_normalized = features_normalized / np.max(features_normalized)

    #Used this to create flattened_data.npz, not as good as h5
    #np.savez_compressed('flattened_data.npz', features=features_normalized, labels=labels)

    # Save the flattened features and labels to an HDF5 file, checkpoint for data before splitting
    with h5py.File('flattened_data.h5', 'w') as f:
        f.create_dataset('features', data=features_normalized)
        f.create_dataset('labels', data=labels)

    
def read_from_flattened_h5(filename, new_features_count):
    with h5py.File(filename, 'r') as f:   
        loaded_features = f['features'][:].astype(np.float32)  # used to be 64 but too large to process
        loaded_labels = f['labels'][:].astype(np.float32) #needs to be 32 because anything less wont be supported by svd

    print("Before PCA data shape:", loaded_features.shape)
    print("Before PCA Y_hat shape:", loaded_labels.shape)

    #Shuffle
    subset_size = int(len(loaded_features) * 1)  # 100% of the data can use 0.1 for 10%
    indices = np.random.choice(len(loaded_features), size=subset_size, replace=False)
    loaded_features = loaded_features[indices]
    loaded_labels = loaded_labels[indices]

    # Split the dataset into 1/3 for training and 2/3 for validation
    split_index = len(loaded_features) // 3
    X_train, X_val = loaded_features[split_index:], loaded_features[:split_index]
    Y_train, Y_val = loaded_labels[split_index:], loaded_labels[:split_index]

    print("Before PCA Training data shape after split:", X_train.shape)
    print("Before PCA Validation data shape after split:", X_val.shape)
    #print("Y_train: ", Y_train[:50])

    num_components = new_features_count # this is a parameter fed into this function
    
    X_train_reduced, variance_retained = pca_svd(X_train, num_components)
    X_val_reduced, _ = pca_svd(X_val, num_components)

    print(f"Variance retained: {variance_retained * 100:.2f}%")

    print("Training data shape after reduction:", X_train_reduced.shape)
    print("Validation data shape after reduction:", X_val_reduced.shape)

    # Save the flattened features and labels to an HDF5 file
    with h5py.File('Training_and_Validation.h5', 'w') as f:
        f.create_dataset('xTrainReduced', data=X_train_reduced)
        f.create_dataset('yTrain', data=Y_train)
        f.create_dataset('xValidationReduced', data=X_val_reduced)
        f.create_dataset('yValidation', data=Y_val)


if __name__ == "__main__":

    # Both functions produce h5 in the local folder they are being run from

    #Warning: This functions is the first step but takes a long time to run, LARGE DATASET
    read_from_sw96_CSDM('sw96_CSDM')


    # Reading from flattened_data.h5 after saving it off locally, still to large to upload to github
    # Continuation of setting up training and validation 
    num_components = 1256 #512 is stable
    read_from_flattened_h5('flattened_data.h5', num_components)
    