import numpy as np
import json


with open('sw96_CSDM/feature_metadata.json', 'r') as f:
    metadata = json.load(f)


feature_shape = metadata["shape"]
feature_dtype = np.complex128  


n_elements = np.prod(feature_shape)

real_part = np.fromfile('sw96_CSDM/features.dat', dtype=np.float64, count=n_elements)
imag_part = np.fromfile('sw96_CSDM/features.dat', dtype=np.float64, count=n_elements)

features = real_part + 1j * imag_part 

features = features.reshape(*feature_shape)
feature_shape = metadata["shape"]
feature_dtype = metadata["dtype"]

frequencies = np.load('sw96_CSDM/frequencies.npy') 
labels = np.load('sw96_CSDM/labels.npy') 

print("Features shape:", features.shape)
print("Frequencies shape:", frequencies.shape)
print("Labels shape:", labels.shape)



n_samples = features.shape[0]
n_features = np.prod(features.shape[1:])  


features_flattened = features.reshape(n_samples, n_features)


features_normalized = np.abs(features_flattened) 
features_normalized = features_normalized / np.max(features_normalized)


#Used this to create flattened_data.npz
#np.savez_compressed('flattened_data.npz', features=features_normalized, labels=labels)

import h5py

# Save the flattened features and labels to an HDF5 file
with h5py.File('flattened_data.h5', 'w') as f:
    f.create_dataset('features', data=features_normalized)
    f.create_dataset('labels', data=labels)