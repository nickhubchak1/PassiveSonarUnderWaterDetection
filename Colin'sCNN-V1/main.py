import numpy as np
import matplotlib.pyplot as plt
import os
import json
from PIL import Image
from Framework import ConvolutionalLayer, MaxPoolLayer, FlatteningLayer, FullyConnectedLayer, LogLoss, LogisticSigmoidLayer, CrossEntropy, SoftmaxLayer


def main():

    csdm_dataset_path = "sw96_CSDM/features.dat"
    csdm_metadata_path = "sw96_CSDM/feature_metadata.json"
    label_path = "sw96_CSDM/labels.npy"


    with open(csdm_metadata_path, "r") as meta_file:
        # metadata = json.load(meta_file)
        csdm_metadata = json.load(meta_file)

    csdm_shape = tuple(csdm_metadata["shape"])
    csdm_dtype = csdm_metadata["dtype"]

    # Switched dtype from np.complex128 to float for speedup. Not sure if good technique.
    features = np.memmap(csdm_dataset_path, dtype=float, mode="r", shape = csdm_shape)
    labels = np.load(label_path, mmap_mode="r")
    
    featuresTest = features[:1, :, :, :]
    
    

    # Initialize kernel with shape (depth, height, width)
    np.random.seed(42)
    kernel = np.random.rand(25, 3, 3) * 0.01
    
    Global_Learning_Rate = 0.01  # Adjusted learning rate

    # Create and configure convolutional layer
    conv_layer = ConvolutionalLayer(kernel_size=(25, 3, 3))
    conv_layer.setKernels(kernel)
    pool_layer = MaxPoolLayer(poolSize=(2, 2, 2), stride=(25, 3, 3))
    flatten_layer = FlatteningLayer()
    fc_layer = FullyConnectedLayer(sizeIn=64, sizeOut=1)
    fc_layer.weights = np.random.randn(64, 1) * 0.01  # Adjusted initialization
    fc_layer.bias = np.zeros((1, 1))
    # conv_out = conv_layer.forward(features)
    # pool_out = pool_layer.forward(conv_out)
    # flatten_out = flatten_layer.forward(pool_out)
    # sizeIn = flatten_out.shape[1]

    print("Running convolutional layer...")
    conv_out = conv_layer.forward(featuresTest)
    print(f"Conv output shape: {conv_out.shape}")

    print("Running pooling layer...")
    pool_out = pool_layer.forward(conv_out)
    print(f"Pool output shape: {pool_out.shape}")

    print("Running flattening layer...")
    flatten_out = flatten_layer.forward(pool_out)
    print(f"Flatten output shape: {flatten_out.shape}")





    ### Just keeping this here and commented out so I can take a look at how I did the yale faces tensor for reference. ###
    ########################################################################################################################


    # ##################################################
    # # PART 3: CNN CLASSIFICATION OF YALE FACES DATA #
    # ##################################################

    # folder_path = r'yalefaces'  # Using the provided full path
    # images = load_images(folder_path)
    # num_images = len(images)

    # # Ensure we have images before stacking
    # if num_images > 0:
    #     # Convert list of images to a NumPy array (tensor)
    #     yale_tensor = np.stack(images, axis=0)  # Use axis=0 to stack along the first axis
    # else:
    #     print("No images were loaded.")

    # np.random.seed(42)
    # kernel = np.random.randn(9, 9) * 0.05
    # initial_kernel = np.copy(kernel)  # Save the initial kernel
    # Global_Learning_Rate = 0.01  # Adjusted learning rate

    # conv_layer = ConvolutionalLayer(9)
    # conv_layer.setKernels(kernel)
    # pool_layer = MaxPoolLayer(poolSize=4, stride=4)
    # flatten_layer = FlatteningLayer()

    # # Determine the correct sizeIn dynamically
    # conv_out = conv_layer.forward(yale_tensor)
    # pool_out = pool_layer.forward(conv_out)
    # flatten_out = flatten_layer.forward(pool_out)
    # sizeIn = flatten_out.shape[1]

    # fc_layer = FullyConnectedLayer(sizeIn=sizeIn, sizeOut=14)  # Adjusted for 14 classes
    # fc_layer.weights = np.random.randn(sizeIn, 14) * 0.01  # Adjusted initialization
    # fc_layer.bias = np.zeros((1, 14))
    # softmaxActivation = SoftmaxLayer()
    # crossEntropy_function = CrossEntropy()

    # normal_tensor = z_score_normalize(yale_tensor)

    # # Generating one-hot encoded labels for 14 subjects
    # Y = np.eye(14)

    # num_epochs = 100  # Adjusted number of epochs
    # losses = []  # To store the CrossEntropy loss values for each epoch


    # for epoch in range(num_epochs):
    #     conv_out = conv_layer.forward(normal_tensor)
    #     pool_out = pool_layer.forward(conv_out)
    #     flatten_out = flatten_layer.forward(pool_out)
    #     fc_out = fc_layer.forward(flatten_out)
    #     SM_out = softmaxActivation.forward(fc_out)
        
    #     loss = crossEntropy_function.eval(Y, SM_out)
    #     losses.append(loss)  # Store the loss

    #     dL_dout = crossEntropy_function.gradient(Y, SM_out)
    #     dL_dfc_out = softmaxActivation.backward(dL_dout)
    #     grad_input, dL_dW = fc_layer.backward(dL_dfc_out)

    #     # Apply gradient update
    #     fc_layer.updateWeights(dL_dW, Global_Learning_Rate)

    #     dL_dpool_out = flatten_layer.backward(grad_input)
    #     dL_dconv_out = pool_layer.backward(dL_dpool_out)
    #     conv_layer.updateKernels(dL_dconv_out, Global_Learning_Rate)

    #     if epoch % 10 == 0:  # Print loss every 10 epochs
    #         print(f"Epoch {epoch}, Loss: {loss}")

    # # Checking the final output, predictions, and true labels
    # conv_out = conv_layer.forward(normal_tensor)
    # pool_out = pool_layer.forward(conv_out)
    # flatten_out = flatten_layer.forward(pool_out)
    # fc_out = fc_layer.forward(flatten_out)
    # SM_out = softmaxActivation.forward(fc_out)

    # predictions = np.argmax(SM_out, axis=1)
    # true_labels = np.argmax(Y, axis=1)

    # accuracy = calculate_accuracy(predictions, true_labels)
    # print(f"Epoch {epoch}, Loss: {loss}")
    # print("Predictions:", predictions)
    # print("True Labels:", true_labels)
    # print(f"Accuracy: {accuracy:.2f}%")

    # # Retrieve the final kernel for plotting
    # final_kernel = conv_layer.getKernels()
    
    # plot_images(yale_tensor[0], yale_tensor[1], initial_kernel, final_kernel, losses, flag=True)

if __name__ == "__main__":
    main()
