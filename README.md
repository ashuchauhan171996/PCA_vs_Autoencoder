# PCA_vs_Autoencoder
Implementation and comparison between PCA and the autoencoder (AE) to a collection of handwritten digit images from the USPS dataset using PyTorch.

# Introduction

Principal Component Analysis (PCA) and autoencoders are both techniques used in dimensionality reduction and feature extraction, but they achieve these goals through different mechanisms and have different applications:

## Principal Component Analysis (PCA):

* PCA is a statistical method used for dimensionality reduction. It works by finding the principal components, which are the orthogonal axes that capture the maximum variance in the data.
* PCA transforms the original high-dimensional data into a new set of orthogonal variables called principal components.
* The principal components are ordered in terms of the amount of variance they explain in the data, with the first principal component explaining the most variance, the second explaining the second most, and so on.
* PCA does not involve any specific neural network architecture and is purely a mathematical/statistical technique.
* PCA is a linear method, meaning it captures linear relationships between variables.
* It is commonly used for data visualization, noise reduction, and speeding up learning algorithms by reducing the dimensionality of the feature space.

## Autoencoder:

* An autoencoder is a type of neural network architecture used for unsupervised learning. It consists of an encoder and a decoder network.
* The encoder network compresses the input data into a lower-dimensional representation, typically called a "latent space" or "encoding."
* The decoder network then attempts to reconstruct the original input data from the compressed representation.
* Autoencoders are trained to minimize the reconstruction error, encouraging them to learn a compact representation of the input data in the latent space.
* Unlike PCA, autoencoders are nonlinear models and can capture complex relationships in the data.
* Autoencoders can learn hierarchical representations, making them powerful tools for feature learning and extraction in tasks such as image denoising, dimensionality reduction, and anomaly detection.
* Variants of autoencoders, such as denoising autoencoders, variational autoencoders (VAEs), and convolutional autoencoders, have been developed to address specific challenges and improve performance in various domains.

In this assignment, I applied PCA and the autoencoder (AE) to a collection of handwritten digit images from the USPS dataset. The data file is stored in the "/data" folder as "\USPS.mat". The whole dataset is loaded and stored in the matrix A with shape 3000x256. Each row of matrix A represents a 16x16 handwritten digit image (between 0 and 9), which is 
flattened to a 256-dimensional vector.

# Observations and Result
<img width="547" alt="image" src="https://github.com/ashuchauhan171996/PCA_vs_Autoencoder/assets/83955120/b24d6227-ea35-43eb-a229-1a2ac9c11f77">
<img width="558" alt="image" src="https://github.com/ashuchauhan171996/PCA_vs_Autoencoder/assets/83955120/bca3defa-33b6-49fa-9ed1-d5e620c9cdc3">
<img width="562" alt="image" src="https://github.com/ashuchauhan171996/PCA_vs_Autoencoder/assets/83955120/ec663f06-27bc-4ba3-819a-8675aabe0458">
<img width="564" alt="image" src="https://github.com/ashuchauhan171996/PCA_vs_Autoencoder/assets/83955120/401abb99-32e7-4eb8-949c-d69855d60955">



# Summary

<img width="552" alt="image" src="https://github.com/ashuchauhan171996/PCA_vs_Autoencoder/assets/83955120/a5400f1e-4933-4a9b-a2aa-b64dde24f51e">
<img width="545" alt="image" src="https://github.com/ashuchauhan171996/PCA_vs_Autoencoder/assets/83955120/13ef8f6f-5f2f-491c-8523-682393ee67cb">



