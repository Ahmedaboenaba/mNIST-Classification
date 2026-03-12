# MNIST Digit Classification using TensorFlow & Keras

This repository contains a Jupyter Notebook that builds, trains, and evaluates a deep neural network to classify handwritten digits (0-9) from the iconic MNIST dataset. It serves as a clear demonstration of a standard machine learning pipeline using Python and TensorFlow/Keras.

## Features

* **Data Exploration:** Analyzes dataset shapes, checks pixel value ranges, and visually plots sample images with their corresponding labels.
* **Data Preprocessing:** Normalizes image pixel values to a [0, 1] range for better convergence and flattens the 28x28 2D images into 1D arrays of 784 pixels.
* **Custom Neural Network:** Implements a Multi-Layer Perceptron (MLP) using the Keras `Sequential` API.
* **Advanced Callbacks:** Utilizes `EarlyStopping` to prevent model overfitting and `ReduceLROnPlateau` to dynamically decay the learning rate when validation loss stops improving.
* **Performance Visualization:** Plots training vs. validation accuracy and loss curves across epochs to monitor learning behavior.
* **Prediction Visualization:** Displays a grid of test images alongside their predicted and true labels, color-coded green for correct predictions and red for incorrect ones.

## Model Architecture

The model is a straightforward feedforward neural network consisting of the following layers:

* **Input:** 784 neurons (flattened 28x28 images)
* **Hidden Layer 1:** 128 neurons, ReLU activation
* **Hidden Layer 2:** 64 neurons, ReLU activation
* **Output Layer:** 10 neurons, Softmax activation (representing probabilities for digits 0-9)
* **Total Trainable Parameters:** 109,386

## Results

After training with the Adam optimizer and sparse categorical crossentropy loss, the model stops early at around 10 epochs (preventing overfitting) and achieves excellent performance on the unseen test set:

* **Test Accuracy:** ~97.50%
* **Test Loss:** ~0.0886

## Requirements

To run this notebook locally, ensure you have Python installed along with the following libraries:

* `numpy`
* `matplotlib`
* `tensorflow` (includes `keras`)

You can install the dependencies using pip:

```bash
pip install numpy matplotlib tensorflow

```

## Usage

1. Clone this repository to your local machine.
2. Open the `mnist_classification.ipynb` file using Jupyter Notebook, JupyterLab, or upload it to Google Colab.
3. Run the cells sequentially to download the dataset, train the network, and visualize the predictions.

---
