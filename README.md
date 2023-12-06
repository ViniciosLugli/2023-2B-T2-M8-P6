# 2023-2B-T2-M8-P6

## Quick Starter

This quick starter provides a simple example of using the MNIST dataset with TensorFlow and Keras. It includes a Convolutional Neural Network (CNN) for digit classification, training the model, and evaluating its performance.

### Requirements

Ensure you have the required dependencies installed by running:

_Currently using Python 3.11_

```bash
pip install -r requirements.txt
```

### Usage

1. Run the `main.py` file to train the model and execute predictions on the test dataset.

```bash
python main.py
```

### Code Overview

The code in `main.py` contains a class `MNISTClassifier` encapsulating the MNIST model and related functions. It includes the following methods:

-   `train_model(epochs)`: Trains the model on the MNIST training dataset for the specified number of epochs.
-   `evaluate_model()`: Evaluates the trained model on the MNIST test dataset and prints the test accuracy.
-   `predict_samples(num_samples)`: Displays predictions for a specified number of samples from the test dataset.

Additionally, the code includes a test class `TestMNISTClassifier` within which there is a test method `test_model_evaluation`. This method checks if the model is initialized, trains it, evaluates its accuracy, and predicts samples, ensuring the accuracy is above a specified threshold.

### MNISTClassifier Class

The `MNISTClassifier` class utilizes TensorFlow and Keras to define, compile, and train a Convolutional Neural Network for digit classification on the MNIST dataset. The architecture consists of a convolutional layer, max-pooling layer, flattening layer, and dense layers.

### Notes

-   The model is saved as `mnist_model.h5` after training.
-   The test accuracy is checked to ensure it is greater than 95%.
-   Predictions for sample images are displayed using Matplotlib.

## Demo
[demo](https://github.com/ViniciosLugli/2023-2B-T2-M8-P6/assets/40807526/a54bb31a-5ab2-4a76-842c-80e8cbd9f881)
