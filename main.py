import tensorflow as tf
from tensorflow.keras import layers, models
import unittest
import matplotlib.pyplot as plt


class MNISTClassifier:
    def __init__(self):
        # Load the MNIST dataset
        (self.x_train, self.y_train), (self.x_test, self.y_test) = tf.keras.datasets.mnist.load_data()

        # Normalize the data (0-255 -> 0-1)
        self.x_train, self.x_test = self.x_train / 255.0, self.x_test / 255.0

        # Create the model
        self.model = models.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),  # 32 filters, 3x3 kernel
            layers.MaxPooling2D((2, 2)),  # 2x2 max pooling
            layers.Flatten(),  # Flatten the 2D image to 1D vector
            layers.Dense(128, activation='relu'),  # 128 neurons
            layers.Dense(10, activation='softmax')  # 10 outputs
        ])

        # Compile
        self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    def train_model(self, epochs=3):
        # Train the model
        self.model.fit(self.x_train.reshape(-1, 28, 28, 1), self.y_train, epochs=epochs,
                       validation_data=(self.x_test.reshape(-1, 28, 28, 1), self.y_test))

    def evaluate_model(self):
        # Evaluate the model on the test set
        test_loss, test_acc = self.model.evaluate(self.x_test.reshape(-1, 28, 28, 1), self.y_test, verbose=2)
        print(f"\nTest accuracy: {test_acc * 100:.2f}%")
        return test_acc

    def predict_samples(self, num_samples=10):
        # Predict samples and display the results
        predictions = self.model.predict(self.x_test.reshape(-1, 28, 28, 1))

        for i in range(num_samples):
            plt.imshow(self.x_test[i], cmap='gray')
            plt.title(f"True Label: {self.y_test[i]}, Predicted Label: {tf.argmax(predictions[i])}")
            plt.show()


class TestMNISTClassifier(unittest.TestCase):
    def setUp(self):
        self.mnist_classifier = MNISTClassifier()

    def test_model_evaluation(self):
        assert self.mnist_classifier.model is not None, "Model is not initialized"
        # Train the model
        self.mnist_classifier.train_model()

        # Evaluate the model
        acc = self.mnist_classifier.evaluate_model()
        assert acc > 0.95, "Accuracy is too low"

        # Predict samples
        self.mnist_classifier.predict_samples()


if __name__ == "__main__":
    unittest.main()
