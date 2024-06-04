import numpy as np

class Perceptron:
    def __init__(self, num_input_features, num_neurons_hidden_layer, num_output_features, learning_rate=0.1, num_epochs=100000):
        self.num_input_features = num_input_features
        self.num_neurons_hidden_layer = num_neurons_hidden_layer
        self.num_output_features = num_output_features
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.parameters = self.initialize_parameters()

    def initialize_parameters(self):
        np.random.seed(None)  # Ensure different random initialization each time
        W1 = np.random.randn(self.num_neurons_hidden_layer, self.num_input_features) * np.sqrt(1 / self.num_input_features)
        W2 = np.random.randn(self.num_output_features, self.num_neurons_hidden_layer) * np.sqrt(1 / self.num_neurons_hidden_layer)
        b1 = np.zeros((self.num_neurons_hidden_layer, 1))
        b2 = np.zeros((self.num_output_features, 1))

        parameters = {"W1": W1, "b1": b1,
                      "W2": W2, "b2": b2}
        return parameters

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def forward_propagation(self, inputs):
        W1 = self.parameters["W1"]
        W2 = self.parameters["W2"]
        b1 = self.parameters["b1"]
        b2 = self.parameters["b2"]

        Z1 = np.dot(W1, inputs) + b1
        A1 = self.sigmoid(Z1)
        Z2 = np.dot(W2, A1) + b2
        predictions = self.sigmoid(Z2)

        cache = (Z1, A1, Z2, predictions)
        return cache, predictions

    def compute_loss(self, predictions, outputs):
        num_examples = outputs.shape[1]
        log_probs = np.multiply(np.log(predictions), outputs) + np.multiply(np.log(1 - predictions), (1 - outputs))
        loss = -np.sum(log_probs) / num_examples
        return loss

    def backward_propagation(self, inputs, outputs, cache):
        num_examples = inputs.shape[1]
        (Z1, A1, Z2, predictions) = cache

        error_output_layer = predictions - outputs
        dW2 = np.dot(error_output_layer, A1.T) / num_examples
        db2 = np.sum(error_output_layer, axis=1, keepdims=True)

        error_hidden_layer = np.dot(self.parameters["W2"].T, error_output_layer)
        dZ1 = np.multiply(error_hidden_layer, A1 * (1 - A1))
        dW1 = np.dot(dZ1, inputs.T) / num_examples
        db1 = np.sum(dZ1, axis=1, keepdims=True) / num_examples

        gradients = {"dW2": dW2, "db2": db2,
                     "dW1": dW1, "db1": db1}
        return gradients

    def update_parameters(self, gradients):
        self.parameters["W1"] -= self.learning_rate * gradients["dW1"]
        self.parameters["W2"] -= self.learning_rate * gradients["dW2"]
        self.parameters["b1"] -= self.learning_rate * gradients["db1"]
        self.parameters["b2"] -= self.learning_rate * gradients["db2"]

    def train(self, X, Y):
        for epoch in range(self.num_epochs):
            cache, predictions = self.forward_propagation(X)
            loss = self.compute_loss(predictions, Y)
            gradients = self.backward_propagation(X, Y, cache)
            self.update_parameters(gradients)

            if epoch % 10000 == 0:
                print(f"Epoch {epoch}, Loss: {loss}")

        print(f"Final loss after {self.num_epochs} epochs: {loss}")

    def predict(self, X):
        _, predictions = self.forward_propagation(X)
        return (predictions > 0.5) * 1.0

    def accuracy(self, X, Y):
        predictions = self.predict(X)
        accuracy = np.mean(predictions == Y)
        print(f"Accuracy: {accuracy * 100:.2f}%")
        return accuracy

# Model to learn the XOR truth table 
X_train = np.array([[0, 0, 1, 1], [0, 1, 0, 1]]) # XOR input
Y_train = np.array([[0, 1, 1, 0]]) # XOR output

# Define model parameters
num_neurons_hidden_layer = 2 # number of hidden layer neurons (2)
num_input_features = X_train.shape[0] # number of input features (2)
num_output_features = Y_train.shape[0] # number of output features (1)

# Try multiple runs to find the best initialization
best_accuracy = 0
best_perceptron = None
best_run = -1

for i in range(10):  # Run the training multiple times
    print(f"Run {i+1}")
    perceptron = Perceptron(num_input_features, num_neurons_hidden_layer, num_output_features)
    perceptron.train(X_train, Y_train)
    accuracy = perceptron.accuracy(X_train, Y_train)
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_perceptron = perceptron
        best_run = i+1

print(f"Best model is from run {best_run} with an accuracy of {best_accuracy * 100:.2f}%")

# Testing
X_test = np.array([[1, 1, 0, 0], [0, 1, 0, 1]]) # XOR input
predictions = best_perceptron.predict(X_test)
print("Best Model Predictions:", predictions)
