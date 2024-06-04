import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Define the neural network model
class Perceptron(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Perceptron, self).__init__()
        self.hidden = nn.Linear(input_size, hidden_size)
        self.output = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.sigmoid(self.hidden(x))
        x = self.sigmoid(self.output(x))
        return x

# Define a function to train the model
def train_model(model, criterion, optimizer, X_train, Y_train, num_epochs=100000):
    for epoch in range(num_epochs):
        # Convert numpy arrays to torch tensors
        inputs = torch.tensor(X_train, dtype=torch.float32)
        labels = torch.tensor(Y_train, dtype=torch.float32)

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print loss every 10000 epochs
        if (epoch+1) % 10000 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
    
    print(f'Final loss after {num_epochs} epochs: {loss.item():.4f}')

# Define a function to evaluate the model
def evaluate_model(model, X):
    with torch.no_grad():
        inputs = torch.tensor(X, dtype=torch.float32)
        outputs = model(inputs)
        predictions = (outputs > 0.5).float()
    return predictions.numpy()

# Define a function to calculate accuracy
def calculate_accuracy(predictions, labels):
    accuracy = np.mean(predictions == labels)
    return accuracy

# XOR input and output data
X_train = np.array([[0, 0, 1, 1], [0, 1, 0, 1]]).T
Y_train = np.array([[0, 1, 1, 0]]).T

# Define model parameters
input_size = 2
hidden_size = 2
output_size = 1

# Try multiple runs to find the best initialization
best_accuracy = 0
best_model = None
best_run = -1

for i in range(10):
    print(f'Run {i+1}')
    model = Perceptron(input_size, hidden_size, output_size)
    criterion = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    
    train_model(model, criterion, optimizer, X_train, Y_train)
    predictions = evaluate_model(model, X_train)
    accuracy = calculate_accuracy(predictions, Y_train)
    
    print(f'Accuracy: {accuracy * 100:.2f}%')
    
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = model
        best_run = i+1

print(f'Best model is from run {best_run} with an accuracy of {best_accuracy * 100:.2f}%')

# Testing the best model
X_test = np.array([[1, 1, 0, 0], [0, 1, 0, 1]]).T
best_predictions = evaluate_model(best_model, X_test)
print("Best Model Predictions:", best_predictions)
