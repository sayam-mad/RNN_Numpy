import numpy as np

# Activation functions and their derivatives
def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1 - np.tanh(x) ** 2

def softmax(x):
    exp_x = np.exp(x - np.max(x))  # stability improvement
    return exp_x / np.sum(exp_x, axis=0)

# RNN class with BPTT
class SimpleRNN:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.001):
        # Initialize weights
        self.Wxh = np.random.randn(hidden_size, input_size) * 0.01  # Input to hidden
        self.Whh = np.random.randn(hidden_size, hidden_size) * 0.01  # Hidden to hidden
        self.Why = np.random.randn(output_size, hidden_size) * 0.01  # Hidden to output
        self.bh = np.zeros((hidden_size, 1))  # Hidden bias
        self.by = np.zeros((output_size, 1))  # Output bias
        
        # Hidden state initialization
        self.h = np.zeros((hidden_size, 1))
        
        # Learning rate for gradient descent
        self.learning_rate = learning_rate
    
    def forward(self, inputs):
        h_prev = self.h
        self.hs = {}  # Store hidden states for each time step
        self.hs[-1] = np.copy(h_prev)  # Initialize h_0
        outputs = []
        
        # Loop through the sequence
        for t in range(len(inputs)):
            x_t = inputs[t].reshape(-1, 1)  # Ensure correct shape
            h_t = tanh(np.dot(self.Wxh, x_t) + np.dot(self.Whh, h_prev) + self.bh)
            y_t = np.dot(self.Why, h_t) + self.by
            output = softmax(y_t)
            outputs.append(output)
            
            # Store hidden state and update for next step
            self.hs[t] = h_t
            h_prev = h_t
        
        self.h = h_prev  # Update hidden state for next sequence
        return outputs

    def loss(self, outputs, targets):
        # Calculate the cross-entropy loss
        total_loss = 0
        for t, output in enumerate(outputs):
            target_idx = np.argmax(targets[t])  # Index of the correct class
            total_loss += -np.log(output[target_idx])
        return total_loss

    def backward(self, inputs, outputs, targets):
        # Initialize gradients for weights and biases
        dWxh, dWhh, dWhy = np.zeros_like(self.Wxh), np.zeros_like(self.Whh), np.zeros_like(self.Why)
        dbh, dby = np.zeros_like(self.bh), np.zeros_like(self.by)
        dh_next = np.zeros_like(self.h)
        
        # Backpropagate through time (from last time step to first)
        for t in reversed(range(len(inputs))):
            dy = np.copy(outputs[t])
            target_idx = np.argmax(targets[t])
            dy[target_idx] -= 1  # Derivative of softmax loss w.r.t. logits
            dWhy += np.dot(dy, self.hs[t].T)
            dby += dy
            
            # Backprop through tanh non-linearity and into hidden state
            dh = np.dot(self.Why.T, dy) + dh_next  # Backprop into hidden state
            dh_raw = dh * tanh_derivative(self.hs[t])  # Backprop through tanh
            dbh += dh_raw
            dWxh += np.dot(dh_raw, inputs[t].reshape(1, -1))
            dWhh += np.dot(dh_raw, self.hs[t-1].T)
            dh_next = np.dot(self.Whh.T, dh_raw)
        
        # Clip gradients to avoid exploding gradients
        for dparam in [dWxh, dWhh, dWhy, dbh, dby]:
            np.clip(dparam, -5, 5, out=dparam)
        
        # Return gradients
        return dWxh, dWhh, dWhy, dbh, dby

    def update_weights(self, dWxh, dWhh, dWhy, dbh, dby):
        # Update weights using gradient descent
        self.Wxh -= self.learning_rate * dWxh
        self.Whh -= self.learning_rate * dWhh
        self.Why -= self.learning_rate * dWhy
        self.bh -= self.learning_rate * dbh
        self.by -= self.learning_rate * dby

    def predict(self, outputs):
        # Get the index with the highest probability for each output
        return [np.argmax(output) for output in outputs]

# Parameters
input_size = 10    # One-hot encoding size (10 different words/phrases)
hidden_size = 15   # Number of hidden units
output_size = 10   # Output size (same as input, predicting the next word)
learning_rate = 0.01  # Learning rate

# Instantiate RNN
rnn = SimpleRNN(input_size, hidden_size, output_size, learning_rate)

# Example long sequence of one-hot encoded words
sequence_length = 20  # A long sequence of 20 steps
vocabulary_size = 10  # Vocabulary size of 10 unique words

# Generate a random input sequence of 20 time steps, where each word is one-hot encoded
inputs = [np.eye(vocabulary_size)[np.random.choice(vocabulary_size)] for _ in range(sequence_length)]

# Target sequence: let's assume the target is the same as the input, shifted by one step
targets = inputs[1:] + [inputs[0]]  # Shifted by one time step

# Training loop
epochs = 100  # Number of iterations to train the model
for epoch in range(epochs):
    # Forward pass
    outputs = rnn.forward(inputs)
    
    # Compute loss
    loss = rnn.loss(outputs, targets)
    
    # Backward pass (BPTT)
    dWxh, dWhh, dWhy, dbh, dby = rnn.backward(inputs, outputs, targets)
    
    # Update weights
    rnn.update_weights(dWxh, dWhh, dWhy, dbh, dby)
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss}")

# Final predictions after training
outputs = rnn.forward(inputs)
predictions = rnn.predict(outputs)

# Print predicted indices
print("Predicted conversation sequence (word indices):", predictions)
