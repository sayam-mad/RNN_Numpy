Forward Pass:

The forward function computes hidden states and outputs for each time step in the sequence.
Loss Calculation:

The cross-entropy loss function computes the error between the predicted and target sequences.
Backward Pass (BPTT):

We calculate gradients for each time step, starting from the last and moving backward through time (BPTT). Gradients are calculated for the weight matrices and biases.
We apply gradient clipping to avoid the issue of exploding gradients.
Weight Updates:

After calculating the gradients, the weights are updated using gradient descent with a specified learning rate.
Training Loop:

We iterate through the input sequence multiple times (epochs), updating the weights at each step.
Every 10 epochs, the loss is printed to track the model's performance.
Predictions:

After training, we predict the sequence of words by selecting the word with the highest probability at each time step.
