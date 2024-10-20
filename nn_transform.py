import numpy as np
from nn_train import SimpleNN

def nn_feature_transform(X_train, X_val, hidden_size=64, epochs=100, learning_rate=0.001):
    # Initialize the neural network
    input_size = X_train.shape[1]
    nn = SimpleNN(input_size=input_size, hidden_size=hidden_size, learning_rate=learning_rate)
    
    # Train the neural network using a dummy loss (just feed-forward and backward)
    grad_output = np.random.randn(X_train.shape[0], nn.output_size)  # Dummy gradients for simplicity
    nn.train(X_train, grad_output, epochs=epochs)
    
    # Transform the features
    X_train_transformed = nn.forward(X_train)
    X_val_transformed = nn.forward(X_val)
    
    return X_train_transformed, X_val_transformed
