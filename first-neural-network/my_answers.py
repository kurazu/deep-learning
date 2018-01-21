import numpy as np


def sigmoid(x):
    """
    Calculate sigmoid
    """
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(sigmoid):
    return sigmoid * (1 - sigmoid)


def identity(x):
    return x


class NeuralNetwork(object):

    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        # Set number of nodes in input, hidden and output layers.
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        # Initialize weights
        self.weights_input_to_hidden = np.random.normal(
            0.0, self.input_nodes**-0.5,
            (self.input_nodes, self.hidden_nodes)
        )

        self.weights_hidden_to_output = np.random.normal(
            0.0, self.hidden_nodes**-0.5,
            (self.hidden_nodes, self.output_nodes)
        )
        self.lr = learning_rate

        self.activation_function = sigmoid
        self.output_activation_function = identity

    def train(self, features, targets):
        ''' Train the network on batch of features and targets.

            Arguments
            ---------

            features: 2D array, each row is one data record,
                each column is a feature
            targets: 1D array of target values

        '''
        n_records = features.shape[0]
        delta_weights_i_h = np.zeros(self.weights_input_to_hidden.shape)
        delta_weights_h_o = np.zeros(self.weights_hidden_to_output.shape)
        for X, y in zip(features, targets):
            # Implement the forward pass function below
            final_outputs, hidden_outputs = self.forward_pass_train(X)
            # Implement the backproagation function below
            delta_weights_i_h, delta_weights_h_o = self.backpropagation(
                final_outputs, hidden_outputs, X, y,
                delta_weights_i_h, delta_weights_h_o
            )
        self.update_weights(delta_weights_i_h, delta_weights_h_o, n_records)

    def forward_pass_train(self, X):
        ''' Implement forward pass here

            Arguments
            ---------
            X: features batch

        '''
        # Forward pass

        # Hidden layer
        # signals into hidden layer
        hidden_inputs = np.matmul(X, self.weights_input_to_hidden)
        # signals from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)

        # Output layer
        # signals into final output layer
        final_inputs = np.matmul(hidden_outputs, self.weights_hidden_to_output)
        # signals from final output layer
        final_outputs = self.output_activation_function(final_inputs)

        return final_outputs, hidden_outputs

    def backpropagation(
        self, final_outputs, hidden_outputs, X, y,
        delta_weights_i_h, delta_weights_h_o
    ):
        ''' Implement backpropagation

            Arguments
            ---------
            final_outputs: output from forward pass
            y: target (i.e. label) batch
            delta_weights_i_h: change in weights from input to hidden layers
            delta_weights_h_o: change in weights from hidden to output layers

        '''
        # ### Implement the backward pass here ####
        # ### Backward pass ###

        # TODO: Output error - Replace this value with your calculations.
        # Output layer error is the difference between desired target and
        # actual output.
        error = np.sum(y - final_outputs)

        # TODO: Backpropagated error terms - Replace these values with your
        # calculations.
        output_error_term = error * 1

        # Transform from column matrix to row vector
        weights_hidden_to_output = self.weights_hidden_to_output.reshape(
            self.weights_hidden_to_output.shape[0]
        )

        hidden_error_term = (
            weights_hidden_to_output *
            output_error_term *
            sigmoid_derivative(hidden_outputs)
        )

        # Weight step (hidden to output)
        delta_weights_h_o += (output_error_term * hidden_outputs)\
            .reshape(hidden_outputs.shape[0], 1)

        # Weight step (input to hidden)
        delta_weights_i_h += hidden_error_term\
            .reshape(1, hidden_error_term.shape[0]) * X.reshape(X.shape[0], 1)

        return delta_weights_i_h, delta_weights_h_o

    def update_weights(self, delta_weights_i_h, delta_weights_h_o, n_records):
        ''' Update weights on gradient descent step

            Arguments
            ---------
            delta_weights_i_h: change in weights from input to hidden layers
            delta_weights_h_o: change in weights from hidden to output layers
            n_records: number of records

        '''
        # update hidden-to-output weights with gradient descent step
        self.weights_hidden_to_output += self.lr * delta_weights_h_o / n_records
        # update input-to-hidden weights with gradient descent step
        self.weights_input_to_hidden += self.lr * delta_weights_i_h / n_records

    def run(self, features):
        ''' Run a forward pass through the network with input features

            Arguments
            ---------
            features: 1D array of feature values
        '''

        final_outputs, _ = self.forward_pass_train(features)
        return final_outputs


#########################################################
# Set your hyperparameters here
##########################################################
iterations = 10000
learning_rate = 0.5
hidden_nodes = 20
output_nodes = 1
