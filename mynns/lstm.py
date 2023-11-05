
import numpy as np

class LSTMCell:
    def __init__(self, input_size, hidden_size):
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Initialize the weights of the LSTM cell
        self.W = np.random.randn(input_size + hidden_size, 4 * hidden_size)
        self.b = np.zeros((4 * hidden_size,))

    def forward(self, inputs, hidden_state, cell_state):
        # Calculate the gates
        gates = np.matmul(inputs, self.W) + self.b
        i_gate = np.tanh(gates[:, :self.hidden_size])
        f_gate = np.tanh(gates[:, self.hidden_size:2 * self.hidden_size])
        o_gate = np.tanh(gates[:, 2 * self.hidden_size:3 * self.hidden_size])
        g_gate = np.tanh(gates[:, 3 * self.hidden_size:])

        # Calculate the new cell state
        new_cell_state = f_gate * cell_state + i_gate * g_gate

        # Calculate the new hidden state
        new_hidden_state = o_gate * np.tanh(new_cell_state)

        return new_hidden_state, new_cell_state

class LSTMModel:
    def __init__(self, input_size, hidden_size):
        self.lstm_cell = LSTMCell(input_size, hidden_size)

    def forward(self, inputs):
        hidden_state = np.zeros((inputs.shape[0], self.lstm_cell.hidden_size))
        cell_state = np.zeros((inputs.shape[0], self.lstm_cell.hidden_size))

        for input in inputs:
            hidden_state, cell_state = self.lstm_cell.forward(input, hidden_state, cell_state)

        return hidden_state

# Create an LSTM model
lstm_model = LSTMModel(input_size=10, hidden_size=100)

# Generate some input data
inputs = np.random.randn(10, 10)

# Calculate the output of the LSTM model
outputs = lstm_model.forward(inputs)

print(outputs)
