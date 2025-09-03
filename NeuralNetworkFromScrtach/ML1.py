'''Part 1 of Video - Single Neuron with 3 input nodes
# Inputs come in from the previous layer of the network
input = [4.1, 6.3, 7.3]
# Each link to the previous neurons (each input) has a weight
weights = [1.2, 3.6, 7.4]
# There is a bias assocaited with each neuron
bias = 3

# Output is the value of the neuron at this layer
output = (input[0]*weights[0])+(input[1]*weights[1])+(input[2]*weights[2])+bias

print(output)'''


'''Part 2 of Video - 3 nodes with 4 inputs
# As all nodes are getting the same inputs, the inputs is still only 1 array of the same values
input = [1, 2,3,2.5]
# Now there are 3 neurons so there are 3 arrays of weights - 1 for each neuron
weights1 = [0.2, 0.8, -0.5, 1.0]
weights2 = [0.5, -0.91, 0.26, -0.5]
weights3 = [-0.26, -0.27, 0.17, 0.87]
# Similar to weights - there are 3 neurons so 3 bias values are needed
bias1 = 2
bias2 = 3
bias3 = 0.5

# Output is the value of the neuron at this layer
output = [(input[0]*weights1[0])+(input[1]*weights1[1])+(input[2]*weights1[2])+(input[3]*weights1[3])+bias1, 
          (input[0]*weights2[0])+(input[1]*weights2[1])+(input[2]*weights2[2])+(input[3]*weights2[3])+bias2, 
          (input[0]*weights3[0])+(input[1]*weights3[1])+(input[2]*weights3[2])+(input[3]*weights3[3])+bias3]

print(output)'''

'''Part 3 of Video - 3 nodes with 4 inputs
# As all nodes are getting the same inputs, the inputs is still only 1 array of the same values
input = [1, 2,3,2.5]
# The weights are now ordered into a single weights list for simplicity
weights = [[0.2, 0.8, -0.5, 1.0], [0.5, -0.91, 0.26, -0.5],[-0.26, -0.27, 0.17, 0.87]]
# Bias for each neuron is now also in a list for simplicity
biases = [2,3,0.5]

layer_outputs = [] # Outputs of the current layer

for neuron_weight, neuron_bias in zip(weights,biases):
    neuron_output = 0 # output for the current neuron 
    for n_input,weight in zip(input,neuron_weight):
        neuron_output += n_input*weight # main calculation (input*weight)
    neuron_output+=neuron_bias # Add the neuron bias
    layer_outputs.append(neuron_output) # Save to the output list
    
print(layer_outputs)'''


'''Part 3 of Video - 3 nodes with 4 inputs - Using the Dot product
# As all nodes are getting the same inputs, the inputs is still only 1 array of the same values
import numpy as np

input = [1, 2,3,2.5]
# The weights are now ordered into a single weights list for simplicity
weights = [[0.2, 0.8, -0.5, 1.0], [0.5, -0.91, 0.26, -0.5],[-0.26, -0.27, 0.17, 0.87]]
# Bias for each neuron is now also in a list for simplicity
biases = [2,3,0.5]

output = np.dot(weights, input)+biases
print(output)'''

'''Part 4 of Video - Quick Addition of new layer and multiple inputs
# As all nodes are getting the same inputs, the inputs is still only 1 array of the same values
import numpy as np

input = [[1, 2, 3, 2.5],
     [2.0, 5.0, -1.0, 2.0],
     [-1.5, 2.7, 3.3, -0.8]]
# The weights are now ordered into a single weights list for simplicity
weights = [[0.2, 0.8, -0.5, 1.0], [0.5, -0.91, 0.26, -0.5],[-0.26, -0.27, 0.17, 0.87]]
# Bias for each neuron is now also in a list for simplicity
biases = [2,3,0.5]

weights2 = [[0.1, -0.14, 0.5], [-0.5, 0.12, -0.33],[-0.44, 0.73, -0.13]]
# Bias for each neuron is now also in a list for simplicity
biases2 = [-1,2,-0.5]

lay1_output = np.dot(input,np.array(weights).T)+biases
lay2_output = np.dot(lay1_output,np.array(weights2).T)+biases2

print(lay2_output)'''

'''Part 4 of Video - Object
# As all nodes are getting the same inputs, the inputs is still only 1 array of the same values
import numpy as np

X = [[1, 2, 3, 2.5],
     [2.0, 5.0, -1.0, 2.0],
     [-1.5, 2.7, 3.3, -0.8]]
# The weights are now ordered into a single weights list for simplicity

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.1*(np.random.randn(n_inputs, n_neurons))
        self.biases = np.zeros((1, n_neurons)) # Double brackets are important - Generate as a tuple
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights)+self.biases
        
        
Layer1 = Layer_Dense(4,5)
Layer2 = Layer_Dense(5,2)

Layer1.forward(X)
#print(Layer1.output)
Layer2.forward(Layer1.output)
print(Layer2.output)'''


'''Part 5 of Video - Actovation Functions'''
# As all nodes are getting the same inputs, the inputs is still only 1 array of the same values
import numpy as np
np.random.seed(0)

def spiral_data(points, classes):
    X = np.zeros((points*classes, 2))
    y = np.zeros(points*classes, dtype='uint8')
    for class_number in range(classes):
        ix = range(points*class_number, points*(class_number+1))
        r = np.linspace(0.0, 1, points)  # radius
        t = np.linspace(class_number*4, (class_number+1)*4, points) + np.random.randn(points)*0.2
        X[ix] = np.c_[r*np.sin(t*2.5), r*np.cos(t*2.5)]
        y[ix] = class_number
    return X, y

#X = [[1, 2, 3, 2.5],
#     [2.0, 5.0, -1.0, 2.0],
#     [-1.5, 2.7, 3.3, -0.8]]

X, y = spiral_data(100,3)

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.1*(np.random.randn(n_inputs, n_neurons))
        self.biases = np.zeros((1, n_neurons)) # Double brackets are important - Generate as a tuple
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights)+self.biases
        
class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0,inputs)

Layer1 = Layer_Dense(2,5)
activation1 = Activation_ReLU()

Layer1.forward(X)
activation1.forward(Layer1.output)

print(activation1.output)