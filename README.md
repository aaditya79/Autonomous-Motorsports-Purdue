# haar_cascade.py

Classification rules help to differentiate between the background pixels and object pixels. 
The Haar Cascade (feature based classifiers) pre-trained OpenCV file will be used to implement cascading windows.

# image_norm1.py

The first model normalizes a color image with min-max normalization and range [0,1]. 

# image_norm2.py

The second model normalizes a color image with binary output values of 0 or 1.

# neural_network.py

Programmed a neural network that predicts the value of output based on pre-trained input and output tables. 
The difference between the neuron’s predicted and expected output is calculated and passed through the 
error derivative formula and Sigmoid function 20,000 times to determine an output value. 

Example:

Pre-trained Inputs: [0,0,0], [1,1,1], [1,0,0], [0,1,1]
Pre-trained Outputs: [0, 1, 0, 1]
Classification Rule: The second element in the pre-trained input is the output. 
User Input: [0,1,0]
Expected Output: [1]
Actual Output: [0.99490848]
