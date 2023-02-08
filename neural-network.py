import numpy as np
class nn():
   def __init__(self):
       np.random.seed(0) # Seed for random number
       self.synaptic_weights = 3 * np.random.random((3, 1)) - 3 # 3x1 array from [-3,0)
   def model(self, inputs):
       arr = (self.curve(np.dot(inputs.astype(float), self.synaptic_weights)))
       return arr
   def training(self, trainedIn, trainedOut, n):
       i = 0
       while (i < n):
           adjustments = np.dot(trainedIn.T, self.deriv(self.model(trainedIn)) * (trainedOut - self.model(trainedIn)))
           self.synaptic_weights = self.synaptic_weights + adjustments
           i += 1;           
   def deriv(self, num):
       return num * (1 - num) # Derivative for Sigmoid function
   def curve(self, num):
       return 1 / (1 + np.exp(-num)) # Sigmoid function


if __name__ == "__main__":
   neuralNet = nn()
   neuralNet.training(np.array([[0,0,0], [1,1,1], [1,0,0], [0,1,1]]), np.array([[0, 1, 0, 1]]).T, 20000) # Array 1 - training input table, Array 2 - training output table, Number - iterations
   print("Array: ")
   in1, in2, in3 = str(input()), str(input()), str(input()) # Testing input
   print("Guess: ")
   print(neuralNet.model(np.array([in1, in2, in3]))) # Testing output
