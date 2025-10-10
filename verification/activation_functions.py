import numpy as np

class Activation_functions:
    def linear(x):
        ''' y = f(x) It returns the input as it is'''
        # print(' Activation function is Linear(x). It returns the input as it is ' + str(x) + '.')
        return x

    def relu(x):
        ''' y = relu(x) It returns the input as it is'''
        if x > 0:
            # print(' Activation function is Relu(x). It returns the input as it is ' + str(x) + '.')
            return x
        else:
            # print(' Activation function is Relu(x). It returns the input as it is ' + str(0) + '.')
            return 0

    def sigmoid(x):
        ''' It returns 1/(1+exp(-x)). where the values lies between zero and one '''
        # print(' Activation function is Sigmoid(x). It returns the input as it is ' + str(1 / (1 + np.exp(-x))) + '.')
        return 1 / (1 + np.exp(float(-x)))

    def tanh(x):
        ''' It returns the value (1-exp(-2x))/(1+exp(-2x)) and the value returned will be lies in between -1 to 1.'''
        # print(' Activation function is Tanh(x). It returns the input as it is ' + str(np.tanh(x)) + '.')
        return np.tanh(float(x))
# class activation_functions:
#     # def __init__(self,input,activation_type):
#     #     self.input = input
#     #     self.activation_type = activation_type
#
#
#
#
#     def relu(self,a):
#         if a > 0:
#             return a
#         else:
#             return 0
#
#
# if __name__ == "__main__":
#     linear(0.8)
#     relu(-0.5)
#     sigmoid(0.5)
#     tanh(0.5)
