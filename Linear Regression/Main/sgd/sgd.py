import numpy as np
import matplotlib.pyplot as mp
import random
import os

class SGD():
    '''
       Name: sgd
       type: class
       objective: This function is used to instantiate SGD class to train data
                  using stochastic gradient descent
    '''
    def __init__(self):
        '''
            Constructor used for enabling SGD class instantiation
        '''
        self.directory = os.path.dirname(__file__)

    def sgd(self,learning_rate,minibatch_size,num_epochs,L2_lambda,design_matrix,output_data):
        '''
           sgd: this function is used to perform training using stochastic gradient descent
           Input:
                learning_rate: learning rate used while training data
                minibatch_size: batch size of the training dataset used in each iteration
                num_epochs: Number of epochs
                L2_lambda: Regularization Factor
                design_matrix: Design matrix of NXM dimension
                output_data: target vector
           Output:
                retuns the weight calculated using stochastic gradient descent
        '''
        N, _ = design_matrix.shape

        weights = np.zeros([1, len(design_matrix[0])])

        lastError=100
        for epoch in range(num_epochs):
            for i in range(N / minibatch_size):
                lower_bound = i * minibatch_size
                upper_bound = min((i+1)*minibatch_size, N)
                Phi = design_matrix[lower_bound : upper_bound, :]
                t = output_data[lower_bound : upper_bound]
                E_D = np.matmul(
                (np.matmul(Phi, weights.T)-t).T,
                Phi
                )
                E = (E_D + L2_lambda * weights) / minibatch_size
                weights = weights - learning_rate * E
            # stopping condition
            if (np.linalg.norm(E) < lastError):
                lastError = np.linalg.norm(E)
            else:
                return weights[len(weights) - 1]
        return weights[len(weights)-1]

sgdObject = SGD()
