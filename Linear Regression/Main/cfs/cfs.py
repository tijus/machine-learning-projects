import numpy as np
import random
import os


class CFS():
    '''
       Name: CFS
       type: class
       objective: This function is used to instantiate CFS class to train data
                  using closed form solution
    '''

    def __init__(self):
        '''
            Constructor used for enabling CFS class instantiation
        '''
        self.directory = os.path.dirname(__file__)

    def cfs(self, L2_lambda, design_matrix, output_data):
        '''
           cfs: this function is used to perform training using closed form solution
           Input:
                L2_Lambda: Regularization Factor
                design_matrix: Design matrix of NXM dimension
                output_data: target vector
           Output:
                retuns the weight calculated from the closed form solution
        '''
        return np.linalg.solve(
        L2_lambda * np.identity(design_matrix.shape[1]) +
        np.matmul(design_matrix.T, design_matrix),
        np.matmul(design_matrix.T, output_data)
        ).flatten()


cfsObject = CFS()
