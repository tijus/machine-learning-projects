import numpy as np
import os

class Validation():
    '''
       Name: Validation
       type: class
       objective: This function is used to instantiate object which enables validation of data points
    '''

    def __init__(self):
        '''
           Constructor
        '''
        self.directory = ""

    def predict(self, W, X):
        '''
           predict: this function is used to generated predicted output
           Input:
                W: Weights generated
                X: Input Matrix
                output_data: target vector
           Output:
                retuns the predicted output
        '''
        ans = 0
        for i in range(0, len(W)):
            ans = ans + W[i] * X[i]
        return ans


    def calculateRMSE(self,W,outputs,Validation):
        '''
           predict: This function is used to calculate the root mean square error
           Input:
                W: Weights generated
                outputs: target values given in the input as labels
                Validation: Validation / Testing Matrix
           Output:
                retuns the predicted output
        '''
        SSE = 0
        for i in range(0, len(Validation)-1):
            predicted = self.predict(W,Validation[i,:])
            actual = outputs[i]
            error = actual - predicted
            SSE += (error**2)
        RMSE = (SSE /(len(Validation)))**0.5
        return RMSE

validationObject = Validation()