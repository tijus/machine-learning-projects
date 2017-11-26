import numpy as np
import os

class extraction():
    '''
       Name: extraction
       type: class
       objective: to instantiate the object enabling extraction of Synthetic dataset
    '''
    def __init__(self):
        '''
            Constructor used for enabling extraction class instantiation
        '''
        self.directory = ""

    def extractDirectory(self):
        '''
           extractDirectory: this function is used to extract the current working directory
           Input:
                None
           Output:
                Returns the current working directory
        '''
        self.directory = os.path.dirname(__file__)
        return self.directory

    def extractInput(self):
        '''
           extractInput: this function is used to extract input of the Synthetic dataset
           Input:
                None
           Output:
                Returns the path of the Synthetic dataset
        '''
        directory = self.extractDirectory()
        self.path = os.path.join(directory,"data/input.csv")
        return self.path

    def extractOutput(self):
        '''
           extractOutput: this function is used to extract output of the Synthetic dataset
           Input:
                None
           Output:
                Returns the path of the Synthetic dataset
        '''
        directory = self.extractDirectory()
        self.path = os.path.join(directory,"data/output.csv")
        return self.path

    def executeExtraction(self):
        '''
           executeExtraction: this function executes the extraction process of the Synthetic dataset
           Input:
                None
           Output:
                Saves the Matrix obtained from the extraction process in the data directory
        '''
        path = self.extractInput()
        pathOutput = self.extractOutput()

        dataFrameInput = np.genfromtxt(path, delimiter=',')
        dataFrameOutput = np.genfromtxt(pathOutput)

        Matrix = np.column_stack((dataFrameInput,dataFrameOutput))

        np.random.shuffle(Matrix) # random shuffling
        # drop rows with all 0 or nan values
        mask = np.all(np.isnan(Matrix) | np.equal(Matrix, 0), axis=1)
        Matrix = Matrix[~mask] # 0 rows were dropped

        np.save("data/Synthetic/SyntheticMatrix.npy",Matrix)
        np.savetxt('data/Synthetic/SyntheticMatrix.csv',Matrix)


extractObject = extraction()

