import numpy as np
import os
from LETOR import extraction

class PartitionData():
    '''
       Name: PartitionData
       type: class
       objective: This function is used to partition data
    '''

    def __init__(self):
        '''
            Constructor
        '''
        self.directory =""


    def executePartitionData(self,directory, dataSet):
        '''
           executePartitionData: this function is used to perform partitioning of data
           Input:
                L2_Lambda: directory
                design_matrix: dataSet

           Output:
                returns training, testing and validation data
        '''
        path = directory
        Matrix = np.load(path)
        Training, Validation, Testing = np.split(Matrix, [int(0.8 * len(Matrix)), int(0.9 * len(Matrix))])  # split from 0 to 0.8 0.8 to 0.9 and 0.9 ro 1.0

        np.save("data/"+dataSet+"/Training.npy",Training)
        np.save("data/"+dataSet+"/Testing.npy",Testing)
        np.save("data/"+dataSet+"/Validation.npy",Validation)

        return


partitionDataObj = PartitionData()





