import numpy as np
import scipy.cluster as spc
import os

class GeneratePhi():
    '''
       Name: GeneratePhi
       type: class
       objective: This function is used to Generate design matrix
    '''

    def __init__(self):
        '''
         Constructor
        '''
        self.directory = ""

    def getCenters(self, M, Data):
        '''
           getCenter: Get Centers
           Input:
                M: number of centers
                Data: data
           Output:
                returns kmean centers and labels
        '''
        Centers,labels = spc.vq.kmeans2(Data,M,minit='points',missing='warn')
        Centers = Centers[:, np.newaxis, :]
        return Centers,labels

    def generateSpread(self, M, centers,label,data):
        '''
           getSpread: Get Spreads
           Input:
                M: number of centers
                data: data
                centers: kmean centers
                label: kmean labels
           Output:
                returns spread
        '''
        spread = []
        for i in range(0,M):
            cluster = np.zeros(len(data[0]));
            for j in range(0,len(data)):
                if(label[j]==i):
                    cluster = np.vstack((cluster,data[j]))
            cluster = cluster[1:len(cluster)]
            sigma = np.multiply(np.cov(cluster.T),np.identity(len(data[0])))*0.1 # 46 x 46
            spread.append(sigma)
        return np.array(spread)

    def compute_design_matrix(self, X, centers, spreads):
        '''
           compute_design_matrix: computes the design matrix
           Input:
                X: Input
                centers: kmean centers
                spreads: Spreads of the gaussian radial basis function
           Output:
                returns design matrix
        '''
        # use broadcast
        X = X[np.newaxis,:,:]
        basis_func_outputs = np.exp(
        np.sum(
        np.matmul(X - centers, spreads) * (X - centers),
        axis=2
        ) / (-2)
        ).T
        # insert ones to the 1st col
        return np.insert(basis_func_outputs, 0, 1, axis=1)

phiObject = GeneratePhi()
