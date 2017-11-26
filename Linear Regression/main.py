import numpy as np
from LETOR import extraction as letorextract
from Main.executables import partition, Generate_Phi, Validation
from Main.cfs import cfs
from Main.sgd import sgd
from Synthetic import extraction as synextract
import os

#extract data
def extractLETORData():
    letorextract.extractobject.executeExtraction()
    return

#extract synthetic data
def extractSyntheticData():
    synextract.extractObject.executeExtraction()
    return

#partition data
def partitionSamples(dataSet):
    if dataSet == "LETOR":
        partition.partitionDataObj.executePartitionData("data/LETOR/LETORMatrix.npy", "LETOR")
    elif dataSet == "Synthetic":
        partition.partitionDataObj.executePartitionData("data/Synthetic/SyntheticMatrix.npy", "Synthetic")
    return

# generate phi
def generatePhi(M,dataSet):
    Training = np.load('data/'+dataSet+'/Training.npy')
    Training = Training[: , 0:len(Training[0])-1] # drop the last column as its output label
    Centers,labels = Generate_Phi.phiObject.getCenters(M,Training)
    Spread = Generate_Phi.phiObject.generateSpread(M,Centers,labels,Training)
    Phi_Training = Generate_Phi.phiObject.compute_design_matrix(Training,Centers,Spread)

    Validation = np.load('data/'+dataSet+'/Validation.npy')
    Validation = Validation[:,0:len(Validation[0])-1]
    Phi_Validation = Generate_Phi.phiObject.compute_design_matrix(Validation,Centers,Spread)

    Testing = np.load('data/'+dataSet+'/Testing.npy')
    Testing = Testing[:,0:-1]
    Phi_Testing = Generate_Phi.phiObject.compute_design_matrix(Testing,Centers,Spread)

    np.save('data/'+dataSet+'/Phi_Training.npy', Phi_Training)
    np.save('data/'+dataSet+'/Phi_Validation.npy', Phi_Validation)
    np.save('data/'+dataSet+'/Phi_Testing.npy', Phi_Testing)

    return

# training cfs
def trainDataCFS(dataSet):
    if dataSet == "LETOR":
        generatePhi(101,dataSet)
    elif dataSet == "Synthetic":
        generatePhi(32,dataSet)
    Training = np.load("data/"+dataSet+"/Phi_Training.npy")
    initial_set = np.load("data/"+dataSet+"/Training.npy")
    outputs = initial_set[:, -1]

    if dataSet == "LETOR":
        W = cfs.cfsObject.cfs(0.0001, Training, outputs)
    elif dataSet == "Synthetic":
        W = cfs.cfsObject.cfs(0.00005, Training, outputs)
    np.save("data/"+dataSet+"/W_cfs.npy", W)
    np.savetxt("data/"+dataSet+"/W_cfs.txt", W)
    return

#training sgd
def trainDataSGD(dataSet):
    generatePhi(16,dataSet)
    Phi_Training = np.load("data/"+dataSet+"/Phi_Training.npy")
    initial_set = np.load("data/"+dataSet+"/Training.npy")
    outputs = initial_set[:,-1]
    #print(Training[1:10,:]);

    '''learning_rate = 0.001
    mini_batchsize =len(Phi_Training)/10
    num_epochs = 100
    L2_lambda = 0.1'''
    design_matrix = Phi_Training

    if dataSet == "LETOR":
        W = sgd.sgdObject.sgd(0.0001,1000,100,0.001,design_matrix,outputs);
    elif dataSet == "Synthetic":
        W = sgd.sgdObject.sgd(0.001, 100, 2000, 0.1, design_matrix, outputs);
    np.save("data/"+dataSet+"/W_sgd.npy",W)
    np.savetxt("data/"+dataSet+"/W_sgd.txt",W)
    return


#validation / testing
def testData(dataSet,method):
    W = np.load("data/"+dataSet+"/W_"+method+".npy");

    intital_data = np.load("data/"+dataSet+"/Validation.npy")
    outputs = intital_data[:,-1]
    testData = np.load("data/"+dataSet+"/Phi_Validation.npy")

    return Validation.validationObject.calculateRMSE(W, outputs, testData)

print("1. Start from Extraction")
print("2. Start from Testing")
ch = int(raw_input("Please Enter your choice\t"))
if ch == 1:
    print("Extracting LETOR data ...")
    extractLETORData()
    print("Partitioning LETOR data ...")
    partitionSamples("LETOR")
    print("Training LETOR data using CFS ...")
    trainDataCFS("LETOR")
    print("Training LETOR data using SGD ..")
    trainDataSGD("LETOR")
    print("Extracting Synthetic data ...")
    extractSyntheticData()
    print("Partitioning Synthetic data ...")
    partitionSamples("Synthetic")
    print("Training Synthetic data using CFS ...")
    trainDataCFS("Synthetic")
    print("Training Synthetic data using SGD ..")
    trainDataSGD("Synthetic")

print "RMSE(LETOR/CFS): ",testData("LETOR","cfs")
print "RMSE(LETOR/SGD): ",testData("LETOR","sgd")
print "RMSE(Synthetic/CFS): ",testData("Synthetic","cfs")
print "RMSE(Synthetic/SGD): ",testData("Synthetic","sgd")






