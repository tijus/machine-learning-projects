import numpy as np
import math as mt
import scipy
from scipy.stats import norm as nm
from scipy.stats import multivariate_normal as mn
import pandas as pd
from pandas import  ExcelWriter
from  pandas import ExcelFile
import matplotlib.pyplot as plt

'''
   This function reads the data from the Excel File
   Inputs: fileName - name of the Excel file
           sheetName - name of the sheet in the Excel File           
   Output: dataFrame - DataFrame Object
'''
def readExcelFile(fileName, sheetName):
    dataFrame = pd.read_excel(fileName, sheetname=sheetName, parse_cols="C:F")
    return dataFrame

'''
    This function allocates data from the particular column to the data array.
    Inputs: df - dataFrame object
            columnName - name of the columns in the sheet
            dataFrame - DataFrame Object    
    Output: dataArray - contains an array of data that is fetched from the dataFrame   
'''
def allocateData(dataFrame,columnName):
    dataArray = []
    for i in dataFrame.index:
        data = round(dataFrame[columnName][i],1)
        if mt.isnan(data) == False:
            dataArray.append(data)
    return dataArray


'''
    This function calculates the standard distribution (mean, variance and standard deviation)
    and outputs it in the form of an array
    Inputs: dataFrame - dataFrame object
            columnName - name of the columns in the sheet
    Output: resultArray - An array containing mean, variance and standard deviation of the
            sample distribution   
'''
def calculate(dataFrame,columnName):
    resultArray = []
    dArray = allocateData(dataFrame,columnName)
    resultArray.append(np.mean(dArray))
    resultArray.append(np.var(dArray))
    resultArray.append(np.std(dArray))
    return resultArray

'''
    This function calculates the 4X4 covariance matrix
    Inputs: dataFrame - dataFrame object
            columnNames - an array of all the columns from the dataFrame
    Output: a 4X4 covariance matrix
'''
def calculateCovariance(dataFrame, columnNames):
    dArray = [];
    for i in range(len(columnNames)):
        dArray.append(allocateData(dataFrame, columnNames[i]))
    verticalStack = np.vstack(dArray)
    covMat = np.cov(verticalStack,ddof=0)
    return covMat

'''
    This function calculates the 4X4 correlation matrix
    Inputs: dataFrame - dataFrame object
            columnNames - an array of all the columns from the dataFrame
    Output: a 4X4 correlation matrix
'''
def calculateCorrelationCoeff(dataFrame, columnNames):
    dArray = [];
    for i in range(len(columnNames)):
        dArray.append(allocateData(dataFrame, columnNames[i]))
    verticalStack = np.vstack(dArray)
    corrcoeffMat = np.corrcoef(verticalStack)
    return corrcoeffMat

'''
    This function calculates the log likelihood (a scalar value) for 
    independent variables.
    Inputs: dataFrame - dataFrame object
            columnNames - an array of all the columns from the dataFrame
    Output: loglikelihood of the data
'''
def calculateLikelihood(dataFrame, columnNames):
    logLikelihood = 0
    data = allocateData(dataFrame, columnNames[0])
    # Standard Distribution Array
    stdist = []
    for i in range(len(columnNames)):
        stdist.append(calculate(dataFrame, columnNames[i]))

    for i in range(0,len(data),1):
        likelihood = 0
        for j in range(0,len(columnNames),1):
            likelihood += nm.logpdf(dataFrame[columnNames[j]][i],(stdist[j])[0],(stdist[j])[2])
        logLikelihood +=likelihood
    return logLikelihood

'''
    This function calculates the log likelihood (a scalar value) for 
    Multivariate functions.
    Inputs: dataFrame - dataFrame object
            columnNames - an array of all the columns from the dataFrame
            covarianceMatrix - Covariance Matrix
    Output: loglikelihood of the data
'''
def calculateMultivariateLikelihood(dataFrame, columnNames, covarianceMatrix):
    logLikelihood = 0
    dArray = []
    # Standard Distribution Array
    stdist = []
    mean = []

    for i in range((len(columnNames))):
        dArray.append(allocateData(dataFrame, columnNames[i]))

    for i in range(len(columnNames)):
        stdist.append(calculate(dataFrame, columnNames[i]))

    for i in range(len(columnNames)):
        mean.append((stdist[i])[0])

    for i in range(0,len(dArray[0]),1):
        data = []
        for j in range(0,len(columnNames),1):
            data.append((dArray[j])[i])
        likelihood = mn.logpdf(data, mean, covarianceMatrix, allow_singular=True)
        logLikelihood += likelihood
    return logLikelihood

'''
    This function plots the bar graph of the given correlation matrix
    Inputs: correlationMatrix - correlationMatrix
    Output: a bar graph plot
'''
def plotBarGraph(correlationMatrix):
    list_correlation = []
    for i in range(0,4,1):
        for j in range(0,4,1):
            if i!=j:
                list_correlation.append(round(correlationMatrix[i,j],3))
    correlationPair = list(set(list_correlation))
    y = correlationPair[::-1]
    N = len(y)
    x = range(N)
    width = 1 / 1.5
    plt.bar(x, y, width, color="blue")
    plt.show()

'''
    This function plots the bar graph of the given correlation matrix
    Inputs: correlationMatrix - correlationMatrix
    Output: a correlation matrix graph plot
'''
def correlationMatrixPlot(correlationMatrix):

    columnNames = ["CS Score", "Research", "BasePay", "Tuition"]
    # plot correlation matrix
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(correlationMatrix, vmin=-1, vmax=1)
    fig.colorbar(cax)
    ticks = np.arange(0, len(columnNames), 1)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(columnNames)
    ax.set_yticklabels(columnNames)
    plt.show()

'''
    This function plots the bar graph of the given correlation matrix
    Inputs: correlationMatrix - correlationMatrix
    Output: a scatter plot matrix
'''
def scatterPlotMatrix(dataFrame):
    columnNames = ["CS Score", "Research", "BasePay", "Tuition"]
    pd.plotting.scatter_matrix(dataFrame)
    plt.show()

## Function calls and implementation of the program starts here ..
dataFrame = readExcelFile("DataSet/university data.xlsx","university_data")
columns = dataFrame.columns;
columnName = [];
result = [];
for i in range(0,len(columns),1):
    columnName.append(columns[i])

for i in range(0,len(columns),1):
    result.append(calculate(dataFrame,columnName[i]))

print "mu1 = %.3f" %(result[0])[0]
print "mu2 = %.3f" %(result[1])[0]
print "mu3 = %.3f" %(result[2])[0]
print "mu4 = %.3f" %(result[3])[0]

print "var1 = %.3f" %(result[0])[1]
print "var2 = %.3f" %(result[1])[1]
print "var3 = %.3f" %(result[2])[1]
print "var4 = %.3f" %(result[3])[1]

print "sigma1 = %.3f" %(result[0])[2]
print "sigma2 = %.3f" %(result[1])[2]
print "sigma3 = %.3f" %(result[2])[2]
print "sigma4 = %.3f" %(result[3])[2]

## Suppresses the scientific notation and makes the precision of numpy
## array to be 3.
np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
covarianceMatrix = calculateCovariance(dataFrame, columnName)
print "covarianceMat = ", np.around(covarianceMatrix,decimals=3)

correlationMatrix = calculateCorrelationCoeff(dataFrame, columnName)
print "correlationMat = ", correlationMatrix.round(3)

logLikelihood= calculateLikelihood(dataFrame,columnName)
print "logLikelihood = ", logLikelihood.round(3)

logLikelihoodMultivariate = calculateMultivariateLikelihood(dataFrame,columnName,covarianceMatrix)
print "logLikelihoodMultivariate = ", logLikelihoodMultivariate.round(3)

# Uncomment these to see the various plots
'''
plotBarGraph(correlationMatrix)
correlationMatrixPlot(correlationMatrix)
scatterPlotMatrix(dataFrame)
'''
