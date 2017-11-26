import cv2 as cv
import numpy as np
import os
import glob

def readUSPSDataset(folderName,filetype):
    '''
        readUSPSDataser: this function reads image file from the folder
        Input:
            folderName: Name of the directory in which images are located
            filetype: type of the image files
        Output:
            returns array consisting of all image matrix
    '''
    pathToImage =os.path.realpath(folderName)
    imgPath = os.path.join(pathToImage,'*.'+ filetype)
    img = []
    fileList = glob.glob(imgPath)
    for i in range(len(fileList)):
        imagesrc = cv.imread(fileList[i],0)
        normalizedImage = normalizeImage(imagesrc)
        img.append(normalizedImage)
    return np.array(img)


def testImage(folderName, filetype):
    '''
        testImage: this function converts the image file to test iamge
        Input:
            folderName: Name of the directory in which images are located
            filetype: type of the image files
        Output:
            returns array consisting of all test image matrix
    '''
    print("Creating Test Images ..")
    imageArray = readUSPSDataset(folderName,filetype)
    testImageArray = []
    for i in range(len(imageArray)):
        testImageArray.append(imageArray[i][:].flatten())
        saveImage(np.array(testImageArray))
        saveImageLabel(np.array(testImageArray))
    #print("TestImage")
    return np.array(testImageArray)

def saveImage(testImageArray):
    '''
        saveImage: saves the image array in a file
        Input:
            testImageArray: test image array
        Output:
            saves test image to the file
    '''
    np.savetxt('test_image.txt',testImageArray)

def saveImageLabel(testImageArray):
    '''
        saveImageLabes: creates and save image label
        Input:
            testImageArray: test image array
        Output:
            creates test labels and saves it to the file
    '''
    print("Creating Test Labels..")
    testLabel  = []
    label = 10
    for j in range(10):
        label=label-1
        for i in range(int(len(testImageArray)/10)):
                       initiallabel = [0]*10
                       initiallabel[label]=1
                       testLabel.append(initiallabel)
    np.savetxt('test_label.txt',testLabel)

def normalizeImage(imagesrc):
    '''
        normalizeImage: normalizes each image in the Test folder
        Input:
            imagesrc: Image to be normalized
        Output:
            returns normalized image
    '''
    resizedImage = cv.resize(imagesrc,(28,28))
    resizedImage = (np.ones(shape=(28,28))*255) - resizedImage
    normalizedImage = resizedImage/255.0;
    return normalizedImage

testImage("Test","png")
print("End of program .. ")
