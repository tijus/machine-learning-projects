import numpy as np
import os

class extraction():
    '''
       Name: extraction
       type: class
       objective: to instantiate the object enabling extraction of LETOR dataset
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

    def extractPath(self):
        '''
           extractPath: this function is used to extract path of the LETOR dataset
           Input:
                None
           Output:
                Returns the path of the LETOR dataset
        '''
        dir = extraction()
        directory = dir.extractDirectory()
        self.path = os.path.join(directory,"data/MQ2007/Querylevelnorm.txt")
        return self.path

    def executeExtraction(self):
        '''
           executeExtraction: this function executes the extraction process of the LETOR dataset
           Input:
                None
           Output:
                Saves the Matrix obtained from the extraction process in the data directory
        '''
        extractionobject = extraction()
        path = extractionobject.extractPath()
        F = open(path, 'r')
        counter = 0
        # Matrix will have 0 45 : features and 46 : label

        while(1):
            current_line = F.readline()  # A is used for reading elements from F file
            if(current_line == ''):
                break

            words = current_line.split()
            current_row = [None] * 47
            current_row[46] = int(words[0])  # this is the output label

            words = words[2:48] # these are the 46 features

            for i in range(0,46) :
                words[i] = words[i].split(':')  # split with respect to ':'
                current_row[i] = float(words[i][1]) # after splitting wrt ':' , get the part to its right

            if (counter == 0):
                Matrix = np.array(current_row)
                counter = counter+1
            else :
                Matrix = np.vstack ( [Matrix, current_row] )
                counter = counter + 1

        np.random.shuffle(Matrix) # random shuffling
        # drop rows with all 0 or nan values
        mask = np.all(np.isnan(Matrix) | np.equal(Matrix, 0), axis=1)
        Matrix = Matrix[~mask] # 3 rows were dropped

        np.save("data/LETOR/LETORMatrix.npy",Matrix)
        np.savetxt('data/LETOR/LETORMatrix.csv',Matrix)
        return


extractobject = extraction()

