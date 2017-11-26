'''
The main.py defines the entry point of the solution to the given problem
(as defined in the project description and report).
The execution of this problem is called externally from the other program.
'''

#Defining the UBIT Name
UBitName = "sujitjit"

#Defining the UB Person Number
personNumber = 50247206

#Printing the author's UBIT Name
print "UBitName = ",UBitName

#Printing the author's UB Person Number
print "personNumber = ",personNumber

'''
The function execFile gives a call to the execution file
of the problem.
This function returns an error if the execution is interrupted or is unspecified.
'''
execfile("exec.py")
