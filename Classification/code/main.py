
print("Name: Gautam and Sujit")
print("PersonNO: 50245840 and 50247206")
print("1. Multinomial / Multiclass Logistic Regression")
print("2. Multilayer Neural Network with one hidden layer")
print("3. Convolution Neural Network")
ch = int(input("Please Enter your choice"))
if(ch==1):
    print("Implementing Multinomial / Multiclass Logistic Regression")
    with open('MLogisticRegression.py') as source_file:
        exec (source_file.read())
if(ch==2):
    print("Implementing Multilayer Neural Network with one hidden layer")
    with open('MultiLayerNN.py') as source_file:
        exec (source_file.read())
if(ch==3):
    print("Implementing Convolution Neural Network")
    with open('ConvolutionNN.py') as source_file:
        exec (source_file.read())

