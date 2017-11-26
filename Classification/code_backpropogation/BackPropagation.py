# Taking a simple example of a backpropogation network 
# with input as 0.05 giving output as 0.01 and
# input 2 as 0.1 giving output as 0.99 on a single
# feed forward neural network

from random import *
import math


def main():
    # inputs
    i1 = 0.05
    i2 = 0.1
    
    # outputs
    o1 = 0.01
    o2 = 0.99
    
    # no of nodes in a hidden layer
    n_nodes = 2
    
    # learning rate
    l_rate = 0.5
    
    # number of epochs
    n_epochs = 100000
    
    w_input = defineInputWeights(n_nodes,2)
    w_output = defineOutputWeights(n_nodes,2)
    
    for i in range(n_epochs):

        # Forward Pass
        # calculating net output to output layer form hidden layer
        netH1 = w_input[2]*i1 + w_input[3]*i2 + w_input[0] # w_input[0] bias for node 1
        netH2 = w_input[4]*i1 + w_input[5]*i2 + w_input[1] # w_input[1] bias for node 2

        # applying activation function to net output from the hidden layer
        outH1 = sigmoid(netH1)
        outH2 = sigmoid(netH2)

        #calculating net output from the output layer
        netO1 = w_output[2]*outH1 + w_output[3]*outH2 + w_output[0] # w_input[0] bias for node 1
        netO2 = w_output[4]*outH1 + w_output[5]*outH2 + w_output[1] # w_input[1] bias for node 2

        outO1 = sigmoid(netO1)
        outO2 = sigmoid(netO2)

        error = ((o1-outO1)**2 + (o2-outO2)**2)/2.0

        #########################################
        # Backword Pass (Backpropogating to hidden layer)
        #calculating total error change w.r.t. output
        dEOut1 = (outO1-o1)

        # calculating total output chnge w.r.t. net input
        dONet1 = outO1*(1-outO1)

        # calculating a component of total error change w.r.t weights
        delta = (outO1-o1)*dONet1
        # applying chain rule for back propogation
        dEWeight5 = delta*outH1
        dEWeight6 = delta*outH2
        dEWeight7 = delta*outH1
        dEWeight8 = delta*outH2

        # Updating Weights
        w_output[2]  = w_output[2]-l_rate*dEWeight5
        w_output[3]  = w_output[3]-l_rate*dEWeight6
        w_output[4]  = w_output[4]-l_rate*dEWeight7
        w_output[5]  = w_output[5]-l_rate*dEWeight8

        # print(w_output[2])
        # print(w_output[3])
        # print(w_output[4])
        # print(w_output[5])

        #############################################
        # Backward Pass (Backpropogating to input layer)

        # calculating rate of change of total error w.r.t. weights
        dEOH1 = dEOut1*dONet1*w_output[1]
        dEOH2 = dEOut1*dONet1*w_output[1]
        dEtotalOH1= dEOH1 + dEOH2
        sigma = dEtotalOH1*outH1*(1-outH1)
        
        w_input[2] = w_input[2]-l_rate*(sigma*i1)
        w_input[3] = w_input[3]-l_rate*(sigma*i2)
        w_input[4] = w_input[4]-l_rate*(sigma*i1)
        w_input[5] = w_input[5]-l_rate*(sigma*i2)


        # calulating accuracy at each epoch
        accuracy =1-error
    print (accuracy)

# defining weights of the hidden layer
def defineInputWeights(n_nodes,inputs):
    n_weights = n_nodes*inputs
    weights = []
    #  +n_nodes for bias
    for i in range(n_weights+n_nodes):
        rand = random()
        weights.append(rand)    
    return weights

# defining weights of the output layer
def defineOutputWeights(n_nodes,outputs):
    n_weights = n_nodes*outputs
    weights = []
    # +n_node for bias
    for i in range(n_weights+n_nodes):
        rand = random()
        weights.append(rand)
    return weights

# defining simoid(activation) function
def sigmoid(x):
    return 1 / (1 + math.exp(-x))

if __name__ == '__main__':main()
