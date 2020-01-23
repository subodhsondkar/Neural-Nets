import random
import time
from matplotlib import pyplot

# ReLU: non-linearity function
def ReLU(x):
    if x > 0:
        return x
    return x / 10
def ReLUdiff(x):
    if x >= 0:
        return 1
    return 0.1

# Initialisations
'''
layersize: number of perceptrons in each layer,
as layersize[i],
where i refers to layer number
'''
layersize = []
for i in range(int(input("Total number of layers: "))):
    print("Enter size of layer ", i, ":", end = " ")
    layersize += [int(input())]
'''
n: learning rate
'''
n = float(input("Enter learning rate: "))
'''
droupout: the probability of any perceptron being deactivated
'''
dropout = float(input("Enter dropout probability: "))
'''
cutoff: the cost should be lower than this number
'''
cutoff = float(input("Enter cutoff cost value: "))
'''
datasize: total size of data
trainsize: size of training data
'''
datasize = int(input("Enter total size of data: "))
trainsize = int(input("Enter size of training data: "))
'''
traindata: input data values for training
trainanswer: output values corresponding to specific input
'''
traindata = []
trainanswer = []
for i in range(datasize):
    traindatatemp = []
    for j in range(layersize[0]):
        traindatatemp += [float(input())]
    traindata += [traindatatemp]
    trainanswertemp = []
    for j in range(layersize[-1]):
        trainanswertemp += [float(input())]
    trainanswer += [trainanswertemp]
span = []
elmin = []
for i in range(layersize[0]):
    tempmax = traindata[0][i]
    tempmin = traindata[0][i]
    for j in range(datasize):
        if tempmax < traindata[j][i]:
            tempmax = traindata[j][i]
        if tempmin > traindata[j][i]:
            tempmin = traindata[j][i]
    span += [tempmax - tempmin]
    elmin += [tempmin]
for i in range(datasize):
    for j in range(layersize[0]):
        traindata[i][j] = (traindata[i][j] - elmin[j]) / span[j]
span = []
elmin = []
for i in range(layersize[-1]):
    tempmax = trainanswer[0][i]
    tempmin = trainanswer[0][i]
    for j in range(datasize):
        if tempmax < trainanswer[j][i]:
            tempmax = trainanswer[j][i]
        if tempmin > trainanswer[j][i]:
            tempmin = trainanswer[j][i]
    span += [tempmax - tempmin]
    elmin += [tempmin]
for i in range(datasize):
    for j in range(layersize[-1]):
        trainanswer[i][j] = (trainanswer[i][j] - elmin[j]) / span[j]
print()
for i in range(datasize):
    print(traindata[i], trainanswer[i])
'''
A: list of perceptrons,
as A[i][j],
where i refers to layer number,
and j refers to perceptron number in the layer.
M: mask layer of perceptrons,
same as A,
but all values are 0,
used to implement dropouts.
'''
An = []
Mn = []
for i in range(layersize[0]):
    An += [0]
    Mn += [1]
A = [An]
M = [Mn]
'''
W: list of weights of edges from one layer to next,
as W[i][j][k],
where i refers to layer number,
j refers to the perceptron in (i + 1)th layer where the edge terminates,
and k refers to the perceptron in the (i)th layer from which the edge originates.
Weights are initialised such that they keep the perceptron value of the next layer,
as the mean of all the perceptron values of current layer.
'''
W = []
'''
B: list of biases for each perceptron in each layer except the first,
as B[i][j],
where i refers to the layer number,
and j refers to the perceptron in the layer to which bias is applied.
'''
B = []
'''
D: contains del/del(A[i][j])*(cost),
as D[i][j],
where i refers to layer number,
and j refers to perceptron number in the layer.
'''
D = []
'''
An, Ann, Bn, Wn, Wnn: temporary lists used to create A, B, W
'''
# Forming the hidden layers and the output layer, the weights of edges, and the biases for the perceptrons
for i in range(len(layersize) - 1):
    An = []
    Bn = []
    Dn = []
    Mn = []
    Wn = []
    for j in range(layersize[i + 1]):
        An += [0]
        Bn += [(random.random())]
        Dn += [0]
        Mn += [1]
        Wnn = []
        for k in range(layersize[i]):
            Wnn += [(random.random()) / layersize[i]]
        Wn += [Wnn]
    A += [An]
    B += [Bn]
    D += [Dn]
    M += [Mn]
    W += [Wn]
'''
cost: the deviation of the predicted answers from the actual answers
'''
testcost = cutoff + 1
steps = 0
stepset = []
start_time = time.time()
costplot = []

# Looping until cost is lower than a specific value
print("Cost after each step:\t")
while testcost >= cutoff:
    cost = 0
    for count in range(trainsize):
        for i in range(layersize[0]):
            A[0][i] = traindata[count][i]
        for i in range(len(layersize) - 2):
            for j in range(layersize[i + 1]):
                M[i + 1][j] = int(random.random() - dropout + 1)
        # Calculating perceptron values according to weights for the hidden layers and the output layer
        for i in range(len(layersize) - 1):
            for j in range(layersize[i + 1]):
                if M[i + 1][j] == 1:
                    A[i + 1][j] = 0
                    for k in range(layersize[i]):
                        A[i + 1][j] += M[i][k] * A[i][k] * W[i][j][k]
                    A[i + 1][j] = ReLU(A[i + 1][j] + B[i][j])
        # Calculating new cost from updated perceptron values
        for i in range(layersize[-1]):
            cost += (trainanswer[count][i] - A[-1][i]) ** 2
        # Updating del/del(A[i][j])*(cost)
        for i in range(layersize[-1]):
            D[-1][i] = n * (A[-1][i] - trainanswer[count][i])
        for i in range(len(layersize) - 2):
            for j in range(layersize[- i - 2]):
                D[- i - 2][j] = 0
                for k in range(layersize[- i - 1]):
                    D[- i - 2][j] += D[- i - 1][k] * ReLUdiff(M[- i - 1][k] * A[- i - 1][k]) * W[- i - 1][k][j]
        # Updating weights with the optimisation of reducing weights inversely proportional to the number of weights that'll affect the cost
        for i in range(len(layersize) - 1):
            for j in range(layersize[- i - 1]):
                for k in range(layersize[- i - 2]):
                    W[- i - 1][j][k] -= D[- i - 1][j] * ReLUdiff(M[- i - 1][j] * A[- i - 1][j]) * A[- i - 2][k] * M[- i - 2][k] / (layersize[- i - 2] ** 2)
        # Updating biases
        for i in range(len(layersize) - 1):
            for j in range(layersize[ - i - 1]):
                B[- i - 1][j] -= D[- i - 1][j] * ReLUdiff(M[- i - 1][j] * A[- i - 1][j])
    print("Average training cost:", int(cost * 1000 / trainsize))
    print("Sum of B:", sum(B[-1]), "Sum of W:", sum(W[-1][-1]))
    # Testing on known and unknown data
    plotx, ploty, plot0, plot1, plot00, plot01 = [], [], [], [], [], []
    testcost = 0
    for testcount in range(datasize):
        for i in range(layersize[0]):
            A[0][i] = traindata[testcount][i]
        for i in range(len(layersize) - 1):
            for j in range(layersize[i + 1]):
                A[i + 1][j] = 0
                for k in range(layersize[i]):
                    A[i + 1][j] += A[i][k] * W[i][j][k]
                A[i + 1][j] = ReLU(A[i + 1][j] + B[i][j])
        for i in range(layersize[-1]):
            testcost += (A[-1][i] - trainanswer[testcount][i]) ** 2
            plotx += [trainanswer[testcount][i]]
            ploty += [A[-1][i]]
    costplot += [testcost]
    stepset += [steps]
    
    for i in range(4):
        plot01 += [i / 2]
        plot00 += [0]
    
    plot1 += [steps]
    plot0 += [0]
    print("Average testing cost:", int(testcost * 1000 / datasize))
    if steps % 1000 == 0:
        pyplot.plot(stepset, costplot)
        pyplot.plot(plot1, plot0)
        pyplot.show()
        pyplot.scatter(plotx, ploty, label  = 'skitscat', color = 'k', s = 5, marker = "o")
        pyplot.plot(plot00, plot01)
        pyplot.plot(plot01, plot01)
        pyplot.plot(plot01, plot00)
        pyplot.xlabel('Actual')
        pyplot.ylabel('Estimated')
        pyplot.show()
    '''
    pyplot.scatter(plotx, ploty, label  = 'skitscat', color = 'k', s = 5, marker = "o")
    pyplot.plot(plot00, plot01)
    pyplot.plot(plot01, plot01)
    pyplot.plot(plot01, plot00)
    pyplot.xlabel('Actual')
    pyplot.ylabel('Estimated')
    pyplot.show()
    '''
    steps += 1
print("\n---SUMMARY---")
print("Final training cost: ", cost)
print("Final testing cost:", testcost)
print("Number of steps: ", steps, "steps")
print("Time of execution: ", time.time() - start_time, " seconds")
print("Number of steps per second: ", steps / (time.time() - start_time), " steps per second")
print("Expected answers:", trainanswer)
#print("Perceptron values:")
#for i in range(len(layersize)):
#    print("\tLayer", i, ":", A[i])
print("Bias values:")
for i in range(len(layersize) - 1):
    print("\tLayer", i + 1, ":", B[i])
#print("Cost gradient wrt perceptron values:")
#for i in range(len(layersize) - 1):
#    print("\tLayer", i + 1, ":", D[i])
print("Weight values:")
for i in range(len(layersize) - 1):
    print("\tConnecting layers", i, "and", i + 1, ":")
    for j in range(layersize[i + 1]):
        print("\t\tReaching node", j, ":", W[i][j])

# Testing
plotx, ploty, plot0, plot1 = [], [], [], []
testcost = 0
for count in range(datasize):
    for i in range(layersize[0]):
        A[0][i] = traindata[count][i]
    for i in range(len(layersize) - 1):
        for j in range(layersize[i + 1]):
            A[i + 1][j] = 0
            for k in range(layersize[i]):
                A[i + 1][j] += A[i][k] * W[i][j][k]
            A[i + 1][j] = ReLU(A[i + 1][j] + B[i][j])
    for i in range(layersize[-1]):
        testcost += (A[-1][i] - trainanswer[count][i]) ** 2
        plotx += [trainanswer[count][i]]
        ploty += [A[-1][i]]
for i in range(3):
    plot1 += [i / 2]
    plot0 += [0]
print("Testing cost:", testcost)
pyplot.scatter(plotx, ploty, label  = 'skitscat', color = 'k', s = 5, marker = "o")
pyplot.plot(plot1, plot1)
pyplot.plot(plot1, plot0)
pyplot.plot(plot0, plot1)
pyplot.xlabel('Actual')
pyplot.ylabel('Estimated')
pyplot.show()

