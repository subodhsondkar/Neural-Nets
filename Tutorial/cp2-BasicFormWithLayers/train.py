import time
import random

# relu: non-linearity function
def relu(x):
    if x > 0:
        return x
    return 0
def relu_diff(x):
    if x >= 0:
        return 1
    return 0

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
cutoff: the cost should be lower than this number
'''
cutoff = float(input("Enter cutoff cost value: "))
'''
Y: actual answer array,
should ideally match with A[-1]
'''
Y = []
for i in range(layersize[-1]):
    print("Enter expected answer of perceptron ", i, ":", end = " ")
    Y += [float(input())] 
'''
A: list of perceptrons,
as A[i][j],
where i refers to layer number,
and j refers to perceptron number in the layer.
'''
An = []
for i in range(layersize[0]):
    An += [0.1 * (i % 2 + 1)]
A = [An]
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
    Wn = []
    for j in range(layersize[i + 1]):
        An += [0]
        Bn += [0]
        Dn += [0]
        Wnn = []
        for k in range(layersize[i]):
            Wnn += [random.random()]
        Wn += [Wnn]
    A += [An]
    B += [Bn]
    D += [Dn]
    W += [Wn]
'''
cost: the deviation of the predicted answers from the actual answers
'''
cost = 0
for i in range(layersize[-1]):
    cost += (Y[i] - A[-1][i]) ** 2

steps = 0

start_time = time.time()

# Looping until cost is lower than a specific value
print("Cost after each step:\t")
while cost >= cutoff:
    # Calculating perceptron values according to weights for the hidden layers and the output layer
    for i in range(len(layersize) - 1):
        for j in range(layersize[i + 1]):
            A[i + 1][j] = 0
            for k in range(layersize[i]):
                A[i + 1][j] += A[i][k] * W[i][j][k]
            A[i + 1][j] = relu(A[i + 1][j] + B[i][j])
    # Calculating new cost from updated perceptron values
    cost = 0
    for i in range(layersize[-1]):
        cost += (Y[i] - A[-1][i]) ** 2
    # Updating del/del(A[i][j])*(cost)
    for i in range(layersize[-1]):
        D[-1][i] = n * (A[-1][i] - Y[i])
    for i in range(len(layersize) - 2):
        for j in range(layersize[- i - 2]):
            D[- i - 2][j] = 0
            for k in range(layersize[- i - 1]):
                D[- i - 2][j] += D[- i - 1][k] * relu_diff(A[- i - 1][k]) * W[- i - 1][k][j]
    # Updating weights with the optimisation of reducing weights inversely proportional to the number of weights that'll affect the cost
    for i in range(len(layersize) - 1):
        for j in range(layersize[- i - 1]):
            for k in range(layersize[- i - 2]):
                W[- i - 1][j][k] -= D[- i - 1][j] * relu_diff(A[- i - 1][j]) * A[- i - 2][k]
    # Updating biases
    for i in range(len(layersize) - 1):
        for j in range(layersize[ - i - 1]):
            B[- i - 1][j] -= D[- i - 1][j] * relu_diff(A[- i - 1][j])
    steps += 1
    #print(int(cost * 10000), end = " | ")
    #if steps % 25 == 0:
        #print()
print("\n\n---SUMMARY---")
print("Final cost:", cost)
print("Number of steps:", steps, "steps")
print("Time of execution:", time.time() - start_time, "seconds")
print("Number of steps per second:", steps / (time.time() - start_time), "steps per second")
print("Expected answers:", Y)
print("Perceptron values:")
for i in range(len(layersize)):
    print("\tLayer", i, ":", A[i])
print("Bias values:")
for i in range(len(layersize) - 1):
    print("\tLayer", i + 1, ":", B[i])
print("Cost gradient wrt perceptron values:")
for i in range(len(layersize) - 1):
    print("\tLayer", i + 1, ":", D[i])
print("Weight values:")
for i in range(len(layersize) - 1):
    print("\tConnecting layers", i, "and", i + 1, ":")
    for j in range(layersize[i + 1]):
        print("\t\tReaching node", j, ":", W[i][j])

