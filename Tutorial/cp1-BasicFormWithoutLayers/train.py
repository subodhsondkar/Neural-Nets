import time

# relu: Non-linearity function
def relu(x):
    if x > 0:
        return x
    return 0
def relu_diff(x):
    if x >= 0:
        return 1
    return 0

# y: expected answer; n: learning rate; s: size of input layer; A: input layer; Y: calculated answer; b: bias; W: weights of edgesi; cost: the lower the cost, the better the accuracy; steps: number of iterations to complete the process; cutoff: cost should be less than this value
y = float(input("Enter expected answer: "))
n = float(input("Enter learning rate: "))
s = int(input("Enter size of input layer: "))
# Scanning input layer for testing and initialising weights and bias to 0
A = []
Y = 0
b = 0
W = []
print("Enter perceptron values: ")
for i in range(s):
    A += [float(input())]
    W += [0]
cost = 1
steps = 0
cutoff = float(input("Enter cutoff on cost: "))
start_time = time.time()
# Looping until cost is reduced enough
while cost >= cutoff:
    # z: storing the summation of multiplication of activations with corresponding weights
    z = 0
    for i in range(s):
        z += A[i] * W[i]
    Y = relu(z + b)
    cost = (Y - y) ** 2
    # Changing weights for better cost (gradient descent)
    for i in range(s):
        W[i] -= n * (Y - y) * relu_diff(Y) * A[i]
    # Changing bias for better cost (gradient descent)
    b -= n * (Y - y) * relu_diff(Y)
    steps += 1
print("\n---SUMMARY---")
print("Final cost:", cost)
print("Number of steps:", steps, "steps")
print("Time of execution:", time.time() - start_time, "seconds")
print("Number of steps per second:", steps / (time.time() - start_time), "steps per second")
print("Expected answer:", y)
print("Perceptron values:")
print("\tLayer 1:", A)
print("\tLayer 2:", Y)
print("Bias value:", b)
print("Weight values:", W)

