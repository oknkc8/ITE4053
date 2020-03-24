import random
import math

m = 1000 # num of train sample
n = 100  # num of evaluation sample
iterations = 100

alpha = 0.0001  # Hyper Parameter

def cross_entroy_loss(y_hat, y):
    a1 = -(y * math.log(y_hat))
    a2 = (1 - y) * math.log(1 - y_hat)
    return a1 + a2

def sigmoid(z):
    return 1 / (1 + math.exp(-z))

def model(x1, x2, w1, w2, b):
    return sigmoid(w1*x1 + w2*x2 + b)


# Step 1. Preprocess

x1_train = []
x2_train = []
y_train = []
for i in range(m):
    x1_train.append(random.randint(-10,10))
    x2_train.append(random.randint(-10,10))

    if x1_train[-1] + x2_train[-1] > 0:
        y_train.append(1)
    else:
        y_train.append(0)

x1_test=[]
x2_test=[]
y_test=[]
for i in range(m):
    x1_test.append(random.randint(-10,10))
    x2_test.append(random.randint(-10,10))

    if x1_test[-1] + x2_test[-1] > 0:
        y_test.append(1)
    else:
        y_test.append(0)


# Initialize Fucntion Parameters
w1 = random.random()
w2 = random.random()
b = random.random()
# w1 = 0
# w2 = 0
# b = 0
print("Initial Function Parameters w1: %.6f, w2: %.6f, b: %.6f"%(w1, w2, b))


for iteration in range(iterations):
    cost = 0
    dw1 = 0
    dw2 = 0
    db = 0
    acc = 0

    # Step 2-1
    print("\n", iteration, "iteration Parameters w1: %.6f, w2: %.6f, b: %.6f"%(w1, w2, b))


    print("######### Training #########")    
    # Step 2-2
    for i in range(m):
        y_hat = model(x1_train[i], x2_train[i], w1, w2, b)
        cost += -cross_entroy_loss(y_hat, y_train[i])
        
        if (y_hat > 0.5 and y_train[i] == 1) or (y_hat <= 0.5 and y_train[i] == 0):
            acc = acc +1

        dz = y_hat - y_train[i]
        dw1 += x1_train[i] * dz
        dw2 += x2_train[i] * dz
        db += dz
    
    cost /= m
    dw1 /= m
    dw2 /= m
    db /= m

    # Step 2-4
    print("Training Accuracy: %f%%" % (acc / m * 100.0))

    acc = 0

    print("######## Evaluation ########")
    # Step 2-3
    for i in range(n):
        y_hat = model(x1_test[i], x2_test[i], w1, w2, b)
        
        if (y_hat > 0.5 and y_test[i] == 1) or (y_hat <= 0.5 and y_test[i] == 0):
            acc = acc +1

    # Step 2-5
    print("Evaluation Accuracy: %f%%" % (acc / n * 100.0))

    # Parameters Update
    w1 = w1 - alpha * dw1
    w2 = w2 - alpha * dw2
    b = b - alpha * db
