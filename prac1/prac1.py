import random
import math
import argparse
import numpy as np
import time

parser = argparse.ArgumentParser()
parser.add_argument('--version', type=str, default='element_wise', required=True, help='version of logistic regression (element_wise, vectorized, compare, find_alpha)')
parser.add_argument('--size_m', type=int, default=1000, help='number of train samples')
parser.add_argument('--size_n', type=int, default=100, help='number of test samples')
parser.add_argument('--size_k', type=int, default=100, help='number of iterations')
parser.add_argument('--log_step', type=int, default=1, help='step for printing log')
parser.add_argument('--alpha', type=float, default=0.001, help='learning rate')
parser.add_argument('-initial_zero', action="store_true", help='set initial parameter as zero')

args = parser.parse_args()

version = args.version
m = args.size_m # num of train sample
n = args.size_n  # num of evaluation sample
iterations = args.size_k
log_step = args.log_step
alpha = args.alpha  # Hyper Parameter

"""
Functions for logistic regression for element-wise version
"""
def cross_entropy_loss_ew(y_hat, y):
    a1 = -(y * math.log(y_hat + 1e-10))
    a2 = (1 - y) * math.log(1 - y_hat + 1e-10)
    return a1 + a2

def sigmoid_ew(z):
    return 1 / (1 + math.exp(-z + 1e-10))

def model_ew(x1, x2, w1, w2, b):
    return sigmoid_ew(w1*x1 + w2*x2 + b)

"""
Functions for logistic regression for vectorized version
"""
def cross_entropy_loss_v(y_hat, y):
    a1 = -(y * np.log(y_hat))
    a2 = (1 - y) * np.log(1 - y_hat + 1e-10)
    return a1 + a2

def sigmoid_v(z):
    return 1 / (1 + np.exp(-z + 1e-10))

def model_v(x, W, b):
    return sigmoid_v(np.dot(W, x) + b)


def preprocess():
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

    x1_test = []
    x2_test = []
    y_test = []
    for i in range(n):
        x1_test.append(random.randint(-10,10))
        x2_test.append(random.randint(-10,10))

        if x1_test[-1] + x2_test[-1] > 0:
            y_test.append(1)
        else:
            y_test.append(0)
    
    return x1_train, x2_train, y_train, x1_test, x2_test, y_test


def train_element_wise(x1_train, x2_train, y_train, x1_test, x2_test, y_test):
    # Initialize Fucntion Parameters
    w1 = random.random()
    w2 = random.random()
    b = random.random()
    if args.initial_zero :
        w1 = w2 = b = 0

    acc_train = 0
    acc_test = 0

    start_time = time.time()
    print("\n\nInitial Function Parameters w1: %.6f, w2: %.6f, b: %.6f"%(w1, w2, b))
    for iteration in range(iterations):
        cost = 0
        dw1 = dw2 = db = 0
        acc = 0

        # Step 2-1
        if (iteration + 1) % log_step == 0:
            print("\n", iteration+1, "iteration Parameters w1: %.6f, w2: %.6f, b: %.6f"%(w1, w2, b))

        if (iteration + 1) % log_step == 0:
            print("######### Training #########")    
        # Step 2-2
        for i in range(m):
            y_hat = model_ew(x1_train[i], x2_train[i], w1, w2, b)
            cost += -cross_entropy_loss_ew(y_hat, y_train[i])
            
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
        if (iteration + 1) % log_step == 0:
            print("Cost: %f" % cost)

        # Step 2-4
        if (iteration + 1) % log_step == 0:
            print("Training Accuracy: %f%%" % (acc / m * 100.0))
        acc_train += (acc / m * 100.0)

        acc = 0
        cost = 0

        if (iteration + 1) % log_step == 0:
            print("######## Evaluation ########")
        # Step 2-3
        for i in range(n):
            y_hat = model_ew(x1_test[i], x2_test[i], w1, w2, b)
            cost += -cross_entropy_loss_ew(y_hat, y_test[i])
            
            if (y_hat > 0.5 and y_test[i] == 1) or (y_hat <= 0.5 and y_test[i] == 0):
                acc = acc +1

        cost /= n
        if (iteration + 1) % log_step == 0:
            print("Cost: %f" % cost)

        # Step 2-5
        if (iteration + 1) % log_step == 0:
            print("Evaluation Accuracy: %f%%" % (acc / n * 100.0))
        acc_test += (acc / n * 100.0)

        # Parameters Update
        w1 = w1 - alpha * dw1
        w2 = w2 - alpha * dw2
        b = b - alpha * db
    
    end_time = time.time()

    return end_time - start_time, acc_train / iterations, acc_test / iterations


def train_vectorized(x1_train, x2_train, y_train, x1_test, x2_test, y_test):
    # Convert to Numpy array
    x_train = np.array((x1_train, x2_train))
    y_train = np.array(y_train)
    x_test = np.array((x1_test, x2_test))
    y_test = np.array(y_test)

    # Initialize Fucntion Parameters
    W = np.random.rand(2)
    b = random.random()
    if args.initial_zero :
        W = np.zeros(2)
        b = 0

    acc_train = 0
    acc_test = 0

    start_time = time.time()
    print("\n\nInitial Function Parameters w1: %.6f, w2: %.6f, b: %.6f"%(W[0], W[1], b))
    for iteration in range(iterations):
        # Step 2-1
        if (iteration + 1) % log_step == 0:
            print("\n", iteration+1, "iteration Parameters w1: %.6f, w2: %.6f, b: %.6f"%(W[0], W[1], b))


        if (iteration + 1) % log_step == 0:
            print("######### Training #########")    
        # Step 2-2
        y_hat = model_v(x_train, W, b)
        cost = np.sum((-cross_entropy_loss_v(y_hat, y_train))) / m
        if (iteration + 1) % log_step == 0:
            print("Cost: %f" % cost)

        dz = y_hat - y_train
        dW = np.dot(x_train, dz) / m
        db = np.sum(dz) / m

        y_hat[y_hat > 0.5] = 1
        y_hat[y_hat <= 0.5] = 0
        acc = np.sum(y_hat == y_train)

        # Step 2-4
        if (iteration + 1) % log_step == 0:
            print("Training Accuracy: %f%%" % (acc / m * 100.0))
        acc_train += (acc / m * 100.0)


        if (iteration + 1) % log_step == 0:
            print("######## Evaluation ########")
        # Step 2-3
        y_hat = model_v(x_test, W, b)
        cost = np.sum((-cross_entropy_loss_v(y_hat, y_test))) / m
        if (iteration + 1) % log_step == 0:
            print("Cost: %f" % cost)
        
        y_hat[y_hat > 0.5] = 1
        y_hat[y_hat <= 0.5] = 0
        acc = np.sum(y_hat == y_test)

        # Step 2-5
        if (iteration + 1) % log_step == 0:
            print("Evaluation Accuracy: %f%%" % (acc / n * 100.0))
        acc_test += (acc / n * 100.0)

        # Parameters Update
        W = W - alpha * dW
        b = b - alpha * db
    
    end_time = time.time()

    return end_time - start_time, acc_train / iterations, acc_test / iterations


if __name__ == "__main__":
    x1_train, x2_train, y_train, x1_test, x2_test, y_test = preprocess()

    if version == 'compare':
        T_ew, acc_train_ew, acc_test_ew = train_element_wise(x1_train, x2_train, y_train, x1_test, x2_test, y_test)
        T_v, acc_train_v, acc_test_v = train_vectorized(x1_train, x2_train, y_train, x1_test, x2_test, y_test)
        print("\n\n")
        print("######## RESULT ########")
        print("num of train sample (m) : %d" % (m))
        print("num of test sample (n) : %d" % (n))
        print("num of iterations (k) : %d" % (iterations))
        print("Alpha : %.6f" % (alpha))
        print("initial_zero :", args.initial_zero)
        print("\n------ Element-wise ------")
        print("Running Time : %.6f" % (T_ew))
        print("Training Accuracy : %.6f" % (acc_train_ew))
        print("Test Accuracy : %.6f" % (acc_test_ew))
        print("\n------ Vectorized ------")
        print("Running Time : %.6f" % (T_v))
        print("Training Accuracy : %.6f" % (acc_train_v))
        print("Test Accuracy : %.6f" % (acc_test_v))

    elif version == 'find_alpha':
        head = 0
        tail = 1
        ACC = 0
        cnt = 0
        while head < tail:
            alpha = (head + tail) / 2.0
            
            T, acc_train, acc_test = train_vectorized(x1_train, x2_train, y_train, x1_test, x2_test, y_test)
            print("######## RESULT ########")
            print("Alpha : %.6f" % (alpha))
            print("Running Time : %.6f" % (T))
            print("Training Accuracy : %.6f" % (acc_train))
            print("Test Accuracy : %.6f" % (acc_test))

            cnt = cnt + 1
            if acc_train == 100:
                break

            if ACC < acc_train:
                ACC = acc_train
                head = alpha
            else:
                tail = alpha
        
        print("\n\n Best value of alpha : %.6f" % (alpha))
        print("Num of search : %d" % (cnt))
    
    elif version == 'element_wise':
        T, acc_train, acc_test = train_element_wise(x1_train, x2_train, y_train, x1_test, x2_test, y_test)
        print("\n\n")
        print("######## RESULT ########")
        print("num of train sample (m) : %d" % (m))
        print("num of test sample (n) : %d" % (n))
        print("num of iterations (k) : %d" % (iterations))
        print("\n------ Element-wise ------")
        print("Running Time : %.6f" % (T))
        print("Training Accuracy : %.6f" % (acc_train))
        print("Test Accuracy : %.6f" % (acc_test))
    
    else:
        T, acc_train, acc_test = train_vectorized(x1_train, x2_train, y_train, x1_test, x2_test, y_test)
        print("\n\n")
        print("######## RESULT ########")
        print("num of train sample (m) : %d" % (m))
        print("num of test sample (n) : %d" % (n))
        print("num of iterations (k) : %d" % (iterations))
        print("\n------ Vectorized ------")
        print("Running Time : %.6f" % (T))
        print("Training Accuracy : %.6f" % (acc_train))
        print("Test Accuracy : %.6f" % (acc_test))