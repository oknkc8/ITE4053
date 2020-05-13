import random
import math
import argparse
import numpy as np
import time
import os
import pdb

parser = argparse.ArgumentParser()
parser.add_argument('--size_m', type=int, default=1000, help='number of train samples')
parser.add_argument('--size_n', type=int, default=100, help='number of test samples')
parser.add_argument('--size_k', type=int, default=100, help='number of iterations')
parser.add_argument('--log_step', type=int, default=1, help='step for printing log')
parser.add_argument('--alpha', type=float, default=0.001, help='learning rate')
parser.add_argument('-initial_zero', action="store_true", help='set initial parameter as zero')
parser.add_argument('-find_best_alpha', action="store_true", help='find alpha for best performance')

args = parser.parse_args()

m = args.size_m # num of train sample
n = args.size_n  # num of evaluation sample
iterations = args.size_k
log_step = args.log_step
alpha = args.alpha  # Hyper Parameter
flag_print = True

W1_initial = np.random.randn(1,2)
b1_initial = np.random.randn(1,1)
W2_initial = np.random.randn(1,1)
b2_initial = np.random.randn(1,1)


"""
Functions for logistic regression for vectorized version
"""
def cross_entropy_loss(y_hat, y):
    a1 = (y * np.log(y_hat))
    a2 = (1 - y) * np.log(1 - y_hat + 1e-10)
    return a1 + a2

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def model(x, W, b):
    return np.dot(W, x) + b


def train_n_test(x_train, y_train, x_test, y_test):

    # Initialize Fucntion Parameters
    W1 = W1_initial
    b1 = b1_initial
    W2 = W2_initial
    b2 = b2_initial

    if args.initial_zero :
        W1 = np.zeros((1,2))
        b1 = np.zeros((1,1))
        W2 = np.zeros((1,1))
        b2 = np.zeros((1,1))

    acc_train = 0
    acc_test = 0
    cost_train = 0
    cost_test = 0

    start_time = time.time()
    if flag_print:
        print("\n\nInitial Function Parameters: ", W1, b1, W2, b2)
        print("\n######### Training #########")
    for iteration in range(iterations):
        # Foward Propagation
        Z1 = model(x_train, W1, b1)
        A1 = sigmoid(Z1)
        Z2 = model(A1, W2, b2)
        A2 = sigmoid(Z2)
        cost = np.sum((-cross_entropy_loss(A2, y_train))) / m

        # Backward Propagation
        dZ2 = A2 - y_train
        dW2 = np.dot(dZ2, A1.T) / m
        db2 = np.sum(dZ2, axis=1, keepdims=True) / m

        dZ1 = np.dot(W2.T, dZ2) * (A1 * (1 - A1))
        dW1 = np.dot(dZ1, x_train.T) / m
        db1 = np.sum(dZ1, axis=1, keepdims=True) / m

        # Calculate Accuracy
        y_hat = A2
        y_hat[y_hat > 0.5] = 1
        y_hat[y_hat <= 0.5] = 0
        acc = np.sum(y_hat == y_train)

        if (iteration + 1) % log_step == 0 and flag_print:
            print("%d iteration => Cost: %f, Training Accuracy: %f%%" % (iteration + 1, cost, acc / m * 100.0))
        acc_train = (acc / m * 100.0)
        cost_train = cost

        # Parameters Update
        W1 = W1 - alpha * dW1
        b1 = b1 - alpha * db1
        W2 = W2 - alpha * dW2
        b2 = b2 - alpha * db2
        
    end_time = time.time()
    train_time = (end_time - start_time) / iterations

    start_time = time.time()
    if flag_print:
        print("\n######### Inference #########")
    Z1 = model(x_test, W1, b1)
    A1 = sigmoid(Z1)
    Z2 = model(A1, W2, b2)
    A2 = sigmoid(Z2)
    cost = np.sum((-cross_entropy_loss(A2, y_test))) / n

    y_hat = A2
    y_hat[y_hat > 0.5] = 1
    y_hat[y_hat <= 0.5] = 0
    acc = np.sum(y_hat == y_test)

    if flag_print:
        print("Cost: %f, Test Accuracy: %f%%" % (cost, acc / n * 100.0))
    acc_test = acc / n * 100.0
    cost_test = cost
    
    end_time = time.time()
    test_time = end_time - start_time

    return train_time, test_time, acc_train, acc_test, cost_train, cost_test


if __name__ == "__main__":
    train_file_name = 'data/train/train_' + str(m) + '_' + str(n) + '.npz'
    test_file_name = 'data/test/test_' + str(m) + '_' + str(n) + '.npz'

    if os.path.exists(train_file_name) == False:
        print('Warning! : No Data File')
        print('Generating Train & Test set file...')
        os.system('python data_Generator.py --size_m ' + str(m) + ' --size_n ' + str(n))
        print('Done!\n')

    train_set = np.load(train_file_name)
    x_train = train_set['x_train']
    y_train = train_set['y_train']

    test_set = np.load(test_file_name)
    x_test = test_set['x_test']
    y_test = test_set['y_test']   

    if args.find_best_alpha :
        # Using Ternary Search
        flag_print = False
        head = 0
        tail = 1.0
        cnt = 0
        best_alpha = 0.001

        print('Finding Best Learning Rate...\n')
        while tail - head > 1e-6:
            p = (2 * head + tail) / 3
            q = (head + 2 * tail) / 3

            alpha = p
            _, _, _, _, p_cost, _ = train_n_test(x_train, y_train, x_test, y_test)
            alpha = q
            _, _, _, _, q_cost, _ = train_n_test(x_train, y_train, x_test, y_test)

            cnt += 1
            print('%d Search: [%.6f, %.6f, %.6f, %.6f] => Cost_p: %.6f, Cost_q: %.6f' % (cnt, head, p, q, tail, p_cost, q_cost))
            if p_cost > q_cost:
                head = p
                best_alpha = q
            elif p_cost <= q_cost:
                tail = q
                best_alpha = p
        
        print('\nBest Learning Rate: %.6f' % best_alpha)
        alpha = best_alpha
        flag_print = True    
    
    T_train, T_test, acc_train, acc_test, cost_train, cost_test = train_n_test(x_train, y_train, x_test, y_test)
    print("\n\n")
    print("######## HYPER-PARAMETERS ########")
    print("num of train sample (m) : %d" % (m))
    print("num of test sample (n) : %d" % (n))
    print("num of iterations (k) : %d" % (iterations))
    print("\n######## TASK 2 RESULT ########")
    print("Training Time : %.6f" % (T_train))
    print("Training Accuracy : %f" % (acc_train))
    print("Training Cost : %f" % (cost_train))
    print("Test Time : %.6f" % (T_test))
    print("Test Accuracy : %f" % (acc_test))
    print("Test Cost : %f" % (cost_test))