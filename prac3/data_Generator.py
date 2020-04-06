import random
import argparse
import numpy as np
import os

parser = argparse.ArgumentParser()
parser.add_argument('--size_m', type=int, default=1000, help='number of train samples')
parser.add_argument('--size_n', type=int, default=100, help='number of test samples')


def generate(m, n):
    if os.path.exists('data/') == False:
        os.mkdir('data')

    x1_train = []
    x2_train = []
    y_train = []
    for i in range(m):
        x1_train.append(random.randint(-2,2))
        x2_train.append(random.randint(-2,2))

        if x1_train[-1] * x1_train[-1] > x2_train[-1]:
            y_train.append(1)
        else:
            y_train.append(0)

    x_train = np.array((x1_train, x2_train))
    y_train = np.array(y_train)

    if os.path.exists('data/train/') == False:
        os.mkdir('data/train')

    file_name = 'train_' + str(m) + '_' + str(n)
    np.savez('data/train/' + file_name, x_train=x_train, y_train=y_train)


    x1_test = []
    x2_test = []
    y_test = []
    for i in range(n):
        x1_test.append(random.randint(-2,2))
        x2_test.append(random.randint(-2,2))

        if x1_test[-1] * x1_test[-1] > x2_test[-1]:
            y_test.append(1)
        else:
            y_test.append(0)
    
    x_test = np.array((x1_test, x2_test))
    y_test = np.array(y_test)

    if os.path.exists('data/test/') == False:
        os.mkdir('data/test')

    file_name = 'test_' + str(m) + '_' + str(n)
    np.savez('data/test/' + file_name, x_test=x_test, y_test=y_test)
    

if __name__ == "__main__":
    args = parser.parse_args()

    m = args.size_m # num of train sample
    n = args.size_n  # num of evaluation sample

    generate(m, n)