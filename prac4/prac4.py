import tensorflow as tf

import random
import argparse
import numpy as np
import time
import os
import pdb

parser = argparse.ArgumentParser()
parser.add_argument('--m', type=int, default=1000, help='number of train samples')
parser.add_argument('--n', type=int, default=100, help='number of test samples')
parser.add_argument('--epoch', type=int, default=1000, help='number of epochs')
parser.add_argument('--batch', type=int, default=1000, help='batch size')
parser.add_argument('--loss', type=str, default='BCE', help='kind of loss function')
parser.add_argument('--optimizer', type=str, default='SGD', help='kind of optimizer')
parser.add_argument('--log_step', type=int, default=1, help='step for printing log')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')

def data_generate(m, n):
    if os.path.exists('data/') == False:
        os.mkdir('data')

    x1_train = []
    x2_train = []
    y_train = []
    for i in range(m):
        x1_train.append(random.uniform(-2,2))
        x2_train.append(random.uniform(-2,2))

        if x1_train[-1] * x1_train[-1] > x2_train[-1]:
            y_train.append(1)
        else:
            y_train.append(0)

    x_train = np.array((x1_train, x2_train), dtype='float32')
    y_train = np.array(y_train, dtype='float32')

    if os.path.exists('data/train/') == False:
        os.mkdir('data/train')

    file_name = 'train_' + str(m) + '_' + str(n)
    np.savez('data/train/' + file_name, x_train=x_train, y_train=y_train)


    x1_test = []
    x2_test = []
    y_test = []
    for i in range(n):
        x1_test.append(random.uniform(-2,2))
        x2_test.append(random.uniform(-2,2))

        if x1_test[-1] * x1_test[-1] > x2_test[-1]:
            y_test.append(1)
        else:
            y_test.append(0)
    
    x_test = np.array((x1_test, x2_test), dtype='float32') 
    y_test = np.array(y_test, dtype='float32')

    if os.path.exists('data/test/') == False:
        os.mkdir('data/test')

    file_name = 'test_' + str(m) + '_' + str(n)
    np.savez('data/test/' + file_name, x_test=x_test, y_test=y_test)

if __name__ == "__main__":
    args = parser.parse_args()

    m = args.m # num of train sample
    n = args.n  # num of evaluation sample
    epochs = args.epoch
    BATCH_SIZE = args.batch
    SHUFFLE_BUFFER_SIZE = int(m / 10)
    LOSS_FUNCTION = args.loss
    OPTIMIZER = args.optimizer
    lr = args.lr
    log_step = args.log_step

    # Load Data
    train_file_name = 'data/train/train_' + str(m) + '_' + str(n) + '.npz'
    test_file_name = 'data/test/test_' + str(m) + '_' + str(n) + '.npz'

    if os.path.exists(train_file_name) == False:
        print('Warning! : No Data File')
        print('Generating Train & Test set file...')
        data_generate(m, n)
        print('Done!\n')

    train_set = np.load(train_file_name)
    x_train = train_set['x_train'].transpose()
    y_train = train_set['y_train'].transpose()

    test_set = np.load(test_file_name)
    x_test = test_set['x_test'].transpose()
    y_test = test_set['y_test'].transpose()


    # Generate Dataset
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))

    train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
    test_dataset = test_dataset.batch(BATCH_SIZE)


    # Genearte Model
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(3, activation='sigmoid'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    # Set Loss Function & Optimizer
    if LOSS_FUNCTION == 'BCE':
        loss = tf.keras.losses.BinaryCrossentropy()
    elif LOSS_FUNCTION == 'MSE':
        loss = tf.keras.losses.MeanSquaredError()
    if OPTIMIZER == 'SGD':
        optimizer = tf.keras.optimizers.SGD(learning_rate=lr)
    elif OPTIMIZER == 'RMS':
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=lr)
    elif OPTIMIZER == 'Adam':
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    test_loss = tf.keras.metrics.Mean(name='test_loss')


    print("######## SETTING ########")
    print("num of train sample (m) : %d" % (m))
    print("num of test sample (n) : %d" % (n))
    print("num of epochs : %d" % (epochs))
    print("learning rate : %f" % (lr))
    print("batch size : %d" % (BATCH_SIZE))
    print("Loss Function : %s" % (LOSS_FUNCTION))
    print("Optimizer : %s" % (OPTIMIZER))

    print("\n")
    print("######## TRAINING ########")

    # Train & Test
    for epoch in range(epochs):
        train_acc = 0
        test_acc = 0

        start_time = time.time()
        for x, y in train_dataset:
            with tf.GradientTape() as tape:
                y_hat = model(x)
                t_loss = loss(y, y_hat)
            gradients = tape.gradient(t_loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

            train_loss(t_loss)
                      
            y_hat = y_hat.numpy()
            y_hat[y_hat > 0.5] = 1.0
            y_hat[y_hat <= 0.5] = 0.0
            train_acc += np.sum(np.expand_dims(y.numpy(), -1) == y_hat)
        end_time = time.time()
        train_time = end_time - start_time

        start_time = time.time()
        for test_x, test_y in test_dataset:
            y_hat = model(test_x)
            t_loss = loss(test_y, y_hat)

            test_loss(t_loss)
            y_hat = y_hat.numpy()
            y_hat[y_hat > 0.5] = 1
            y_hat[y_hat <= 0.5] = 0
            test_acc += np.sum(np.expand_dims(test_y.numpy(), -1) == y_hat)
        end_time = time.time()
        test_time = end_time - start_time


        if (epoch + 1) % args.log_step == 0:
            print('Epoch: %d => train loss: %.6f, train acc: %.3f, train time: %.6f, test loss: %.6f, test acc: %.3f, test time %.6f'
                % (epoch+1, train_loss.result(), train_acc/m*100, train_time, test_loss.result(), test_acc/n*100, test_time))