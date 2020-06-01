import tensorflow as tf
from tensorflow.keras import Model
import matplotlib.pyplot as plt
from PIL import Image

import numpy as np
import argparse
import pdb
import easydict

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def plot_img(epoch, status, img_input, img_output, img_GT):
	plt.figure(figsize=(9,3))
	
	plt.subplot(1,3,1)
	plt.imshow(img_input)
	plt.xlabel("{}, Input, Epoch: {}".format(status, epoch+1))

	plt.subplot(1,3,1)
	plt.imshow(img_output)
	plt.xlabel("{}, Output, Epoch: {}".format(status, epoch+1))
	
	plt.subplot(1,3,3)
	plt.imshow(img_GT)
	plt.xlabel("{}, GT, Epoch: {}".format(status, epoch+1))
	plt.show()


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--epoch', type=int, default=1000, help='number of epochs')
	parser.add_argument('--batch', type=int, default=1000, help='batch size')
	parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
	parser.add_argument('--log_step', type=int, default=1, help='step for printing log')
	parser.add_argument('--img_show', type=str2bool, default=False, help='Show images while training')
	parser.add_argument('--inference', type=str2bool, default=True, help='Running inference code')
	parser.add_argument('--inf_img', type=str, default='', help='Path of inference input image')
	parser.add_argument('--output', type=str, default='', help='Path of inference result')

	args = parser.parse_args()
	args = easydict.EasyDict({
		'epoch' : 100,
		'batch' : 32,
		'lr' : 0.01,
		'log_step' : 10,
		'img_show' : True,
		'inference' : False,
		'inf_img' : None,
		'output' : None
	})

	epochs = args.epoch
	BATCH_SIZE = args.batch
	lr = args.lr
	log_step = args.log_step

	# Load Data
	cifar10 = tf.keras.datasets.cifar10
	(x_train, _), (x_test, _) = cifar10.load_data()
	x_train, x_test = x_train / 255.0, x_test / 255.0
	y_train, y_test = x_train, y_test

	# Generate Dataset
	train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000).batch(BATCH_SIZE)
	test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(BATCH_SIZE)

	# Generate Model
	model_1 = tf.keras.Sequential([
		tf.keras.layers.GaussianNoise(0.1, input_shape=(28, 28)),
		tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
		tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
		tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
		tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
		tf.keras.layers.Conv2D(3, 3, padding='same', activation=None)
	])

	# Set Loss Function & Optimizer
	loss = tf.keras.losses.MeanSquaredError()
	optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

	train_loss = tf.keras.metrics.Mean(name='train_loss')
	test_loss = tf.keras.metrics.Mean(name='test_loss')

	# Train & Evalute
	for epoch in range(epochs):
		# Training
		for idx, (img_input, img_GT) in enumerate(train_dataset):
			with tf.GradientTape() as tape:
				img_output = model_1(img_input)
				t_loss = loss(img_GT, img_output)
			gradients = tape.gradient(t_loss, model_1.trainable_variables)
			optimizer.apply_gradients(zip(gradients, model_1.trainable_variables))
			
			train_loss(t_loss)
			if (epoch + 1) % args.log_step == 0 and idx == 0 and args.img_show:
				plot_img(epoch, 'training', img_input, img_output, img_GT)


		# Evalution
		for idx, (test_img_input, test_img_GT) in enumerate(test_dataset):
			test_img_output = model_1(test_img_input)
			t_loss = loss(test_img_GT, test_img_output)
			
			test_loss(t_loss)
			if (epoch + 1) % args.log_step == 0 and idx == 0 and args.img_show:
				plot_img(epoch, 'training', img_input, img_output, img_GT)

		if (epoch + 1) % args.log_step == 0:
			print('Epoch: %d => train loss: %.6f, test loss: %.6f'
                % (epoch+1, train_loss.result(), test_loss.result()))
	
	# Inference
	if args.inference:
		inf_img_input = np.array(Image.open(args.inf_img))

		inf_img_output = model_1(inf_img_input)
		plt.figure(figsize=(6,3))
		plt.subplot(1,2,1)
		plt.imshow(inf_img_input)
		plt.xlabel("Inference Input")
		plt.subplot(1,2,2)
		plt.imshow(inf_img_output)
		plt.xlabel("Inference Output")