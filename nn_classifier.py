# code modified from
#	https://www.tensorflow.org/tutorials/images/classification 

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

import tensorflow as tf

from tensorflow import keras
from keras import layers
from keras.layers import convolutional
from keras.models import Sequential
from sklearn import metrics, preprocessing

import pathlib

from termcolor import colored

CLASS_NAMES = ['Metal', 'Blues', 'Reggae', 'Country', 'Jazz', 'Pop', 'Electronic', 'Rock', 'Hip-Hop', 'Classical']

def spectrogram_cnn():
	print()

	data_dir = "./melspecs/"
	data_dir = pathlib.Path(data_dir)
	print(data_dir)

	batch_size = 32		# not sure what batch size is for
	img_width = 496
	img_height = 369

	# get training data
	train_ds = tf.keras.utils.image_dataset_from_directory(
		data_dir,
		validation_split=0.2,
		subset="training",
		seed=420,
		image_size=(img_height, img_width),
		batch_size=batch_size,
		# labels='inferred',
		# label_mode='categorical',
		# class_names=CLASS_NAMES
	)

	# get validation data
	val_ds = tf.keras.utils.image_dataset_from_directory(
		data_dir,
		validation_split=0.2,
		subset="validation",
		seed=420,
		image_size=(img_height, img_width),
		batch_size=batch_size,
		# labels='inferred',
		# label_mode='categorical',
		# class_names=CLASS_NAMES
	)

	class_names = train_ds.class_names

	# visualizing some of the images
	# plt.figure(figsize=(10, 10))
	# for images, labels in train_ds.take(1):
	#   for i in range(10):
	#     ax = plt.subplot(2, 5, i + 1)
	#     plt.imshow(images[i].numpy().astype("uint8"))
	#     plt.title(class_names[labels[i]])
	#     plt.axis("off")
	# plt.show()

	# configuring for performance
	AUTOTUNE = tf.data.AUTOTUNE

	train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
	val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

	# next, need to normalize data
	#	RGB goes from [0, 255], want to normalize to [0,1]
	normalization_layer = layers.Rescaling(1./255)

	normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
	image_batch, labels_batch = next(iter(normalized_ds))


	# next, create basic Keras model
	#	3 convolution blocks
	#		max pooling layer in each
	#	fully connected layer with 128 units on top of it that is activated by ReLU

	num_classes = len(class_names)

	model = Sequential([
		layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
		layers.Conv2D(16, 3, padding='same', activation='relu'),
		layers.MaxPooling2D(),
		layers.Conv2D(32, 3, padding='same', activation='relu'),
		layers.MaxPooling2D(),
		layers.Conv2D(64, 3, padding='same', activation='relu'),
		layers.MaxPooling2D(),
		layers.Flatten(),
		layers.Dense(128, activation='relu'),
		layers.Dense(num_classes)
	])

	# compile the model
	#	to view training and validation accuracy for each training epoch, pass `metrics` arg to `Model.compile`
	model.compile(
		optimizer='adam',
		loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
		# loss='categorical_crossentropy',
		metrics=['accuracy']
	)

	print()
	# model summary
	model.summary()
	print()

	# train the model
	epochs=10	# TODO: change this (probably to ~10)
	history = model.fit(
		train_ds,
		validation_data=val_ds,
		epochs=epochs
	)

	# print(colored(val_ds, "green"))

	# https://stackoverflow.com/questions/64687375/get-labels-from-dataset-when-using-tensorflow-image-dataset-from-directory 
	# preds = np.array([])
	# y_true =  np.array([])
	# for x, y in val_ds:
	# 	preds = np.concatenate([preds, np.argmax(model.predict(x), axis = -1)])
	# 	y_true = np.concatenate([y_true, np.argmax(y.numpy(), axis=-1)])

	# print(colored(f"y_true {y_true}", "blue"))
	# print(colored(f"preds  {preds}", "green"))
	# print(colored(f"Accuracy for Spectrogram CNN: {round(accuracy_score(y_true, preds), 5)}", "magenta"))
	# confusion_matrix = metrics.confusion_matrix(y_true, preds)
	# confusionmatrix = (confusion_matrix / confusion_matrix.astype(np.float).sum(axis=1))
	# # print(y_test[0:30])
	# cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusionmatrix, display_labels=CLASS_NAMES)
	# cm_display.plot()
	# plt.show()

	# visualizing training results
	acc = history.history['accuracy']
	val_acc = history.history['val_accuracy']

	loss = history.history['loss']
	val_loss = history.history['val_loss']

	epochs_range = range(epochs)

	plt.figure(figsize=(8, 8))
	plt.subplot(1, 2, 1)
	plt.plot(epochs_range, acc, label='Training Accuracy')
	plt.plot(epochs_range, val_acc, label='Validation Accuracy')
	plt.legend(loc='lower right')
	plt.title('Training and Validation Accuracy')

	plt.subplot(1, 2, 2)
	plt.plot(epochs_range, loss, label='Training Loss')
	plt.plot(epochs_range, val_loss, label='Validation Loss')
	plt.legend(loc='upper right')
	plt.title('Training and Validation Loss')
	plt.show()


# couldn't get this above ~22% accuracy with some experimentation
def numerical_cnn():
	print()
	TRAIN_PERCENT = 0.8

	df = pd.read_csv('./data_with_splits.csv')
	rng = np.random.RandomState()

	train = df.sample(frac=TRAIN_PERCENT, random_state=rng)
	test = df.loc[~df.index.isin(train.index)]
	# print(train)
	# print(test)

	train = train.values
	test = test.values

	# print(colored(train.shape, "red"))
	trainX = train[:, 1:].reshape(train.shape[0], 38).astype('float32')
	X_train = trainX / 24912115
	print(colored(X_train.shape, "red"))

	y_train = train[:,0]
	# print()
	# print(X_train)
	# print()
	# print(y_train)

	# print(colored(test.shape, "red"))
	testX = test[:, 1:].reshape(test.shape[0], 38).astype('float32')
	X_test = testX / 24912115
	print(colored(X_test.shape, "red"))

	y_test = test[:,0]

	# lb = preprocessing.LabelBinarizer()
	# y_train = lb.fit_transform(y_train)
	# y_test = lb.fit_transform(y_test)
	label_encoder = preprocessing.LabelEncoder()
	y_train = np.array(label_encoder.fit_transform(y_train))
	y_test = np.array(label_encoder.fit_transform(y_test))

	# K.set_image_dim_ordering('th')
	model = Sequential()
	model.add(layers.Conv1D(16, 3, padding='same', activation='relu', input_shape=(38, 1)))
	model.add(layers.MaxPooling1D())
	model.add(layers.Conv1D(32, 3, padding='same', activation='relu'))
	model.add(layers.MaxPooling1D())
	model.add(layers.Conv1D(64, 3, padding='same', activation='relu'))
	model.add(layers.MaxPooling1D())
	model.add(layers.Conv1D(32, 3, padding='same', activation='relu'))
	model.add(layers.MaxPooling1D())
	model.add(layers.Conv1D(16, 3, padding='same', activation='relu'))
	model.add(layers.MaxPooling1D())
	model.add(layers.Flatten())
	model.add(layers.Dense(16, activation='relu'))
	model.add(layers.Dense(10, activation='softmax'))

	# compile the model
	#	to view training and validation accuracy for each training epoch, pass `metrics` arg to `Model.compile`
	model.compile(
		optimizer='adam',
		loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
		metrics=['accuracy']
	)

	print()
	# model summary
	model.summary()
	print()

	# train the model
	epochs=20	# TODO: change this (probably to ~10)
	fit_model = model.fit(
		X_train,
		y_train,
		epochs=epochs
	)

	score = model.evaluate(X_test, y_test, batch_size=128)
	print()
	print(score)

	# epochs_range = range(epochs)

	# plt.figure(figsize=(8, 8))
	# plt.subplot(1, 2, 1)
	# plt.plot(epochs_range, acc, label='Training Accuracy')
	# plt.plot(epochs_range, val_acc, label='Validation Accuracy')
	# plt.legend(loc='lower right')
	# plt.title('Training and Validation Accuracy')

	# plt.subplot(1, 2, 2)
	# plt.plot(epochs_range, loss, label='Training Loss')
	# plt.plot(epochs_range, val_loss, label='Validation Loss')
	# plt.legend(loc='upper right')
	# plt.title('Training and Validation Loss')
	# plt.show()


# def model_assess(model, title="Default"):
#     model.fit(X_train, y_train)
#     preds = model.predict(X_test)
#     preds = le.inverse_transform(preds)
#     print('Accuracy', title, ':', round(accuracy_score(y_test, preds), 5))
#     confusion_matrix = metrics.confusion_matrix(y_test, preds)
#     confusionmatrix = (confusion_matrix / confusion_matrix.astype(np.float).sum(axis=1))
#     print(y_test[0:30])
#     cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusionmatrix, display_labels=['Metal', 'Blues', 'Reggae', 'Country', 'Jazz', 'Pop', 'Electronic', 'Rock', 'Hip-Hop', 'Classical'])
#     cm_display.plot()
#     plt.show()


if __name__ == "__main__":
	spectrogram_cnn()

