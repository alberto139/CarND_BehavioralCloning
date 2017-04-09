#train.py
# Author: Alberto Rivera
# Date: April 9, 2017
# Description: train.py is the python code used to extract the images made
# by the Udacity SDC simulator and use them to train a Convolutional Neural Network

import csv
import cv2
import matplotlib.pyplot as plt
import numpy as np
from keras.models import Sequential 
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.convolutional import Convolution2D

# TODO: Open and read driving log
print ("Training the Model....")
lines = []
with open("./data/IMG/driving_log.csv") as csvfile:
	reader = csv.reader(csvfile)
	first_line = True
	for line in reader:
		if first_line:
			header = line
			first_line = False
		else:
			lines.append(line)

# TODO: For each line in the driving log, read the corresponding image files
images = []
measurements = []
# For testing perpuses use: for line in lines[0:10]:
for line in lines[0:2]:
	# TODO: Incorporate side camera images

	filename = line[0].split('/')[-1]
	left_filename = line[1].split('/')[-1]
	right_filename = line[2].split('/')[-1]

	current_path = "./data/IMG/IMG/" + filename
	left_path = "./data/IMG/IMG/" + left_filename
	right_path = "./data/IMG/IMG/" + right_filename

	img_center = cv2.imread(current_path) 
	img_left   = cv2.imread(left_path)
	img_right  = cv2.imread(right_path)
	images.append(img_center)
	images.append(img_left)
	images.append(img_right)
	#images.append(img_right)
	#images.extend([img_center, img_left, img_right])

	# TODO: Get stearing messurements
	angle_center = float(line[3])
	angle_left = angle_center + 0.1
	angle_right = angle_center - 0.1
	measurements.extend([angle_center, angle_left, angle_right]) #Right


#plt.axis("off")
#plt.imshow(cv2.cvtColor(images[0], cv2.COLOR_BGR2RGB))
#plt.show()

# TODO: Augment Data
augmented_images = []
augmented_mesurements = []

for image, measurement in zip(images, measurements):
	augmented_images.append(image)
	augmented_mesurements.append(measurement)
	# Append a flipped version of the image
	augmented_images.append(cv2.flip(image,1))
	augmented_mesurements.append(measurement*-1.0)

# TODO: Convert data to numpy arrays to be used with Keras
print("# of images in the data set: " + str(len(images)))
print("# of images + augmented:     " + str(len(augmented_images)))
X_train = np.array(augmented_images)
y_train = np.array(augmented_mesurements)


# TODO: Build Neural Network
# NVIDIA Architecture
# Simple Regression Model for Testing 
model = Sequential()
# TODO: Preprocess (Normalize and Mean Center)
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape = (160,320,3)))
model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160,320,3)))
model.add(Convolution2D(24, (5, 5), strides=(2, 2), activation="relu"))
model.add(Convolution2D(36, (5, 5), strides=(2, 2), activation="relu"))
model.add(Convolution2D(48, (5, 5), strides=(2, 2), activation="relu"))
model.add(Convolution2D(64, (3, 3), activation="relu"))
model.add(Convolution2D(64, (3, 3), activation="relu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss = 'mse', optimizer = 'adam')
model.fit(X_train, y_train, validation_split = 0.2, shuffle = True, epochs = 5)

# Save model
model.save('model.h5')
print("DONE Training")