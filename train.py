#train.py
# Author: Alberto Rivera
# Date: April 6, 2017
# Description: train.py is the python code used to extract the images made
# by the Udacity SDC simulator and use them to train a Convolutional Neural Network

import csv
import cv2
import matplotlib.pyplot as plt
import numpy as np
from keras.models import Sequential 
from keras.layers import Flatten, Dense, Lambda

# TODO: Open and read driving log
print ("Training the Model")
lines = []
with open("./data/driving_log.csv") as csvfile:
	reader = csv.reader(csvfile)
	first_line = True
	for line in reader:
		if first_line:
			header = line
			first_line = False
		else:
			lines.append(line)
#print(lines[0])

# TODO: For each line in the driving log, read the corresponding image files
images = []
measurements = []
# For testing perpuses use: for line in lines[0:10]:
for line in lines:
	source_path = line[0]
	filename = source_path.split('/')[-1]
	current_path = "./data/IMG/" + filename
	image = cv2.imread(current_path) 
	images.append(image)

	# TODO: Get stearing messurements
	measurement = float(line[3])
	measurements.append(measurement)

#plt.axis("off")
#plt.imshow(cv2.cvtColor(images[0], cv2.COLOR_BGR2RGB))
#plt.show()

# TODO: Augment Data
augmented_images = []
augmented_mesurements = []

for image, measurement in zip(images, measurements):
	augmented_images.append(image)
	augmented_mesurements.append(measurements)
	# Append a flipped version of the image
	augmented_images.append(cv2.flip(iamge,1))
	augmented_mesurements.append(measurement*-1.0)

# TODO: Convert data to numpy arrays to be used with Keras
print("# of images in the data set: " + str(len(images)))
print("# of images + augmented:     " + str(len(augmented_images)))
X_train = np.array(augmented_images)
y_train = np.array(augmented_mesurements)

# TODO: Build Neural Network
# Simple Regression Model for Testing 
model = Sequential()
# TODO: Preprocess (Normalize and Mean Center)
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape = (160,320,3)))
model.add(Flatten())
model.add(Dense(1))

model.compile(loss = 'mse', optimizer = 'adam')
model.fit(X_train, y_train, validation_split = 0.2, shuffle = True)

# Save model
model.save('model.h5')










