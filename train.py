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
from keras.layers import Flatten, Dense

# TODO: Open and read driving log
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
print(lines[0])

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

# TODO: Convert data to numpy arrays to be used with Keras
X_train = np.array(images)
y_train = np.array(measurements)

# TODO: Build Neural Network
# Simple Regression Model for Testing 
model = Sequential()
model.add(Flatten(input_shape=(160,320,3)))
model.add(Dense(1))

model.compile(loss = 'mse', optimizer = 'adam')
model.fit(X_train, y_train, validation_split = 0.2, shuffle = True)

# Save model
model.save('model.h5')










