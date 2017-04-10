#train.py
# Author: Alberto Rivera
# Date: April 9, 2017
# Description: train.py is the python code used to extract the images made
# by the Udacity SDC simulator and use them to train a Convolutional Neural Network
import os
import sklearn
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import csv
import cv2
import matplotlib.pyplot as plt
import numpy as np
from keras.models import Sequential 
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.convolutional import Convolution2D


# generator()
# input: samples - is an array of lines from the csv file
#		 btach_size
def generator(samples, batch_size=128):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            correction = 0.5
            for batch_sample in batch_samples:
            	# Center Image
                name = "./data/IMG/IMG/"+batch_sample[0].split('/')[-1]
                center_image = cv2.imread(name)
                center_angle = float(batch_sample[3])
                images.append(center_image)
                images.append(cv2.flip(center_image,1))
                angles.append(center_angle)
                angles.append(center_angle*-1.0)
                # Left Image
                name = "./data/IMG/IMG/"+batch_sample[1].split('/')[-1]
                left_image = cv2.imread(name)
                images.append(left_image)
                images.append(cv2.flip(left_image,1))
                angles.append(center_angle + correction)
                angles.append((center_angle + correction) *-1.0)
                # Right Image
                name = "./data/IMG/IMG/"+batch_sample[2].split('/')[-1]
                right_image = cv2.imread(name)
                images.append(right_image)
                images.append(cv2.flip(right_image,1))
                angles.append(center_angle - correction)
                angles.append((center_angle - correction) *-1.0)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield (X_train, y_train)


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
samples = lines
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

#plt.axis("off")
#plt.imshow(cv2.cvtColor(images[0], cv2.COLOR_BGR2RGB))
#plt.show()

print("Number of Samples: " + str(len(samples)))
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

# TODO: Build Neural Network
# NVIDIA Architecture
# Simple Regression Model for Testing 
model = Sequential()
# TODO: Preprocess (Normalize and Mean Center)
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape = (160,320,3)))
model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160,320,3)))
model.add(Convolution2D(16, 3, 3, subsample=(2, 2), activation="relu"))
model.add(Convolution2D(32, 3, 3, subsample=(2, 2), activation="relu"))
model.add(Convolution2D(64, 5, 5, subsample=(2, 2), activation="relu"))
model.add(Convolution2D(64, 3, 3, activation="relu"))
model.add(Convolution2D(64, 3, 3, activation="relu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss = 'mse', optimizer = 'adam')
#model.fit(X_train, y_train, validation_split = 0.2, shuffle = True, epochs = 5, batch_size=32)
model.fit_generator(train_generator, samples_per_epoch= len(train_samples), validation_data=validation_generator, nb_val_samples=len(validation_samples), nb_epoch=3)

# Save model
model.save('model.h5')
print("DONE Training")










