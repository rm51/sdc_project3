import csv
import cv2
import numpy as np

def resize(data):
	from keras.backend import tf as ktf
	return ktf.image.resize_images(data, (64, 64))

lines = []
with open('twolapsaugmented/driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	for line in reader:
		lines.append(line)

images = []


measurements = []
for line in lines:
	for i in range(3):
		source_path = line[i]
		filename = source_path.split('/')[-1]
		current_path = 'twolapsaugmented/IMG/' + filename
		image = cv2.imread(current_path)
		measurement = float(line[3])
		images.append(image)
		measurements.append(measurement)

#images = resize(images)

augmented_images, augmented_measurements = [], []
for image, measurement in zip(images, measurements):
	augmented_images.append(image)
	augmented_measurements.append(measurement)
	flip_prob = np.random.random()
	# change flip_prob maybe make lower so more images flipped
	# 0.3, goes on dirt road and also rides on the lane
	# 0.5 perofrmed much worse and go stuck coming off the bridge
	if flip_prob >  .3:
	#  when flipping only if measurement greater than .3 for turning result is worse
		augmented_images.append(cv2.flip(image,1))
		augmented_measurements.append(measurement*-1.0)

X_train = np.array(augmented_images)
y_train = np.array(augmented_measurements)



from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers import Convolution2D
from keras.layers.pooling  import MaxPooling2D
from keras.layers.core import Dropout
from keras.models import Model
import matplotlib.pyplot as plt

model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((50,20),(0,0))))
model.add(Lambda(resize))
model.add(Convolution2D(24,(5,5), strides=(2,2),activation="relu"))
model.add(Convolution2D(36,(5,5),strides=(2,2),activation="relu"))
#model.add(Dropout(0.5))
model.add(Convolution2D(48,(5,5),strides=(2,2),activation="relu"))
model.add(Convolution2D(64,(3,3),activation="relu"))
model.add(Convolution2D(64,(3,3),activation="relu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
#model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=7)
history_object = model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=5, verbose=2)
print (history_object.history.keys())
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()
model.save('model.h5')
exit(0)

