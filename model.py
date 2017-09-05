import csv
import cv2
import numpy as np
from PIL import Image

def resize(data):
	from keras.backend import tf as ktf
	return ktf.image.resize_images(data, (64, 64))

def process_image(image):
	filename = source_path.split('/')[-1]
	current_path = './aug31/IMG/' + filename
	image = cv2.imread(current_path)
	image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	return image_rgb

lines = []
with open('./aug31/driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	for line in reader:
		lines.append(line)


images = []


measurements = []
for line in lines:
	steering_center = float(line[3])
	correction = 0.15 
	steering_left = steering_center + correction
	steering_right = steering_center - correction

	source_path = line[0]
	filename = source_path.split('/')[-1]
	current_path = './aug31/IMG/' + filename
	img_center = process_image(line[0])
	img_left = process_image(line[1])
	img_right = process_image(line[2])
	images.extend((img_center, img_left, img_right))
	measurements.extend((steering_center, steering_left, steering_right))

augmented_images, augmented_measurements = [], []
for image, measurement in zip(images, measurements):
	augmented_images.append(image)
	augmented_measurements.append(measurement)
	flip_prob = np.random.random()
	if flip_prob >  .3 or abs(measurement) < 0.05:
		pass
	else:
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
model.add(Dropout(0.7))
model.add(Convolution2D(48,(5,5),strides=(2,2),activation="relu"))
model.add(Convolution2D(64,(3,3),activation="relu"))
model.add(Dropout(0.7))
model.add(Convolution2D(64,(3,3),activation="relu"))
model.add(Dropout(0.3))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
history_object = model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=15, verbose=2)
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
