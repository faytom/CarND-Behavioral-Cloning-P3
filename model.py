import csv
import cv2
import numpy as np
import sklearn
import keras
from keras.models import Sequential
from keras.layers import Lambda
from keras.layers import Flatten, Dense, Convolution2D, MaxPooling2D, Dropout, Cropping2D, ELU

samples = []
with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    firstLineFlag = True
    for line in reader:
        if not firstLineFlag:
            samples.append(line)
        else:
            # does not append first label line
            firstLineFlag = False

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

def generator(samples, resize_shape=None, batch_size=128):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        samples = sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, int(batch_size)):
            batch_samples = samples[offset:offset+int(batch_size)]

            images = []
            angles = []
            for batch_sample in batch_samples:

                for selection in range(3):
                    name = './data/IMG/'+batch_sample[selection].split('/')[-1]
                    image = cv2.imread(name)
                    angle = 0
                    if selection == 0:
                        angle = float(batch_sample[3]) 
                    elif selection == 1:
                        angle = float(batch_sample[3]) + 0.2
                    else:
                        angle = float(batch_sample[3]) - 0.2

                    if resize_shape is not None:
                        image = cv2.resize(image, resize_shape)

                    images.append(image)
                    angles.append(angle)

                    image, angle = augmentFlip(image, angle)

                    images.append(image)
                    angles.append(angle)

            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

def augmentFlip(image, angle):
    return cv2.flip(image,1), angle * -1

def LeNet(input_shape=(32,32,3)):
    model = Sequential()

    model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))
    model.add(Cropping2D(cropping=((70,25),(0,0))))
    
    model.add(Convolution2D(6, 5, 5, name='convolution_1', subsample=(1, 1), border_mode="valid"))
    model.add(ELU())
    model.add(MaxPooling2D(pool_size=(2, 2), border_mode='valid'))
    
    model.add(Convolution2D(16, 5, 5, name='convolution_2', subsample=(1, 1), border_mode="valid"))
    model.add(ELU())
    model.add(MaxPooling2D(pool_size=(2, 2), border_mode='valid'))

    model.add(Flatten())

    model.add(Dropout(0.5))
    model.add(Dense(120, name='hidden1'))
    model.add(ELU())
    
    model.add(Dropout(0.5))
    model.add(Dense(84, name='hidden2'))
    model.add(ELU())
    
    model.add(Dense(10, name='hidden3'))
    model.add(ELU())

    model.add(Dense(1, name='steering_angle'))
        
    return model


def NvidiaModel(input_shape=(160,320,3)):
    # model start here
    model = Sequential()

    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=input_shape))
    model.add(Cropping2D(cropping=((70,25),(0,0))))

    model.add(Convolution2D(24, 5, 5, border_mode='valid', activation='relu', subsample=(2, 2)))
    model.add(Convolution2D(36, 5, 5, border_mode='valid', activation='relu', subsample=(2, 2)))
    model.add(Convolution2D(48, 5, 5, border_mode='valid', activation='relu', subsample=(2, 2)))
    model.add(Convolution2D(64, 3, 3, border_mode='valid', activation='relu', subsample=(1, 1)))
    model.add(Convolution2D(64, 3, 3, border_mode='valid', activation='relu', subsample=(1, 1)))

    model.add(Flatten())
    model.add(Dense(1164, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(0.15))
    model.add(Dense(50, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(10, activation='relu'))

    model.add(Dense(1, activation='linear'))
    
    return model

   

model = NvidiaModel()
# model = LeNet()
model.compile(optimizer='adam', loss='mse')

train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)
model.fit_generator(train_generator, samples_per_epoch=len(train_samples)*6, validation_data=validation_generator,nb_val_samples=len(validation_samples)*6, nb_epoch=6)

model.summary()
model.save('model.h5')
print("saved model")