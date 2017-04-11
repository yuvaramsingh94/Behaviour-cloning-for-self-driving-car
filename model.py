import cv2
import sys
import csv
import numpy as np

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

def generator (samples ,train_data_path , batch_size = 32  ): # define the generator to return image and steering label
    num_sample = len(samples)
    while 1:
        samples = shuffle(samples)
        for offset in range(0,num_sample,batch_size):
            batch_sample = samples [offset:offset+batch_size]
            img = []
            ang = []
            for bat in batch_sample:
                # get center image
                name = train_data_path + sep + 'IMG'+sep+bat[0].split(sep)[-1] # this is the path of the center image
                center_image = cv2.imread(name,1)
                center_angle = float(bat[3])

                img.append(center_image)
                ang.append(center_angle)

                # get right image
                name = train_data_path + sep + 'IMG'+sep+ bat[2].split(sep)[-1]
                right_image = cv2.imread(name)
                right_angle = float(bat[3]) - 0.5 # add additional value to steer back to center
                img.append(right_image)
                ang.append(right_angle)

                # get left image
                name = train_data_path + sep + 'IMG'+sep+ bat[1].split(sep)[-1]
                left_image = cv2.imread( name)
                left_angle = float(bat[3]) + 0.5 # add additional value to steer back to center
                img.append(left_image)
                ang.append(left_angle)

            X_train = np.array(img)
            y_train = np.array(ang)


        yield shuffle(X_train, y_train)



if len(sys.argv )== 4:# check if correct argument has been entered in the command line
    # import the needed lib
    from keras.models import Sequential
    from keras.layers import Flatten, Dense, Lambda, Activation
    from keras.layers import Convolution2D, MaxPooling2D, Dropout
    from keras.layers.convolutional import Conv2D
    from keras.layers import Cropping2D


    ip_shape = (160, 320, 3)# shape of hte input image
    os_based_sep = sys.argv[3] # os type 'windows' or 'linux'
    train_data_path = sys.argv[1]# path of training data
    path_to_save = sys.argv[2]# path to save my final model

    # based on the os selected ....... set the seperator (sep) to '/' if linux or '\\' if windows
    sep = '\\' if os_based_sep == 'windows' else '/'

    # start extracting the csv file
    lines = []
    with open(train_data_path + sep + 'driving_log.csv') as file:
        read = csv.reader(file)
        for line in read:
            lines.append(line)
    train, validation = train_test_split(lines, test_size=0.2)

    batch = 32 # batch size of training

    ### initialize the generators for training and testing
    train_generator = generator(train, batch_size=batch,train_data_path = train_data_path)
    validation_generator = generator(validation, batch_size=batch ,train_data_path = train_data_path)

    ##### model  : Nvidia model for Self Driving Car
    modelNvidia = Sequential()
    modelNvidia.add(Lambda(lambda x: (x / 255) - 0.5, input_shape=ip_shape))
    modelNvidia.add(Cropping2D(cropping=((70, 25), (0, 0))))
    modelNvidia.add(
        Conv2D(filters=24, kernel_size=(5, 5), strides=(2, 2), padding='valid'))  # can add an activation function
    modelNvidia.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same'))
    modelNvidia.add(
        Conv2D(filters=36, kernel_size=(5, 5), strides=(2, 2), padding='valid'))  # can add an activation function
    modelNvidia.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same'))
    modelNvidia.add(
        Conv2D(filters=48, kernel_size=(5, 5), strides=(2, 2), padding='valid'))  # can add an activation function
    modelNvidia.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same'))
    modelNvidia.add(
        Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='valid'))  # can add an activation function
    modelNvidia.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same'))
    modelNvidia.add(
        Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='valid'))  # can add an activation function
    modelNvidia.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same'))
    modelNvidia.add(Flatten())
    modelNvidia.add(Dense(1164))
    modelNvidia.add(Dense(100))
    modelNvidia.add(Dense(50))
    modelNvidia.add(Dense(10))
    modelNvidia.add(Dense(1))
    modelNvidia.compile(loss='mse', optimizer='adam')
    modelNvidia.fit_generator(train_generator, samples_per_epoch= len(train), validation_data=validation_generator, nb_val_samples=len(validation), nb_epoch=10)
    modelNvidia.save(path_to_save+sep+'modelNvidiaMax_2.h5')
    print ('model saved at location : ',path_to_save)
    #### model ends
else:
    print(
        'you forgot to mention the training data path or the path to save the model or the os of your system  ...\n please enter it and try again')
    print('it looks like this ')
    print('python model.py /path/to/training/data /path/to/save/data windows (or) linux')
    print('please read this section : how to run model.py in the writeup.md')

