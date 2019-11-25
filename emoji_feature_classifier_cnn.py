import keras
import glob
import os
import argparse
import numpy as np
import random
import matplotlib.pyplot as plt
from cv2 import cv2
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD, RMSprop, Adam
from sklearn.model_selection import train_test_split

def resize_image(img, d=300):
    dim = (d, d)
    return cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

def get_train_images():
    X = []
    Y = []
    
    try:
        for i in range(0, 7):
            print('reading: ' + str(i))
            for file in glob.glob(f'roi/{i}/*.jpg'):
                img = cv2.imread(file)
                if img.shape[0] != img.shape[1]: continue # Skip non-square images
                resized = resize_image(img)
                X.append(resized)
                Y.append(0 if i == 0 else 1)
    except Exception as e:
        print(e)
          
    X = np.array(X) / 255
    Y = np.reshape(np.array(Y), (-1, 1))

    return X, Y



def createModel(n_classes, input_shape):
    model = Sequential()
    # The first two layers with 32 filters of window size 3x3
    model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=input_shape))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(n_classes, activation='softmax'))
    
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Feature Extraction')
    parser.add_argument('-t', action='store_true', dest='retrain', help='Retrain?')
    args = parser.parse_args()
    
    if args.retrain or not 'filter_model.h5' in os.listdir():
        X, Y = get_train_images()
        
        test_size = 0.33
        seed = 5
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)
        
        nRows,nCols,nDims = X_train.shape[1:]
        input_shape = (nRows, nCols, nDims)
        
        X_train = X_train.reshape(X_train.shape[0], nRows, nCols, nDims)
        X_test = X_test.reshape(X_test.shape[0], nRows, nCols, nDims)
        Y_cat_train = to_categorical(Y_train)
        Y_cat_test = to_categorical(Y_test)
        
        n_classes = 2
        
        model = createModel(n_classes, input_shape)
        print('Got model')
        opt = SGD()
        model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
        print('Compile model')

        batch_size = 10
        epochs = 100
        datagen = ImageDataGenerator(
                zoom_range=0.1, # randomly zoom into images
                rotation_range=20,  # randomly rotate images in the range (degrees, 0 to 180)
                width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
                height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
                # horizontal_flip=True,  # randomly flip images
                vertical_flip=True)  # randomly flip images
        print('Got datagen')
        
        # history = model.fit(X_train, Y_cat_train, batch_size=batch_size, epochs=epochs, verbose=1, 
        #                validation_data=(X_test, Y_cat_test))
        # Fit the model on the batches generated by datagen.flow().
        history = model.fit_generator(datagen.flow(X_train, Y_cat_train, batch_size=batch_size),
                                    steps_per_epoch=int(np.ceil(X_train.shape[0] / float(batch_size))),
                                    epochs=epochs,
                                    validation_data=(X_test, Y_cat_test),
                                    workers=4)
        print('Got history')
        
        model.evaluate(X_test, Y_cat_test)
        print('Got evaluation')

        model.save('filter_model.h5')
        print('Model saved')
    else:
        model = keras.models.load_model('filter_model.h5')
        print('Model extracted')
    
        X, Y = get_train_images()
        test_size = 0.33
        seed = 5
        
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)
        
        nRows,nCols,nDims = X_train.shape[1:]
        X_train = X_train.reshape(X_train.shape[0], nRows, nCols, nDims)
        X_test = X_test.reshape(X_test.shape[0], nRows, nCols, nDims)
        Y_cat_train = to_categorical(Y_train)
        Y_cat_test = to_categorical(Y_test)
        
        t = random.randint(0, len(X_test)-1)
        # a = model.evaluate(X, to_categorical(Y))
        b = np.array([X_test[t]])
        a = model.predict(b)
        print(a)
        plt.imshow(X_test[t])
        plt.show()