#Fidel Garcia
#Spring.2020
#WildLife - Image Classifier: gather camera trap images to classify and
#           organize such information using a Convolutional Neural Network
import time
import numpy as np
import os
import cv2
import random
import pickle

################################################################################
#creating the dataset for training
def create_dataset():
    #create a path to the data folder
    data_dir = "data"

    #create a list of categories
    cat = ["badger", "bird", "bobcat", "car", "cat", "cow", "coyote", "deer", "dog", "empty", "fox", "lizard", "mountainlion", "opossum", "rabbit", "raccoon", "rodent", "skunk", "squirrel"]

    img_size = 100

    #create an array for the training data
    train_data_array = []

    #helper function - insert, grayscale, and resize the images
    def import_training_data():
        for category in cat:
            path = os.path.join(data_dir, category)
            class_number = cat.index(category)
            for image in os.listdir(path):
                try:
                    image_array = cv2.imread(os.path.join(path, image), cv2.IMREAD_GRAYSCALE)
                    resize_array = cv2.resize(image_array, (img_size, img_size))
                    train_data_array.append([resize_array, class_number])
                except Exception as e:
                    pass


    import_training_data()
    random.shuffle(train_data_array)

    #arrays for features (categories - help predict) and the labels
    #(assign. the features)
    x = []
    y = []

    for feature, label in train_data_array:
        x.append(feature)           #the category or images (abtraction data type)
        y.append(label)             #the label thats assigned (number)

    #make the arrays into a numpy array for manipulation
    x = np.array(x).reshape(-1, img_size, img_size, 1)
    y = np.array(y)

    #using pickle to create byte pattern files for our images
    #(allows for easy download/use of the dataset)
    pickle_out = open("x.pickle", "wb")    #write to the bit file
    pickle.dump(x, pickle_out)
    pickle_out.close()

    pickle_out = open("y.pickle", "wb")
    pickle.dump(y, pickle_out)
    pickle_out.close()

    pickle_in = open("x.pickle", "rb")      #read from the bit file
    x = pickle.load(pickle_in)

    print("--------------------Dataset is ready for use---------------------")


################################################################################
#creating the Convolutional Nueral Network
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
import pickle
from keras.models import model_from_json
from keras.models import load_model

def create_network():
    #open up the dataset for reading and training
    x = pickle.load(open("x.pickle", "rb"))
    y = pickle.load(open("y.pickle", "rb"))

    #noramlize the data to a fixed pixel size
    x = x / 255.0

    #start  building the model
    model = Sequential()

    #create Convolutional layers (filters)
    model.add(Conv2D(32, (3, 3), input_shape = x.shape[1:]))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    #these layers are hidden
    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation("relu"))

    model.add(Dense(128))
    model.add(Activation("relu"))

    # the final layer needs to have the same amount of neurons as our categories
    model.add(Dense(len(category)))
    model.add(Activation("softmax"))

    #copmile the model
    model.compile(loss = "sparse_categorical_crossentropy",
                optimizer = "adam",
                metrics = ["accuracy"])

    #training the model with 10 iterations (epochs)
    history = model.fit(x, y, batch_size = 32, epochs = 25, validation_split = 0.1)

    #save the model
    model.save_weights("model.h5")
    print("Saved model to disk")
    model.save('CNN.model')


################################################################################
#predict the images - needs to be a funtion

#Needs to be done or at least try:
#   Save the file names in the testing dir into a list or array and then
#   pass each index into the prepare method  or change the prepare function
#   to handle a dir instead of a file

import cv2
import tensorflow as tf
category = ["badger", "bird", "bobcat", "car", "cat", "cow", "coyote", "deer", "dog", "empty", "fox", "lizard", "mountainlion", "opossum", "rabbit", "raccoon", "rodent", "skunk", "squirrel"]

img_size = 100

def prepare(file):
    test_data_array = []
    img_size = 100
    img_array = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
    new_array = cv2.resize(img_array, (img_size, img_size))
    test_data_array.append([new_array])
    test_data_array = np.array([test_data_array])
    return test_data_array.reshape(-1, img_size, img_size, 1)
###############################################################################

#setup the training dataset
print('Currently creating the dataset ..... Please wait (Approx: 5 min.)')
data_start = time.clock()
create_dataset()
data_elapsed = (time.clock() - data_start)
print("Time to create the dataset was: ", int(data_elapsed / 60),
"minutes and ", int(data_elapsed % 60), "seconds")



#create/train and save the neural network
print('Currently creating a CNN model ...... Approx: 25 min.')
cnn_start = time.clock()
create_network()
cnn_elapsed = (time.clock() - cnn_start)
print("Time to create and train the neural network was: ", int(cnn_elapsed / 360),
"minutes and ", int(cnn_elapsed % 40), "seconds")



#predict the images
from tkinter import Tk
from tkinter.filedialog import askdirectory
path = askdirectory(title='Select the testing folder.')
dirs = os.listdir( path )
#create a file


model = tf.keras.models.load_model("CNN.model")
for image in dirs:
    try:
        image = os.path.join(path, image)               #path to the images
        image_change = prepare(image)                   #change the img for the CNN
        prediction = model.predict([image_change])      #prediction
        prediction = list(prediction[0])
        final_predict = category[prediction.index(max(prediction))] #final_prediction
        print(image," is: ", final_predict)
    except:
        pass