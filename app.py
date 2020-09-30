#Author: Fidel Garica
#Date: April, 28th, 2020
#Description: 
# 1. Gather test images for creating a dataset to train a CNN for image prediction. 
# 2. Train the CNN to recognize image classifications. 
# 3. Based on the prediction percentage provide the best prediction.

#imports that are needed
import os
from cv2 import cv2
import random
from pandas.compat.numpy import np
import pickle
from keras.engine.sequential import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.core import Activation, Dense, Dropout, Flatten
from keras.layers.pooling import MaxPooling2D
from tkFileDialog import askdirectory
from tensorboard.compat import tf
import csv

#create main method to ask for cat.
categories = []         #global list of species
def start():
    n = int(input("How many species are you trying to differentiate? (Including empty): ")) 

    for i in range(0, n): 
        ele = raw_input()
        categories.append(ele)
    print(categories, " with a total of: " , len(categories), "species") 
    return


#creating the dataset for training
def create_dataset(a = []):
    #create a path to the data folder
    data_dir = askdirectory(title='Select the training folder.')

    #create a list of categories
    cat = a
    #["badger", "bird", "bobcat", "car", "cat", "cow", "coyote", "deer", "dog", "empty", "fox", "lizard", "mountainlion", "opossum", "rabbit", "raccoon", "rodent", "skunk", "squirrel"]

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

    print("Dataset is ready for use")
    return


#creating the Convolutional Nueral Network
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
    model.add(Dense(len(categories)))
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
    return

#Prepare the images for prediction
def prepare(file):
    test_data_array = []
    img_size = 100
    img_array = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
    new_array = cv2.resize(img_array, (img_size, img_size))
    test_data_array.append([new_array])
    test_data_array = np.array([test_data_array])
    return test_data_array.reshape(-1, img_size, img_size, 1)

#make the prediction
def prediction():
    path = askdirectory(title='Select the testing folder.')
    dirs = os.listdir( path )

    #predict
    model = tf.keras.models.load_model("CNN.model")
    for image in dirs:
        try:
            #path to the images
            image = os.path.join(path, image)               
            image_change = prepare(image)
            #change the img for the CNN                   
            prediction = model.predict([image_change])
            prediction = list(prediction[0])
            #final_prediction
            final_predict = categories[prediction.index(max(prediction))] 
            #open the CSV file
            with open("prediction.csv", 'w') as csvfile:
                csvwriter = csv.writer(csvfile) 
                #writing the prediction
                csvwriter.writerow(image, final_predict)
                #close the file
                csvfile.close()
        except:
            pass
        print("Prediction is made an csv file is updated.")
    return




#actual program being ran
def menu():
    runMenu = True
    while runMenu:
        print("\n 1. Distingish species\n\n 2. Create the dataset\n\n 3. Create/Train the Neural Network\n\n 4. Create a prediction\n\n 5. Exit\n")
        ans =raw_input("What would you like to do? ")
        if ans == "1":
            start()
        elif ans == "2":
            create_dataset(categories)
        elif ans == "3":
            create_network()
        elif ans == "4":
            prediction()
        elif ans == "5":
            runMenu = False
    return

#run the whole project
menu()