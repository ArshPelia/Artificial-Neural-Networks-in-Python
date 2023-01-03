# This program designs and trains artificial neural networs for classification task of 3 datasets
# Dataset 1: Iris - iris.csv
# the dataset will have a corresponding neural network with unique number of hidden layers, neurons and activation functions
# the program will train the neural network and test it on the test set
# data is randomly split into 70% training and 30% testing
# the program will display the training plots and confusion matrix for each dataset
# the program will also Report train and test accuracy of classification
# the program will also Report the time taken to train the network

# import libraries
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import time, os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder


#setup the dataset
def setup():
    iris_data = pd.read_csv('iris.csv')
    #create a dataframe 
    iris_df = pd.DataFrame(iris_data)
    #Encode species label to convert String to numeric values for the target (required for neural network)
    le = LabelEncoder()
    iris_df['species'] = le.fit_transform(iris_df['species'])

    #split the data into training and testing
    X = iris_df.iloc[:, 0:4].values #features (sepal length, sepal width, petal length, petal width)
    y = iris_df.iloc[:, 4].values #target (species)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    species_df = pd.DataFrame({'Species': ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']}) #create a dataframe for species names where 0 = Iris-setosa, 1 = Iris-versicolor, 2 = Iris-virginica

    #scale the data
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    #return the data
    return X_train, X_test, y_train, y_test, species_df


#globals
X_train, X_test, y_train, y_test, species_df = setup()

#design model for iris dataset
def iris_model():
    #design the model
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(8, activation='relu', input_shape=(4,)), #input layer with 4 neurons and relu activation function (rectified linear unit) 
        tf.keras.layers.Dense(8, activation='relu'), #hidden layer with 8 neurons and relu activation function
        tf.keras.layers.Dense(3, activation='softmax') #output layer with 3 neurons and softmax activation function (used for classification) 
    ])

    #compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy']) #adam optimizer, sparse_categorical_crossentropy loss function, accuracy metric

    #train the model
    history = model.fit(X_train, y_train, epochs=100, validation_data=(X_test, y_test)) #train the model for 100 epochs and validate on the test set (30% of the data) 

    #test the model
    y_pred = model.predict(X_test) #predict the target values for the test set (30% of the data) 
    y_pred = np.argmax(y_pred, axis=1) #convert the predicted values to 0, 1 or 2 (0 = Iris-setosa, 1 = Iris-versicolor, 2 = Iris-virginica) 

    #display the training plots
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0.5, 1])
    plt.legend(loc='lower right')
    plt.show()

    #display the confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    #plot the confusion matrix
    plt.figure(figsize=(10, 7))
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.imshow(cm, cmap='Blues')
    plt.colorbar()
    plt.show()

    #display the classification report
    print(classification_report(y_test, y_pred, target_names=species_df['Species'].values))

    #display the accuracy
    print('Accuracy: ', accuracy_score(y_test, y_pred))

    #display the time taken to train the model
    print('Time taken to train the model: ', time.process_time(), 'seconds')

    #return the model
    return model


#def main
def main():
    #train the model
    iris_model()

#call main 
if __name__ == '__main__':
    main()

#end of program
# explanation of the code: 
# we begin by importing the required libraries
# we then define the setup function which will load the dataset, split the data into training and testing, encode the target values, scale the data and return the data
# the setup function is required because we need to prepare the data for the neural network and we need to prepare the data for each dataset separately 
# the scale function is used to standardize the data (it is required for neural network) since the neural network is sensitive to the scale of the data
# we then define the globals which will store the data returned by the setup function
# we then define the iris_model function which will design the model, compile the model, train the model, test the model, display the training plots, display the confusion matrix, display the classification report, display the accuracy and display the time taken to train the model
# the model consists of an input layer with 4 neurons and relu activation function (rectified linear unit), a hidden layer with 8 neurons and relu activation function and an output layer with 3 neurons and softmax activation function (used for classification)
# the model is compiled using the adam optimizer, sparse_categorical_crossentropy loss function and accuracy metric
# the adam optimizer is a gradient descent algorithm that is used to update the weights of the neural network
# the sparse_categorical_crossentropy loss function is used for classification problems
# the relu activation function is used to introduce non-linearity into the model (it is used for the input and hidden layers)
# The model is then trained for 100 epochs and validated on the test set (30% of the data) 
# training consists of updating the weights of the neural network using the adam optimizer and calculating the loss using the sparse_categorical_crossentropy loss function
# the model is then tested by predicting the target values for the test set (30% of the data)
# prediction is done by passing the test set through the neural network and calculating the output values

# the training plots are displayed using matplotlib 
# training plots are used to visualize the training process of the neural network (it is used to check if the model is overfitting or underfitting)
# the loss plot shows the loss of the model on the training set and the validation set (the validation set is the test set)
# the accuracy plot shows the accuracy of the model on the training set and the validation set (the validation set is the test set)
# the loss plot shows that the model is overfitting since the loss of the model on the training set is decreasing while the loss of the model on the validation set is increasing
# the accuracy plot shows that the model is overfitting since the accuracy of the model on the training set is increasing while the accuracy of the model on the validation set is decreasing





