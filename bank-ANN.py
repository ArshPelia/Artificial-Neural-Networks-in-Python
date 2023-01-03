# This program designs and trains artificial neural networs for classification task of the bank marketing dataset (bank.csv)
# The program uses the Keras library to design and train the neural network
# The program uses the scikit-learn library to evaluate the performance of the neural network
# The program uses the matplotlib library to plot the learning curves of the neural network
# The program uses the pandas library to load the dataset
# The program uses the numpy library to perform mathematical operations on the dataset

# bANK.CSV DATASET DESCRIPTION
# The dataset contains 20 attributes and 4521 instances 
# The dataset contains both categorical and numerical attribute (except the target attribute)
# All categorical attributes: job, marital, education, default, housing, loan, contact, month, day_of_week, poutcome (in total 10 attributes not including the target attribute)
# All numerical attributes: age, campaign, pdays, previous, emp.var.rate, cons.price.idx, cons.conf.idx, euribor3m, nr.employed (in total 9 attributes not including the target attribute)
# The target attribute: y (yes or no)

# the program will train the neural network and test it on the test set
# data is randomly split into 70% training and 30% testing
# the program will display the training plots and confusion matrix for each dataset
# the program will also Report train and test accuracy of classification
# the program will also Report the time taken to train the network

# import the necessary libraries

from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.models import load_model

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
from sklearn.preprocessing import OneHotEncoder
from imblearn.over_sampling import SMOTE

import seaborn as sns

categorical_columns = ['job', 'marital', 'education', 'housing', 'loan', 'contact', 'month', 'day_of_week', 'poutcome', 'y', 'pdays']
numerical_columns = ['age', 'campaign', 'previous', 'emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed']
binary_columns = ['housing', 'loan', 'contact', 'y', 'default']

# Begin by loading the dataset and preprocessing it 
def preProcess():
    bank_df = pd.read_csv('bank.csv')
    
    #first remove unnecessary columns ('contact','month','day_of_week','default','pdays') 
    # bank_df = bank_df.drop(['contact','month','day_of_week','default','pdays'], axis=1)
    bank_df = bank_df.dropna() #drop rows with missing values
    bank_df = bank_df.drop_duplicates() #drop duplicate rows
    bank_df.replace(['basic.6y','basic.4y', 'basic.9y'], 'basic', inplace=True) #replace basic.6y, basic.4y, basic.9y with basic
    bank_df.replace(['unknown'], np.nan, inplace=True) #replace unknown with NaN

    #replace unknown with most frequent value
    bank_df['job'].fillna(bank_df['job'].value_counts().idxmax(), inplace=True)
    bank_df['marital'].fillna(bank_df['marital'].value_counts().idxmax(), inplace=True)
    bank_df['education'].fillna(bank_df['education'].value_counts().idxmax(), inplace=True)
    bank_df['housing'].fillna(bank_df['housing'].value_counts().idxmax(), inplace=True)
    bank_df['loan'].fillna(bank_df['loan'].value_counts().idxmax(), inplace=True)
    bank_df['poutcome'].fillna(bank_df['poutcome'].value_counts().idxmax(), inplace=True)
    bank_df['y'].fillna(bank_df['y'].value_counts().idxmax(), inplace=True)
    bank_df['pdays'].fillna(bank_df['pdays'].value_counts().idxmax(), inplace=True)
    bank_df['default'].fillna(bank_df['default'].value_counts().idxmax(), inplace=True)

    Label_dict = {'default': {'yes': 1, 'no': 0}, 
                    'housing': {'yes': 1, 'no': 0}, 
                    'loan': {'yes': 1, 'no': 0}, 
                    'contact': {'cellular': 1, 'telephone': 0}, 
                    'poutcome': {'failure': 0, 'nonexistent': 1, 'success': 2},
                    'marital': {'single': 0, 'married': 1, 'divorced': 2},
                    'education': {'illiterate': 0, 'basic': 1, 'high.school': 2, 'professional.course': 3, 'university.degree': 4},
                    'job': {'unemployed': 0, 'student': 1, 'housemaid': 2, 'retired': 3, 'admin.': 4, 'blue-collar': 5, 'technician': 6, 'services': 7, 'management': 8, 'entrepreneur': 9, 'self-employed': 10},
                    'month': {'jan': 0, 'feb': 1, 'mar': 2, 'apr': 3, 'may': 4, 'jun': 5, 'jul': 6, 'aug': 7, 'sep': 8, 'oct': 9, 'nov': 10, 'dec': 11},
                    'day_of_week': {'mon': 0, 'tue': 1, 'wed': 2, 'thu': 3, 'fri': 4}}
    #encode categorical data based on the dictionary
    for key in Label_dict:
        bank_df[key] = bank_df[key].map(Label_dict[key])

    #encode target data
    bank_df['y'] = bank_df['y'].map({'yes': 1, 'no': 0})

    #normalize numerical data
    bank_df[numerical_columns] = (bank_df[numerical_columns] - bank_df[numerical_columns].mean()) / bank_df[numerical_columns].std()

    # print(bank_df.head())

    #split data into train and test
    y = bank_df['y']
    x = bank_df.drop(['y'], axis=1)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

    #balance the data with SMOTE
    sm = SMOTE(random_state=0)
    x_train, y_train = sm.fit_resample(x_train, y_train)

    #scale data
    scaler = StandardScaler()
    scaler.fit(x)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)

    # print(x_train.shape)
    # print(x_test.shape)

    # X_train = pd.DataFrame(X_train)
    # X_test = pd.DataFrame(X_test) #for PCA

    return x_train, x_test, y_train, y_test

def ANN(x_train, x_test, y_train, y_test):
        #create ANN model 
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(32, activation='relu', input_dim=19)) #input layer with 19 neurons using relu activation function
    model.add(tf.keras.layers.Dense(16, activation='relu')) #hidden layer with 16 neurons using relu activation function
    model.add(tf.keras.layers.Dense(8, activation='relu')) #hidden layer with 8 neurons using relu activation function
    model.add(tf.keras.layers.Dense(1, activation='sigmoid')) #output layer with 1 neuron using sigmoid activation function 
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy']) #binary_crossentropy for binary classification instead of categorical_crossentropy because we have only 2 classes


    #train model
    history = model.fit(x_train, y_train, epochs=20, batch_size=32)

    #plot accuracy and loss 
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['loss'])
    plt.title('model accuracy and loss')
    plt.ylabel('accuracy and loss')
    plt.xlabel('epoch')
    plt.legend(['accuracy', 'loss'], loc='upper left')
    plt.show()

    #evaluate model
    y_pred = model.predict(x_test)
    y_pred = (y_pred > 0.5)
    print()
    print("Classification report: ")
    print(classification_report(y_test, y_pred))

#import sn 
    import seaborn as sn
    #plot confusion matrix with namve labels not numbers (0, 1)
    cm = confusion_matrix(y_test, y_pred)
    df_cm = pd.DataFrame(cm, index = [i for i in ['No', 'Yes']],
                  columns = [i for i in ['No', 'Yes']])
    plt.figure(figsize = (10,7))
    sn.heatmap(df_cm, annot=True, fmt='g')
    plt.show()

    return model

# def MLP(x_train, x_test, y_train, y_test):
#     #create model
#     model = MLPClassifier(hidden_layer_sizes=(10, 10, 10), max_iter=1000)
#     model.fit(x_train, y_train)

#     #predict
#     y_pred = model.predict(x_test)

#     #evaluate
#     print(confusion_matrix(y_test, y_pred))
#     print(classification_report(y_test, y_pred))
#     print(accuracy_score(y_test, y_pred))

#     return model

def main():
    x_train, x_test, y_train, y_test = preProcess()
    model_ann = ANN(x_train, x_test, y_train, y_test)
    # model_mlp = MLP(x_train, x_test, y_train, y_test

if __name__ == "__main__":
    main()



                    




