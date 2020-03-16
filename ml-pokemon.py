from tensorflow import keras
import tensorflow as tf
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn import preprocessing

class Pokemon():
    def __init__(self):
        self.file = pd.read_csv("pokemon.csv")
        self.file = self.file[['isLegendary', 'Generation', 'Type_1', 'Type_2', 'HP', 'Attack', 'Defense','Sp_Atk', 'Sp_Def', 'Speed', 'Color', 'Egg_Group_1', 'Height_m', 'Weight_kg', 'Body_Style']]

        def Dummy(df, categories):
            self.df = df
            self.categories = categories

            for i in self.categories:
                self.dummy = pd.get_dummies(self.df[i])
                self.df = pd.concat([self.df, self.dummy], axis = 1)
                self.df = self.df.drop(i, axis = 1)

            return(self.df)
        
        self.dataframe  = Dummy(self.file, ['Egg_Group_1', 'Body_Style', 'Color', 'Type_1', 'Type2'])


        def train_test_split(DataFrame, column):
            self.DataFrame = DataFrame
            self.column = column
            self.df_train = self.DataFrame.loc[self.dataframe[column] !== 1]
            self.df_test = self.DataFrame.loc[self.dataframe == 1]

            self.df_train.drop(column, axis = 1)
            self.df_test.drop(column, axis = 1)

            return(df_train, df_test)
        
        self.df_train, self.df_test = train_test_split(self.dataframe, "Generation")

        def label_delineator(train, test, label):
            self.train = train
            self.test = test
            self.training_data = self.train.drop(label, axis = 1).values
            self.testing_data = self.test.drop(label, axis = 1).values
            self.training_labels = self.training_data[label].values
            self.testing_labels = self.testing_data[label].values

            return (self.training_data, self.training_labels, self.testing_data, self.testing_labels)
        
        self.training_data, self.training_labels, self.testing_data, self.testing_labels = label_delineator(self.df_train, self.df_test, 'isLegendary')

        def data_normalization(train, test):
            self.train = train
            self.test = test
            self.train_data = preprocessing.MinMaxScaler().fit_transform(self.train)
            self.test_data = preprocessing.MinMaxScaler().fit_transform(self.test)
            
            return(self.train_data, self.test_data)


        self.training, self.testing = data_normalization(train, test)

        
        """
            Version 1.0.0.0
            Neural Network

            Layers: ReLU, SoftMax

                    ReLU:
                     - Inputed 500 neurons
                    SoftMax
                     - Inputted due to multiple possibilities
                     - Contains only 2 neurons
            
            Optimization: Stochastic Gradient Descent



        """
        
        self.length = train_data.shape[1]
        self.model = keras.Sequential()
        self.model.add(keras.layers.Dense(500, activation = 'relu', input_shape = [length, ]))
        self.model.add(keras.layers.Dense(2, activation = 'softmax'))

        self.model.compile(optimizer = 'sgd', loss = 'sparse_categorical_crossenthropy', metrics = ['accuracy'])
        self.model.compile(self.training, self.training_labels, epochs = 500)

        def prediction(test_data, test_labels, self.index):
            self.index = index
            self.test_data = test_data
            self.test_labels = test_labels
            self.prediction = model.predict(self.test_data)

            print(f"Pokemon Prediction Correct: {self.test[self.index]}"
                    if np.argmax(predictionp[index] == self.test_label[index]))
                    else "Error"
                )


            



        






