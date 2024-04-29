import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score, log_loss
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import pickle
import statistics

class DataHandler:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None
        self.input_df = None
        self.output_df = None

    def load_data(self):
        self.data = pd.read_csv(self.file_path)

    def remove(self, target_column = ['id', 'CustomerId', 'Surname']):
        self.data = self.data.drop(columns=target_column)
        
    def create_input_output(self, target_column):
        self.output_df = self.data[target_column]
        self.input_df = self.data.drop(target_column, axis=1)
        
class ModelHandler:
    def __init__(self, input_data, output_data):
        self.input_data = input_data
        self.output_data = output_data
        self.createModel()
        self.x_train, self.x_test, self.y_train, self.y_test, self.y_predict = [None] * 5
    
    def createMeanFromColumn(self, col):
        return np.mean(self.x_train[col])
    
    def createModeFromColumn(self, col):
        return statistics.mode(self.x_train[col])

    def fillNA(self, columns, number):
        self.x_train[columns].fillna(number, inplace=True)
        self.x_test[columns].fillna(number, inplace=True)

    def split_data(self, test_size = 0.2, random_state = 42):
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            self.input_data, self.output_data, test_size = test_size, random_state = random_state)

    def BinaryEncode(self,col):
        self.train_encode={"Gender" : {"Male" : 1, "Female" : 0}, "Geography" : {"France":1, "Germany":2, "Spain":3}}
        self.test_encode={"Gender" : {"Male" : 1, "Female" : 0}, "Geography" : {"France":1, "Germany":2, "Spain":3}}
        self.x_train=self.x_train.replace(self.train_encode)
        self.x_test=self.x_test.replace(self.test_encode)
    
    def createModel(self,criteria='gini',maxdepth=8):
         self.model = RandomForestClassifier()
         
    def train_model(self):
        self.model.fit(self.x_train, self.y_train)
         
    def makePrediction(self):
        self.y_predict = self.model.predict(self.x_test) 
        
    def createReport(self):
        print('\nClassification Report\n')
        print(classification_report(self.y_test, self.y_predict, target_names=['0', '1']))

    def evaluate_model(self):
        predictions = self.model.predict(self.x_test)
        return accuracy_score(self.y_test, predictions)

    def save_model_to_file(self, filename):
        with open(filename, 'wb') as file:
            pickle.dump(self.model, file) 

file_path = 'data_D.csv'  
data_handler = DataHandler(file_path)
data_handler.load_data()
data_handler.remove()
data_handler.create_input_output('churn')
input_df = data_handler.input_df
output_df = data_handler.output_df

model_handler = ModelHandler(input_df, output_df)
model_handler.split_data()

CreditScore_replace_na = model_handler.createModeFromColumn('CreditScore')

model_handler.fillNA('CreditScore', CreditScore_replace_na)

model_handler.BinaryEncode(['Gender', 'Geography'])

model_handler.train_model()
model_handler.makePrediction()
model_handler.createReport()
print("Model Accuracy:", model_handler.evaluate_model())

model_handler.save_model_to_file('UTSMODELOOP.pkl')