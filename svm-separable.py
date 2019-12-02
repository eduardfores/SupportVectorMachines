import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas import DataFrame
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix


data_x=[];
data_y=[];

print("Loading Data");
ringdata = pd.read_csv("./ring-separable.csv");
ringtest = pd.read_csv("./ring-test.csv");

ringdata.shape;
ringdata.head();

ringtest.shape;
ringtest.head();

X_train = ringdata.drop('Class', axis=1);
y_train = ringdata['Class'];

X_test = ringtest.drop('Class', axis=1);
y_test = ringtest['Class'];

#svclassifier = SVC(kernel='sigmoid', gamma=2);  #Sigmoid
svclassifier = SVC(gamma=2, C=1); #RBF (The best in this type of dataset)

print("Training..");
svclassifier.fit(X_train, y_train);

print("Predicting...");

y_pred = svclassifier.predict(X_test);

X_test=X_test.values.tolist();

for item in X_test:
	data_x.append(item[0]);
	data_y.append(item[1]);

Result ={ 'X' : data_x,
	  'Y' : data_y,
	  'Output' : y_pred
}

df = DataFrame(Result, columns= ['X', 'Y', 'Output'])
export_csv = df.to_csv (r'./result_separable.csv', index = None, header=True) 

print(svclassifier.score(X_test,y_test));
print("\nconfusion matrix");
print(confusion_matrix(y_test,y_pred));
print("\nclassification_report");
print(classification_report(y_test,y_pred));
