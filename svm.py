import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

#matplotlib inline

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

#svclassifier = SVC(kernel='sigmoid', gamma=2);
svclassifier = SVC(gamma=2, C=1);
print("Training..");
svclassifier.fit(X_train, y_train);

print("Predicting...");

y_pred = svclassifier.predict(X_test);

print("\nconfusion matrix")
print(confusion_matrix(y_test,y_pred));
print("\nclassification_report");
print(classification_report(y_test,y_pred));
