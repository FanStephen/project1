import pandas as pd
import numpy as np
import time
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

#read the train data
df=pd.read_csv('new_data.csv',header=None)
df= df.drop([0,1,2,3], axis = 1)
dataset=df.values

x_train = []
for i  in range(30904):
    x_temp = dataset[i+1][1:16]
    x_train.append(x_temp)
x_train= np.array(x_train)

y_train = []
for i in range(30904):
    y_temp = dataset[i+1][0]
    y_train.append(y_temp)
y_train = np.array(y_train)

#read the test data
tf = pd.read_csv('test_set.csv', header = None)
tf= tf.drop([0,1,2,3], axis = 1)
testdata = tf.values

x_test = []
for i in range(20586):
    tes_temp = testdata[i+1][1:16]
    x_test.append(tes_temp)
x_test = np.array(x_test)

 
y_test = []
for i in range(20586):
    tes_temp = testdata[i+1][0]
    y_test.append(tes_temp)
y_test = np.array(y_test)

#DT
print('decision tree')
DT0 = time.time()
DT = DecisionTreeClassifier(max_depth=15,splitter='random')
DT.fit(x_train,y_train)
DT1=time.time()
acc_DT=DT.score(x_test,y_test)
DTtime=DT1-DT0
print('The accuracy is:',acc_DT,' ','The time is:',DTtime)

#KNN
print('KNN')
KNN0=time.time()
KNN = KNeighborsClassifier(n_neighbors = 10,weights='uniform',algorithm='auto')
KNN.fit(x_train,y_train)
KNN1=time.time()
acc_KNN=KNN.score(x_test,y_test)
KNNtime=KNN1-KNN0
print('The accuracy is:',acc_KNN,' ','The time is:',KNNtime)

#SVM
print('SVM')
SVM0= time.time()
SVM = SVC(kernel ='rbf',gamma = 'auto' )
SVM.fit(x_train,y_train)
SVM1=time.time()
acc_SVM = SVM.score(x_test,y_test)
SVMtime=SVM1-SVM0
print('The accuracy is:',acc_SVM,' ','The time is:',SVMtime)