# Import all the libraries need
from __future__ import absolute_import, division, print_function, unicode_literals
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from flask import Flask,request, url_for, redirect, render_template
# %matplotlib inline
sns.set(style='whitegrid', color_codes=True)
import os
from six.moves import urllib
from IPython.display import clear_output
import tensorflow as tf
from sklearn import preprocessing
from sklearn.preprocessing import Binarizer
from sklearn.preprocessing import StandardScaler
from tkinter import *


print('All Libraries Imported')


'''
app = Flask(__name__)
@app.route('/')
def routefunc():
    return render_template('hec.html')
'''

# Define columns name
colnames = ['faceshape', 'eyeshape', 'eyecolor', 'noseshape', 'lipshape', 'skincolor', 'hairtype', 'height', 'weight',
            'ethnicity']
# Define class Labels
ethnicity = ['NB', 'BIA', 'BARUA', 'CHAKMA', 'BSP', 'TGMST']

# Read train & test data
train = pd.read_csv('C:\\Users\\nnabi\\.keras\\datasets\\ptrain.csv', header=0, names= colnames)  # from local PC
test = pd.read_csv('C:\\Users\\nnabi\\.keras\\datasets\\ptest.csv', header=0, names= colnames)  # from local PC

print(train)

x = train.values  # returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
dftrain = pd.DataFrame(x_scaled,columns=colnames)
#print(dftrain)
'''
binarizer = Binarizer(threshold = 0.0).fit(dftrain)
binaryX = binarizer.transform(dftrain)
#print(binaryX)
dftrainbinary=pd.DataFrame(binaryX, columns=colnames)
#print(dftrainbinary)
'''
'''
scalerx = StandardScaler().fit(dftrain)
rescaledX = scalerx.transform(dftrain)
dftrainrescaled = pd.DataFrame(rescaledX, columns=colnames)
print(dftrainrescaled)
'''

y = test.values  # returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
y_scaled = min_max_scaler.fit_transform(y)
dftest = pd.DataFrame(y_scaled,columns=colnames)
#print(dftest)
'''
binarizer = Binarizer(threshold = 0.0).fit(dftest)
binaryy = binarizer.transform(dftest)
#print(binaryy)
dftestbinary=pd.DataFrame(binaryy, columns=colnames)
#print(dftestbinary)
'''
'''
scalery = StandardScaler().fit(dftest)
rescaledy = scalery.transform(dftest)
dftestrescaled = pd.DataFrame(rescaledy, columns=colnames)
print(dftestrescaled)
'''




'''
plt.figure(figsize=(6,6))
sns.countplot(x='ethnicity', data=train)
plt.show()
'''
'''
plt.figure(figsize=(10,.8))
ax = sns.countplot(train, palette="icefire")
plt.title("Number of digit classes")
'''
#x_scaled.describe()
#plt.show()
'''
train['faceshape'].plot(figsize=(10,6), grid=True)
train['eyeshape'].plot(figsize=(10,6), grid=True)
train['eyecolor'].plot(figsize=(10,6), grid=True)
train['noseshape'].plot(figsize=(10,6), grid=True)
train['lipshape'].plot(figsize=(10,6), grid=True)
train['skincolor'].plot(figsize=(10,6), grid=True)
train['hairtype'].plot(figsize=(10,6), grid=True)
train['height'].plot(figsize=(10,6), grid=True)
train['weight'].plot(figsize=(10,6), grid=True)
train['ethnicity'].plot(figsize=(10,6), grid=True)
'''
'''
plt.figure(figsize=(10, .8))
sns.heatmap(
    data=train.corr("kendall").iloc[:1, 1:],
    annot=True,
    fmt='.0%',
    cmap='coolwarm'
)
'''
# Pop target column from train & test dataset to input features in estimetor
train_y = train.pop('ethnicity')
test_y = test.pop('ethnicity')



# Input fucntion to preprocess data
def input_func(features, labels, trainning=True, batch_size=400):
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))  # dataframe created
    if trainning:
        dataset = dataset.shuffle(10000).repeat()
    return dataset.batch(batch_size)


featurecols = []

for key in train.keys():
    featurecols.append(tf.feature_column.numeric_column(key=key))
print(featurecols)

classifier = tf.estimator.DNNClassifier(feature_columns=featurecols, hidden_units=[500, 400, 300, 200, 100, 50],n_classes=7)
classifier.train(input_fn=lambda: input_func(train, train_y, trainning=True), steps=10)

eval_result = classifier.evaluate(input_fn=lambda: input_func(test, test_y, trainning=False))
# clear_output()
print('\nTest Set Acc: {accuracy:0.3f}\n'.format(**eval_result))

print(eval_result)

#print('accuracy:' [eval_result['accuracy']])
#print('accuracy_baseline:' [eval_result['accuracy_baseline']])

'''
# Model accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('accuracy')
plt.xlabel('steps')
plt.legend(['train', 'test'])
plt.show()
'''


def input_fn(features, batch_size=250):
    return tf.data.Dataset.from_tensor_slices(dict(features)).batch(batch_size)


features = ['faceshape', 'eyeshape', 'eyecolor', 'noseshape', 'lipshape', 'skincolor', 'hairtype', 'height', 'weight']
predict = {}
'''
print('Please Type Neumeric values as prompted.')
for feature in features:
    #valid = True
    #while valid:
    val = int(input(feature + ': '))

    if val.isnumeric():
        pass
    else:
        valid = False


    predict[feature] = [int(val)]
    print(predict)
'''

'''
@app.route('/finput',methods=['POST','GET'])


def finput():
    for x in request.form.values():
        c= 0
        c = c + 1
        predict[features[c]] = [int(x)]

print(predict)
'''

a = int(input('Face Shape: '))
predict[features[0]] = [a]

print(predict)

b = int(input('Eye Shape: '))
predict[features[1]] = [b]

print(predict)

c = int(input('Eye Color: '))
predict[features[2]] = [c]

print(predict)

d = int(input('Nose Shape: '))
predict[features[3]] = [d]

print(predict)

e = int(input('Lip Shape: '))
predict[features[4]] = [e]
print(predict)

f = int(input('Skin Color: '))
predict[features[5]] = [f]
print(predict)

g = int(input('Hair Type: '))
predict[features[6]] = [g]
print(predict)

h = int(input('Height: '))
predict[features[7]] = [h]
print(predict)

i = int(input('Weight: '))
predict[features[8]] = [i]

print(predict)


predictions = classifier.predict(input_fn=lambda: input_fn(predict))
for pred_dict in predictions:
    class_id = pred_dict['class_ids'][0]
    probability = pred_dict['probabilities'][class_id]
print('Prediction is "{}" ({:.1f}%)'.format(ethnicity[class_id], 100 * probability))



