# Import all the libraries need
from __future__ import absolute_import, division, print_function, unicode_literals
import pandas as pd
import matplotlib.pyplot as plt
from PIL import ImageTk
import tensorflow as tf
from sklearn import preprocessing
from sklearn.metrics import plot_confusion_matrix
from tkinter import *
from tkinter import messagebox

window = Tk()
window.title('Ethnicity Classification of Bangladeshi Inhabitants Using Deep Learning')
window.geometry('1000x800')
#scrollbar = Scrollbar(window)
#scrollbar.pack(side = RIGHT, fill = Y)

labelfshape = Label(window, text='Face shape', font=('Arial', 14))
labelfshape.grid(row=0, column=0, padx=10, pady=10)
faceshapeimage = ImageTk.PhotoImage(file = "C:\\Users\\nnabi\\Downloads\\face-final.jpeg")
faceshapeimagelabel = Label(image = faceshapeimage)
faceshapeimagelabel.grid(row=0, column=2, padx=1, pady=1)
datafshape = IntVar()
tboxfshape = Entry(window, textvariable=datafshape, font=('Arial', 14))
tboxfshape.grid(row=0, column=1)
rr=str(datafshape.get())

labeleyeshape = Label(window, text='Eye Shape', font=('Arial', 14))
labeleyeshape.grid(row=1, column=0, padx=10, pady=10)
eyeshapeimage = ImageTk.PhotoImage(file = "C:\\Users\\nnabi\\Downloads\\eyeshape-final.jpeg")
eyeshapeimagelabel = Label(image = eyeshapeimage)
eyeshapeimagelabel.grid(row=1, column=2, padx=1, pady=1)
dataeyeshape = IntVar()
tboxeyeshape = Entry(window, textvariable=dataeyeshape, font=('Arial', 14))
tboxeyeshape.grid(row=1, column=1)

ss=str(dataeyeshape.get())

labeleyecolor = Label(window, text='Eye Color', font=('Arial', 14))
labeleyecolor.grid(row=2, column=0, padx=10, pady=10)
eyecolorimage = ImageTk.PhotoImage(file = "C:\\Users\\nnabi\\Downloads\\eyecolor-final.jpeg")
eyecolorimagelabel = Label(image = eyecolorimage)
eyecolorimagelabel.grid(row=2, column=2, padx=1, pady=1)
dataeyecolor = IntVar()
tboxeyecolor = Entry(window, textvariable=dataeyecolor, font=('Arial', 14))
tboxeyecolor.grid(row=2, column=1)
tt=str(dataeyecolor.get())

labelnshape = Label(window, text='Nose Shape', font=('Arial', 14))
labelnshape.grid(row=3, column=0, padx=10, pady=10)
noseshapeimage = ImageTk.PhotoImage(file = "C:\\Users\\nnabi\\Downloads\\noseshape-final.jpeg")
noseshapeimagelabel = Label(image = noseshapeimage)
noseshapeimagelabel.grid(row=3, column=2, padx=1, pady=1)
datanshape = IntVar()
tboxnshape = Entry(window, textvariable=datanshape, font=('Arial', 14))
tboxnshape.grid(row=3, column=1)
uu=str(datanshape.get())

labellshape = Label(window, text='Lips Shape', font=('Arial', 14))
labellshape.grid(row=4, column=0, padx=10, pady=10)
lipshapeimage = ImageTk.PhotoImage(file = "C:\\Users\\nnabi\\Downloads\\lipshape-final.jpeg")
lipshapeimagelabel = Label(image = lipshapeimage)
lipshapeimagelabel.grid(row=4, column=2, padx=1, pady=1)
datalshape = IntVar()
tboxlshape = Entry(window, textvariable=datalshape, font=('Arial', 14))
tboxlshape.grid(row=4, column=1)
vv=str(datalshape.get())

labelskincolor = Label(window, text='Skin Color', font=('Arial', 14))
labelskincolor.grid(row=5, column=0, padx=10, pady=10)
skincolorimage = ImageTk.PhotoImage(file = "C:\\Users\\nnabi\\Downloads\\skincolor-final.jpeg")
skincolorimagelabel = Label(image = skincolorimage)
skincolorimagelabel.grid(row=5, column=2, padx=1, pady=1)
dataskincolor = IntVar()
tboxskincolor = Entry(window, textvariable=dataskincolor, font=('Arial', 14))
tboxskincolor.grid(row=5, column=1)
ww=str(dataskincolor.get())

labelhtype = Label(window, text='Hair Type', font=('Arial', 14))
labelhtype.grid(row=6, column=0, padx=10, pady=10)
hairtypeimage = ImageTk.PhotoImage(file = "C:\\Users\\nnabi\\Downloads\\hairtype-final.jpeg")
hairtypeimagelabel = Label(image = hairtypeimage)
hairtypeimagelabel.grid(row=6, column=2, padx=1, pady=1)
datahtype = IntVar()
tboxhtype = Entry(window, textvariable=datahtype, font=('Arial', 14))
tboxhtype.grid(row=6, column=1)
xx=str(datahtype.get())

labelheight = Label(window, text='Height', font=('Arial', 14))
labelheight.grid(row=7, column=0, padx=10, pady=10)
dataheight = IntVar()
tboxheight = Entry(window, textvariable=dataheight, font=('Arial', 14))
tboxheight.grid(row=7, column=1)
labelheighttext = Label(window, text='(Enter Height in Inch, Min-30::Max-96)', font=('Arial', 14))
labelheighttext.grid(row=7, column=2, padx=1, pady=1)
yy=str(dataheight.get())

labelweight = Label(window, text='Weight', font=('Arial', 14))
labelweight.grid(row=8, column=0, padx=10, pady=10)
dataweight = IntVar()
tboxweight = Entry(window, textvariable=dataweight, font=('Arial', 14))
tboxweight.grid(row=8, column=1)
labelweighttext = Label(window, text='(Enter Weight in Kg, Min-10::Max-200)', font=('Arial', 14))
labelweighttext.grid(row=8, column=2, padx=1, pady=1)
zz=str(dataweight.get())

colnames = ['faceshape', 'eyeshape', 'eyecolor', 'noseshape', 'lipshape', 'skincolor', 'hairtype', 'height', 'weight',
            'ethnicity']
# Define class Labels
ethnicity = ['Native Bangladeshi', 'Bangladeshi Indian Ancestor', 'Barua', 'Chakma', 'Bangladeshi Stranded Pakistani', 'Tribal']

# Read train & test data
train = pd.read_csv('C:\\Users\\nnabi\\.keras\\datasets\\ptrain.csv', header=0, names=colnames)  # from local PC
test = pd.read_csv('C:\\Users\\nnabi\\.keras\\datasets\\ptrain.csv', header=0, names=colnames)  # from local PC

x = train.values  # returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
dftrain = pd.DataFrame(x_scaled)

#print(dftrain)

y = test.values  # returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
y_scaled = min_max_scaler.fit_transform(y)
dftest = pd.DataFrame(y_scaled)
#print(dftest)

# Pop target column from train & test dataset to input features in estimetor
train_y = train.pop('ethnicity')
test_y = test.pop('ethnicity')

# Display the header of columns
#print(train.head())
# Display the data shape
#print(train.shape)


# Input fucntion to preprocess data
def input_func(features, labels, trainning=True, batch_size=400):
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))  # dataframe created
    if trainning:
        dataset = dataset.shuffle(10000).repeat()
    return dataset.batch(batch_size)


featurecols = []

for key in train.keys():
    featurecols.append(tf.feature_column.numeric_column(key=key))
#print(featurecols)

classifier = tf.estimator.DNNClassifier(feature_columns=featurecols, hidden_units=[500, 400, 300, 200, 100, 50], n_classes=7)
classifier.train(input_fn=lambda: input_func(train, train_y, trainning=True), steps=1)
eval_result = classifier.evaluate(input_fn=lambda: input_func(test, test_y, trainning=False))

#print('\nTest Set Acc: {accuracy:0.3f}\n'.format(**eval_result))

def input_fn(features, batch_size=250):
    return tf.data.Dataset.from_tensor_slices(dict(features)).batch(batch_size)


features = ['faceshape', 'eyeshape', 'eyecolor', 'noseshape', 'lipshape', 'skincolor', 'hairtype', 'height', 'weight']
predict = {}


def inputdata():
    try:
        if 1 <= datafshape.get() <= 6 and 1 <= dataeyeshape.get() <= 8 and 1 <= dataeyecolor.get() <= 5 and 1 <= datanshape.get() <= 6 and 1 <= datalshape.get() <= 6 and \
                1 <= dataskincolor.get() <= 5 and 1 <= datahtype.get() <= 4 and 30 <= dataheight.get() <= 96 and 10 <= dataweight.get() <= 200:
            inputwindow = Tk()
            inputwindow.title('Ethnicity Classification of Bangladeshi Inhabitants Using Deep Learning')
            inputwindow.geometry('400x600')

            labelinputtext = Label(inputwindow, text='Review Input: ', font=('Arial', 20))
            labelinputtext.grid(row=0, column=0, padx=10, pady=10)

            labelfshape = Label(inputwindow, text='Face shape: ', font=('Arial', 14))
            labelfshape.grid(row=1, column=0, padx=10, pady=10)
            emptylabelfshape = Label(inputwindow, font=('Arial', 14))
            emptylabelfshape.grid(row=1, column=1)
            emptylabelfshape.config(text=datafshape.get())

            labeleyeshape = Label(inputwindow, text='Eye shape: ', font=('Arial', 14))
            labeleyeshape.grid(row=2, column=0, padx=10, pady=10)
            emptylabeleyeshape = Label(inputwindow, font=('Arial', 14))
            emptylabeleyeshape.grid(row=2, column=1)
            emptylabeleyeshape.config(text=dataeyeshape.get())

            labeleyecolor = Label(inputwindow, text='Eye Color: ', font=('Arial', 14))
            labeleyecolor.grid(row=3, column=0, padx=10, pady=10)
            emptylabeleyecolor = Label(inputwindow, font=('Arial', 14))
            emptylabeleyecolor.grid(row=3, column=1)
            emptylabeleyecolor.config(text=dataeyecolor.get())

            labelnshape = Label(inputwindow, text='Nose Shape: ', font=('Arial', 14))
            labelnshape.grid(row=4, column=0, padx=10, pady=10)
            emptylabelnoseshape = Label(inputwindow, font=('Arial', 14))
            emptylabelnoseshape.grid(row=4, column=1)
            emptylabelnoseshape.config(text=datanshape.get())

            labellipshape = Label(inputwindow, text='Lips Shape: ', font=('Arial', 14))
            labellipshape.grid(row=5, column=0, padx=10, pady=10)
            emptylabellipshape = Label(inputwindow, font=('Arial', 14))
            emptylabellipshape.grid(row=5, column=1)
            emptylabellipshape.config(text=datalshape.get())

            labelskincolor = Label(inputwindow, text='Skin Color: ', font=('Arial', 14))
            labelskincolor.grid(row=6, column=0, padx=10, pady=10)
            emptylabelskincolor = Label(inputwindow, font=('Arial', 14))
            emptylabelskincolor.grid(row=6, column=1)
            emptylabelskincolor.config(text=dataskincolor.get())

            labelhairtype = Label(inputwindow, text='Hairtype: ', font=('Arial', 14))
            labelhairtype.grid(row=7, column=0, padx=10, pady=10)
            emptylabelhairtype = Label(inputwindow, font=('Arial', 14))
            emptylabelhairtype.grid(row=7, column=1)
            emptylabelhairtype.config(text=datahtype.get())

            labelheight = Label(inputwindow, text='Height: ', font=('Arial', 14))
            labelheight.grid(row=8, column=0, padx=10, pady=10)
            emptylabelheight = Label(inputwindow, font=('Arial', 14))
            emptylabelheight.grid(row=8, column=1)
            emptylabelheight.config(text=dataheight.get())

            labelweight = Label(inputwindow, text='Weight: ', font=('Arial', 14))
            labelweight.grid(row=9, column=0, padx=10, pady=10)
            emptylabelweight = Label(inputwindow, font=('Arial', 14))
            emptylabelweight.grid(row=9, column=1)
            emptylabelweight.config(text=dataweight.get())

            a = datafshape.get()
            predict[features[0]] = [a]

            b = dataeyeshape.get()
            predict[features[1]] = [b]

            c = dataeyecolor.get()
            predict[features[2]] = [c]

            d = datanshape.get()
            predict[features[3]] = [d]

            e = datalshape.get()
            predict[features[4]] = [e]

            f = dataskincolor.get()
            predict[features[5]] = [f]

            g = datahtype.get()
            predict[features[6]] = [g]

            h = dataheight.get()
            predict[features[7]] = [h]

            i = dataweight.get()
            predict[features[8]] = [i]

            predictions = classifier.predict(input_fn=lambda: input_fn(predict))
            for pred_dict in predictions:
                class_id = pred_dict['class_ids'][0]
                probability = pred_dict['probabilities'][class_id]

            def getresult():
                resultwindow = Tk()
                resultwindow.title('Ethnicity Classification of Bangladeshi Inhabitants Using Deep Learning')
                resultwindow.geometry('400x600')

                labelresulttext = Label(resultwindow, text='Prediction Result: ', font=('Arial', 20))
                labelresulttext.grid(row=0, column=0, padx=10, pady=10)

                emptylabelresult = Label(resultwindow, font=('Arial', 14))
                emptylabelresult.grid(row=1, column=0, padx=10, pady=10)
                emptylabelresult.config(
                    text='Prediction is "{}" ({:.1f}%)'.format(ethnicity[class_id], 100 * probability))

            resultbutton = Button(inputwindow, command=getresult, text='Get Prediction', font=('Arial', 14))
            resultbutton.grid(row=10, column=1)

        else:
            messagebox.showerror("Invalid Input","Please Give Input as Per Instructions in The Interface")

    except:
        messagebox.showerror("Value Missing", "Please Fill All The Input Fields")



submitbutton = Button(window, command=inputdata, text='Submit Data', font=('Arial', 14))
submitbutton.grid(row=9, column=1)

window.mainloop()