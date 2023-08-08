import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm
import tkinter as tk    
from tkinter import filedialog


def open_file():
    global filepath
    filepath = filedialog.askopenfilename()
    print(filepath)

root = tk.Tk()
button = tk.Button(root, text="Open file", command=open_file)
button.pack()
root.mainloop()

spam = pd.read_csv(filepath)

y = spam["Label"]
z = spam['EmailText']
z_train, z_test,y_train, y_test = train_test_split(z,y,test_size = 0.2)

cv = CountVectorizer()
features = cv.fit_transform(z_train)

model = svm.SVC()
model.fit(features,y_train)

features_test = cv.transform(z_test)
print("Accuarcy: ",model.score(features_test,y_test))
