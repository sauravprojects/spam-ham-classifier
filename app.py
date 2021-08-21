# -*- coding: utf-8 -*-
"""
Created on Sat Aug 21 16:43:23 2021

@author: Saurav
"""

from flask import Flask,render_template,url_for,request
import pickle
import joblib

pickle_in = open("classifier.pkl","rb")
classifier=pickle.load(pickle_in)

pickle_in1 = open("Vectorizer.pkl","rb")
transform_model = pickle.load(pickle_in1)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        data = [message]
        num_vector = transform_model.transform(data).toarray()
        my_prediction = classifier.predict(num_vector)
    return render_template('result.html',prediction = my_prediction)

if __name__=="__main__":
    app.run()

    