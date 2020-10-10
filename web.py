from flask import Flask, render_template, url_for, request
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import numpy as np

from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
import pickle

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods = ['POST'])
def predict():
    model = pickle.load(open('SupportVectorMachines.pkl', 'rb'))
    text_query = request.form['textquery']
    prediction = model.predict([text_query])
    prediction_proba = model.predict_proba([text_query])
    probability = np.max(prediction_proba)
    return render_template('result.html', prediction = prediction, pred = probability)




if __name__ == '__main__':
    app.run(debug = True)
