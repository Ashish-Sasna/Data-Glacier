# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 14:08:45 2023

@author: Ashish S
"""

import os
import sys
import numpy as np
import pickle as pkl
from flask import Flask
from flask import request 
from flask import render_template

app = Flask(__name__)

# Load the model
model_path = os.path.join(os.getcwd(), 'model.pkl')
model = pkl.load(open(model_path, 'rb'))

# Route to the homepage
@app.route('/')
def home():
    
    return render_template('home.html')

# Prediction of inputs from the form
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    
    features = [float(x) for x in request.form.values()]
    features = [np.array(features)]
    result = model.predict(features)[0]
    result = round(result[0], 3)
    
    return render_template('home.html', prediction_result='Diabetes progression result is {}'.format(result))

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)