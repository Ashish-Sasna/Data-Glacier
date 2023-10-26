# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 14:08:45 2023

@author: Ashish S
"""

import os
import numpy as np
import pickle as pkl
from flask import Flask
from flask import request 
from flask import render_template

app = Flask(__name__)

#os.chdir('..')
model = pkl.load(open('..\model.pkl', 'rb'))

@app.route('/')
def home():
    
    return render_template('home.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    
    features = [float(x) for x in request.form.values()]
    features = [np.array(features)]
    result = model.predict(features)[0]
    
    return render_template('../Template/home.html', **locals)

if __name__ == "__main__":
    app.run(debug=True)