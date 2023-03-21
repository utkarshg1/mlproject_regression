from flask import Flask, request, render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler

application = Flask(__name__)

app = application

@app.route('/')
def home_page():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
