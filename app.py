# from xgboost.sklearn import XGBClassifier
import pickle
# from sklearn.externals import joblib
from sklearn.ensemble import RandomForestClassifier
# import xgboost as xgb
import pandas as pd
import jinja2
from flask import Flask, request, render_template
import numpy as np




# import joblib


app = Flask(__name__)
model = pickle.load(open('cancerpkl.pickle', 'rb'))


# model = XGBClassifier.Booster({'nthread':4})
# model.load_model('new.pkl')


@app.route('/')
def home():
    return render_template('index1.html')


@app.route('/predict', methods=['POST'])
def predict():
    input_features = [float(x) for x in request.form.values()]
    features_value = [np.array(input_features)]

    features_name = ['radius', 'texture', 'perimeter', 'area',
                     'smoothness', 'compactness', 'concavity',
                     'concave_points', 'symmetry', 'fractal_dimension']

    df = pd.DataFrame(features_value, columns=features_name)
    output = model.predict(df)

    if output == 0:
        res_val = "** Cancer Detected **"
    else:
        res_val = "No Cancer Detected"

    return render_template('index1.html', prediction_text='In Patient data {}'.format(res_val))

if __name__ == "__main__":
    #     app.debug = True
    app.run()

