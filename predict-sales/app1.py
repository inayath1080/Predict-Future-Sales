# Refernce link: https://github.com/krishnaik06/Deployment-flask

import numpy as np
from flask import Flask, request,render_template
import pickle
import pandas as pd
from xgboost import  XGBRegressor

app = Flask(__name__)
model = pickle.load(open("xgb.dat", "rb"))
data=pd.read_csv('preprocessed_data.csv')
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    f1=int(request.form.get("SHOPID"))
    f2=int(request.form.get('ITEMID'))
    X=data.loc[(data['shop_id']==f1)&(data['item_id']==f2)]
    if X.empty:
        return render_template('index.html', prediction_text='Enter valid values. Shop:{} and item:{} are invalid'.format(f1,f2))
    else:
        X=X.drop(['ID','shop_id','item_id','b6','b0','b1','b2','b3','b4','b5'],axis=1)
        prediction = model.predict(X)

        output = round(prediction[0].clip(0,20))

        return render_template('index.html', prediction_text='Total sales for next month for shop:{},item:{} may be {}'.format(f1,f2,output))


if __name__ == "__main__":
    
    app.run(debug=True)