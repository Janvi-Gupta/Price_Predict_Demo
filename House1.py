import numpy as np
import pickle
from random import random
from flask import Flask, render_template, request

app = Flask(__name__)

model = pickle.load(open('model.pkl','rb'))

@app.route('/')
def House():
    return render_template('House.html')

@app.route('/Price_Predict', methods=['POST'])
def Price_Predict():
    try:
        def predict(predict_list):
            to_predict = np.array(predict_list).reshape(1, 12)
            result = model.predict(to_predict)
            return result[0]
        
        if request.method == 'POST':
            predict_list = request.form.to_dict()
            predict_list = list(predict_list.values())
            predict_list = list(map(int, predict_list))
            result = predict(predict_list)
            return render_template('House.html' , prediction_text = f'The price of House is: {int(result)}/-')
  
    except:
            return render_template('House.html' , prediction_text = "Please enter valid data!")

if __name__ == '__main__':
    app.run(debug=True)