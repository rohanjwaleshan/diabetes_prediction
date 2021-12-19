from flask import Flask, render_template, request
import pickle
import numpy as np

model = pickle.load(open('rf_model.pkl','rb'))

app = Flask(__name__)

@app.route('/')
def main():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def homepage():
    input1 = request.form['a']
    input2 = request.form['b']
    input3 = request.form['c']
    input4 = request.form['d']
    input5 = request.form['e']
    input6 = request.form['f']
    input7 = request.form['g']
    input8 = request.form['h']
    array = np.array([[input1,input2,input3,input4,input5,input6,input7,input8]])
    pred = model.predict(array)
    return render_template('pred.html',data=pred[0])
    
if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)
    

