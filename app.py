from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os

app = Flask(__name__)
model = load_model('cnn.weight.h5')


class_names = ['urochloa', 'brachiaria', 'panicum', 'setaria']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/prediction.html')
def predict_page():
    return render_template('prediction.html', result='')

@app.route('/logout.html')
def logout():
    return render_template('logout.html')

@app.route('/result', methods=['POST'])
def result():
    if 'file' not in request.files:
        return 'No file uploaded.', 400

    file = request.files['file']
    if file.filename == '':
        return 'No selected file.', 400

    filepath = os.path.join('static', file.filename)
    file.save(filepath)

   
    img = load_img(filepath, target_size=(128, 128)) 
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0) / 255.0

    prediction = model.predict(x)
    predicted_class = class_names[np.argmax(prediction)]

    return render_template('prediction.html', result=predicted_class)

