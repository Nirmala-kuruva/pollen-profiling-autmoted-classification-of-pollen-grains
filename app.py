from flask import Flask, render_template, request, url_for
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os
import tensorflow as tf

app = Flask(__name__)
model = tf.keras.models.load_model('pollen_model.h5')

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/prediction', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        f = request.files.get('pc_image')
        if not f:
            return "No file uploaded", 400

        img_path = "static/uploads/" + f.filename
        f.save(img_path)

        
        img = load_img(img_path, color_mode='grayscale', target_size=(128, 128))
        image_array = img_to_array(img)
        image_array = image_array / 255.0  # Normalize
        image_array = np.expand_dims(image_array, axis=0)


        predictions = model.predict(image_array)
        print("Raw predictions:", predictions)

        pred = np.argmax(predictions, axis=1)


        pred = np.argmax(model.predict(image_array), axis=1)

        class_names = ['anadenanthera', 'arecaceae', 'arrabidaea' ,'cecropia', 'chromolaena',
         'combretum', 'croton', 'dipteryx' ,'eucalipto', 'eugenia', 'handroanthus',
         'invasoras' ,'laubertia', 'melastomataceae', 'miconia' ,'mimosa', 'myrcia',
         'myrtaceae', 'psidium' ,'senna' ,'tapirira', 'urochloa', 'vitex']
        prediction = class_names[int(pred)]
        img_filename = f.filename
        img_url = url_for('static', filename=f'uploads/{img_filename}')

        return render_template('logout.html', prediction=prediction, img_url=img_url)


    
    return render_template('predict.html')

if __name__ == '__main__':
    app.run(debug=True, port=2222)
