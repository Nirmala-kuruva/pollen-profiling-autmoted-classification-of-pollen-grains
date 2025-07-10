from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np


model = load_model('C:/Users/Nirma/Downloads/model.h5')


img_path = 'C:/Users/Nirma/Downloads/pollen_data/eucalipto_23.jpg'
img = load_img(img_path, color_mode='grayscale', target_size=(128, 128))  
x = img_to_array(img) / 255.0
x = np.expand_dims(x, axis=0)  
print("Image shape:", x.shape)  

probs = model.predict(x)[0]
predicted_index = np.argmax(probs)


labels = [
    'anadenanthera', 'arecaceae', 'arrabidaea', 'cecropia', 'chromolaena',
    'combretum', 'croton', 'dipteryx', 'eucalipto', 'faramea', 'hyptis', 'mabea',
    'matayba', 'mimosa', 'myrcia', 'protium', 'qualea', 'schinus',
    'senegalia', 'serjania', 'syagrus', 'tridax', 'urochloa'
]

# === Output ===
print("Prediction probabilities:", probs)  
print("Predicted label:", labels[predicted_index])
