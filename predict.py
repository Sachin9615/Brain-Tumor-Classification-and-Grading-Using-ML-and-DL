# predict.py
import tensorflow as tf
import numpy as np
from PIL import Image

model = keras.models.load_model('brain_tumor_model.h5')  # or .keras

class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']

def predict(image_path):
    img = Image.open(image_path).convert('RGB').resize((224, 224))
    img_array = np.expand_dims(np.array(img) / 255.0, axis=0)
    
    pred = model.predict(img_array)
    print(f"🧠 Result     : {class_names[np.argmax(pred)]}")
    print(f"📊 Confidence : {np.max(pred)*100:.2f}%")

predict('test_mri.jpg')  # 👈 your image here