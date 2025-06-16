import os
import gdown
from flask import Flask, render_template, request
from keras.models import load_model
from keras.preprocessing import image
import numpy as np

app = Flask(__name__)

model_path = 'soldadura_model.h5'
if not os.path.exists(model_path):
    print("Descargando modelo desde Google Drive...")
    file_id = '1d5KflP43wrcwyqExbpSiJOzqYVDkzai7'
    gdown.download(f'https://drive.google.com/uc?id={file_id}', model_path, quiet=False)

model = load_model(model_path)

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    image_path = None

    if request.method == 'POST':
        f = request.files['file']
        file_path = os.path.join('static', f.filename)
        f.save(file_path)

        img = image.load_img(file_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        prediction = model.predict(img_array)
        result = "Soldadura BUENA ✅" if prediction[0][0] > 0.5 else "Soldadura DEFECTUOSA ❌"
        image_path = file_path

    return render_template('index.html', result=result, image_path=image_path)

if __name__ == '__main__':
    app.run(debug=True)
