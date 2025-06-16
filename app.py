from flask import Flask, render_template, request
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import io
import base64

app = Flask(__name__)

# Carga tu modelo (ajusta la ruta)
model = load_model('C:/Users/luis1/OneDrive/Documentos/Procesamiento de imagenes/codigos/detector-soldaduras/soldadura_model.h5')

def prepare_image(image, target_size=(150,150)):
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(target_size)
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)  # Shape: (1, H, W, 3)
    return image

def pil_image_to_base64(img):
    # Convertir imagen PIL a base64 para mostrar en HTML
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files.get('image')
    if not file:
        return render_template('index.html', error="No se seleccionó ninguna imagen.")

    img = Image.open(file)
    img_prepared = prepare_image(img)

    pred = model.predict(img_prepared)[0][0]
    resultado = 'Buena soldadura' if pred > 0.5 else 'Soldadura mala'

    # Convertir imagen original a base64 para mostrarla
    img_base64 = pil_image_to_base64(img)

    return render_template('result.html', resultado=resultado, img_data=img_base64)

@app.route('/quienes-somos')
def quienes_somos():
    return render_template('quienes_somos.html')

if __name__ == '__main__':
    app.run(debug=True)
