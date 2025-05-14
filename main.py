import os
import csv
import cv2
import numpy as np
from datetime import datetime
from flask import Flask, request, render_template
from tensorflow.keras.models import load_model

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
RESULT_FILE = 'results.csv'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load model
model = load_model("model/mobilenet_model.h5")

# Dự đoán cho nhiều lớp
def predict_image(path):
    img = cv2.imread(path)
    img = cv2.resize(img, (224, 224))
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)
    pred = model.predict(img)[0]

    # Gán nhãn theo lớp có xác suất cao nhất
    class_names = [
        'Tomato Bacterial spot',
        'Tomato Early blight',
        'Tomato healthy',
        'Tomato Late blight',
        'Tomato Leaf Mold',
        'Tomato Septoria leaf spot',
        'Tomato Spider mites Two-spotted spider mite',
        'Tomato Target Spot',
        'Tomato Tomato mosaic virus',
        'Tomato Tomato Yellow Leaf Curl Virus'
    ]
    predicted_class = class_names[np.argmax(pred)]

    return predicted_class

def save_result(filename, prediction):
    file_exists = os.path.isfile(RESULT_FILE)
    with open(RESULT_FILE, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(["Thời gian", "Tên ảnh", "Kết quả"])
        writer.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), filename, prediction])

@app.route("/", methods=['GET'])
def index():
    return render_template("index.html")

@app.route("/predict", methods=['POST'])
def predict():
    file = request.files['image']
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    prediction = predict_image(filepath)
    save_result(file.filename, prediction)

    return render_template("index.html", prediction=prediction, image_url=filepath)

if __name__ == "__main__":
    app.run(debug=True)
