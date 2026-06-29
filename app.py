from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import os

app = Flask(__name__)

model = load_model("models/skin_disease_model.h5")

classes = [
    "bkl",
    "nv",
    "df",
    "mel",
    "vasc",
    "bcc",
    "akiec",
]

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = ""

    if request.method == "POST":
        file = request.files["image"]

        if file:
            path = "temp.jpg"
            file.save(path)

            img = cv2.imread(path)
            img = cv2.resize(img, (64, 64))
            img = img / 255.0
            img = np.expand_dims(img, axis=0)

            pred = model.predict(img)
            print(pred)
            print(np.argmax(pred))
            prediction = classes[np.argmax(pred)]

            os.remove(path)

    return f"""
    <h2>AI Skin Disease Detection</h2>
    <form method='POST' enctype='multipart/form-data'>
        <input type='file' name='image'>
        <input type='submit' value='Predict'>
    </form>
    <h3>{prediction}</h3>
    """

if __name__ == "__main__":
    app.run(debug=True)