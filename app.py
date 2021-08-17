from flask import Flask, render_template, request, send_from_directory
import numpy as np
from keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model


model = load_model('/Users/deveshvishal/Document/Project/BrainTumor/model/model.h5')

app = Flask(__name__, static_folder="/Users/deveshvishal/Document/Project/BrainTumor/images/")

@app.route("/", methods=["GET"])
def hello():
    return render_template("index.html")

@app.route("/", methods = ["POST"])
def prediction():

    imagefile = request.files['imagefile']
    path = "/Users/deveshvishal/Document/Project/BrainTumor/images/" + imagefile.filename
    imagefile.save(path)

    img = load_img(path, target_size=(224, 224))
    input_arr = img_to_array(img) / 255

    input_arr = np.expand_dims(input_arr, axis=0)

    pred = model.predict(input_arr)[0][0]
    # classification = "Choose Image"
    if pred < 0.30:
        classification = "The MRI is having Tumour"
    else:
        classification = "The MRI is not having Tumour"

    return render_template("index.html", prediction= classification)

if __name__ == "__main__":
    app.run(debug= True)