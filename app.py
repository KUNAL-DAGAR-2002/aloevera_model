from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np

app = Flask(__name__)

#class names
class_names = ['healthy_leaf', 'Rot', 'Rust']
model = tf.keras.models.load_model("models/1")
image_size = 375

def predict_label(img_path):
    image = tf.keras.preprocessing.image.load_img(img_path,target_size=(image_size,image_size))
    image = np.array(image)
    image = image.reshape(1,image_size,image_size,3)
    p = model.predict(image)
    

    return  class_names[np.argmax(p[0])]
        
def confi_label(img_path):
    image = tf.keras.preprocessing.image.load_img(img_path,target_size=(image_size,image_size))
    image = np.array(image)
    image = image.reshape(1,image_size,image_size,3)
    a = model.predict(image)
    

    return "{:.2f}".format((np.max(a[0]))*100)
    

@app.route("/")
def main():
    return render_template("index.html")


@app.route("/submit", methods=['POST'])
def get_output():
    if request.method == 'POST':
        image = request.files['my_image']
        img_path = "static/" + image.filename
        image.save(img_path)
        p = predict_label(img_path)
        a = confi_label(img_path)

    return render_template("index.html", prediction = p ,confidence = a, img_path = img_path)

if __name__ == "__main__":
    app.run(debug=True)