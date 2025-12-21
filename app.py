from flask import Flask, render_template, request
import joblib
import cv2
import numpy as np
from skimage.feature import hog
import os

app = Flask(__name__)

knn_class = joblib.load("knn_class_model.joblib")
knn_shape = joblib.load("knn_shape_model.joblib")
knn_color = joblib.load("knn_color_model.joblib")

UPLOAD_FOLDER = "static"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

def extract_hog(img_path, image_size=(64, 64)):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, image_size)
    features = hog(img,
                   orientations=9,
                   pixels_per_cell=(8, 8),
                   cells_per_block=(2, 2),
                   block_norm='L2-Hys')
    return features

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files["file"]
        if file:
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(filepath)

            # Trích xuất đặc trưng HOG
            feat = extract_hog(filepath).reshape(1, -1)

            # Dự đoán bằng KNN
            pred_class = knn_class.predict(feat)[0]
            pred_shape = knn_shape.predict(feat)[0]
            pred_color = knn_color.predict(feat)[0]

            # Trả về template cùng kết quả
            return render_template("index.html",
                                   filename=file.filename,
                                   pred_class=pred_class,
                                   pred_shape=pred_shape,
                                   pred_color=pred_color,
                                   class_name=f"Biển số {pred_class}")
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
