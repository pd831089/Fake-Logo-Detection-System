from flask import Flask, redirect, render_template, request, send_from_directory, url_for
import cv2
from keras.utils import load_img
from keras.models import load_model
from keras.utils import img_to_array
from PIL import Image
import numpy as np
import os


app = Flask(__name__)
model = load_model("static/models/fake_logo.h5")
fpath = 'static/images/testing_img.jpg'
classes =['fake (1)', 'fake (10)', 'fake (2)', 'fake (3)', 'fake (4)', 'fake (5)', 'fake (6)', 'fake (7)', 'fake (8)', 'fake (9)', 'real (1)', 'real (10)', 'real (2)', 'real (3)', 'real (4)', 'real (5)', 'real (6)', 'real (7)', 'real (8)', 'real (9)']


# index page
@app.route("/")
def hello_world():
    return render_template("index.html")

# for check the image |Testng
def check(res):
        p1 = classes
        path = p1
        pred = model.predict(res)
        res = np.argmax(pred)
        res = path[res]
        return (res)


def convert_img_to_tensor2(fpath):
    # ______________________________
    img = cv2.imread(fpath)
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite('static/images/gray.jpg', gray_image)
    # apply the Canny edge detection
    edges = cv2.Canny(img, 100, 200)
    cv2.imwrite('static/images/edges.jpg', edges)
    # apply thresholding to segment the image
    retval2, threshold2 = cv2.threshold(gray_image, 200, 255, cv2.THRESH_BINARY)
    cv2.imwrite('static/images/threshold.jpg', threshold2)
    # apply the sharpening kernel to the image
    kernel_sharpening = np.array([[-1, -1, -1],
                                  [-1, 9, -1],
                                  [-1, -1, -1]])
    sharpened = cv2.filter2D(img, -1, kernel_sharpening)
    cv2.imwrite('static/images/sharpened.jpg', sharpened)

    # ______________________________
    img = cv2.imread(fpath)
    img = cv2.resize(img, (256, 256))
    res = img_to_array(img)
    res = np.array(res, dtype=np.float16) / 255.0
    res = res.reshape(-1, 256, 256, 3)
    res = res.reshape(1, 256, 256, 3)
    return res


# prediction pagge
@app.route('/prediction', methods=['POST'])
def predictor():
    if request.method == 'POST':

        img = request.files['image']
        img.save(fpath)


        res = convert_img_to_tensor2(fpath)
        msg = check(res)
        msg=msg.split(" ")[0].upper()
        print(msg)

        return render_template('detector.html', data=[fpath, msg])


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0')
    # app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 1
