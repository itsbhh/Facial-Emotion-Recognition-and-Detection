from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import base64
import tensorflow as tf
from flask_cors import CORS

app = Flask(__name__)
CORS(app, origins='*')

# Load the pre-trained model
model = tf.keras.models.load_model("emotiondetector.h5")
haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)

# Extract facial features
def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1, 48, 48, 1)
    return feature / 255.0

# Route to serve HTML file
@app.route('/')
def home():
    return render_template('index.html')

# Route for emotion prediction
@app.route('/predictemotion', methods=['POST'])
def predictemotion():
    data = request.get_json()
    if data is None:
        return jsonify({"error": "invalid request"}), 400

    # Decode the base64 image
    string_image = data['image'].split(',')[1]
    string_image = base64.b64decode(string_image)
    nparr_image = np.frombuffer(string_image, np.uint8)
    img = cv2.imdecode(nparr_image, cv2.IMREAD_GRAYSCALE)

    faces = face_cascade.detectMultiScale(img, 1.3, 5)
    for (p, q, r, s) in faces:
        image = img[q:q+s, p:p+r]
        image = cv2.resize(image, (48, 48))
        im = extract_features(image)
        pred = model.predict(im)
        labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}
        prediction_label = labels[pred.argmax()]
        return jsonify({"emotion": prediction_label, "x1": str(p), "y1": str(q), "x2": str(p+r), "y2": str(q+s)}), 200

    return jsonify({"emotion": "face not detected", "x1": "0", "y1": "0", "x2": "0", "y2": "0"}), 200

if __name__ == '__main__':
    app.run(debug=True)
