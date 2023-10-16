import tensorflow_text
from flask import Flask, request, jsonify
from inference import Model
import tensorflow as tf

sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))  # Set to use GPU
app = Flask(__name__)
model = Model()


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    # Validate data based on existence and type
    if "text" not in data or type(data["text"]) is not str:
        return jsonify({"error": "'text' field missing or type is incorrect.'"}), 400
    text = data["text"]
    prediction = model.predict(text)

    return jsonify({"prediction": prediction}), 200


@app.route("/ping", methods=["GET"])
def ping():
    return jsonify({"message": "pong"})


if __name__ == "__main__":
    model.load()
    app.run()