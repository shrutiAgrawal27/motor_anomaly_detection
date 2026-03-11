from flask import Flask, request, jsonify
import numpy as np
from src.models.predict import detect_anomaly

app = Flask(__name__)


@app.route("/detect", methods=["POST"])
def detect():

    data = request.get_json()

    signal = np.array(data["signal"])

    result = detect_anomaly(signal)

    return jsonify({"status": result})


if __name__ == "__main__":
    app.run(debug=True)
