from io import BytesIO
import tensorflow as tf
from flask import Flask, request, jsonify
from tensorflow.keras.preprocessing import image
import numpy as np
from .model import model_moon_1_0, CONFIDENCE_THRESHOLD, class_names, img_size

from flask import Blueprint, request, jsonify

moon_1_0 = Blueprint("moon_1_0", __name__)

@moon_1_0.route("/predict", methods=["POST"])
def predict_moon_1_0():
    if "file" not in request.files:
        return jsonify({"error": "No enviaste archivo"}), 400

    file = request.files["file"]

    try:
        # Cargar y preprocesar imagen
        img_bytes = BytesIO(file.read())

        # Cargar imagen desde BytesIO
        img = image.load_img(img_bytes, target_size=img_size)
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)

        # Predicción
        preds = model_moon_1_0.predict(img_array)
        pred_idx = np.argmax(preds[0])
        confidence = float(preds[0][pred_idx])

        # Detección de "Desconocido"
        if confidence < CONFIDENCE_THRESHOLD:
            predicted_label = "Desconocido"
            print(f"Confianza baja ({confidence*100:.2f}%). No coincide con ninguna clase.")
        else:
            predicted_label = f"{class_names[pred_idx]}"

        # Mostrar probabilidades detalladas
        print("\nProbabilidades:")
        for i, c in enumerate(class_names):
            print(f"{c:30s}: {preds[0][i]*100:.2f}%")

        return jsonify({
            "predicted_label": predicted_label,
            "confidence": confidence
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@moon_1_0.route("/healthcheck", methods=["GET"])
def health():
    return {"status": "ok"}
