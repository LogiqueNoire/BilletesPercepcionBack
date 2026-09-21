import tempfile

from flask import Flask, request, jsonify
from .functions import extract_frames
from .model import model_sun_1_0, device
import torch

from flask import Blueprint, request, jsonify

sun_1_0 = Blueprint("sun_1_0", __name__)
THRESHOLD = 3.961789e-06

@sun_1_0.route("/predict", methods=["POST"])
def predict_sun_1_0():
    if "file" not in request.files:
        return jsonify({
            "error": "No enviaste archivo"
        }), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({
            "error": "No seleccionaste ningún archivo"
        }), 400

    temp_path = None

    try:
        # Crear un archivo temporal con el video recibido
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".mp4"
        ) as temp_file:

            file.save(temp_file.name)
            temp_path = temp_file.name

        # Extraer frames del video recibido
        frames = extract_frames(temp_path)

        x = torch.tensor(frames, dtype=torch.float32).unsqueeze(0).to(device)

        x = x.permute(0, 1, 4, 2, 3)

        with torch.no_grad():
            reconstructed, original = model_sun_1_0(x)

            mse = (reconstructed - original).pow(2)
            score = mse.mean().item()

        print(f"Error de reconstrucción: {score:.6f}")

        if score > THRESHOLD:
            return "Billete falso"
        else:
            return "Billete normal"
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@sun_1_0.route("/healthcheck", methods=["GET"])
def health():
    return {"status": "ok"}

