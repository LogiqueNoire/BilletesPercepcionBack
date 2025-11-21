from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# PRUEBA DE MODELO EfficientNetB0 - Billetes


# CARGAR MODELO
model_path = "EfficientNet_Billetes_FineTuned.keras"  # Ajusta si el nombre difiere
model = tf.keras.models.load_model(model_path)
print("Modelo cargado correctamente")

# Las clases deben coincidir con las del entrenamiento
class_names = [
    'billete100_anverso_antiguo',
    'billete100_anverso_nuevo',
    'billete100_reverso_antiguo',
    'billete10_anverso_antiguo',
    'billete10_reverso_antiguo',
    'billete20_anverso_nuevo',
    'billete20_reverso_nuevo',
    'billete50_reverso_nuevo'
]

img_size = (224, 224)
CONFIDENCE_THRESHOLD = 0.60  # Ajusta este valor (0.6 = 60%)
       

#=============       ENDPOINT       =============#
@app.route("/predict", methods=["POST"])
def predictSVM():
    if "file" not in request.files:
        return jsonify({"error": "No enviaste archivo"}), 400

    file = request.files["file"]

    try:
        # Cargar y preprocesar imagen
        img = image.load_img(file.stream, target_size=img_size)
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)

        # Predicción
        preds = model.predict(img_array)
        pred_idx = np.argmax(preds[0])
        confidence = preds[0][pred_idx]

        # Detección de "Desconocido"
        if confidence < CONFIDENCE_THRESHOLD:
            predicted_label = "Desconocido"
            print(f"Confianza baja ({confidence*100:.2f}%). No coincide con ninguna clase.")
        else:
            predicted_label = f"{class_names[pred_idx]} ({confidence*100:.2f}%)"

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


@app.route("/healthcheck")
def health():
    return {"status": "ok"}

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
