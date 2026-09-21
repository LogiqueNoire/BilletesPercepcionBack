import tensorflow as tf
from pathlib import Path

CONFIDENCE_THRESHOLD = 0.60  # Ajusta este valor (0.6 = 60%)

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

moon_1_0_path = Path(__file__).resolve().parent / "EfficientNet_Billetes_FineTuned.keras"

model_moon_1_0 = None
model_loaded = False
img_size = (224, 224)

try:
    model_moon_1_0 = tf.keras.models.load_model(moon_1_0_path)
    model_loaded = True
    print("Modelo Moon 1.0 cargado correctamente")
except Exception as e:
    print(f"Error cargando el modelo Moon 1.0: {e}")