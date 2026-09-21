from flask import Flask
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# PRUEBA DE MODELO EfficientNetB0 - Billetes
from moon.v1_0.endpoint import moon_1_0
from sun.v1_0.endpoint import sun_1_0

app.register_blueprint(
    moon_1_0,
    url_prefix="/moon/1.0"
)

app.register_blueprint(
    sun_1_0,
    url_prefix="/sun/1.0"
)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
