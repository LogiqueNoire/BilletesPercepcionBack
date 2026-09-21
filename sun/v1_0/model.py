import torch
from pathlib import Path

CONFIDENCE_THRESHOLD = 0.60  # Ajusta este valor (0.6 = 60%)

from .classes import GRUAutoencoder

hidden_size = 64

sun_1_0_path = Path(__file__).resolve().parent / "/sun/v1_0/EF_1300_0.91_2026-09-11_14-16-57.pth"

model_sun_1_0 = None
model_loaded = False
img_size = (224, 224)

try:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_sun_1_0 = GRUAutoencoder(
        feature_dim=64,
        hidden_dim=hidden_size
    ).to(device)

    model_sun_1_0.load_state_dict(
        torch.load(sun_1_0_path)
    )

    model_sun_1_0.eval()
except Exception as e:
    print(f"Error cargando el modelo Sun 1.0: {e}")


