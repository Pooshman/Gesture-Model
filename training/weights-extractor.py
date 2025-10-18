from tensorflow.keras.models import load_model
import numpy as np

model = load_model('models/gesture_mlp_model.h5', compile=False)
print("[INFO] Loaded models/gesture_mlp_model.h5 for extraction")

# Loop over layers
for i, layer in enumerate(model.layers):
    weights = layer.get_weights()
    if weights:
        w, b = weights
        np.savetxt(f'models/layer_{i}_weights.csv', w, delimiter=',')
        np.savetxt(f'models/layer_{i}_bias.csv', b, delimiter=',')
        print(f"[INFO] Exported models/layer_{i}_weights.csv / models/layer_{i}_bias.csv")
