label_map = {
    0: 'None',
    1: 'Thumb_Up',
    2: 'Thumb_Down',
    3: 'Victory',
    4: 'Pointing_Up',
    5: 'Closed_Fist',
    6: 'Open_Palm',
    7: 'ILoveYou'
}
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical

# ----------------------------------------
# ✅ Load CSV
# ----------------------------------------

df = pd.read_csv('data/landmarks_filtered.csv', header=None)
print(f"[INFO] Loaded dataset shape: {df.shape}")

# Last column = labels
X = df.iloc[:, :-1].values  # landmark vectors
y = df.iloc[:, -1].values   # labels

print(f"[INFO] Label distribution: {np.bincount(y)}")

# One-hot encode labels with fixed-size output layer (8 classes)
num_classes = 8  # Always 8 gestures
y_cat = to_categorical(y, num_classes=num_classes)

# ----------------------------------------
# ✅ Train/test split
# ----------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y_cat, test_size=0.2, random_state=42, stratify=y)

print(f"[INFO] Train: {X_train.shape}, Test: {X_test.shape}")

# Optional: normalize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
np.savez('models/scaler_params.npz', mean=scaler.mean_, scale=scaler.scale_)
print("[INFO] Saved scaler parameters to models/scaler_params.npz")

# ----------------------------------------
# ✅ Build MLP model
# ----------------------------------------

model = models.Sequential()
model.add(layers.Input(shape=(63,)))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(8, activation='softmax'))

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# ----------------------------------------
# ✅ Compute class weights for imbalanced training
# ----------------------------------------
from sklearn.utils.class_weight import compute_class_weight

# Get the unique labels that are actually present
present_classes = np.unique(y)

# Compute weights only for present classes
class_weights_array = compute_class_weight(
    class_weight='balanced',
    classes=present_classes,
    y=y
)

# Map the weights to all 8 labels, with 1.0 as neutral default for missing ones
class_weights = {i: 1.0 for i in range(num_classes)}
for cls, weight in zip(present_classes, class_weights_array):
    class_weights[cls] = weight

# ----------------------------------------
# ✅ Train
# ----------------------------------------

history = model.fit(
    X_train, y_train,
    epochs=30,
    batch_size=32,
    validation_data=(X_test, y_test),
    class_weight=class_weights
)

# ----------------------------------------
# ✅ Evaluate
# ----------------------------------------

loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"[INFO] Test Accuracy: {acc:.3f}")

# ----------------------------------------
# ✅ Save weights for FPGA or later
# ----------------------------------------

model.save('models/gesture_mlp_model.h5')
print("[INFO] Model saved to models/gesture_mlp_model.h5")
