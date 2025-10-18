#@title Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# We're using Mediapipe
# This model can recognize 7 hand gestures: 👍, 👎, ✌️, ☝️, ✊, 👋, 🤟

# -----------------------------------------
# ✅ Imports
# -----------------------------------------

import urllib.request
import cv2
import math
import mediapipe as mp
from matplotlib import pyplot as plt
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import struct


def load_quantized_weights_bin(path):
    with open(path, 'rb') as f:
        buf = f.read()
    offset = 0
    def read_array():
        nonlocal offset
        length = struct.unpack_from('I', buf, offset)[0]
        offset += 4
        arr = struct.unpack_from(f'{length}f', buf, offset)
        offset += 4 * length
        return np.array(arr, dtype=np.float32)

    w0 = read_array().reshape(63, 128)
    b0 = read_array().reshape(128,)
    w1 = read_array().reshape(128, 64)
    b1 = read_array().reshape(64,)
    w2 = read_array().reshape(64, 8)
    b2 = read_array().reshape(8,)
    return w0, b0, w1, b1, w2, b2


def load_scaler_params(path):
    data = np.load(path)
    return data['mean'], data['scale']


def relu(x):
    return np.maximum(0, x)

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def run_mlp(x, w0, b0, w1, b1, w2, b2):
    x = relu(np.dot(x, w0) + b0)
    x = relu(np.dot(x, w1) + b1)
    x = np.dot(x, w2) + b2
    return softmax(x)

# -----------------------------------------
# ✅ Matplotlib + Landmark Drawing Setup
# -----------------------------------------

# Clean up matplotlib plot style
plt.rcParams.update({
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.spines.left': False,
    'axes.spines.bottom': False,
    'xtick.labelbottom': False,
    'xtick.bottom': False,
    'ytick.labelleft': False,
    'ytick.left': False,
    'xtick.labeltop': False,
    'xtick.top': False,
    'ytick.labelright': False,
    'ytick.right': False
})

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

def display_one_image(image, title, subplot, titlesize=16):
    """Displays one image along with the predicted category name and score."""
    plt.subplot(*subplot)
    plt.imshow(image)
    if len(title) > 0:
        plt.title(
            title,
            fontsize=int(titlesize),
            color='black',
            fontdict={'verticalalignment': 'center'},
            pad=int(titlesize/1.5)
        )
    return (subplot[0], subplot[1], subplot[2]+1)


def display_batch_of_images_with_gestures_and_hand_landmarks(images, results):
    """Displays a batch of images with the gesture category and its score along with the hand landmarks."""
    # Convert images to numpy view if needed.
    images = [image.numpy_view() for image in images]
    gestures = [top_gesture for (top_gesture, _) in results]
    multi_hand_landmarks_list = [multi_hand_landmarks for (_, multi_hand_landmarks) in results]

    # Auto-squaring: drop excess if grid is not square-ish.
    rows = int(math.sqrt(len(images)))
    cols = len(images) // rows

    # Figure size and spacing.
    FIGSIZE = 13.0
    SPACING = 0.1
    subplot = (rows, cols, 1)
    if rows < cols:
        plt.figure(figsize=(FIGSIZE, FIGSIZE/cols*rows))
    else:
        plt.figure(figsize=(FIGSIZE/rows*cols, FIGSIZE))

    # Loop over batch and draw.
    for i, (image, gesture) in enumerate(zip(images[:rows*cols], gestures[:rows*cols])):
        title = f"{gesture.category_name} ({gesture.score:.2f})"
        dynamic_titlesize = FIGSIZE * SPACING / max(rows, cols) * 40 + 3

        # Convert RGB → BGR for OpenCV drawing
        annotated_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        for hand_landmarks in multi_hand_landmarks_list[i]:
            hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            hand_landmarks_proto.landmark.extend([
                landmark_pb2.NormalizedLandmark(
                    x=landmark.x, y=landmark.y, z=landmark.z
                ) for landmark in hand_landmarks
            ])

            mp_drawing.draw_landmarks(
                annotated_image,
                hand_landmarks_proto,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

        subplot = display_one_image(annotated_image, title, subplot, titlesize=dynamic_titlesize)

    plt.tight_layout()
    plt.subplots_adjust(wspace=SPACING, hspace=SPACING)
    plt.show()

# -----------------------------------------
# ✅ Run the Gesture Recognizer
# -----------------------------------------

# STEP 1: Configure the recognizer
base_options = python.BaseOptions(model_asset_path='gesture_recognizer.task')
options = vision.GestureRecognizerOptions(base_options=base_options)
recognizer = vision.GestureRecognizer.create_from_options(options)


#FOR WEBCAM:
import time

w0, b0, w1, b1, w2, b2 = load_quantized_weights_bin('quantized_weights.bin')
try:
    scaler_mean, scaler_scale = load_scaler_params('scaler_params.npz')
    scaler_scale = np.maximum(scaler_scale, 1e-6)
except FileNotFoundError:
    scaler_mean = np.zeros(63, dtype=np.float32)
    scaler_scale = np.ones(63, dtype=np.float32)
    print("[WARN] scaler_params.npz not found. Falling back to per-sample normalization.")
GESTURE_LABELS = [
    'None',         # 0
    'Thumb_Up',     # 1
    'Thumb_Down',   # 2
    'Victory',      # 3
    'Pointing_Up',  # 4
    'Closed_Fist',  # 5
    'Open_Palm',    # 6
    'ILoveYou'      # 7
]

# Open webcam
cap = cv2.VideoCapture(0)  # 0 = default Mac webcam

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip BGR → RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Wrap as MediaPipe Image
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )

    # Run recognition
    recognition_result = recognizer.recognize(mp_image)

    # Draw landmarks on frame
    if recognition_result.hand_landmarks:
        for hand_landmarks in recognition_result.hand_landmarks:
            hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            hand_landmarks_proto.landmark.extend([
                landmark_pb2.NormalizedLandmark(
                    x=landmark.x, y=landmark.y, z=landmark.z
                ) for landmark in hand_landmarks
            ])
            mp_drawing.draw_landmarks(
                frame,  # ← Draw directly on BGR webcam frame
                hand_landmarks_proto,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

    if recognition_result.hand_landmarks:
        hand_landmarks = recognition_result.hand_landmarks[0]
        x_input = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks]).flatten()

        # Normalize using training scaler statistics
        x_input = (x_input - scaler_mean) / scaler_scale

        probs = run_mlp(x_input, w0, b0, w1, b1, w2, b2)
        gesture_id = np.argmax(probs)
        gesture_name = GESTURE_LABELS[gesture_id]
        gesture_score = probs[gesture_id]

        print(f"[MLP] Recognized: {gesture_name} ({gesture_score:.2f})")
        cv2.putText(frame, f"{gesture_name} ({gesture_score:.2f})", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow("Webcam Feed", frame)

    # Quit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
