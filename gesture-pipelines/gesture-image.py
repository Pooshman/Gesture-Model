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
# ✅ Download test images
# -----------------------------------------

IMAGE_FILENAMES = ['thumbs_down.jpg', 'victory.jpg', 'thumbs_up.jpg', 'pointing_up.jpg']

for name in IMAGE_FILENAMES:
    url = f'https://storage.googleapis.com/mediapipe-tasks/gesture_recognizer/{name}'
    urllib.request.urlretrieve(url, name)

# -----------------------------------------
# ✅ Preview the images locally with OpenCV
# -----------------------------------------

DESIRED_HEIGHT = 480
DESIRED_WIDTH = 480

def resize_and_show(image):
    """Resize image to a fixed dimension and show using OpenCV."""
    h, w = image.shape[:2]
    if h < w:
        img = cv2.resize(image, (DESIRED_WIDTH, math.floor(h / (w / DESIRED_WIDTH))))
    else:
        img = cv2.resize(image, (math.floor(w / (h / DESIRED_HEIGHT)), DESIRED_HEIGHT))

    cv2.imshow('Image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Preview images
images = {name: cv2.imread(name) for name in IMAGE_FILENAMES}
for name, image in images.items():
    print(f"Previewing: {name}")
    resize_and_show(image)

# -----------------------------------------
# ✅ Run the Gesture Recognizer
# -----------------------------------------

# STEP 1: Configure the recognizer
base_options = python.BaseOptions(model_asset_path='models/gesture_recognizer.task')
options = vision.GestureRecognizerOptions(base_options=base_options)
recognizer = vision.GestureRecognizer.create_from_options(options)

# STEP 2: Run inference and collect results
images_for_result = []
results = []

for image_file_name in IMAGE_FILENAMES:
    # Load image
    image = mp.Image.create_from_file(image_file_name)

    # Run recognition
    recognition_result = recognizer.recognize(image)

    # Collect results
    images_for_result.append(image)
    top_gesture = recognition_result.gestures[0][0]
    hand_landmarks = recognition_result.hand_landmarks
    results.append((top_gesture, hand_landmarks))

# -----------------------------------------
# ✅ Visualize all results together
# -----------------------------------------

display_batch_of_images_with_gestures_and_hand_landmarks(images_for_result, results)