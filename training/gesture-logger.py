import cv2
import csv
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2


# Map label indices to gesture names
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

# ----------------------------------------
# ✅ Setup recognizer
# ----------------------------------------

base_options = python.BaseOptions(model_asset_path='gesture_recognizer.task')
options = vision.GestureRecognizerOptions(base_options=base_options)
recognizer = vision.GestureRecognizer.create_from_options(options)

# ----------------------------------------
# ✅ Open webcam + CSV
# ----------------------------------------

cap = cv2.VideoCapture(0)
csv_file = open('landmarks.csv', 'a', newline='')
csv_writer = csv.writer(csv_file)

print("[INFO] Auto-logger started.")
print("Press 'q' to stop logging and close webcam.")

frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )

    recognition_result = recognizer.recognize(mp_image)

    if recognition_result.hand_landmarks:
        vector = []
        for landmark in recognition_result.hand_landmarks[0]:
            vector.extend([landmark.x, landmark.y, landmark.z])

        # Use the model's best guess or None
        if recognition_result.gestures:
            gesture_name = recognition_result.gestures[0][0].category_name
        else:
            gesture_name = 'None'

        # Map to integer
        gesture_map = {
            'Thumb_Up': 1,
            'Thumb_Down': 2,
            'Victory': 3,
            'Pointing_Up': 4,
            'Closed_Fist': 5,
            'Open_Palm': 6,
            'ILoveYou': 7,
            'None': 0
        }

        label_index = gesture_map.get(gesture_name, 0)
        row = vector + [label_index]
        csv_writer.writerow(row)

        frame_count += 1
        print(f"[LOGGED] Frame: {frame_count} | Label: {label_index} | {gesture_name}")

        # Visual feedback
        cv2.putText(frame, f"{gesture_name}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

        # Draw landmarks
        hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        hand_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(
                x=landmark.x, y=landmark.y, z=landmark.z
            ) for landmark in recognition_result.hand_landmarks[0]
        ])
        mp.solutions.drawing_utils.draw_landmarks(
            frame,
            hand_landmarks_proto,
            mp.solutions.hands.HAND_CONNECTIONS
        )

    cv2.imshow("Auto Logger Feed", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
csv_file.close()
cv2.destroyAllWindows()
print(f"[INFO] Logging done. Frames saved: {frame_count}. File: landmarks.csv")