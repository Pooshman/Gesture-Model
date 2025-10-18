# count_labels.py

import pandas as pd

CSV_FILE = 'landmarks_filtered.csv'

df = pd.read_csv(CSV_FILE, header=None)
label_counts = df.iloc[:, -1].value_counts().sort_index()

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

print(f"For file {CSV_FILE}:")
print("[INFO] Sample count per label index:\n")
for label, count in label_counts.items():
    gesture_name = label_map.get(label, "Unknown")
    print(f"Label {int(label)} ({gesture_name}): {count} samples")