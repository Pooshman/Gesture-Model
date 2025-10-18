# delete_labels.py

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

INPUT_FILE = 'landmarks.csv'
REMOVE_LABELS = {0,1,2,3,4,6,7}  # Labels to delete

# First, read and filter
filtered_lines = []
deleted, kept = 0, 0

with open(INPUT_FILE, 'r') as fin:
    for line in fin:
        label = int(line.strip().split(',')[-1])
        if label not in REMOVE_LABELS:
            filtered_lines.append(line)
            kept += 1
        else:
            deleted += 1

# Overwrite the original file
with open(INPUT_FILE, 'w') as fout:
    fout.writelines(filtered_lines)

print(f"[INFO] Done. Kept: {kept} rows, Deleted: {deleted} rows")
print(f"[INFO] Modified the file ({INPUT_FILE})")