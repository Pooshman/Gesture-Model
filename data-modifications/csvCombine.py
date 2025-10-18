import pandas as pd

# File paths
source_csv = 'landmarks.csv'     # CSV to append from
target_csv = 'landmarks_filtered.csv'      # CSV to append into (will be overwritten)

# Load both CSVs
df_source = pd.read_csv(source_csv, header=None)
df_target = pd.read_csv(target_csv, header=None)

# Concatenate
df_combined = pd.concat([df_target, df_source], ignore_index=True)

# Overwrite the target CSV
df_combined.to_csv(target_csv, index=False, header=False)

print(f"[INFO] Appended {len(df_source)} rows from '{source_csv}' to '{target_csv}'.")