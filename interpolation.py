import numpy as np
import pandas as pd

def fill_missing_joints_linear(skeleton_sequence):
    """
    Given skeleton_sequence of shape (T, 2*J), fill missing (0,0) by linear interpolation.
    ...
    (Same code as shown before)
    """
    filled_sequence = skeleton_sequence.copy()
    num_frames, xy_dims = filled_sequence.shape
    num_joints = xy_dims // 2
    
    # Iterate over each joint
    for j in range(num_joints):
        x_vals = filled_sequence[:, 2*j]
        y_vals = filled_sequence[:, 2*j + 1]
        missing_mask = (x_vals == 0) & (y_vals == 0)

        # If all frames are missing, continue
        if missing_mask.all():
            continue

        # Indices
        valid_idx = np.where(~missing_mask)[0]
        invalid_idx = np.where(missing_mask)[0]

        # Clamp beginning & end (optional)
        first_valid, last_valid = valid_idx[0], valid_idx[-1]
        for idx in range(0, first_valid):
            x_vals[idx] = x_vals[first_valid]
            y_vals[idx] = y_vals[first_valid]
        for idx in range(last_valid + 1, num_frames):
            x_vals[idx] = x_vals[last_valid]
            y_vals[idx] = y_vals[last_valid]

        # Recompute missing_mask after clamping
        missing_mask = (x_vals == 0) & (y_vals == 0)
        valid_idx = np.where(~missing_mask)[0]
        invalid_idx = np.where(missing_mask)[0]

        # Linear interpolation
        x_vals[missing_mask] = np.interp(invalid_idx, valid_idx, x_vals[valid_idx])
        y_vals[missing_mask] = np.interp(invalid_idx, valid_idx, y_vals[valid_idx])

        # Put back
        filled_sequence[:, 2*j] = x_vals
        filled_sequence[:, 2*j + 1] = y_vals
    
    return filled_sequence

def main():
    csv_path = "/Users/fedor/Desktop/skeletal fedor/Avenue/testing/trajectories/01-01/00008.csv"  # <-- Adjust this!

    # 1) Read the CSV
    df = pd.read_csv(csv_path, header=None)

    # df shape is at least (N, 35) => 1 frame_index + 34 coords
    # We'll take the first 50 rows, columns [1..35)
    # if your CSV has exactly 35 columns total
    raw_sequence = df.iloc[:20, 1:].values  # shape: (50, 34)

    # Convert to float (just in case)
    raw_sequence = raw_sequence.astype(float)

    print("\n--- RAW (first 50 rows, skipping frame_idx) ---")
    print(raw_sequence)

    # 2) Call fill_missing_joints_linear
    filled_data = fill_missing_joints_linear(raw_sequence)

    print("\n--- AFTER FILLING MISSING POINTS ---")
    print(filled_data)

if __name__ == "__main__":
    main()